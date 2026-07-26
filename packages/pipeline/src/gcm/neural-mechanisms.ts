/**
 * Neural Causal Mechanisms — FFN-based nonlinear mechanisms for SCM.
 *
 * Extends StructuralCausalModel with:
 *   1. FFNMechanism — 2-layer feedforward network for nonlinear X_i = f(pa(X_i)) + ε
 *   2. EnsembleMechanism — weighted ensemble of ANM + FFN for robust fitting
 *
 * The FFN is intentionally small (2 hidden layers, ≤16 neurons) to fit in
 * production SCM pipelines without heavy ML dependencies. Training uses the
 * Adam optimizer from @agentix-e/causality-analyzer-core.
 *
 * References:
 *   - Pawlowski et al. (2020). "SCM with Neural Mechanisms." NeurIPS.
 *   - Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization."
 *
 * @packageDocumentation
 */
import { adam } from '@agentix-e/causality-analyzer-core';

// ── Types ────────────────────────────────────────────────────────────

export interface NeuralMechanismConfig {
  /** Hidden layer sizes (default: [8, 4]) */
  hiddenLayers?: number[];
  /** Learning rate (default: 0.01) */
  learningRate?: number;
  /** Max training iterations (default: 500) */
  maxIter?: number;
  /** L2 regularization (default: 1e-4) */
  l2Reg?: number;
  /** Convergence tolerance on gradient norm (default: 1e-5) */
  gtol?: number;
}

/** Trained FFN mechanism: X = FFN(parents) + ε */
export class FFNMechanism {
  readonly nodeName: string;
  /** Layer weights [W1, W2, W3] where W3 maps to scalar output */
  private weights: Float64Array[];
  private biases: Float64Array[];
  readonly noiseStd: number;
  private readonly inputDim: number;

  constructor(
    nodeName: string,
    weights: Float64Array[],
    biases: Float64Array[],
    noiseStd: number,
    inputDim: number,
  ) {
    this.nodeName = nodeName;
    this.weights = weights;
    this.biases = biases;
    this.noiseStd = noiseStd;
    this.inputDim = inputDim;
  }

  /** Forward pass: E[X | parents] */
  forward(parentValues: number[]): number {
    return forwardPass(
      new Float64Array(parentValues.slice(0, this.inputDim)),
      this.weights,
      this.biases,
    );
  }

  /** Inverse: recover noise from observation */
  invert(x: number, parentValues: number[]): number {
    return x - this.forward(parentValues);
  }

  /** Serialize for persistence */
  toJSON(): {
    nodeName: string;
    weights: number[][];
    biases: number[][];
    noiseStd: number;
    inputDim: number;
  } {
    return {
      nodeName: this.nodeName,
      weights: this.weights.map(w => Array.from(w)),
      biases: this.biases.map(b => Array.from(b)),
      noiseStd: this.noiseStd,
      inputDim: this.inputDim,
    };
  }

  static fromJSON(json: {
    nodeName: string;
    weights: number[][];
    biases: number[][];
    noiseStd: number;
    inputDim: number;
  }): FFNMechanism {
    return new FFNMechanism(
      json.nodeName,
      json.weights.map(w => new Float64Array(w)),
      json.biases.map(b => new Float64Array(b)),
      json.noiseStd,
      json.inputDim,
    );
  }
}

// ── FFN Training ─────────────────────────────────────────────────────

/**
 * Train a FFN mechanism for a causal node.
 *
 * Architecture: input_dim → hidden[0] → hidden[1] → 1 (scalar output)
 * Activation: ReLU for hidden layers, identity for output
 * Loss: MSE + L2 regularization
 *
 * @param X — parent values (n × d matrix as flat Float64Array, row-major)
 * @param y — target values (length n)
 * @param n — number of samples
 * @param inputDim — number of parent variables
 * @param nodeName — node identifier
 * @param config — training configuration
 */
export function trainFFNMechanism(
  X: Float64Array,
  y: Float64Array,
  n: number,
  inputDim: number,
  nodeName: string,
  config: NeuralMechanismConfig = {},
): FFNMechanism {
  const hiddenLayers = config.hiddenLayers ?? [8, 4];
  const lr = config.learningRate ?? 0.01;
  const maxIter = config.maxIter ?? 500;
  const l2Reg = config.l2Reg ?? 1e-4;
  const gtol = config.gtol ?? 1e-5;

  // Edge case: no parents — use mean prediction
  if (inputDim === 0) {
    const mean = y.reduce((a, b) => a + b, 0) / Math.max(1, n);
    let ss = 0;
    for (let i = 0; i < n; i++) ss += (y[i]! - mean) ** 2;
    const std = Math.sqrt(ss / Math.max(1, n - 1)) || 1;
    const w = new Float64Array([0]);
    const b = new Float64Array([mean]);
    return new FFNMechanism(nodeName, [w], [b], std, 0);
  }

  // Edge case: small samples → use mean prediction
  if (n < inputDim * 4) {
    const mean = y.reduce((a, b) => a + b, 0) / Math.max(1, n);
    let ss = 0;
    for (let i = 0; i < n; i++) ss += (y[i]! - mean) ** 2;
    const std = Math.sqrt(ss / Math.max(1, n - 1)) || 1;
    // Output = mean regardless of input
    const w = new Float64Array(inputDim);
    const b = new Float64Array([mean]);
    return new FFNMechanism(nodeName, [w, new Float64Array(1)], [new Float64Array(1), b], std, inputDim);
  }

  // Initialize architecture: [input→h1, h1→h2, h2→1]
  const layerSizes = [inputDim, ...hiddenLayers, 1];
  const weights: Float64Array[] = [];
  const biases: Float64Array[] = [];

  // Xavier initialization
  for (let l = 0; l < layerSizes.length - 1; l++) {
    const fanIn = layerSizes[l]!;
    const fanOut = layerSizes[l + 1]!;
    const scale = Math.sqrt(2 / (fanIn + fanOut));
    const w = new Float64Array(fanIn * fanOut);
    const b = new Float64Array(fanOut);
    for (let i = 0; i < w.length; i++) w[i] = (Math.random() * 2 - 1) * scale;
    weights.push(w);
    biases.push(b);
  }

  // Flatten all parameters for Adam
  const paramCount = weights.reduce((s, w) => s + w.length, 0) + biases.reduce((s, b) => s + b.length, 0);
  const params = new Float64Array(paramCount);

  // Pack into flat array
  let offset = 0;
  for (const w of weights) { for (let i = 0; i < w.length; i++) params[offset + i] = w[i]!; offset += w.length; }
  for (const b of biases) { for (let i = 0; i < b.length; i++) params[offset + i] = b[i]!; offset += b.length; }

  // Objective function for Adam
  const objective = (theta: Float64Array): [number, Float64Array] => {
    // Unpack
    const wTmp: Float64Array[] = [];
    const bTmp: Float64Array[] = [];
    let off = 0;
    for (let l = 0; l < weights.length; l++) {
      const wLen = weights[l]!.length;
      wTmp.push(theta.slice(off, off + wLen));
      off += wLen;
    }
    for (let l = 0; l < biases.length; l++) {
      const bLen = biases[l]!.length;
      bTmp.push(theta.slice(off, off + bLen));
      off += bLen;
    }

    // Forward pass for all samples
    let loss = 0;
    const gradW: Float64Array[] = wTmp.map(w => new Float64Array(w.length));
    const gradB: Float64Array[] = bTmp.map(b => new Float64Array(b.length));

    for (let i = 0; i < n; i++) {
      // Extract row i of X
      const row = new Float64Array(inputDim);
      for (let j = 0; j < inputDim; j++) row[j] = X[i * inputDim + j]!;

      // Forward pass with activations
      interface LayerState { preAct: Float64Array; postAct: Float64Array }
      const states: LayerState[] = [];
      let current: Float64Array = row;

      for (let l = 0; l < weights.length; l++) {
        const w = wTmp[l]!;
        const b = bTmp[l]!;
        const fanIn = l === 0 ? inputDim : (hiddenLayers[l - 1] ?? inputDim);
        const fanOut = layerSizes[l + 1]!;

        const preAct = new Float64Array(fanOut);
        // W^T × current + b
        for (let o = 0; o < fanOut; o++) {
          let sum = b[o]!;
          for (let j = 0; j < fanIn; j++) sum += w[j * fanOut + o]! * current[j]!;
          preAct[o] = sum;
        }

        const postAct = new Float64Array(fanOut);
        if (l < weights.length - 1) {
          // ReLU for hidden layers
          for (let o = 0; o < fanOut; o++) postAct[o] = Math.max(0, preAct[o]!);
        } else {
          // Identity for output layer
          for (let o = 0; o < fanOut; o++) postAct[o] = preAct[o]!;
        }

        states.push({ preAct, postAct });
        current = postAct;
      }

      // MSE loss: (pred - y)²
      const pred = current[0]!;
      const err = pred - y[i]!;
      loss += 0.5 * err * err;

      // Backprop
      let delta = new Float64Array([err]); // output layer delta

      for (let l = weights.length - 1; l >= 0; l--) {
        const w = wTmp[l]!;
        const state = states[l]!;
        const fanIn = l === 0 ? inputDim : (hiddenLayers[l - 1] ?? inputDim);
        const fanOut = layerSizes[l + 1]!;
        const prevAct = l === 0 ? row : states[l - 1]!.postAct;

        // Gradient for ReLU: multiply by indicator(preAct > 0)
        if (l < weights.length - 1) {
          const deltaNew = new Float64Array(fanOut);
          for (let o = 0; o < fanOut; o++) {
            deltaNew[o] = state.preAct[o]! > 0 ? delta[o]! : 0;
          }
          delta = deltaNew;
        }

        // Accumulate gradients
        const gW = gradW[l]!;
        const gB = gradB[l]!;
        for (let o = 0; o < fanOut; o++) {
          gB[o] = (gB[o] ?? 0) + delta[o]!;
          for (let j = 0; j < fanIn; j++) {
            gW[j * fanOut + o] = (gW[j * fanOut + o] ?? 0) + prevAct[j]! * delta[o]!;
          }
        }

        // Propagate delta to previous layer
        if (l > 0) {
          const deltaPrev = new Float64Array(fanIn);
          for (let j = 0; j < fanIn; j++) {
            let sum = 0;
            for (let o = 0; o < fanOut; o++) sum += w[j * fanOut + o]! * delta[o]!;
            deltaPrev[j] = sum;
          }
          delta = deltaPrev;
        }
      }
    }

    // Average and add L2 regularization
    loss /= n;
    for (const w of wTmp) {
      for (let i = 0; i < w.length; i++) loss += 0.5 * l2Reg * w[i]! * w[i]!;
    }

    // Pack gradients
    const grad = new Float64Array(paramCount);
    let gOff = 0;
    for (let l = 0; l < gradW.length; l++) {
      const g = gradW[l]!;
      for (let i = 0; i < g.length; i++) grad[gOff + i] = g[i]! / n + l2Reg * wTmp[l]![i]!;
      gOff += g.length;
    }
    for (let l = 0; l < gradB.length; l++) {
      const g = gradB[l]!;
      for (let i = 0; i < g.length; i++) grad[gOff + i] = g[i]! / n;
      gOff += g.length;
    }

    return [loss, grad];
  };

  const result = adam(objective, params, { lr, maxIter, gtol });

  // Unpack optimized parameters
  offset = 0;
  const optWeights: Float64Array[] = [];
  const optBiases: Float64Array[] = [];
  for (const w of weights) {
    optWeights.push(result.x.slice(offset, offset + w.length));
    offset += w.length;
  }
  for (const b of biases) {
    optBiases.push(result.x.slice(offset, offset + b.length));
    offset += b.length;
  }

  // Compute noiseStd from residuals
  const residuals: number[] = [];
  for (let i = 0; i < Math.min(n, 200); i++) {
    const row = new Float64Array(inputDim);
    for (let j = 0; j < inputDim; j++) row[j] = X[i * inputDim + j]!;
    const pred = forwardPass(row, optWeights, optBiases);
    residuals.push(y[i]! - pred);
  }
  let noiseSs = 0;
  for (const r of residuals) noiseSs += r * r;
  const noiseStd = Math.sqrt(noiseSs / Math.max(1, residuals.length - 1)) || 0.01;

  return new FFNMechanism(nodeName, optWeights, optBiases, noiseStd, inputDim);
}

// ── Forward Pass ──────────────────────────────────────────────────────

function forwardPass(
  input: Float64Array,
  weights: Float64Array[],
  biases: Float64Array[],
): number {
  if (weights.length === 0) return biases[0]?.[0] ?? 0;

  let current: Float64Array = input;

  for (let l = 0; l < weights.length; l++) {
    const w = weights[l]!;
    const b = biases[l]!;
    const fanIn = current.length;
    const fanOut = b.length;

    const next = new Float64Array(fanOut);
    for (let o = 0; o < fanOut; o++) {
      let sum = b[o]!;
      for (let j = 0; j < fanIn; j++) sum += w[j * fanOut + o]! * current[j]!;
      next[o] = sum;
    }

    // ReLU for hidden, identity for output
    if (l < weights.length - 1) {
      for (let o = 0; o < fanOut; o++) next[o] = Math.max(0, next[o]!);
    }

    current = next;
  }

  return current[0]!;
}
