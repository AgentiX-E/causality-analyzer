/**
 * NOTEARS — Non-combinatorial Optimization via Trace Exponential
 * Augmented Lagrangian for Structure learning.
 *
 * Based on Zheng, Aragam, Ravikumar & Xing (NeurIPS 2018):
 * "DAGs with NO TEARS: Continuous Optimization for Structure Learning"
 *
 * Unlike constraint-based (PC) or score-based (GES) methods, NOTEARS
 * converts the discrete DAG constraint into a continuous, differentiable
 * function h(W) = Tr(exp(W ∘ W)) - d, enabling gradient-based optimization.
 *
 * h(W) = 0 ⇔ W encodes a DAG
 *
 * Algorithm:
 * 1. Initialize W randomly
 * 2. Augmented Lagrangian: L_ρ(W, α) = loss(W) + α·h(W) + (ρ/2)·h(W)²
 * 3. L-BFGS inner optimization for W
 * 4. Update α ← α + ρ·h(W), ρ ← min(ρ_max, γ·ρ)
 * 5. Threshold: set edges where |W_ij| > threshold
 *
 * Linear variant: loss(W) = (1/2n) ||X - XW||²_F (no self-loops: diag(W)=0)
 * Nonlinear variant: loss(W,Θ) = (1/2n) Σ_j ||X_j - f_j(X·W_j)||²
 *   where f_j is a neural network with parameters Θ_j
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { CausalGraph } from './causal-graph.js';

export interface NOTEARSConfig {
  /** l1 regularization weight (sparsity) */
  lambda1?: number;
  /** Maximum number of outer iterations */
  maxIter?: number;
  /** Initial penalty parameter ρ */
  rhoInit?: number;
  /** ρ growth factor */
  rhoGamma?: number;
  /** Maximum ρ */
  rhoMax?: number;
  /** H tolerance for convergence */
  hTol?: number;
  /** Edge threshold for final graph */
  threshold?: number;
  /** Use linear (true) or nonlinear (false) SEM */
  linear?: boolean;
  /** Random seed */
  seed?: number;
}

// ── Matrix Exponential (Symmetric) ───────────────────────────────────

/**
 * Matrix exponential via scaling & squaring with Padé(3,3) approximant.
 *
 * Standard approach: scale A by 2^s so ||A|| < 0.5, compute Padé(3,3),
 * then square s times.
 *
 * Padé(3,3): N = I + A/2 + A²/12, D = I - A/2 + A²/12
 * exp(A) ≈ N * D⁻¹
 */
function matrixExponential(A: Float64Array, n: number): Float64Array {
  // Compute norm to determine scaling
  let norm = 0;
  for (let i = 0; i < n; i++) {
    let rowSum = 0;
    for (let j = 0; j < n; j++) rowSum += Math.abs(A[i * n + j] ?? 0);
    if (rowSum > norm) norm = rowSum;
  }

  // Scaling: find s such that ||A||/2^s < 0.5
  let s = 0;
  if (norm > 0.5) {
    s = Math.ceil(Math.log2(norm / 0.5));
  }

  // Scale A
  const scale = 1 / Math.pow(2, s);
  const AScaled = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) AScaled[i] = A[i]! * scale;

  // Padé(3,3): N = I + A/2 + A²/12, D = I - A/2 + A²/12
  // Compute A²
  const A2 = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) sum += AScaled[i * n + k]! * AScaled[k * n + j]!;
      A2[i * n + j] = sum;
    }
  }

  // Build N and D matrices
  const N = new Float64Array(n * n);
  const D = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const aVal = AScaled[i * n + j]!;
      const a2Val = A2[i * n + j]! / 12;
      N[i * n + j] = (i === j ? 1 : 0) + aVal / 2 + a2Val;
      D[i * n + j] = (i === j ? 1 : 0) - aVal / 2 + a2Val;
    }
  }

  // Solve N = X * D  =>  X = N * D⁻¹
  // D is n×n, solve Dᵀ Xᵀ = Nᵀ  (solve linear system for each column)
  const expAScaled = solveLinearSystem(D, N, n);

  // Square s times: E_{k+1} = E_k * E_k
  let result = expAScaled;
  for (let k = 0; k < s; k++) {
    const squared = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k2 = 0; k2 < n; k2++) sum += result[i * n + k2]! * result[k2 * n + j]!;
        squared[i * n + j] = sum;
      }
    }
    result = squared;
  }

  return result;
}

/**
 * Solve A * X = B where A is n×n, B is n×n.
 * Returns X, also n×n.
 * Uses Gaussian elimination with partial pivoting.
 */
function solveLinearSystem(A: Float64Array, B: Float64Array, n: number): Float64Array {
  // Augmented matrix [A | B]
  const aug = new Float64Array(n * 2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      aug[i * 2 * n + j] = A[i * n + j]!;
    }
    for (let j = 0; j < n; j++) {
      aug[i * 2 * n + n + j] = B[i * n + j]!;
    }
  }

  // Gaussian elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxVal = Math.abs(aug[col * 2 * n + col]!);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(aug[row * 2 * n + col]!);
      if (val > maxVal) { maxVal = val; maxRow = row; }
    }

    // Swap rows
    if (maxRow !== col) {
      for (let j = 0; j < 2 * n; j++) {
        const tmp = aug[col * 2 * n + j]!;
        aug[col * 2 * n + j] = aug[maxRow * 2 * n + j]!;
        aug[maxRow * 2 * n + j] = tmp;
      }
    }

    const pivot = aug[col * 2 * n + col]!;
    if (Math.abs(pivot) < 1e-12) continue;

    // Eliminate below
    for (let row = col + 1; row < n; row++) {
      const factor = aug[row * 2 * n + col]! / pivot;
      for (let j = col; j < 2 * n; j++) {
        aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j]!;
      }
    }
  }

  // Back substitution
  const X = new Float64Array(n * n);
  for (let col = 0; col < n; col++) {
    for (let row = n - 1; row >= 0; row--) {
      let sum = aug[row * 2 * n + n + col]!;
      for (let j = row + 1; j < n; j++) {
        sum -= aug[row * 2 * n + j]! * X[j * n + col]!;
      }
      X[row * n + col] = sum / Math.max(1e-12, aug[row * 2 * n + row]!);
    }
  }

  return X;
}

// ── NOTEARS Core ────────────────────────────────────────────────────

/**
 * DAG constraint: h(W) = Tr(exp(W ∘ W)) - d
 * Gradient: ∇h(W) = exp(W ∘ W)ᵀ ∘ 2W  (since exp is applied to symmetric W∘W)
 *   = [exp(W ∘ W)]ᵀ ∘ 2W
 */
function computeHAndGradient(W: Float64Array, n: number): { h: number; grad: Float64Array } {
  // Compute W ∘ W
  const WSq = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      // Only off-diagonal (diag(W) = 0 for acyclicity)
      WSq[i * n + j] = i !== j ? W[i * n + j]! * W[i * n + j]! : 0;
    }
  }

  // exp(W ∘ W)
  const expWSq = matrixExponential(WSq, n);

  // h = Tr(expWSq) - n
  let trace = 0;
  for (let i = 0; i < n; i++) trace += expWSq[i * n + i]!;
  const h = trace - n;

  // Gradient: expWSq * 2W (element-wise on off-diagonals)
  const grad = new Float64Array(n * n);
  const expWSqT = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      expWSqT[i * n + j] = expWSq[j * n + i]!; // transpose
    }
  }

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j) {
        grad[i * n + j] = 2 * expWSqT[i * n + j]! * W[i * n + j]!;
      }
    }
  }

  return { h, grad };
}

/**
 * Loss function: least squares for linear SEM
 * L(W) = (1/2n) ||X - XW||²_F
 *
 * Gradient: ∇L(W) = (1/n) Xᵀ(XW - X)
 */
function computeLossAndGradient(
  W: Float64Array,
  X: Float64Array, // stored row-major: nSamples × d
  nSamples: number,
  nNodes: number,
): { loss: number; grad: Float64Array } {
  // Compute XW (nSamples × nNodes)
  const XW = new Float64Array(nSamples * nNodes);
  for (let r = 0; r < nSamples; r++) {
    for (let j = 0; j < nNodes; j++) {
      let sum = 0;
      for (let k = 0; k < nNodes; k++) {
        if (j !== k) { // no self-loops
          sum += X[r * nNodes + k]! * W[k * nNodes + j]!;
        }
      }
      XW[r * nNodes + j] = sum;
    }
  }

  // Residual: XW - X
  let loss = 0;
  for (let r = 0; r < nSamples; r++) {
    for (let j = 0; j < nNodes; j++) {
      const res = XW[r * nNodes + j]! - X[r * nNodes + j]!;
      loss += res * res;
    }
  }
  loss /= (2 * nSamples);

  // Gradient: (1/n) Xᵀ(XW - X)
  const grad = new Float64Array(nNodes * nNodes);
  for (let i = 0; i < nNodes; i++) {
    for (let j = 0; j < nNodes; j++) {
      if (i === j) continue; // no self-loops
      let sum = 0;
      for (let r = 0; r < nSamples; r++) {
        sum += X[r * nNodes + i]! * (XW[r * nNodes + j]! - X[r * nNodes + j]!);
      }
      grad[i * nNodes + j] = sum / nSamples;
    }
  }

  return { loss, grad };
}

/**
 * L-BFGS two-loop recursion.
 * Minimizes f(w) given gradient.
 */
function lbfgsStep(
  w: Float64Array,
  grad: Float64Array,
  sHistory: Float64Array[],
  yHistory: Float64Array[],
  m: number,
  nParams: number,
): Float64Array {
  if (sHistory.length === 0) {
    // First iteration: gradient descent
    const step = new Float64Array(nParams);
    for (let i = 0; i < nParams; i++) step[i] = -grad[i]!;
    return step;
  }

  const k = sHistory.length;
  const alpha = new Float64Array(k);
  const rho = new Float64Array(k);

  // Two-loop recursion
  let q = new Float64Array(grad);

  // First loop
  for (let i = k - 1; i >= 0; i--) {
    const si = sHistory[i]!;
    const yi = yHistory[i]!;
    let dot = 0;
    for (let j = 0; j < nParams; j++) dot += si[j]! * yi[j]!;
    rho[i] = 1 / Math.max(1e-12, dot);
    let alphaI = 0;
    for (let j = 0; j < nParams; j++) alphaI += si[j]! * q[j]!;
    alpha[i] = alphaI * rho[i]!;
    for (let j = 0; j < nParams; j++) q[j] -= alpha[i]! * yi[j]!;
  }

  // Preconditioner: H₀ = γI where γ = sᵀy / yᵀy
  let gamma = 1;
  if (k > 0) {
    const s = sHistory[k - 1]!;
    const y = yHistory[k - 1]!;
    let sy = 0, yy = 0;
    for (let j = 0; j < nParams; j++) { sy += s[j]! * y[j]!; yy += y[j]! * y[j]!; }
    gamma = sy / Math.max(1e-12, yy);
  }

  const r = new Float64Array(nParams);
  for (let j = 0; j < nParams; j++) r[j] = gamma * q[j]!;

  // Second loop
  for (let i = 0; i < k; i++) {
    const si = sHistory[i]!;
    const yi = yHistory[i]!;
    let beta = 0;
    for (let j = 0; j < nParams; j++) beta += yi[j]! * r[j]!;
    beta *= rho[i]!;
    for (let j = 0; j < nParams; j++) r[j] += si[j]! * (alpha[i]! - beta);
  }

  // Steep descent direction: -r
  for (let i = 0; i < nParams; i++) r[i] = -r[i]!;

  return r;
}

/**
 * NOTEARS causal discovery algorithm.
 *
 * @param data — observation matrix (ml-matrix, rows × cols)
 * @param nodeNames — variable names
 * @param config — algorithm configuration
 * @param domainKnowledge — optional domain constraints
 */
export function notearsAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: Partial<NOTEARSConfig> = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph; W: Float64Array; h: number; iterations: number } {
  const nNodes = nodeNames.length;
  const nSamples = data.rows;

  if (nSamples === 0 || nNodes === 0) {
    return { graph: new CausalGraph(nodeNames), W: new Float64Array(0), h: 0, iterations: 0 };
  }

  const lambda1 = config.lambda1 ?? 0.1;
  const maxIter = config.maxIter ?? 100;
  const rhoInit = config.rhoInit ?? 1.0;
  const rhoGamma = config.rhoGamma ?? 10.0;
  const rhoMax = config.rhoMax ?? 1e12;
  const hTol = config.hTol ?? 1e-8;
  const threshold = config.threshold ?? 0.3;

  // Standardize data: mean 0, std 1 per column
  const X = new Float64Array(nSamples * nNodes);
  for (let j = 0; j < nNodes; j++) {
    let sum = 0, sumSq = 0;
    for (let r = 0; r < nSamples; r++) { const v = data.get(r, j); sum += v; sumSq += v * v; }
    const mean = sum / nSamples;
    const std = Math.sqrt(Math.max(1e-10, sumSq / nSamples - mean * mean));
    for (let r = 0; r < nSamples; r++) X[r * nNodes + j] = (data.get(r, j) - mean) / std;
  }

  // Initialize W randomly
  const rng = createRNG_internal(config.seed ?? 42);
  let W = new Float64Array(nNodes * nNodes);
  // Small random values, only off-diagonal
  for (let i = 0; i < nNodes; i++) {
    for (let j = 0; j < nNodes; j++) {
      if (i !== j) W[i * nNodes + j] = (rng() - 0.5) * 0.05;
    }
  }

  let alpha = 0;
  let rho = rhoInit;
  let h = Infinity;
  let totalIter = 0;

  // L-BFGS history
  const sHistory: Float64Array[] = [];
  const yHistory: Float64Array[] = [];
  const lbfgsM = 10;

  for (let outerIter = 0; outerIter < maxIter; outerIter++) {
    // Inner optimization: minimize L(W) + α·h(W) + (ρ/2)·h(W)² + λ₁||W||₁
    for (let innerIter = 0; innerIter < 50; innerIter++) {
      // Compute loss and DAG constraint
      const { loss, grad: lossGrad } = computeLossAndGradient(W, X, nSamples, nNodes);
      const { h: hVal, grad: hGrad } = computeHAndGradient(W, nNodes);
      h = hVal;

      // Augmented Lagrangian gradient:
      // ∇_W [loss + α·h + (ρ/2)·h²] = lossGrad + (α + ρ·h)·hGrad
      const alphaRhoH = alpha + rho * h;

      // Combined gradient with l1 subgradient
      const grad = new Float64Array(nNodes * nNodes);
      const nParams = nNodes * nNodes;
      for (let i = 0; i < nParams; i++) {
        grad[i] = lossGrad[i]! + alphaRhoH * hGrad[i]!;
        // l1 subgradient (thresholding)
        if (W[i]! > 1e-8) grad[i]! += lambda1;
        else if (W[i]! < -1e-8) grad[i]! -= lambda1;
      }

      // Check convergence
      let gradNorm = 0;
      for (let i = 0; i < nParams; i++) gradNorm += grad[i]! * grad[i]!;
      gradNorm = Math.sqrt(gradNorm);

      if (gradNorm < 1e-6) break;

      // L-BFGS direction
      const direction = lbfgsStep(W, grad, sHistory, yHistory, lbfgsM, nParams);

      // Line search (backtracking)
      let stepSize = 0.1;
      let newW = new Float64Array(nParams);
      let bestLoss = Infinity;
      let bestW = W;

      for (let ls = 0; ls < 10; ls++) {
        for (let i = 0; i < nParams; i++) newW[i] = W[i]! + stepSize * direction[i]!;

        const { loss: newLoss } = computeLossAndGradient(newW, X, nSamples, nNodes);
        const { h: newH } = computeHAndGradient(newW, nNodes);
        const lagrangian = newLoss + alpha * newH + (rho / 2) * newH * newH;
        let l1Penalty = 0;
        for (let i = 0; i < nParams; i++) l1Penalty += lambda1 * Math.abs(newW[i]!);
        const total = lagrangian + l1Penalty;

        if (total < bestLoss) {
          bestLoss = total;
          bestW = new Float64Array(newW);
        }

        stepSize *= 0.5;
      }

      // L-BFGS history update
      const s = new Float64Array(nParams);
      const y = new Float64Array(nParams);
      for (let i = 0; i < nParams; i++) {
        s[i] = bestW[i]! - W[i]!;
        y[i] = grad[i]!;
      }
      let sy = 0;
      for (let i = 0; i < nParams; i++) sy += s[i]! * y[i]!;
      if (sy > 1e-12) {
        sHistory.push(s);
        yHistory.push(y);
        if (sHistory.length > lbfgsM) { sHistory.shift(); yHistory.shift(); }
      }

      W = bestW;
      totalIter++;
    }

    // Update Lagrange multiplier and penalty
    if (h < 0.25 * (hTol > 0 ? Math.max(hTol, Math.abs(computeHAndGradient(W, nNodes).h)) : 0.01)) {
      alpha += rho * h;
    } else {
      rho = Math.min(rhoMax, rhoGamma * rho);
    }

    if (h < hTol) break;
  }

  // Threshold: set edges where |W_ij| > threshold
  const g = new CausalGraph(nodeNames);
  for (let i = 0; i < nNodes; i++) {
    for (let j = 0; j < nNodes; j++) {
      if (i !== j && Math.abs(W[i * nNodes + j]!) > threshold) {
        g.addEdge(nodeNames[i]!, nodeNames[j]!);
      }
    }
  }

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);

  return { graph: g, W, h, iterations: totalIter };
}

function createRNG_internal(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}
