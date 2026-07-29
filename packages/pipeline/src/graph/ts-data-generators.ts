/**
 * Time-series data generators for testing PCMCI+ and time-series causal
 * discovery algorithms.
 *
 * Provides deterministic, seeded generators for:
 *   - Linear VAR(p) models with known ground-truth graphs
 *   - Nonlinear VAR(p) with sigmoid/sine/quadratic/cubic transforms
 *   - SCM-based time series with arbitrary mechanisms
 *   - Convenience helpers for common test scenarios
 *
 * All generators return both the simulated data and a TimeSeriesGraph
 * containing the true causal structure.
 *
 * @packageDocumentation
 */

import { createRNG } from '@agentix-e/causality-analyzer-core';
import type { TimeSeriesEdge, TimeSeriesGraph } from '@agentix-e/causality-analyzer-core';

// ── Types ────────────────────────────────────────────────────────────────

/** Configuration for the linear VAR(p) time-series generator. */
export interface VARGeneratorConfig {
  /** Number of time steps */
  readonly T: number;
  /** Number of variables (must match nodeNames.length) */
  readonly d: number;
  /** Maximum lag order */
  readonly maxLag: number;
  /**
   * Lagged coefficient matrices.
   * coeffMatrices[lag][source][target] = effect of X_source[t-(lag+1)] → X_target[t]
   */
  readonly coeffMatrices: ReadonlyArray<ReadonlyArray<ReadonlyArray<number>>>;
  /**
   * Contemporaneous coefficients (lower-triangular to ensure DAG).
   * contemporaneousCoeffs[source][target] = effect of X_source[t] → X_target[t]
   */
  readonly contemporaneousCoeffs?: ReadonlyArray<ReadonlyArray<number>>;
  /** Noise standard deviation per variable (scalar = same for all) */
  readonly noiseStd: number | ReadonlyArray<number>;
  /** Random seed for reproducibility (null = non-deterministic) */
  readonly seed?: number | null;
}

/** Nonlinearity types for the nonlinear VAR generator. */
export type NonlinearityType = 'tanh' | 'sin' | 'quadratic' | 'cubic';

/** Configuration for the nonlinear VAR generator. */
export interface NonlinearVARConfig extends VARGeneratorConfig {
  /** Nonlinear transformation to apply to parent contributions */
  readonly nonlinearity: NonlinearityType;
  /** Strength of nonlinear contribution [0, 1] (0 = purely linear, 1 = purely nonlinear) */
  readonly nonlinearityStrength: number;
}

/** A single causal mechanism for the SCM generator. */
export interface SCMMechanism {
  /** Target variable name */
  readonly target: string;
  /** Causal parents as [name, lag] tuples */
  readonly parents: ReadonlyArray<readonly [string, number]>;
  /**
   * Mechanism function.
   * Receives past values (Map<varName, Record<lag, value>>) and current
   * values (Map<varName, value>), returns the value for the target.
   */
  readonly fn: (
    pastValues: Readonly<Map<string, Readonly<Record<number, number>>>>,
    currentValues: Readonly<Map<string, number>>,
  ) => number;
}

/** Configuration for the SCM-based time-series generator. */
export interface SCMTimeSeriesConfig {
  /** Number of time steps */
  readonly T: number;
  /** Maximum lag */
  readonly maxLag: number;
  /** Noise standard deviation per variable */
  readonly noiseStd: number | ReadonlyArray<number>;
  /** Ordered causal mechanisms */
  readonly mechanisms: ReadonlyArray<SCMMechanism>;
  /** Random seed */
  readonly seed?: number | null;
}

/** Result of a test time-series generation. */
export interface TestTimeSeries {
  /** (T × d) data matrix */
  readonly data: number[][];
  /** Variable names */
  readonly nodeNames: string[];
  /** Ground-truth causal graph */
  readonly truthGraph: TimeSeriesGraph;
}

// ── Linear VAR Generator ────────────────────────────────────────────────

/**
 * Generate time-series data from a linear Vector Autoregressive model
 * with known ground-truth structure.
 *
 * VAR(p) model:
 *   X[t] = Σ B_τ X[t-τ-1] + C X[t] + ε[t]
 *
 * where B_τ are lagged coefficient matrices and C is the contemporaneous
 * coefficient matrix (lower-triangular for DAG). ε[t] is Gaussian noise.
 *
 * @param nodeNames - variable names (length d)
 * @param config - VAR configuration
 * @returns simulated data and truth graph
 */
export function generateVARTimeSeries(
  nodeNames: string[],
  config: VARGeneratorConfig,
): TestTimeSeries {
  const { T, d, maxLag, coeffMatrices, contemporaneousCoeffs, noiseStd, seed } = config;
  const rng = createRNG(seed ?? Date.now());

  // Normalize noiseStd to per-variable array
  const stds: number[] = Array.isArray(noiseStd)
    ? [...noiseStd]
    : new Array(d).fill(noiseStd);

  // Build truth graph edges
  const truthEdges: TimeSeriesEdge[] = [];

  // Lagged edges from coeffMatrices
  for (let lag = 0; lag < coeffMatrices.length; lag++) {
    const matrix = coeffMatrices[lag]!;
    for (let src = 0; src < d; src++) {
      for (let tgt = 0; tgt < d; tgt++) {
        if (matrix[src]![tgt] !== 0) {
          truthEdges.push({
            source: nodeNames[src]!,
            target: nodeNames[tgt]!,
            lag: lag + 1,
            strength: Math.min(1, Math.abs(matrix[src]![tgt]!)) * Math.sign(matrix[src]![tgt]!),
            pValue: 0,
            sourceMark: 'tail',
            targetMark: 'arrow',
            phase: 'mci',
          });
        }
      }
    }
  }

  // Contemporaneous edges
  if (contemporaneousCoeffs) {
    for (let src = 0; src < d; src++) {
      for (let tgt = 0; tgt < d; tgt++) {
        if (contemporaneousCoeffs[src]![tgt] !== 0) {
          truthEdges.push({
            source: nodeNames[src]!,
            target: nodeNames[tgt]!,
            lag: 0,
            strength: Math.min(1, Math.abs(contemporaneousCoeffs[src]![tgt]!)) * Math.sign(contemporaneousCoeffs[src]![tgt]!),
            pValue: 0,
            sourceMark: 'tail',
            targetMark: 'arrow',
            phase: 'mci',
          });
        }
      }
    }
  }

  const truthGraph: TimeSeriesGraph = {
    nodes: nodeNames,
    edges: truthEdges,
    tauMax: maxLag,
    timeSteps: T,
    isCPDAG: false, // Truth graph is always fully directed
  };

  // Simulate data
  const data: number[][] = [];
  const coeffLag = coeffMatrices.length;

  for (let t = 0; t < T; t++) {
    const row: number[] = new Array(d).fill(0);

    for (let vi = 0; vi < d; vi++) {
      let val = 0;

      // Lagged contributions
      for (let lag = 0; lag < coeffLag; lag++) {
        const tau = lag + 1;
        if (t - tau >= 0) {
          const matrix = coeffMatrices[lag]!;
          for (let src = 0; src < d; src++) {
            val += (data[t - tau]![src]! ?? 0) * (matrix[src]![vi]! ?? 0);
          }
        }
      }

      row[vi] = val;
    }

    // Contemporaneous contributions (lower-triangular: src < tgt)
    if (contemporaneousCoeffs) {
      for (let tgt = 0; tgt < d; tgt++) {
        for (let src = 0; src < tgt; src++) {
          row[tgt]! += row[src]! * (contemporaneousCoeffs[src]![tgt]! ?? 0);
        }
      }
    }

    // Add noise (Box-Muller transform for Gaussian)
    for (let vi = 0; vi < d; vi++) {
      const u1 = rng() || 1e-10;
      const u2 = rng() || 1e-10;
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      row[vi]! += z * stds[vi]!;
    }

    data.push(row);
  }

  return { data, nodeNames, truthGraph };
}

// ── Nonlinear VAR Generator ─────────────────────────────────────────────

/**
 * Generate time-series data from a nonlinear VAR(p) model.
 *
 * Applies a nonlinear transformation to the sum of parent contributions
 * before adding noise.
 *
 * @param nodeNames - variable names
 * @param config - nonlinear VAR configuration
 * @returns simulated data and truth graph
 */
export function generateNonlinearVARTimeSeries(
  nodeNames: string[],
  config: NonlinearVARConfig,
): TestTimeSeries {
  const { T, d, maxLag, coeffMatrices, contemporaneousCoeffs, noiseStd, seed, nonlinearity, nonlinearityStrength } = config;
  const rng = createRNG(seed ?? Date.now());

  // Normalize noiseStd
  const stds: number[] = Array.isArray(noiseStd)
    ? [...noiseStd]
    : new Array(d).fill(noiseStd);

  // Build truth graph (same as linear — edges are present regardless of nonlinearity)
  const truthEdges: TimeSeriesEdge[] = [];

  for (let lag = 0; lag < coeffMatrices.length; lag++) {
    const matrix = coeffMatrices[lag]!;
    for (let src = 0; src < d; src++) {
      for (let tgt = 0; tgt < d; tgt++) {
        if (matrix[src]![tgt] !== 0) {
          truthEdges.push({
            source: nodeNames[src]!,
            target: nodeNames[tgt]!,
            lag: lag + 1,
            strength: Math.min(1, Math.abs(matrix[src]![tgt]!)) * Math.sign(matrix[src]![tgt]!),
            pValue: 0,
            sourceMark: 'tail',
            targetMark: 'arrow',
            phase: 'mci',
          });
        }
      }
    }
  }

  if (contemporaneousCoeffs) {
    for (let src = 0; src < d; src++) {
      for (let tgt = 0; tgt < d; tgt++) {
        if (contemporaneousCoeffs[src]![tgt] !== 0) {
          truthEdges.push({
            source: nodeNames[src]!,
            target: nodeNames[tgt]!,
            lag: 0,
            strength: Math.min(1, Math.abs(contemporaneousCoeffs[src]![tgt]!)) * Math.sign(contemporaneousCoeffs[src]![tgt]!),
            pValue: 0,
            sourceMark: 'tail',
            targetMark: 'arrow',
            phase: 'mci',
          });
        }
      }
    }
  }

  const truthGraph: TimeSeriesGraph = {
    nodes: nodeNames,
    edges: truthEdges,
    tauMax: maxLag,
    timeSteps: T,
    isCPDAG: false,
  };

  // Simulate data with nonlinear transforms
  const data: number[][] = [];
  const coeffLag = coeffMatrices.length;

  // Nonlinearity functions
  const applyNonlinearity = (x: number): number => {
    switch (nonlinearity) {
      case 'tanh': return Math.tanh(x * 2) * 0.5;
      case 'sin': return Math.sin(x) * 0.5;
      case 'quadratic': return x * x * Math.sign(x) * 0.5;
      case 'cubic': return x * x * x * 0.3;
      default: return x;
    }
  };

  for (let t = 0; t < T; t++) {
    const row: number[] = new Array(d).fill(0);

    for (let vi = 0; vi < d; vi++) {
      let linearPart = 0;
      let nonlinearPart = 0;

      for (let lag = 0; lag < coeffLag; lag++) {
        const tau = lag + 1;
        if (t - tau >= 0) {
          const matrix = coeffMatrices[lag]!;
          for (let src = 0; src < d; src++) {
            const contrib = (data[t - tau]![src]! ?? 0) * (matrix[src]![vi]! ?? 0);
            linearPart += contrib * (1 - nonlinearityStrength);
            nonlinearPart += applyNonlinearity(contrib) * nonlinearityStrength;
          }
        }
      }

      row[vi] = linearPart + nonlinearPart;
    }

    // Contemporaneous
    if (contemporaneousCoeffs) {
      for (let tgt = 0; tgt < d; tgt++) {
        for (let src = 0; src < tgt; src++) {
          row[tgt]! += row[src]! * (contemporaneousCoeffs[src]![tgt]! ?? 0);
        }
      }
    }

    // Noise
    for (let vi = 0; vi < d; vi++) {
      const u1 = rng() || 1e-10;
      const u2 = rng() || 1e-10;
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      row[vi]! += z * stds[vi]!;
    }

    data.push(row);
  }

  return { data, nodeNames, truthGraph };
}

// ── SCM Generator ───────────────────────────────────────────────────────

/**
 * Generate time-series data from custom Structural Causal Models.
 *
 * Each mechanism is an arbitrary function of its parents' (past and current)
 * values plus noise.
 *
 * @param nodeNames - variable names
 * @param config - SCM configuration with custom mechanisms
 * @returns simulated data and truth graph
 */
export function generateSCMTimeSeries(
  nodeNames: string[],
  config: SCMTimeSeriesConfig,
): TestTimeSeries {
  const { T, maxLag, noiseStd, mechanisms, seed } = config;
  const rng = createRNG(seed ?? Date.now());

  const d = nodeNames.length;
  const stds: number[] = Array.isArray(noiseStd)
    ? [...noiseStd]
    : new Array(d).fill(noiseStd);

  // Build node name → index map
  const nameToIdx = new Map<string, number>();
  nodeNames.forEach((n, i) => nameToIdx.set(n, i));

  // Build truth graph edges from mechanisms
  const truthEdges: TimeSeriesEdge[] = [];
  for (const mech of mechanisms) {
    for (const [parent, lag] of mech.parents) {
      truthEdges.push({
        source: parent,
        target: mech.target,
        lag,
        strength: 0.8,
        pValue: 0,
        sourceMark: lag > 0 ? 'tail' : 'tail',
        targetMark: 'arrow',
        phase: 'mci',
      });
    }
  }

  const truthGraph: TimeSeriesGraph = {
    nodes: nodeNames,
    edges: truthEdges,
    tauMax: maxLag,
    timeSteps: T,
    isCPDAG: false,
  };

  // Simulate data
  const data: number[][] = [];
  const nameValues = new Map<string, number[]>();

  // Initialize history
  for (const n of nodeNames) {
    nameValues.set(n, new Array(T).fill(0));
  }

  for (let t = 0; t < T; t++) {
    // Build current values map
    const currentValues = new Map<string, number>();
    for (const n of nodeNames) {
      currentValues.set(n, 0);
    }

    // Build past values map: Map<varName, Record<lag, value>>
    const pastValues = new Map<string, Record<number, number>>();
    for (const n of nodeNames) {
      const rec: Record<number, number> = {};
      for (let lag = 1; lag <= maxLag; lag++) {
        if (t - lag >= 0) {
          rec[lag] = nameValues.get(n)![t - lag]!;
        } else {
          rec[lag] = 0;
        }
      }
      pastValues.set(n, rec);
    }

    // Apply mechanisms in order
    for (const mech of mechanisms) {
      // Mechanisms see partial current values (those already computed)
      // and full past values
      const val = mech.fn(pastValues, currentValues);
      currentValues.set(mech.target, val);
      nameValues.get(mech.target)![t] = val;
    }

    // Add noise to all variables
    const row: number[] = [];
    for (let vi = 0; vi < d; vi++) {
      const name = nodeNames[vi]!;
      const baseVal = nameValues.get(name)![t]!;
      const u1 = rng() || 1e-10;
      const u2 = rng() || 1e-10;
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const noisyVal = baseVal + z * stds[vi]!;
      nameValues.get(name)![t] = noisyVal;
      row.push(noisyVal);
    }
    data.push(row);
  }

  return { data, nodeNames, truthGraph };
}

// ── Convenience Helpers ─────────────────────────────────────────────────

/**
 * Generate a simple 3-variable test time series with known structure:
 *   X0[t-1] → X0[t] (autocorrelation)
 *   X0[t-1] → X1[t] (lagged causal)
 *   X1[t] ↔ X2[t] (contemporaneous correlated)
 *
 * @param T - number of time steps (default: 200)
 * @returns test data and truth graph
 */
export function simpleTestTimeSeries(T: number = 200): TestTimeSeries {
  const nodeNames = ['X0', 'X1', 'X2'];
  const d = 3;
  const maxLag = 1;

  // B0: lag=1 coefficients: X0→X0 (0.6), X0→X1 (0.5)
  const coeffMatrices = [
    [
      [0.6, 0.0, 0.0],  // X0 source
      [0.5, 0.0, 0.0],  // X1 source
      [0.0, 0.0, 0.0],  // X2 source
    ],
  ];

  // C: contemporaneous: X1→X2 (0.4)
  const contemporaneousCoeffs = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.4],
    [0.0, 0.0, 0.0],
  ];

  return generateVARTimeSeries(nodeNames, {
    T, d, maxLag, coeffMatrices, contemporaneousCoeffs,
    noiseStd: 0.3, seed: 42,
  });
}

/**
 * Generate a chain time series:
 *   X0[t-1] → X1[t], X1[t-1] → X2[t], ..., X_{d-2}[t-1] → X_{d-1}[t]
 *
 * @param T - number of time steps
 * @param d - number of variables (≥2)
 * @returns test data and truth graph
 */
export function chainTimeSeries(T: number, d: number): TestTimeSeries {
  const nodeNames = Array.from({ length: d }, (_, i) => `X${i}`);
  const maxLag = 1;

  // Build coefficient matrices: X_i → X_{i+1} at lag=1
  const coeffMatrices = [Array.from({ length: d }, () => new Array(d).fill(0))];
  for (let i = 0; i < d - 1; i++) {
    coeffMatrices[0]![i]![i + 1] = 0.5;
  }

  return generateVARTimeSeries(nodeNames, {
    T, d, maxLag, coeffMatrices,
    noiseStd: 0.3, seed: 42,
  });
}

/**
 * Generate a fully connected VAR(1) time series with specified density.
 *
 * @param T - number of time steps
 * @param d - number of variables
 * @param density - proportion of edges present [0, 1]
 * @param seed - random seed
 * @returns test data and truth graph
 */
export function fullyConnectedVAR1(
  T: number,
  d: number,
  density: number,
  seed: number,
): TestTimeSeries {
  const nodeNames = Array.from({ length: d }, (_, i) => `X${i}`);
  const maxLag = 1;
  const rng = createRNG(seed);

  // Random coefficients with specified density
  const coeffMatrices = [Array.from({ length: d }, () => new Array(d).fill(0))];
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      if (rng() < density) {
        coeffMatrices[0]![i]![j] = (rng() - 0.5) * 1.5;
      }
    }
  }

  return generateVARTimeSeries(nodeNames, {
    T, d, maxLag, coeffMatrices,
    noiseStd: 0.3, seed: seed + 1,
  });
}
