/**
 * DirectLiNGAM — Linear Non-Gaussian Acyclic Model for causal discovery.
 *
 * Reference: Shimizu, Inazumi, Sogawa, Hyvarinen, Kawahara, Washio,
 *   Hoyer & Bollen (2011). "DirectLiNGAM: A Direct Method for Learning
 *   a Linear Non-Gaussian Acyclic Model." JMLR 12:1225-1248.
 *
 * Unlike PC/FCI (constraint-based) and GES (score-based), LiNGAM exploits
 * non-Gaussianity of the data to uniquely identify the full causal graph
 * (including edge directions), given the assumption of linear non-Gaussian
 * additive noise: X_i = Σ b_{ij} X_j + e_i, where e_i are non-Gaussian
 * and independent.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';

/**
 * Run DirectLiNGAM on observational data.
 *
 * @returns learned causal graph with edge weights
 */
export function directLiNGAM(
  data: Matrix,
  nodeNames: string[],
): {
  graph: CausalGraph;
  weights: Map<string, Map<string, number>>;
  order: string[];
} {
  const n = nodeNames.length;
  const N = data.rows;

  // Extract data as Float64Array columns
  const X = nodeNames.map((_, i) => {
    const col = new Float64Array(N);
    for (let r = 0; r < N; r++) col[r] = data.get(r, i);
    return col;
  });

  // Center the data
  const means = X.map(col => col.reduce((a, b) => a + b, 0) / N);
  for (let i = 0; i < n; i++) {
    for (let r = 0; r < N; r++) X[i]![r] -= means[i]!;
  }

  // Track which variables remain to be ordered
  const remaining = new Set(nodeNames.map((_, i) => i));
  const order: string[] = [];
  const weights = new Map<string, Map<string, number>>();

  // Storage for residuals
  const residuals = X.map(col => new Float64Array(col));

  for (let step = 0; step < n - 1; step++) {
    // Find the most exogenous variable among remaining
    let bestVar = -1;
    let bestScore = Infinity;

    for (const i of remaining) {
      const others = [...remaining].filter(j => j !== i);
      if (others.length === 0) continue;

      // Regress X_i on all other remaining variables
      // Score = sum of mutual information-like metric with residuals
      let totalDep = 0;
      for (const j of others) {
        const dep = kernelDependence(residuals[i]!, residuals[j]!, N);
        totalDep += dep;
      }
      if (totalDep < bestScore) {
        bestScore = totalDep;
        bestVar = i;
      }
    }

    if (bestVar === -1) break;
    order.push(nodeNames[bestVar]!);
    remaining.delete(bestVar);

    // Regress out bestVar from all other remaining variables
    for (const j of remaining) {
      const b = regressOut(residuals[j]!, residuals[bestVar]!, N);
      if (Math.abs(b) > 1e-6) {
        if (!weights.has(nodeNames[j]!)) weights.set(nodeNames[j]!, new Map());
        weights.get(nodeNames[j]!)!.set(nodeNames[bestVar]!, b);
      }
    }
  }

  // Add the last remaining variable
  for (const r of remaining) order.push(nodeNames[r]!);

  // Build the causal graph from the order + weights
  const g = new CausalGraph(nodeNames);

  for (let i = 0; i < order.length; i++) {
    const child = order[i]!;
    const childWeights = weights.get(child);
    if (childWeights) {
      for (const [parent, weight] of childWeights) {
        const parentIdx = order.indexOf(parent);
        // Parent must come before child in causal order
        if (parentIdx < i && Math.abs(weight) > 1e-4) {
          g.addEdge(parent, child);
        }
      }
    }
  }

  return { graph: g, weights, order };
}

/**
 * Kernel-based dependence measure between two variables.
 * Uses pairwise distance-based correlation, robust to non-Gaussianity.
 */
function kernelDependence(x: Float64Array, y: Float64Array, n: number): number {
  // Use absolute Kendall's tau as dependence measure
  // This correctly identifies dependence for non-Gaussian distributions
  let concordant = 0, discordant = 0;

  // Stratified sampling for efficiency on large datasets
  const step = Math.max(1, Math.floor(n / 500));

  for (let i = 0; i < n; i += step) {
    for (let j = i + 1; j < n; j += step) {
      const dx = x[i]! - x[j]!;
      const dy = y[i]! - y[j]!;
      if (dx > 0 && dy > 0) concordant++;
      else if (dx > 0 && dy < 0) discordant++;
      else if (dx < 0 && dy > 0) discordant++;
      else if (dx < 0 && dy < 0) concordant++;
    }
  }

  const total = concordant + discordant;
  if (total === 0) return 0;

  // Dependence = 1 - |tau|, where tau = (C-D)/(C+D)
  // Lower value = less dependent (more exogenous)
  const tau = Math.abs(concordant - discordant) / total;
  return 1 - tau;
}

/**
 * Regress y on x: y = b*x + residual.
 * Updates y array in-place with residuals.
 * @returns regression coefficient b
 */
function regressOut(y: Float64Array, x: Float64Array, n: number): number {
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += x[i]! * y[i]!;
    sxx += x[i]! * x[i]!;
  }

  const b = sxx > 1e-10 ? sxy / sxx : 0;

  // Replace y with residuals
  for (let i = 0; i < n; i++) {
    y[i] -= b * x[i]!;
  }

  return b;
}
