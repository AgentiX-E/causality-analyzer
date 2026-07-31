/**
 * GScore — Production Evaluation Engine for Causal Discovery
 *
 * Evaluates causal graphs without ground truth using a hybrid of:
 *   - StARS (Liu et al. 2010, NeurIPS): subsampling-based edge stability
 *   - Per-algorithm fit term: BIC, p-value, or MSE depending on algorithm type
 *
 * GScore = α · StARS_stability + (1-α) · fit_score
 *   α = 0.5 (equal weight)
 *   Range: [0, 1], higher = better
 *
 * Applicable to ALL causal discovery algorithms:
 *   - Score-based (BOSS, GES):    StARS + normalized BIC
 *   - Constraint-based (PC, FCI): StARS + mean CI p-value
 *   - Functional (NOTEARS, LiNGAM): StARS + normalized R²
 *
 * @packageDocumentation
 */

import type { CausalGraph } from '../graph/causal-graph.js';
import { Matrix } from 'ml-matrix';
import { fisherZTest } from '@agentix-e/causality-analyzer-core';
import { bicLocal } from '../graph/ges.js';

// ── StARS: Stability Approach to Regularization Selection ──────────

/**
 * StARS edge stability via subsampling.
 *
 * Algorithm (Liu et al. 2010):
 *   1. Create m subsamples of the data (each b = 0.8·N rows)
 *   2. Run discovery algorithm with same params on each subsample
 *   3. For each edge e in the full-data graph G_full:
 *        p̂(e) = fraction of subsample graphs containing e
 *   4. Total instability D = 2 · mean(p̂) · (1 - mean(p̂)) / |E|
 *   5. Stability score = 1 - D ∈ [0, 1]
 *
 * Interpretation:
 *   stability = 1.0 → every subsample recovers the exact same edges
 *   stability = 0.5 → edges appear in ~50% of subsamples (random)
 *   stability = 0.0 → no edge appears in any two subsamples
 *
 * @param pred — Causal graph discovered from full data
 * @param data — original data matrix (n × p)
 * @param runDiscovery — function that runs discovery and returns graph
 * @param m — number of subsamples (default: 20)
 * @param b — subsample fraction (default: 0.8)
 */
export function computeStARS(
  pred: CausalGraph,
  data: number[][],
  runDiscovery: (subsample: number[][]) => CausalGraph,
  m: number = 20,
  b: number = 0.8,
): number {
  if (pred.edges.length === 0) return 1.0; // empty graph is trivially stable
  const n = data.length;
  const subsampleSize = Math.max(10, Math.floor(n * b));

  const edgeSet = new Set(pred.edges.map(e => `${e.source}→${e.target}`));

  // For each subsample, count how many of the full-data edges appear
  const edgeFrequencies = new Map<string, number>();
  for (const key of edgeSet) edgeFrequencies.set(key, 0);

  let validSubsamples = 0;
  for (let s = 0; s < m; s++) {
    // Random subsample (with basic seeded RNG)
    const indices = reservoirSample(n, subsampleSize, 42 + s);
    const subsample = indices.map(i => data[i]);

    try {
      const subGraph = runDiscovery(subsample);
      for (const key of edgeSet) {
        const [src, tgt] = key.split('→');
        const found = subGraph.edges.some(e => e.source === src && e.target === tgt);
        if (found) edgeFrequencies.set(key, (edgeFrequencies.get(key) ?? 0) + 1);
      }
      validSubsamples++;
    } catch {
      // Skip failed runs
    }
  }

  if (validSubsamples === 0) return 0.5; // can't evaluate

  // Total instability D = 2ξ(1-ξ) averaged over all edges
  let totalD = 0;
  for (const count of edgeFrequencies.values()) {
    const p = count / validSubsamples;
    totalD += 2 * p * (1 - p);
  }
  const meanD = totalD / edgeFrequencies.size;
  return 1 - meanD; // stability = 1 - total instability
}

/** Reservoir sampling — O(n) random subset without replacement */
function reservoirSample(n: number, k: number, seed: number): number[] {
  let s = seed;
  const rng = (): number => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return s / 0x7FFFFFFF; };

  const reservoir: number[] = [];
  for (let i = 0; i < n; i++) {
    if (i < k) {
      reservoir.push(i);
    } else {
      const j = Math.floor(rng() * (i + 1));
      if (j < k) reservoir[j] = i;
    }
  }
  return reservoir;
}

// ── GScore: Combined Production Evaluation ─────────────────────────

/**
 * GScore — unified production evaluation for causal discovery.
 *
 * GScore(pred, data, algorithm) = 0.5 · StARS + 0.5 · fit(pred, data, algorithm)
 *
 * This replaces SHD for production use where ground truth is unavailable.
 *
 * Threshold interpretation:
 *   GScore ≥ 0.8 → excellent (edges stable, good fit)
 *   GScore ≥ 0.6 → acceptable
 *   GScore ≥ 0.4 → marginal (consider retune)
 *   GScore  < 0.4 → poor (drift likely, trigger recovery)
 */
export function computeGScore(
  pred: CausalGraph,
  data: number[][],
  algorithm: string,
  runDiscovery: (subsample: number[][]) => CausalGraph,
): number {
  const stability = computeStARS(pred, data, runDiscovery, 20, 0.8);
  const fit = computeFitScore(pred, data, algorithm);
  return 0.5 * stability + 0.5 * fit;
}

// ── Per-Algorithm Fit Terms ────────────────────────────────────────

type AlgorithmFamily = 'score' | 'constraint' | 'functional';

function classifyAlgorithm(algorithm: string): AlgorithmFamily {
  switch (algorithm) {
    case 'BOSS': case 'GES': return 'score';
    case 'PC': case 'FCI': case 'GFCI': return 'constraint';
    case 'NOTEARS': case 'LiNGAM': return 'functional';
    default: return 'score';
  }
}

/**
 * Compute algorithm-specific fit score ∈ [0, 1].
 *
 * Score-based (BOSS, GES):
 *   BIC normalized to [0, 1]: lower BIC → higher score
 *
 * Constraint-based (PC, FCI, GFCI):
 *   Mean CI p-value across edges: higher p-value → more confident edges
 *
 * Functional (NOTEARS, LiNGAM):
 *   R² of the structural equations: higher R² → better fit
 */
function computeFitScore(pred: CausalGraph, data: number[][], algorithm: string): number {
  switch (classifyAlgorithm(algorithm)) {
    case 'score': return computeBICFit(pred, data);
    case 'constraint': return computePValueFit(pred, data);
    case 'functional': return computeMSEFit(pred, data);
    default: return 0.5;
  }
}

/** BIC-based fit: lower BIC relative to null model → higher score */
function computeBICFit(pred: CausalGraph, data: number[][]): number {
  if (data.length === 0 || pred.edges.length === 0) return 0.5;

  const d = data[0]?.length ?? 0;
  const n = data.length;

  // Compute covariance matrix
  const cov = computeCovariance(data, n, d);

  // Compute total BIC of the graph = sum of per-node BIC
  const cache = new Map<string, number>();
  let totalBIC = 0;
  for (let y = 0; y < d; y++) {
    const parents = pred.edges
      .filter(e => e.target === pred.nodes[y])
      .map(e => pred.nodes.indexOf(e.source))
      .filter(i => i !== -1);
    totalBIC += bicLocal(y, parents, cov, n, cache, 1.0);
  }

  // Null model BIC: independent nodes (no parents)
  let nullBIC = 0;
  for (let y = 0; y < d; y++) {
    nullBIC += bicLocal(y, [], cov, n, cache, 1.0);
  }

  // Normalized: how much better than null model
  // BIC is negative; better models have HIGHER BIC (less negative)
  if (nullBIC >= totalBIC) return 0.5; // no improvement over null
  const improvement = (totalBIC - nullBIC) / Math.abs(nullBIC);
  return Math.min(1, 0.5 + improvement * 0.5);
}

/** P-value based fit: mean p-value across CI tests implied by the graph */
function computePValueFit(pred: CausalGraph, data: number[][]): number {
  if (pred.edges.length === 0) return 0.5;

  const d = data[0]?.length ?? 0;
  if (d < 2) return 0.5;

  // For each edge, test conditional independence given all other nodes
  let totalPValue = 0;
  let tests = 0;

  for (const edge of pred.edges) {
    const i = pred.nodes.indexOf(edge.source);
    const j = pred.nodes.indexOf(edge.target);
    if (i === -1 || j === -1) continue;

    // Test: are i and j conditionally independent?
    const condSet = pred.nodes
      .map((_, idx) => idx)
      .filter(idx => idx !== i && idx !== j);

    if (condSet.length === 0) continue;

    const z = fisherZTest(data, i, j, condSet);
    const pValue = 2 * (1 - normalCDF(Math.abs(z)));

    totalPValue += pValue;
    tests++;
  }

  if (tests === 0) return 0.5;
  return Math.min(1, totalPValue / tests);
}

/** MSE-based fit: R² of structural equations for functional models */
function computeMSEFit(pred: CausalGraph, data: number[][]): number {
  if (pred.edges.length === 0) return 0.5;

  const d = data[0]?.length ?? 0;
  const n = data.length;
  if (n < 2) return 0.5;

  let totalR2 = 0;
  let nodes = 0;

  for (let y = 0; y < d; y++) {
    const parents = pred.edges
      .filter(e => e.target === pred.nodes[y])
      .map(e => pred.nodes.indexOf(e.source))
      .filter(i => i !== -1);

    if (parents.length === 0) continue;

    // Simple linear regression: predict y from parents
    const yVals = data.map(r => r[y]);
    const yMean = yVals.reduce((a, b) => a + b, 0) / n;
    const ssTotal = yVals.reduce((a, v) => a + (v - yMean) ** 2, 0);

    // Predict y as linear combination of parents (simple OLS)
    const X = parents.map(p => data.map(r => r[p]));
    const coeffs = solveLinearRegression(X, yVals);
    const preds = data.map(r => parents.reduce((s, p, k) => s + coeffs[k] * r[p], 0));
    const ssResidual = yVals.reduce((s, v, i) => s + (v - preds[i]) ** 2, 0);

    const r2 = ssTotal > 0 ? 1 - ssResidual / ssTotal : 0;
    totalR2 += Math.max(0, Math.min(1, r2));
    nodes++;
  }

  if (nodes === 0) return 0.5;
  return totalR2 / nodes;
}

// ── Helpers ────────────────────────────────────────────────────────

function computeCovariance(data: number[][], n: number, d: number): number[][] {
  const means = new Array(d).fill(0);
  for (let j = 0; j < d; j++) {
    for (let i = 0; i < n; i++) means[j] += data[i][j];
    means[j] /= n;
  }
  const cov: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      let val = 0;
      for (let r = 0; r < n; r++) val += (data[r][i] - means[i]!) * (data[r][j] - means[j]!);
      cov[i][j] = cov[j][i] = val / n;
    }
  }
  return cov;
}

function normalCDF(x: number): number {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.sqrt(2);
  const t = 1 / (1 + p * x);
  const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return 0.5 * (1 + sign * y);
}

function solveLinearRegression(X: number[][], y: number[]): number[] {
  const p = X.length;
  const n = y.length;
  // Normal equations: X^T X beta = X^T y
  // For simplicity with bounded size: use direct solve
  const XtX: number[][] = Array.from({ length: p + 1 }, () => new Array(p + 1).fill(0));
  const Xty = new Array(p + 1).fill(0);

  for (let r = 0; r < n; r++) {
    XtX[0][0] += 1;
    Xty[0]! += y[r];
    for (let k = 0; k < p; k++) {
      XtX[0][k + 1] += X[k][r];
      XtX[k + 1][0] += X[k][r];
      Xty[k + 1]! += X[k][r] * y[r];
    }
  }
  for (let i = 0; i <= p; i++) {
    for (let j = i + 1; j <= p; j++) {
      XtX[j][i] = XtX[i][j]!;
    }
  }
  // Ridge regularization
  for (let i = 0; i <= p; i++) XtX[i][i] += 1e-8;

  // Gaussian elimination
  const aug = XtX.map((row, ri) => [...row, Xty[ri]!]);
  const k = p + 1;
  for (let col = 0; col < k; col++) {
    let pivot = col;
    for (let row = col + 1; row < k; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[pivot][col])) pivot = row;
    }
    [aug[col], aug[pivot]] = [aug[pivot], aug[col]];
    if (Math.abs(aug[col][col]) < 1e-12) continue;
    for (let j2 = col; j2 <= k; j2++) aug[col][j2]! /= aug[col][col]!;
    for (let row = 0; row < k; row++) {
      if (row === col) continue;
      const f = aug[row][col]!;
      for (let j2 = col; j2 <= k; j2++) aug[row][j2]! -= f * aug[col][j2]!;
    }
  }
  return aug.slice(1).map(r => r[k]!);
}
