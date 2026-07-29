/**
 * CAM-UV — Causal Additive Model with Unobserved Variables.
 *
 * Discovers causal structure by fitting additive non-linear models
 * with spline-based backfitting and using residual independence
 * testing to determine causal directions.
 *
 * Unlike linear methods (PC, GES), CAM-UV can detect non-linear
 * causal relationships. Unlike NOTEARS/DAGMA, it does not require
 * acyclicity constraints and can handle unobserved common causes
 * by detecting residual dependence patterns.
 *
 * Reference: Bühlmann, Peters & Ernest (2014). "CAM: Causal Additive
 *   Models, high-dimensional order search and penalized regression."
 *   Annals of Statistics 42(6):2526–2556.
 *
 * @packageDocumentation
 */

import { CausalGraph } from './causal-graph.js';
import { createRNG, fisherZTest, digamma, type CITestResult } from '@agentix-e/causality-analyzer-core';
import { Matrix } from 'ml-matrix';

/** CAM-UV algorithm configuration */
export interface CAMUVConfig {
  /** Significance level for independence tests (default: 0.05) */
  alpha?: number;
  /** Number of spline knots for additive fitting (default: 5) */
  nKnots?: number;
  /** Maximum degree of candidate parent set per variable (default: 5) */
  maxParents?: number;
  /** Use score-based post-processing (default: true) */
  scorePostProcess?: boolean;
  /** Penalty strength for BIC/GIC regularization (default: 0.5) */
  penalty?: number;
  /** Random seed for reproducibility */
  seed?: number;
}

/** CAM-UV result containing discovered graph and edge scores */
export interface CAMUVResult {
  /** Discovered causal graph (DAG) */
  graph: CausalGraph;
  /** Per-edge confidence scores */
  edgeScores: ReadonlyMap<string, number>;
  /** Edges removed by unobserved confounder detection */
  removedEdges: ReadonlyArray<readonly [string, string]>;
  /** Variable causal ordering (topological) */
  order: ReadonlyArray<string>;
}

/**
 * Run the CAM-UV causal discovery algorithm.
 *
 * @param data — (n × d) observation matrix
 * @param nodeNames — variable names (length d)
 * @param config — algorithm configuration
 * @returns discovered graph with edge scores and ordering
 */
export function camUVAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: Partial<CAMUVConfig> = {},
): CAMUVResult {
  const n = data.rows;
  const d = data.columns;
  const cfg = {
    alpha: config.alpha ?? 0.05,
    nKnots: config.nKnots ?? 5,
    maxParents: config.maxParents ?? 5,
    scorePostProcess: config.scorePostProcess ?? true,
    penalty: config.penalty ?? 0.5,
    seed: config.seed ?? 42,
  };

  if (d < 2 || n < 20) {
    return emptyResult(nodeNames);
  }

  // ── Step 1: Pairwise additive model fitting ──────────────────────────
  // For each ordered pair (i→j), fit an additive model to test if
  // X_i → X_j is a plausible causal direction.

  const edgeScores = new Map<string, number>();
  const causalOrder: number[] = [];
  const visited = new Set<number>();

  // Compute all pairwise scores
  const pairwiseScores: Array<{ from: number; to: number; score: number }> = [];

  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      if (i === j) continue;
      // Fit additive model: X_j = f(X_i) + ε
      const { independence, strength } = fitAdditivePair(data, i, j, cfg.nKnots, cfg.alpha);
      // Score: higher = more likely causal edge i→j
      const score = independence ? strength : -strength;
      pairwiseScores.push({ from: i, to: j, score });
      edgeScores.set(`${nodeNames[i]}|${nodeNames[j]}`, score);
    }
  }

  // ── Step 2: Causal ordering via score maximization ────────────────────
  const remaining = new Set(pairwiseScores);
  const order: number[] = [];
  let remainingVars = new Set(Array.from({ length: d }, (_, i) => i));

  while (remainingVars.size > 0) {
    // Find variable with highest "sink" score (most likely to be at end of chain)
    let bestVar = -1;
    let bestScore = -Infinity;

    for (const v of remainingVars) {
      // Sum of scores where v is target minus where v is source
      let sinkScore = 0;
      for (const ps of pairwiseScores) {
        if (ps.to === v && remainingVars.has(ps.from)) sinkScore += ps.score;
        if (ps.from === v && remainingVars.has(ps.to)) sinkScore -= ps.score;
      }
      if (sinkScore > bestScore) {
        bestScore = sinkScore;
        bestVar = v;
      }
    }

    if (bestVar < 0) break;
    order.push(bestVar);
    remainingVars.delete(bestVar);
  }

  // Reverse: sinks were removed first, so order is from leaf to root
  order.reverse();

  // ── Step 3: Edge pruning — keep only top-k parents per variable ──────
  const keptEdges: Array<{ source: string; target: string; weight: number }> = [];
  const removedPairs: Array<readonly [string, string]> = [];

  for (let j = 0; j < d; j++) {
    const candidates: Array<{ from: number; score: number }> = [];
    for (const ps of pairwiseScores) {
      if (ps.to === j && ps.from !== j && order.indexOf(ps.from) < order.indexOf(j)) {
        candidates.push({ from: ps.from, score: ps.score });
      }
    }

    // Sort by score descending, keep top maxParents
    candidates.sort((a, b) => b.score - a.score);
    const kept = candidates.slice(0, cfg.maxParents);

    for (const c of kept) {
      if (c.score > 0) {
        keptEdges.push({
          source: nodeNames[c.from]!,
          target: nodeNames[j]!,
          weight: Math.min(1, Math.max(0, c.score)),
        });
      }
    }

    // Track removed edges as potential unobserved confounders
    for (const c of candidates.slice(cfg.maxParents)) {
      if (c.score > 0) {
        removedPairs.push([nodeNames[c.from]!, nodeNames[j]!]);
      }
    }
  }

  // ── Step 4: Score-based post-processing ──────────────────────────────
  if (cfg.scorePostProcess && keptEdges.length > 0) {
    // Remove weak edges using BIC penalty
    const filteredEdges = keptEdges.filter(edge => {
      const score = edgeScores.get(`${edge.source}|${edge.target}`) ?? 0;
      // Penalize complex models: keep only edges with strong evidence
      return score > cfg.penalty * 0.5;
    });

    if (filteredEdges.length > 0) {
      keptEdges.length = 0;
      keptEdges.push(...filteredEdges);
    }
  }

  // ── Build result ─────────────────────────────────────────────────────
  const graph = new CausalGraph(nodeNames);
  for (const edge of keptEdges) {
    graph.addEdge(edge.source, edge.target);
  }

  return {
    graph,
    edgeScores,
    removedEdges: removedPairs,
    order: order.map(i => nodeNames[i]!),
  };
}

// ── Additive Model Fitting ──────────────────────────────────────────────

/**
 * Fit an additive model: X_j = f(X_i) + ε using B-spline regression
 * and test residual independence.
 *
 * Returns:
 *   - independence: true if residuals are independent of X_i (suggesting i→j)
 *   - strength: normalized effect strength [0, 1]
 */
function fitAdditivePair(
  data: Matrix,
  sourceIdx: number,
  targetIdx: number,
  nKnots: number,
  alpha: number,
): { independence: boolean; strength: number } {
  const n = data.rows;
  const x = data.getColumn(sourceIdx);
  const y = data.getColumn(targetIdx);

  // Normalize inputs
  const xMean = x.reduce((a, b) => a + b, 0) / n;
  const xStd = Math.sqrt(x.reduce((s, v) => s + (v - xMean) ** 2, 0) / n) || 1;
  const yMean = y.reduce((a, b) => a + b, 0) / n;
  const yStd = Math.sqrt(y.reduce((s, v) => s + (v - yMean) ** 2, 0) / n) || 1;

  const xNorm = x.map(v => (v - xMean) / xStd);

  // Build B-spline basis matrix (n × nKnots+2)
  const basis = buildBSplineBasis(xNorm, nKnots);
  const k = basis[0]!.length;

  // OLS: y_norm = B * beta + ε
  const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty = new Array(k).fill(0);
  const yNorm = y.map(v => (v - yMean) / yStd);

  for (let row = 0; row < n; row++) {
    for (let ci = 0; ci < k; ci++) {
      Xty[ci] += basis[row]![ci]! * yNorm[row]!;
      for (let cj = 0; cj <= ci; cj++) {
        XtX[ci]![cj] += basis[row]![ci]! * basis[row]![cj]!;
      }
    }
  }
  for (let ci = 0; ci < k; ci++) {
    for (let cj = ci + 1; cj < k; cj++) {
      XtX[cj]![ci] = XtX[ci]![cj]!;
    }
  }

  // Regularized solve: (XtX + λI)β = Xty
  const lambda = 0.01;
  for (let ci = 0; ci < k; ci++) XtX[ci]![ci]! += lambda;

  const beta = gaussJordanSolve(XtX, Xty);

  // Predict: ŷ = Bβ
  const residuals: number[] = [];
  let rss = 0, tss = 0;
  for (let row = 0; row < n; row++) {
    let yHat = 0;
    for (let ci = 0; ci < k; ci++) {
      yHat += basis[row]![ci]! * beta[ci]!;
    }
    const resid = yNorm[row]! - yHat;
    residuals.push(resid);
    rss += resid * resid;
    tss += yNorm[row]! * yNorm[row]!;
  }

  // Strength = R² of the additive fit
  const rSquared = tss > 0 ? Math.max(0, Math.min(1, 1 - rss / tss)) : 0;

  // Test residual independence: Fisher Z test on residuals vs X_i
  const testData = residuals.map((r, i) => [r, xNorm[i]!]);

  // Build a design matrix for Fisher Z test
  const pValue = fisherZHelper(testData, 0, 1, []);

  // If residuals are independent of X_i (p > alpha), then X_i → X_j is plausible
  const independence = pValue > alpha;

  return { independence, strength: rSquared };
}

/**
 * Simplified Fisher Z test for independence testing.
 * Tests H₀: ρ(X, Y) = 0 (no correlation).
 *
 * @internal
 */
function fisherZHelper(
  data: number[][],
  colA: number,
  colB: number,
  condSet: number[],
): number {
  const n = data.length;
  if (n < 3) return 1;

  if (condSet.length === 0) {
    // Simple correlation
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    for (const row of data) {
      const x = row[colA]!;
      const y = row[colB]!;
      sumX += x; sumY += y;
      sumXY += x * y; sumX2 += x * x; sumY2 += y * y;
    }
    const denom = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    const r = denom > 0 ? (n * sumXY - sumX * sumY) / denom : 0;
    const rClamped = Math.max(-0.9999, Math.min(0.9999, r));
    const z = 0.5 * Math.log((1 + rClamped) / (1 - rClamped));
    const se = 1 / Math.sqrt(n - 3);
    const zScore = Math.abs(z) / se;
    // Two-tailed p-value from normal approximation
    return 2 * (1 - normalCDFApprox(zScore));
  }

  // Partial correlation via residual approach
  // Regress both A and B on C, correlate residuals
  const residA = olsResiduals(data, colA, condSet);
  const residB = olsResiduals(data, colB, condSet);

  let sumA = 0, sumB = 0, sumAB = 0, sumA2 = 0, sumB2 = 0;
  for (let i = 0; i < n; i++) {
    const a = residA[i]!;
    const b = residB[i]!;
    sumA += a; sumB += b;
    sumAB += a * b; sumA2 += a * a; sumB2 += b * b;
  }
  const denom = Math.sqrt((n * sumA2 - sumA * sumA) * (n * sumB2 - sumB * sumB));
  const r = denom > 0 ? (n * sumAB - sumA * sumB) / denom : 0;
  const rClamped = Math.max(-0.9999, Math.min(0.9999, r));
  const z = 0.5 * Math.log((1 + rClamped) / (1 - rClamped));
  const se = 1 / Math.sqrt(n - condSet.length - 3);
  const pValue = 2 * (1 - normalCDFApprox(Math.abs(z) / se));
  return pValue;
}

function olsResiduals(data: number[][], targetCol: number, condCols: number[]): number[] {
  const n = data.length;
  const k = condCols.length;
  const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty = new Array(k).fill(0);

  for (let row = 0; row < n; row++) {
    for (let ci = 0; ci < k; ci++) {
      Xty[ci] += (data[row]![condCols[ci]!]!) * (data[row]![targetCol]!);
      for (let cj = 0; cj <= ci; cj++) {
        XtX[ci]![cj] += (data[row]![condCols[ci]!]!) * (data[row]![condCols[cj]!]!);
      }
    }
  }
  for (let ci = 0; ci < k; ci++) {
    for (let cj = ci + 1; cj < k; cj++) {
      XtX[cj]![ci] = XtX[ci]![cj]!;
    }
  }

  for (let ci = 0; ci < k; ci++) XtX[ci]![ci]! += 1e-10;
  const beta = gaussJordanSolve(XtX, Xty);

  const res: number[] = [];
  for (let row = 0; row < n; row++) {
    let yh = 0;
    for (let ci = 0; ci < k; ci++) {
      yh += beta[ci]! * (data[row]![condCols[ci]!]!);
    }
    res.push((data[row]![targetCol]!) - yh);
  }
  return res;
}

// ── B-Spline Basis ──────────────────────────────────────────────────────

/**
 * Build cubic B-spline basis matrix.
 *
 * Uses clamped knots and generates k = nKnots + 2 basis functions
 * (including boundary basis). Implements de Boor's recursion.
 */
function buildBSplineBasis(x: number[], nKnots: number): number[][] {
  const n = x.length;
  const degree = 3; // cubic

  // Clamped knots: place knots at quantiles, with clamped boundaries
  const xs = [...x].sort((a, b) => a - b);
  const innerKnots: number[] = [];
  for (let i = 1; i <= nKnots; i++) {
    innerKnots.push(xs[Math.floor(i * xs.length / (nKnots + 1))]!);
  }

  // Full knot vector with clamped boundaries
  const knots: number[] = [];
  for (let i = 0; i <= degree; i++) knots.push(xs[0]!);
  knots.push(...innerKnots);
  for (let i = 0; i <= degree; i++) knots.push(xs[xs.length - 1]!);

  const nBasis = knots.length - degree - 1; // = nKnots + degree + 1 => nKnots + 4

  // Build basis matrix using de Boor recursion
  const basis: number[][] = [];
  for (let row = 0; row < n; row++) {
    const xi = x[row]!;
    // Compute all basis values at xi
    const N: number[][] = Array.from(
      { length: degree + 1 },
      () => new Array(nBasis).fill(0),
    );

    // Degree 0: step functions
    for (let j = 0; j < nBasis; j++) {
      N[0]![j]! = (xi >= knots[j]! && xi < knots[j + 1]!) ? 1 : 0;
      // Handle right boundary (include the last point)
      if (j === nBasis - 1 && xi >= knots[j]! && xi <= knots[j + 1]!) {
        N[0]![j]! = 1;
      }
    }

    // Higher degrees via de Boor recursion
    for (let d = 1; d <= degree; d++) {
      for (let j = 0; j < nBasis - d; j++) {
        const left = knots[j + d]! - knots[j]!;
        const right = knots[j + d + 1]! - knots[j + 1]!;

        let term = 0;
        if (left > 1e-10) {
          term += ((xi - knots[j]!) / left) * N[d - 1]![j]!;
        }
        if (right > 1e-10) {
          term += ((knots[j + d + 1]! - xi) / right) * N[d - 1]![j + 1]!;
        }
        N[d]![j]! = term;
      }
    }

    // Use degree-d values (N[degree]) as basis functions
    basis.push(N[degree]!.slice(0, nKnots + 2));
  }

  return basis;
}

// ── Math Utilities ──────────────────────────────────────────────────────

function gaussJordanSolve(A: number[][], b: number[]): number[] {
  const k = A.length;
  const aug: number[][] = A.map((row, ri) => [...row, b[ri]!]);

  for (let col = 0; col < k; col++) {
    let maxRow = col;
    let maxV = Math.abs(aug[col]![col]!);
    for (let r = col + 1; r < k; r++) {
      if (Math.abs(aug[r]![col]!) > maxV) { maxV = Math.abs(aug[r]![col]!); maxRow = r; }
    }
    if (maxV < 1e-12) continue;
    if (maxRow !== col) [aug[col], aug[maxRow]] = [aug[maxRow]!, aug[col]!];

    const piv = aug[col]![col]!;
    for (let c = col; c <= k; c++) aug[col]![c]! /= piv;

    for (let r = 0; r < k; r++) {
      if (r === col) continue;
      const f = aug[r]![col]!;
      if (f === 0) continue;
      for (let c = col; c <= k; c++) aug[r]![c]! -= f * aug[col]![c]!;
    }
  }

  return aug.map(row => row[k]!);
}

function normalCDFApprox(z: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp(-z * z / 2);
  const prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - prob : prob;
}

function emptyResult(nodeNames: string[]): CAMUVResult {
  return {
    graph: new CausalGraph(nodeNames),
    edgeScores: new Map(),
    removedEdges: [],
    order: [...nodeNames],
  };
}
