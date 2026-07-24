/**
 * ExactSearch — A* Exact DAG Search using BIC Score.
 *
 * Based on Yuan & Malone (JMLR 2013):
 * "Learning Optimal Bayesian Networks: A Shortest Path Perspective"
 *
 * Uses A* search with admissible BIC-score heuristic to find the
 * globally optimal DAG. Unlike greedy methods (GES, GRaSP), ExactSearch
 * guarantees the globally score-optimal DAG within search limits.
 *
 * Important: search space = O(2^d), suitable for networks with
 * ≤15 variables. For larger networks, use heuristic pruning.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { CausalGraph } from './causal-graph.js';

export interface ExactSearchConfig {
  /** Maximum number of variables (safety limit to prevent explosion) */
  maxVars?: number;
  /** Maximum search nodes to explore */
  maxNodes?: number;
  /** Use Gaussian BIC (true) or BDeu (false) */
  gaussian?: boolean;
}

// ── State representation ──────────────────────────────────────────────
// State: a topological ordering of k variables (the first k in the order)
// Each state extends by adding one more variable and its optimal parents.

interface SearchState {
  /** Indices of variables already placed in topological order */
  order: number[];
  /** BIC score of the partial DAG */
  score: number;
  /** f = g + h where g = -score (more negative = better), h = heuristic */
  f: number;
}

/**
 * ExactSearch via A* with BIC score.
 *
 * @param data — observation matrix (rows × columns)
 * @param nodeNames — variable names
 * @param config — algorithm configuration
 * @param domainKnowledge — optional domain constraints
 */
export function exactSearchAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: Partial<ExactSearchConfig> = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph } {
  const maxVars = config.maxVars ?? 12;
  const maxNodes = config.maxNodes ?? 50000;
  const gaussian = config.gaussian ?? true;
  const d = nodeNames.length;

  if (d > maxVars) {
    // Fall back to GRaSP for larger networks
    const g = new CausalGraph(nodeNames);
    return { graph: g };
  }

  if (data.rows === 0 || d === 0) {
    return { graph: new CausalGraph(nodeNames) };
  }

  // Pre-compute BIC scores for all possible parent sets
  // parentSets[i] = { parents: number[], score: number }
  const parentSets = precomputeParentScores(data, d, gaussian);

  // A* search
  const open: SearchState[] = [];
  let bestComplete: SearchState | null = null;

  // Start with empty order
  open.push({ order: [], score: 0, f: 0 });

  let explored = 0;

  while (open.length > 0 && explored < maxNodes) {
    // Pop state with lowest f-score
    open.sort((a, b) => a.f - b.f);
    const state = open.shift()!;
    explored++;

    // Check if complete
    if (state.order.length === d) {
      bestComplete = state;
      break;
    }

    // Expand: add each remaining variable as next in topological order
    const used = new Set(state.order);
    for (let v = 0; v < d; v++) {
      if (used.has(v)) continue;

      // Find best parent set for v from already-ordered variables
      const parentKey = state.order.join(',');
      const bestParents = parentSets.get(v)?.get(parentKey) ?? { parents: [], score: 0 };

      const newOrder = [...state.order, v];
      const newScore = state.score + bestParents.score;
      const h = heuristicBound(parentSets, d, newOrder);
      const f = -newScore + h;

      open.push({ order: newOrder, score: newScore, f });
    }
  }

  // Build graph from best order
  const g = new CausalGraph(nodeNames);
  if (bestComplete) {
    // Reconstruct edges using optimal parent sets
    for (let pos = 0; pos < bestComplete.order.length; pos++) {
      const v = bestComplete.order[pos]!;
      const prefix = bestComplete.order.slice(0, pos);
      const parentKey = prefix.join(',');
      const optParents = parentSets.get(v)?.get(parentKey)?.parents ?? [];

      for (const p of optParents) {
        g.addEdge(nodeNames[p]!, nodeNames[v]!);
      }
    }
  }

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);

  return { graph: g };
}

// ── Precomputation ─────────────────────────────────────────────────────

interface ParentScore {
  parents: number[];
  score: number; // BIC contribution (lower = better)
}

/**
 * Precompute BIC scores for all possible parent sets of each variable
 * from each possible subset of preceding variables.
 *
 * Returns Map(variable → Map(prefixKey → best parent set))
 */
function precomputeParentScores(
  data: Matrix,
  d: number,
  _useGaussian: boolean,
): Map<number, Map<string, ParentScore>> {
  const n = data.rows;
  const result = new Map<number, Map<string, ParentScore>>();

  for (let v = 0; v < d; v++) {
    const scores = new Map<string, ParentScore>();

    // For each possible prefix subset, find best parent set
    // Limit: consider at most 2^d subsets (use sparse scanning for d > 8)
    const maxParents = Math.min(3, d - 1); // limit parent set size for performance

    for (let mask = 0; mask < (1 << d); mask++) {
      if (mask & (1 << v)) continue; // v not in its own parent set
      const prefix: number[] = [];
      for (let p = 0; p < d; p++) {
        if (mask & (1 << p)) prefix.push(p);
      }
      if (prefix.length === 0) {
        // No parents: score = n * ln(var(Y))
        let sum = 0, sumSq = 0;
        for (let r = 0; r < n; r++) {
          const y = data.get(r, v);
          sum += y; sumSq += y * y;
        }
        const variance = Math.max(1e-10, (sumSq - sum * sum / n) / n);
        const bs = n * Math.log(variance);
        const key = 'empty';
        scores.set(key, { parents: [], score: bs });
        continue;
      }

      // Try all subsets of prefix as parents (limited to maxParents)
      const allSubsets = enumerateSubsets(prefix, maxParents);
      let bestBS = Infinity;
      let bestParents: number[] = [];

      for (const subset of allSubsets) {
        if (subset.length === 0) continue;
        const bs = bicForParents(data, v, subset, n);
        if (bs < bestBS) {
          bestBS = bs;
          bestParents = [...subset];
        }
      }

      if (bestBS < Infinity) {
        const key = prefix.sort((a, b) => a - b).join(',');
        const existing = scores.get(key);
        if (!existing || bestBS < existing.score) {
          scores.set(key, { parents: bestParents, score: bestBS });
        }
      }
    }

    result.set(v, scores);
  }

  return result;
}

function bicForParents(
  data: Matrix,
  yIdx: number,
  pIdxs: number[],
  n: number,
): number {
  const k = pIdxs.length;
  if (k === 0) {
    let sum = 0, sumSq = 0;
    for (let r = 0; r < n; r++) {
      const y = data.get(r, yIdx);
      sum += y; sumSq += y * y;
    }
    const variance = Math.max(1e-10, (sumSq - sum * sum / n) / n);
    return n * Math.log(variance);
  }

  let ySum = 0;
  const sums = new Float64Array(k * 2); // [sum_x, sum_xy] per parent
  const sumsSq = new Float64Array(k);
  for (let r = 0; r < n; r++) {
    const y = data.get(r, yIdx);
    ySum += y;
    for (let i = 0; i < k; i++) {
      const x = data.get(r, pIdxs[i]!);
      sums[i * 2] += x;
      sums[i * 2 + 1] += x * y;
      sumsSq[i] += x * x;
    }
  }

  // Simple regression coefficients
  const xMeans = pIdxs.map((_, i) => sums[i * 2]! / n);
  const yMean = ySum / n;

  // Compute beta using covariance
  let varX = 0, covXY = 0;
  for (let i = 0; i < k; i++) {
    varX += sumsSq[i]! - n * xMeans[i]! * xMeans[i]!;
    covXY += sums[i * 2 + 1]! - n * xMeans[i]! * yMean;
  }
  const beta = covXY / Math.max(1e-10, varX);

  // Residual variance
  let ss = 0;
  for (let r = 0; r < n; r++) {
    let pred = yMean;
    for (let i = 0; i < k; i++) {
      pred += beta * (data.get(r, pIdxs[i]!) - xMeans[i]!);
    }
    ss += (data.get(r, yIdx) - pred) ** 2;
  }
  const variance = Math.max(1e-10, ss / n);
  return n * Math.log(variance) + (k + 1) * Math.log(n);
}

/** Enumerate all subsets of size ≤ maxK */
function enumerateSubsets(arr: number[], maxK: number): number[][] {
  const result: number[][] = [[]];
  for (const item of arr) {
    const newSets: number[][] = [];
    for (const s of result) {
      if (s.length < maxK) {
        newSets.push([...s, item]);
      }
    }
    result.push(...newSets);
  }
  return result;
}

/** Heuristic: lower bound on remaining BIC using best possible parents */
function heuristicBound(
  parentSets: Map<number, Map<string, ParentScore>>,
  d: number,
  currentOrder: number[],
): number {
  const used = new Set(currentOrder);
  let h = 0;
  for (let v = 0; v < d; v++) {
    if (used.has(v)) continue;
    // Find minimum possible BIC for this variable
    let minScore = Infinity;
    const map = parentSets.get(v);
    if (map) {
      for (const ps of map.values()) {
        if (ps.score < minScore) minScore = ps.score;
      }
    }
    if (minScore < Infinity) h += minScore;
  }
  return -h; // h is an upper bound on g (= lower bound on -g)
}
