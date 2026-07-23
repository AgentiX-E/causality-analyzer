/**
 * GES (Greedy Equivalence Search) — score-based causal discovery.
 *
 * Reference: Chickering (2002). "Optimal Structure Identification With Greedy Search."
 *            Ramsey et al. (2017). "A Million Variables and More: the Fast Greedy
 *            Equivalence Search Algorithm for Learning High-Dimensional Graphical
 *            Causal Models, with an Application to Functional Magnetic Resonance Images."
 *
 * GES searches the space of CPDAGs (Markov equivalence classes) using a
 * two-phase greedy approach:
 *   1. Forward phase: greedily add edges to maximize BIC score
 *   2. Backward phase: greedily remove edges to maximize BIC score
 *
 * Unlike PC (constraint-based), GES uses score optimization and works well
 * with moderate sample sizes where CI tests may lack power.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { combinations } from '@agentix-e/causality-analyzer-core';

export interface GESConfig {
  /** Maximum number of parents per node (-1 = unlimited) */
  maxDegree?: number;
}

/** Internal BIC cache entry */
interface ScoreCache {
  node: string;
  parents: string[];
  bic: number;
}

/**
 * Run GES on observational data.
 *
 * @returns the learned CPDAG (use .pdag2dag() to convert to DAG)
 */
export function gesAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: GESConfig = {},
  domainKnowledge?: DomainKnowledge,
): CausalGraph {
  const n = nodeNames.length;
  const N = data.rows;
  const maxDegree = config.maxDegree ?? -1;

  // Start with empty graph
  const g = new CausalGraph(nodeNames);

  // Pre-compute all combinations of parents for each node (cached)
  const scoreCache = new Map<string, number>();
  const scoreKey = (node: string, parents: string[]) =>
    `${node}|${[...parents].sort().join(',')}`;

  const computeBIC = (node: string, parents: string[]): number => {
    const key = scoreKey(node, parents);
    if (scoreCache.has(key)) return scoreCache.get(key)!;

    const nodeIdx = nodeNames.indexOf(node);
    const pIdx = parents.map(p => nodeNames.indexOf(p));
    const k = parents.length;

    if (k === 0) {
      // BIC for empty parent set: just variance of the node
      let ss = 0, sum = 0;
      for (let r = 0; r < N; r++) { const v = data.get(r, nodeIdx); sum += v; }
      const mean = sum / N;
      for (let r = 0; r < N; r++) { const v = data.get(r, nodeIdx); ss += (v - mean) ** 2; }
      const bic = -N * Math.log(Math.max(1e-10, ss / N)) - k * Math.log(N);
      scoreCache.set(key, bic);
      return bic;
    }

    // OLS regression: X ~ parents
    const XtX = Array.from({ length: k }, () => new Float64Array(k));
    const Xty = new Float64Array(k);
    let ySum = 0;

    for (let r = 0; r < N; r++) {
      const y = data.get(r, nodeIdx);
      ySum += y;
      for (let i = 0; i < k; i++) {
        const xi = data.get(r, pIdx[i]!);
        Xty[i] += xi * y;
        for (let j = 0; j < k; j++) {
          XtX[i]![j] += xi * data.get(r, pIdx[j]!);
        }
      }
    }

    // Solve XtX * beta = Xty
    const XtXArr = XtX.map(r => Array.from(r));
    const XtyArr = Array.from(Xty);
    const beta = solveOLS(XtXArr, XtyArr, k);

    // Compute RSS and BIC
    let rss = 0;
    for (let r = 0; r < N; r++) {
      const y = data.get(r, nodeIdx);
      let pred = 0;
      for (let i = 0; i < k; i++) pred += (beta[i] ?? 0) * data.get(r, pIdx[i]!);
      rss += (y - pred) ** 2;
    }

    const bic = -N * Math.log(Math.max(1e-10, rss / N)) - k * Math.log(N);
    scoreCache.set(key, bic);
    return bic;
  };

  // ── Phase 1: Forward (add edges greedily) ──────────────────────────

  let improved = true;
  let iter = 0;
  while (improved && iter++ < 100) {
    improved = false;
    let bestDelta = -Infinity;
    let bestAdd: [string, string] | null = null;

    // Get current CPDAG skeleton
    const skeleton = new Map<string, Set<string>>();
    for (const node of nodeNames) skeleton.set(node, new Set());

    for (const u of nodeNames) {
      for (const v of nodeNames) {
        if (u === v) continue;
        if (g.hasEdge(u, v) || g.hasEdge(v, u)) {
          skeleton.get(u)!.add(v);
        }
      }
    }

    // Try adding each non-adjacent edge, orienting both ways
    for (let i = 0; i < n; i++) {
      const u = nodeNames[i]!;
      const currentParents = [...g.parents(u)];

      // Respect max degree
      if (maxDegree >= 0 && currentParents.length >= maxDegree) continue;

      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const v = nodeNames[j]!;
        if (skeleton.get(u)!.has(v)) continue; // already adjacent

        // Try: v → u (add edge v→u)
        const newParents = [...currentParents, v];
        const bicNew = computeBIC(u, newParents);
        const bicOld = computeBIC(u, currentParents);
        const delta = bicNew - bicOld;

        if (delta > bestDelta) {
          bestDelta = delta;
          bestAdd = [v, u];
        }
      }
    }

    if (bestAdd && bestDelta > 1e-6) {
      g.addEdge(bestAdd[0], bestAdd[1]);
      improved = true;
    }
  }

  // ── Phase 2: Backward (remove edges greedily) ──────────────────────

  improved = true;
  iter = 0;
  while (improved && iter++ < 100) {
    improved = false;
    let bestDelta = -Infinity;
    let bestRemove: [string, string] | null = null;

    for (let i = 0; i < n; i++) {
      const u = nodeNames[i]!;
      const currentParents = [...g.parents(u)];

      // Try removing each parent
      for (const v of currentParents) {
        const newParents = currentParents.filter(p => p !== v);
        const bicNew = computeBIC(u, newParents);
        const bicOld = computeBIC(u, currentParents);
        const delta = bicNew - bicOld;

        if (delta > bestDelta) {
          bestDelta = delta;
          bestRemove = [v, u];
        }
      }
    }

    if (bestRemove && bestDelta > 1e-6) {
      g.removeEdge(bestRemove[0], bestRemove[1]);
      improved = true;
    }
  }

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);

  return g;
}

// ── OLS Solver ────────────────────────────────────────────────────────

function solveOLS(A: number[][], b: number[], k: number): number[] {
  // Gaussian elimination with partial pivoting
  const aug = A.map((row, i) => [...row, b[i] ?? 0]);

  for (let col = 0; col < k; col++) {
    let pivot = col;
    for (let row = col + 1; row < k; row++) {
      if (Math.abs(aug[row]![col]!) > Math.abs(aug[pivot]![col]!)) pivot = row;
    }
    [aug[col], aug[pivot]] = [aug[pivot]!, aug[col]!];

    const pv = aug[col]![col]!;
    if (Math.abs(pv) < 1e-12) continue;

    for (let j = col; j <= k; j++) aug[col]![j]! /= pv;
    for (let row = 0; row < k; row++) {
      if (row === col) continue;
      const factor = aug[row]![col]!;
      for (let j = col; j <= k; j++) aug[row]![j]! -= factor * aug[col]![j]!;
    }
  }

  return aug.map(row => row[k]!);
}
