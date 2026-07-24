/**
 * GRaSP — Greedy Relaxation of Sparsity.
 *
 * Extends GES (Chickering 2002) with L1-regularized BIC scoring.
 * The L1 penalty provides soft sparsity control, making GRaSP
 * more stable than GES on small-to-moderate sample sizes where
 * unregularized BIC may overfit by adding spurious edges.
 *
 * Reference: Lam et al. (2022). "GRaSP: Greedy Relaxation of the
 * Sparsity Penalty for Score-Based Structure Learning."
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { combinations } from '@agentix-e/causality-analyzer-core';

export interface GRaSPConfig {
  /** Maximum parents per node (-1 = unlimited) */
  maxDegree?: number;
  /** L1 regularization strength for edge count penalty */
  lambda1?: number;
}

/**
 * Run GRaSP on observational data.
 *
 * Two-phase greedy search with L1-regularized BIC:
 *   1. Forward: add edges that improve (BIC - λ₁·|new_edges|)
 *   2. Backward: remove edges when (BIC + λ₁) ≤ (BIC_removed)
 */
export function graspAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: GRaSPConfig = {},
  domainKnowledge?: DomainKnowledge,
): CausalGraph {
  const n = nodeNames.length;
  const N = data.rows;
  const maxDegree = config.maxDegree ?? -1;
  const lambda1 = config.lambda1 ?? 0.5 * Math.log(N) / N; // adaptive default

  // Start with empty DAG
  const g = new CausalGraph(nodeNames);

  const scoreCache = new Map<string, number>();
  const scoreKey = (node: string, parents: string[]) =>
    `${node}|${[...parents].sort().join(',')}`;

  const computeBIC = (node: string, parents: string[]): number => {
    const key = scoreKey(node, parents);
    const cached = scoreCache.get(key);
    if (cached !== undefined) return cached;

    const k = parents.length + 1; // +1 for intercept
    const pIndices = parents.map(p => nodeNames.indexOf(p));
    const tIdx = nodeNames.indexOf(node);
    const nodeVec: number[] = [];
    for (let i = 0; i < N; i++) nodeVec.push(data.get(i, tIdx));

    const yMean = nodeVec.reduce((a, b) => a + b, 0) / N;
    const sst = nodeVec.reduce((s, v) => s + (v - yMean) ** 2, 0);

    if (parents.length === 0) {
      const bic = N * Math.log(Math.max(1e-10, sst / N)) + k * Math.log(N);
      scoreCache.set(key, bic);
      return bic;
    }

    // Build design matrix
    const X: number[][] = [];
    for (let i = 0; i < N; i++) {
      const row: number[] = [1]; // intercept
      for (const p of pIndices) row.push(data.get(i, p));
      X.push(row);
    }

    // OLS
    const coef = solveOLSMat(X, nodeVec, k);
    const yHat = nodeVec.map((_, i) =>
      coef.reduce((s, c, j) => s + c * (X[i]?.[j] ?? 0), 0),
    );
    const sse = nodeVec.reduce((s, v, i) => s + (v - (yHat[i] ?? 0)) ** 2, 0);
    const bic = N * Math.log(Math.max(1e-10, sse / N)) + k * Math.log(N);
    scoreCache.set(key, bic);
    return bic;
  };

  const penalizedBIC = (node: string, parents: string[]): number =>
    computeBIC(node, parents) + lambda1 * parents.length;

  // ── Forward Phase ────────────────────────────────────────────────
  let improved = true;
  while (improved) {
    improved = false;
    let bestDelta = -Infinity, bestAdd: [string, string] | null = null;

    for (let i = 0; i < n; i++) {
      const target = nodeNames[i]!;
      const currentParents = g.parents(target);
      if (maxDegree > 0 && currentParents.length >= maxDegree) continue;

      const candidates = nodeNames.filter(
        (c, ci) => ci !== i && !currentParents.includes(c) && !g.hasDirectedPath(target, c),
      );

      for (const cand of candidates) {
        if (g.hasEdge(cand, target)) continue;

        const newParents = [...currentParents, cand];
        const newScore = penalizedBIC(target, newParents);
        const oldScore = penalizedBIC(target, currentParents);
        const delta = oldScore - newScore;

        if (delta > bestDelta) {
          bestDelta = delta; bestAdd = [cand, target];
        }
      }
    }

    if (bestAdd && bestDelta > 0) {
      g.addEdge(bestAdd[0], bestAdd[1]);
      improved = true;
    }
  }

  // ── Backward Phase ───────────────────────────────────────────────
  improved = true;
  while (improved) {
    improved = false;
    let bestDelta = -Infinity;
    let bestRemove: [string, string] | null = null;

    for (const node of nodeNames) {
      const parents = g.parents(node);
      if (parents.length === 0) continue;

      for (const p of parents) {
        const withoutP = parents.filter(pa => pa !== p);
        const oldScore = penalizedBIC(node, parents);
        const newScore = penalizedBIC(node, withoutP);
        const delta = oldScore - newScore;

        if (delta < bestDelta) {
          bestDelta = delta; bestRemove = [p, node];
        }
      }
    }

    if (bestRemove && bestDelta < 0) {
      g.removeEdge(bestRemove[0], bestRemove[1]);
      improved = true;
    }
  }

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);

  // Cycle safety
  if (g.hasCycle()) {
    for (const e of [...g.edges].filter(e => e.directed)) {
      g.removeEdge(e.source, e.target);
      if (!g.hasCycle()) break;
    }
  }

  return g;
}

// ── OLS Solver ──────────────────────────────────────────────────────

function solveOLSMat(A: number[][], b: number[], k: number): number[] {
  const aug = A.map((row, i) => [...row, b[i] ?? 0]);
  for (let col = 0; col < k; col++) {
    let pivot = col;
    for (let row = col + 1; row < k; row++)
      if (Math.abs(aug[row]![col]!) > Math.abs(aug[pivot]![col]!)) pivot = row;
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
