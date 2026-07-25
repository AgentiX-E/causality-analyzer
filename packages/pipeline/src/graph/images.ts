/**
 * IMaGES — Independent Multi-sample Greedy Equivalence Search.
 *
 * Extends FGES to multi-dataset scenarios (e.g., fMRI across subjects,
 * multi-site clinical trials). Combines BIC scores from multiple
 * independent datasets sharing the same causal graph.
 *
 * Algorithm:
 *  1. For each dataset, compute per-variable BIC given parents
 *  2. Sum BIC across all datasets
 *  3. Run GES greedy search on the summed BIC landscape
 *
 * This produces a single consensus CPDAG that balances evidence
 * across all datasets.
 *
 * Reference: Ramsey et al. (NeuroImage 2010).
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { solveLinear } from '@agentix-e/causality-analyzer-core';

export function imagesAlgorithm(
  datasets: Matrix[],
  nodeNames: string[],
  config: { maxDegree?: number } = {},
  domainKnowledge?: DomainKnowledge,
): CausalGraph {
  const maxDegree = config.maxDegree ?? -1;
  const _d = nodeNames.length;

  if (datasets.length === 0) return new CausalGraph([...nodeNames]);

  // Multi-dataset BIC: sum of per-dataset BIC
  const scoreCache = new Map<string, number>();

  const computeMultiBIC = (node: string, parents: string[]): number => {
    const key = `${node}|${[...parents].sort().join(',')}`;
    if (scoreCache.has(key)) return scoreCache.get(key)!;

    let totalBIC = 0;
    const k = parents.length + 1;

    for (const data of datasets) {
      const N = data.rows;
      if (N < 2) continue;
      const nodeIdx = nodeNames.indexOf(node);
      const pIdx = parents.map(p => nodeNames.indexOf(p));

      if (parents.length === 0) {
        let sum = 0; for (let r = 0; r < N; r++) sum += data.get(r, nodeIdx);
        const mean = sum / N;
        let rss = 0; for (let r = 0; r < N; r++) rss += (data.get(r, nodeIdx) - mean) ** 2;
        totalBIC += N * Math.log(Math.max(1e-10, rss / N)) + k * Math.log(N);
        continue;
      }

      const X: number[][] = [];
      for (let r = 0; r < N; r++) {
        const row = [1];
        for (const p of pIdx) row.push(data.get(r, p));
        X.push(row);
      }
      const y: number[] = [];
      for (let r = 0; r < N; r++) y.push(data.get(r, nodeIdx));

      const XtX = Array.from({ length: k }, () => new Float64Array(k));
      const Xty = new Float64Array(k);
      for (let r = 0; r < N; r++) {
        for (let i = 0; i < k; i++) {
          Xty[i] += (X[r][i] ?? 0) * y[r];
          for (let j = 0; j < k; j++)
            XtX[i][j] += (X[r][i] ?? 0) * (X[r][j] ?? 0);
        }
      }
      const beta = solveLinear(XtX.map(r => Array.from(r)), Array.from(Xty));
      let rss = 0;
      for (let r = 0; r < N; r++) {
        let pred = 0;
        for (let i = 0; i < k; i++) pred += (beta[i] ?? 0) * (X[r][i] ?? 0);
        rss += (y[r] - pred) ** 2;
      }
      totalBIC += N * Math.log(Math.max(1e-10, rss / N)) + k * Math.log(N);
    }

    scoreCache.set(key, totalBIC);
    return totalBIC;
  };

  // Greedy equivalence search on summed BIC
  const g = new CausalGraph([...nodeNames]);
  let improved = true, iter = 0;

  while (improved && iter++ < 100) {
    improved = false;
    let bestDelta = 1e-6, bestAdd: [string, string] | null = null;

    for (const u of nodeNames) {
      const cp = [...g.parents(u)];
      if (maxDegree >= 0 && cp.length >= maxDegree) continue;
      for (const v of nodeNames) {
        if (u === v) continue;
        if (g.hasEdge(v, u) || g.hasEdge(u, v)) continue;
        const newBIC = computeMultiBIC(u, [...cp, v]);
        const oldBIC = computeMultiBIC(u, cp);
        if (oldBIC - newBIC > bestDelta) {
          bestDelta = oldBIC - newBIC;
          bestAdd = [v, u];
        }
      }
    }

    if (bestAdd) {
      g.addEdge(bestAdd[0], bestAdd[1]);
      improved = true;
    }
  }

  const cpdag = g.pdag2dag();
  if (domainKnowledge) cpdag.applyDomainKnowledge(domainKnowledge);
  return cpdag;
}
