/**
 * ExactSearch — exhaustive BIC search over DAG space for small graphs.
 *
 * Enumerates all possible DAGs (up to ~6 variables) and selects
 * the one with minimum BIC score.  Provides the ground-truth
 * optimum for benchmarking approximate methods.
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import { solveLinear } from '@agentix-e/causality-analyzer-core';

export function exactSearch(
  data: Matrix,
  nodeNames: string[],
): CausalGraph {
  const d = nodeNames.length;
  if (d > 6) return new CausalGraph([...nodeNames]); // too large
  const N = data.rows;

  // Enumerate all subsets of possible edges (0 to d*(d-1) edges)
  const allPairs: Array<[number, number]> = [];
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      if (i !== j) allPairs.push([i, j]);
  const M = allPairs.length;

  let bestBIC = Infinity;
  let bestAdj: Array<[string, string]> = [];

  // Compute BIC for a given adjacency
  const computeTotalBIC = (edges: Array<[number, number]>): number => {
    let total = 0;
    for (let v = 0; v < d; v++) {
      const parents = edges.filter(([_, to]) => to === v).map(([f]) => f);
      const k = parents.length;
      const y: number[] = [];
      for (let r = 0; r < N; r++) y.push(data.get(r, v));
      if (k === 0) {
        const mean = y.reduce((a, b) => a + b, 0) / N;
        const ss = y.reduce((s, yi) => s + (yi - mean) ** 2, 0);
        total += N * Math.log(Math.max(1e-10, ss / N)) + Math.log(N);
        continue;
      }
      const X = y.map((_, r) => [1, ...parents.map(p => data.get(r, p))]);
      const XtX = Array.from({ length: k + 1 }, () => new Float64Array(k + 1));
      const Xty = new Float64Array(k + 1);
      for (let r = 0; r < N; r++)
        for (let a = 0; a <= k; a++) {
          Xty[a] += (X[r][a] ?? 0) * y[r];
          for (let b = 0; b <= k; b++)
            XtX[a][b] += (X[r][a] ?? 0) * (X[r][b] ?? 0);
        }
      const beta = solveLinear(XtX.map(r => Array.from(r)), Array.from(Xty));
      let rss = 0;
      for (let r = 0; r < N; r++) {
        let pred = beta[0] ?? 0;
        for (let a = 1; a <= k; a++) pred += (beta[a] ?? 0) * (X[r][a] ?? 0);
        rss += (y[r] - pred) ** 2;
      }
      total += N * Math.log(Math.max(1e-10, rss / N)) + (k + 1) * Math.log(N);
    }
    return total;
  };

  // Check if edges form a DAG (topological sort)
  const isDAG = (edges: Array<[number, number]>): boolean => {
    const inDegree: number[] = new Array<number>(d).fill(0) as number[];
    const adj = Array.from({ length: d }, () => new Set<number>());
    for (const [f, t] of edges) { adj[f].add(t); inDegree[t]++; }
    const queue: number[] = [];
    for (let i = 0; i < d; i++) if (inDegree[i] === 0) queue.push(i);
    let visited = 0;
    while (queue.length > 0) {
      const v = queue.shift()!;
      visited++;
      for (const w of adj[v]) if (--inDegree[w] === 0) queue.push(w);
    }
    return visited === d;
  };

  // Exhaustive enumeration over all 2^M possible edge sets
  for (let mask = 0; mask < (1 << M); mask++) {
    const edges: Array<[number, number]> = [];
    for (let b = 0; b < M; b++)
      if (mask & (1 << b)) edges.push(allPairs[b]);
    if (!isDAG(edges)) continue;
    const bic = computeTotalBIC(edges);
    if (bic < bestBIC) {
      bestBIC = bic;
      bestAdj = edges.map(e => [nodeNames[e[0]], nodeNames[e[1]]]);
    }
  }

  const g = new CausalGraph([...nodeNames]);
  for (const [from, to] of bestAdj) g.addEdge(from, to);
  return g;
}
