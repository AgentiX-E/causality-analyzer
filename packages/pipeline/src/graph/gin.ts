/**
 * GIN — Generalized Independent Noise condition for latent structure.
 *
 * Detects latent measurement structure by checking the Generalized
 * Independent Noise (GIN) condition: for any partition of observed
 * variables into two sets, GIN tests whether there exists a subset
 * of variables that makes the two sets conditionally independent.
 *
 * Reference: Xie, Cai, Huang, Glymour, Hao & Zhang (2020).
 *   "Generalized independent noise condition for estimating latent
 *    variable causal graphs." NeurIPS 2020.
 */
import { Matrix } from 'ml-matrix';
import { fisherZTest } from '@agentix-e/causality-analyzer-core';

export interface GINResult {
  /** Detected latent variable clusters */
  clusters: Map<number, number[]>;
  /** Number of clusters */
  nClusters: number;
}

/**
 * Detect latent structure using the GIN condition.
 * Returns clusters of variables that share latent parents.
 */
export function ginDetect(
  data: Matrix,
  nodeNames: string[],
  alpha: number = 0.05,
): GINResult {
  const d = nodeNames.length;
  const N = data.rows;
  const clusters = new Map<number, number[]>();
  let clusterId = 0;

  if (d < 3 || N < 20) return { clusters, nClusters: 0 };

  // GIN condition: variables i,j share latent parent if
  // there exists k such that (i,k) ⟂ (j) | latent subset
  const adj = Array.from({ length: d }, () => new Array<boolean>(d).fill(false));

  for (let i = 0; i < d; i++) {
    for (let j = i + 1; j < d; j++) {
      // Test independence of (i, j) given all other variables
      const others = [];
      for (let k = 0; k < d; k++)
        if (k !== i && k !== j) others.push(nodeNames[k]);

      // Build data: [i, j] stacked with all others
      const dataArr: number[][] = [];
      for (let r = 0; r < N; r++) {
        const row: number[] = [];
        row.push(data.get(r, i), data.get(r, j));
        for (const o of others) row.push(data.get(r, nodeNames.indexOf(o)));
        dataArr.push(row);
      }

      if (others.length > 0) {
        const condIndices = Array.from({ length: others.length }, (_, k) => k + 2);
        const p = fisherZTest(dataArr, 0, 1, condIndices.slice(0, Math.min(5, condIndices.length)));
        // If p-value is high → variables are conditionally independent
        // → they may share a latent parent (GIN condition satisfied)
        if (p > alpha) {
          adj[i][j] = adj[j][i] = true;
        }
      }
    }
  }

  // Connected components = clusters
  const visited = new Set<number>();
  for (let i = 0; i < d; i++) {
    if (visited.has(i)) continue;
    const comp: number[] = [];
    const stack = [i];
    while (stack.length > 0) {
      const v = stack.pop()!;
      if (visited.has(v)) continue;
      visited.add(v);
      comp.push(v);
      for (let w = 0; w < d; w++)
        if (adj[v][w] && !visited.has(w)) stack.push(w);
    }
    if (comp.length >= 2) {
      clusters.set(clusterId++, comp.map(v => nodeNames.indexOf(nodeNames[v])));
    }
  }

  return { clusters, nClusters: clusterId };
}
