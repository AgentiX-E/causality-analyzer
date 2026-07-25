/**
 * PC-Max — PC variant using maximum p-value across all conditioning sets.
 *
 * Unlike standard PC which uses the FIRST separating set found,
 * PC-Max computes p-values for ALL conditioning sets and uses
 * the maximum p-value.  This makes it more conservative (fewer
 * false positive edges) at the cost of more computations.
 *
 * Reference: Ramsey (2016). "Improving accuracy and scalability
 *   of the PC algorithm by maximizing p-value." UAI 2015 Workshop.
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import { fisherZTest } from './pc.js';
import { combinations } from '@agentix-e/causality-analyzer-core';

export function pcMaxAlgorithm(
  data: Matrix,
  nodeNames: string[],
  alpha: number = 0.05,
): CausalGraph {
  const n = nodeNames.length;
  const g = new CausalGraph(nodeNames);
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      g.undirectedEdge(nodeNames[i], nodeNames[j]);

  let depth = 0;
  while (depth < n) {
    let removed = false;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (!g.hasEdge(nodeNames[i], nodeNames[j])) continue;
        const neighbors = g.neighbors(nodeNames[i]).filter(c => c !== nodeNames[j]);
        if (neighbors.length < depth) continue;

        const subsets = combinations(neighbors, depth);
        let maxP = 0;
        for (const S of subsets) {
          const p = fisherZTest(data, i, j, S.map(s => nodeNames.indexOf(s)));
          maxP = Math.max(maxP, p);
        }
        if (maxP > alpha) {
          g.removeEdge(nodeNames[i], nodeNames[j]);
          g.removeEdge(nodeNames[j], nodeNames[i]);
          removed = true;
        }
      }
    }
    if (!removed) depth++;
    else if (depth === 0) break; // redo current depth
  }
  return g.pdag2dag();
}
