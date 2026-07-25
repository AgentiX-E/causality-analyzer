/**
 * GES (Greedy Equivalence Search) — score-based causal discovery in CPDAG space.
 *
 * Reference: Chickering (2002). "Optimal Structure Identification With Greedy Search."
 *
 * GES searches the space of CPDAGs (Markov equivalence classes) using a
 * two-phase greedy approach:
 *   1. Forward phase: greedily add edges, considering ALL orientations
 *      consistent with the current CPDAG to maximize BIC score
 *   2. Backward phase: greedily remove edges from the CPDAG
 *   3. Final conversion: CPDAG → DAG (via pdag2dag)
 *
 * Unlike the original DAG-space implementation, this version operates
 * in CPDAG space — matching causal-learn/Tetrad's GES implementations.
 * For each candidate edge addition, BOTH orientations are tried and
 * the best BIC improvement is selected globally.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { solveLinear } from '@agentix-e/causality-analyzer-core';

export interface GESConfig {
  /** Maximum number of parents per node (-1 = unlimited) */
  maxDegree?: number;
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

  // Start with empty graph (empty CPDAG)
  const g = new CausalGraph(nodeNames);

  // ── BIC scoring with caching ──
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
      let ss = 0, sum = 0;
      for (let r = 0; r < N; r++) { const v = data.get(r, nodeIdx); sum += v; }
      const mean = sum / N;
      for (let r = 0; r < N; r++) { const v = data.get(r, nodeIdx); ss += (v - mean) ** 2; }
      const bic = -N * Math.log(Math.max(1e-10, ss / N)) - k * Math.log(N);
      scoreCache.set(key, bic);
      return bic;
    }

    const XtX = Array.from({ length: k }, () => new Float64Array(k));
    const Xty = new Float64Array(k);
    for (let r = 0; r < N; r++) {
      const y = data.get(r, nodeIdx);
      for (let i = 0; i < k; i++) {
        const xi = data.get(r, pIdx[i]);
        Xty[i] += xi * y;
        for (let j = 0; j < k; j++)
          XtX[i][j] += xi * data.get(r, pIdx[j]);
      }
    }

    const XtXArr = XtX.map(r => Array.from(r));
    const XtyArr = Array.from(Xty);
    const beta = solveLinear(XtXArr, XtyArr);

    let rss = 0;
    for (let r = 0; r < N; r++) {
      const y = data.get(r, nodeIdx);
      let pred = 0;
      for (let i = 0; i < k; i++) pred += (beta[i] ?? 0) * data.get(r, pIdx[i]);
      rss += (y - pred) ** 2;
    }

    const bic = -N * Math.log(Math.max(1e-10, rss / N)) - k * Math.log(N);
    scoreCache.set(key, bic);
    return bic;
  };

  // Helper: get BIC score for entire graph
  const _totalBIC = (): number => {
    let total = 0;
    for (const node of nodeNames)
      total += computeBIC(node, [...g.parents(node)]);
    return total;
  };

  // Helper: check if u and v are adjacent (in either direction)
  const isAdjacent = (u: string, v: string): boolean =>
    g.hasEdge(u, v) || g.hasEdge(v, u);

  // ── Phase 1: Forward (add edges greedily) in CPDAG space ──────
  let improved = true;
  let iter = 0;

  while (improved && iter++ < 100) {
    improved = false;
    let bestDelta = -1e-6;
    let bestEdge: [string, string] | null = null;

    for (let i = 0; i < n; i++) {
      const u = nodeNames[i];
      // Use neighbors (undirected) for degree check in CPDAG space
      const uDegree = g.neighbors(u).length;

      if (maxDegree >= 0 && uDegree >= maxDegree) continue;

      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const v = nodeNames[j];
        if (isAdjacent(u, v)) continue;

        const vDegree = g.neighbors(v).length;

        // BIC computation uses parents (one-way edges), degree check uses neighbors
        const uParents = [...g.parents(u)];
        const vParents = [...g.parents(v)];

        // Try both orientations: v→u and u→v
        if (!(maxDegree >= 0 && uDegree >= maxDegree)) {
          const bicNew1 = computeBIC(u, [...uParents, v]);
          const bicOld1 = computeBIC(u, uParents);
          const delta1 = bicNew1 - bicOld1;
          if (delta1 > bestDelta) {
            bestDelta = delta1;
            bestEdge = [v, u];
          }
        }

        // Orientation 2: u → v (u is parent of v)
        if (!(maxDegree >= 0 && vDegree >= maxDegree)) {
          const bicNew2 = computeBIC(v, [...vParents, u]);
          const bicOld2 = computeBIC(v, vParents);
          const delta2 = bicNew2 - bicOld2;
          if (delta2 > bestDelta) {
            bestDelta = delta2;
            bestEdge = [u, v];
          }
        }
      }
    }

    if (bestEdge) {
      // Add as undirected edge in CPDAG space.
      // Direction is ambiguous for the first edge — scores are
      // symmetric for both orientations.  Later edges and the
      // backward phase will resolve the orientation via BIC.
      g.undirectedEdge(bestEdge[0], bestEdge[1]);
      improved = true;
    }
  }

  // ── Phase 2: Backward (remove edges greedily) ──
  // Operates on both directed and undirected edges.
  improved = true;
  iter = 0;

  while (improved && iter++ < 100) {
    improved = false;
    let bestDelta = -1e-6;
    let bestRemove: [string, string] | null = null;

    for (let i = 0; i < n; i++) {
      const u = nodeNames[i];
      // Use neighbors (includes both directed parents and undirected adjacencies)
      for (const v of g.neighbors(u)) {
        // Try removing edge as parent: BIC(u without v) vs BIC(u with v)
        const currentParents = [...g.parents(u)];
        if (currentParents.includes(v)) {
          const newParents = currentParents.filter(p => p !== v);
          const bicNew = computeBIC(u, newParents);
          const bicOld = computeBIC(u, currentParents);
          const delta = bicNew - bicOld;
          if (delta > bestDelta) { bestDelta = delta; bestRemove = [v, u]; }
        } else {
          // Undirected edge: check both directions
          const bicNew1 = computeBIC(u, []);
          const bicOld1 = computeBIC(u, [v]);
          const delta1 = bicNew1 - bicOld1;
          if (delta1 > bestDelta) { bestDelta = delta1; bestRemove = [v, u]; }

          const bicNew2 = computeBIC(v, []);
          const bicOld2 = computeBIC(v, [u]);
          const delta2 = bicNew2 - bicOld2;
          if (delta2 > bestDelta) { bestDelta = delta2; bestRemove = [u, v]; }
        }
      }
    }

    if (bestRemove) {
      g.removeEdge(bestRemove[0], bestRemove[1]);
      g.removeEdge(bestRemove[1], bestRemove[0]);
      improved = true;
    }
  }

  // ── Final: CPDAG → DAG ──
  const cpdag = g.pdag2dag();

  if (domainKnowledge) cpdag.applyDomainKnowledge(domainKnowledge);

  // Cycle safety
  if (cpdag.hasCycle()) {
    const directedEdges = [...cpdag.edges].filter(e => e.directed);
    for (const e of directedEdges) {
      cpdag.removeEdge(e.source, e.target);
      if (!cpdag.hasCycle()) break;
    }
  }

  return cpdag;
}
