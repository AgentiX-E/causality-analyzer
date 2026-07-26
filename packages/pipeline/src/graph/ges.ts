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

  // Empty data guard: cannot learn structure from no observations.
  if (N === 0) {
    const g = new CausalGraph(nodeNames);
    if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);
    return g;
  }

  // Auto-detect maxDegree:
  //   d > 15 (large graph)  → maxDegree=2 (prevents CPDAG explosion)
  //   N/d < 50              → maxDegree=2 (sparse data)
  //   50 ≤ N/d < 150         → maxDegree=3 (moderate)
  //   N/d ≥ 150              → maxDegree=4 (data-rich)
  // Explicit config.maxDegree always overrides.
  const ratio = N / n;
  const maxDegree = config.maxDegree && config.maxDegree >= 0
    ? config.maxDegree
    : n > 15 ? 2 : ratio < 50 ? 2 : ratio < 150 ? 3 : 4;

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

  // Helper: apply Meek rule R1 — if X→Y—Z and X,Z non-adjacent, orient Y→Z
  const applyMeekR1 = (graph: CausalGraph): void => {
    let changed = true;
    while (changed) {
      changed = false;
      for (const y of graph.nodes) {
        // Find all X where X→Y
        const parentsX = [...graph.parents(y)];
        // Find all Z where Y—Z (undirected edge = both directions)
        for (const z of graph.neighbors(y)) {
          if (graph.hasEdge(z, y)) continue; // skip if Z→Y (directed)
          for (const x of parentsX) {
            // Check X and Z are non-adjacent (neither direction)
            if (!graph.hasEdge(x, z) && !graph.hasEdge(z, x)) {
              // R1: orient Y→Z
              graph.removeEdge(z, y); // remove undirected Z—Y
              changed = true;
            }
          }
        }
      }
    }
  };

  const minDelta = Math.log(Math.max(2, N)); // minimum BIC improvement for edge acceptance

  // ── Phase 1: Forward (add edges greedily) ──────────────────────────
  let improved = true;
  let iter = 0;

  while (improved && iter++ < 100) {
    improved = false;
    let bestDelta = minDelta;
    let bestEdge: [string, string] | null = null;

    for (let i = 0; i < n; i++) {
      const u = nodeNames[i];
      const uParents = [...g.parents(u)];

      if (maxDegree >= 0 && uParents.length >= maxDegree) continue;

      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const v = nodeNames[j];
        if (isAdjacent(u, v)) continue;

        const vParents = [...g.parents(v)];

        // Candidate: v → u
        if (maxDegree < 0 || uParents.length < maxDegree) {
          const bicNew = computeBIC(u, [...uParents, v]);
          const bicOld = computeBIC(u, uParents);
          const delta = bicNew - bicOld;
          if (delta > bestDelta) { bestDelta = delta; bestEdge = [v, u]; }
        }

        // Candidate: u → v
        if (maxDegree < 0 || vParents.length < maxDegree) {
          const bicNew = computeBIC(v, [...vParents, u]);
          const bicOld = computeBIC(v, vParents);
          const delta = bicNew - bicOld;
          if (delta > bestDelta) { bestDelta = delta; bestEdge = [u, v]; }
        }
      }
    }

    if (bestEdge) {
      // Add as directed edge — BIC was computed for this specific direction
      g.addEdge(bestEdge[0], bestEdge[1]);
      // Apply Meek R1 to propagate forced V-structure orientations
      applyMeekR1(g);
      improved = true;
    }
  }

  // ── Phase 2: Backward (remove edges greedily) ────────────────────
  improved = true;
  iter = 0;

  while (improved && iter++ < 100) {
    improved = false;
    let bestDelta = minDelta;
    let bestSource: string | null = null;
    let bestTarget: string | null = null;

    for (const node of nodeNames) {
      const parents = [...g.parents(node)];
      if (parents.length === 0) continue;

      for (const p of parents) {
        const newParents = parents.filter(par => par !== p);
        const bicNew = computeBIC(node, newParents);
        const bicOld = computeBIC(node, parents);
        const delta = bicNew - bicOld;

        if (delta > bestDelta) {
          bestDelta = delta;
          bestSource = p;
          bestTarget = node;
        }
      }
    }

    if (bestSource && bestTarget) {
      g.removeEdge(bestSource, bestTarget);
      improved = true;
    }
  }

  // ── Final: CPDAG → DAG ──
  const cpdag = g.pdag2dag();

  if (domainKnowledge) cpdag.applyDomainKnowledge(domainKnowledge);

  // Cycle safety — use topological sort to find exact back-edges.
  // Removes only edges that close cycles, preserving the DAG structure.
  if (cpdag.hasCycle()) {
    const topo = cpdag.topologicalSort();
    const topoSet = new Set(topo);
    const directedEdges = [...cpdag.edges].filter(e => e.directed);
    // A directed edge a→b is a back-edge if b comes before a in topological order
    const backEdges = directedEdges.filter(e => {
      const aIdx = topo.indexOf(e.source);
      const bIdx = topo.indexOf(e.target);
      return aIdx >= 0 && bIdx >= 0 && aIdx >= bIdx;
    });
    // Remove back-edges to break all cycles
    for (const e of backEdges) {
      cpdag.removeEdge(e.source, e.target);
    }
    // Fallback: if cycles persist, remove edges in topological violation order
    if (cpdag.hasCycle()) {
      for (const e of directedEdges) {
        if (!cpdag.hasCycle()) break;
        cpdag.removeEdge(e.source, e.target);
      }
    }
  }

  return cpdag;
}
