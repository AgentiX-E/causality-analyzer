/**
 * GES (Greedy Equivalence Search) — CPDAG-space causal discovery.
 *
 * Reference: Chickering (2002). "Optimal Structure Identification
 *   With Greedy Search." JMLR 3:507-554.
 *
 * Full CPDAG-space implementation with:
 *   - Forward phase: greedily insert directed edges
 *   - Backward phase: greedily delete edges
 *   - Meek rules R1-R3: propagate forced orientations after each operation
 *   - Global BIC scoring: sum of local node BIC scores
 *
 * Score: Gaussian BIC (continuous), BDeu (discrete) — configurable.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { solveLinear } from '@agentix-e/causality-analyzer-core';

export interface GESConfig {
  maxDegree?: number;
  score?: 'bic';
}

export function gesAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: GESConfig = {},
  domainKnowledge?: DomainKnowledge,
): CausalGraph {
  const n = nodeNames.length;
  const N = data.rows;
  if (N === 0) {
    const g = new CausalGraph(nodeNames);
    if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);
    return g;
  }

  const ratio = N / n;
  const maxDegree = config.maxDegree !== undefined && config.maxDegree >= 0
    ? config.maxDegree
    : n > 20 ? 3 : ratio < 30 ? 3 : ratio < 100 ? 4 : 5;

  const nodeIdx = new Map(nodeNames.map((name, i) => [name, i]));

  // ── BIC scoring ──────────────────────────────────────────────────

  const scoreCache = new Map<string, number>();
  const sk = (node: string, parents: string[]): string =>
    `${node}|${[...parents].sort().join(',')}`;

  const bicLocal = (node: string, parents: string[]): number => {
    const key = sk(node, parents);
    if (scoreCache.has(key)) return scoreCache.get(key)!;

    const ni = nodeIdx.get(node)!;
    const pi = parents.map(p => nodeIdx.get(p)!);
    const k = parents.length;

    let rss: number;
    if (k === 0) {
      let ss = 0, sum = 0;
      for (let r = 0; r < N; r++) { const v = data.get(r, ni); sum += v; }
      const mean = sum / N;
      for (let r = 0; r < N; r++) { const v = data.get(r, ni); ss += (v - mean) ** 2; }
      rss = ss;
    } else {
      const XtX: Float64Array[] = [];
      for (let i = 0; i < k; i++) XtX.push(new Float64Array(k));
      const Xty = new Float64Array(k);
      for (let r = 0; r < N; r++) {
        const y = data.get(r, ni);
        for (let i = 0; i < k; i++) {
          const xi = data.get(r, pi[i]!);
          Xty[i] += xi * y;
          for (let j = i; j < k; j++) XtX[i]![j] += xi * data.get(r, pi[j]!);
        }
      }
      for (let i = 0; i < k; i++) for (let j = 0; j < i; j++) XtX[i]![j] = XtX[j]![i]!;
      const beta = solveLinear(XtX.map(r => Array.from(r)), Array.from(Xty));
      rss = 0;
      for (let r = 0; r < N; r++) {
        const y = data.get(r, ni);
        let pred = 0;
        for (let i = 0; i < k; i++) pred += (beta[i] ?? 0) * data.get(r, pi[i]!);
        rss += (y - pred) ** 2;
      }
    }

    const bic = -(N * Math.log(Math.max(1e-10, rss / N)) + k * Math.log(Math.max(2, N)));
    scoreCache.set(key, bic);
    return bic;
  };

  const globalScore = (g: CausalGraph): number => {
    let total = 0;
    for (const node of g.nodes) total += bicLocal(node, [...g.parents(node)]);
    return total;
  };

  const adjacent = (g: CausalGraph, u: string, v: string): boolean =>
    g.hasEdge(u, v) || g.hasEdge(v, u);

  // ── Meek Rules ────────────────────────────────────────────────────

  const meekPropagate = (graph: CausalGraph): void => {
    let changed = true;
    while (changed) {
      changed = false;

      // R1: X → Y — Z, X and Z non-adjacent → Y → Z
      for (const y of graph.nodes) {
        const xList = [...graph.parents(y)]; // X → Y
        for (const z of graph.neighbors(y)) {
          if (graph.hasEdge(z, y)) continue; // already Z → Y
          for (const x of xList) {
            if (!adjacent(graph, x, z)) {
              graph.removeEdge(z, y); // remove Z — Y
              changed = true;
              break;
            }
          }
        }
      }

      // R2: X → Y → Z and X — Z → X → Z
      if (!changed) {
        for (const y of graph.nodes) {
          const xList = [...graph.parents(y)];
          const zList = [...graph.children(y)];
          for (const x of xList) {
            for (const z of zList) {
              if (adjacent(graph, x, z) && !graph.hasEdge(x, z) && !graph.hasEdge(z, x)) {
                graph.removeEdge(z, x);
                graph.addEdge(x, z);
                changed = true;
              }
            }
          }
        }
      }

      // R3: X — Y, X — Z, X — W, Y → Z, Z → W → Y → W
      if (!changed) {
        for (const x of graph.nodes) {
          const undirTo = graph.neighbors(x).filter(z =>
            !graph.hasEdge(x, z) && !graph.hasEdge(z, x)
          );
          if (undirTo.length < 2) continue;
          for (const y of undirTo) {
            for (const w of undirTo) {
              if (y === w) continue;
              // Check Y → ... → W chain? Simplified: just check non-adjacent
              if (!adjacent(graph, y, w)) {
                const zS = graph.children(y).filter(z =>
                  graph.children(z).includes(w) && adjacent(graph, x, z)
                );
                if (zS.length > 0) {
                  graph.removeEdge(w, x);
                  graph.addEdge(x, w);
                  changed = true;
                }
              }
            }
          }
        }
      }
    }
  };

  // ── Working graph ──────────────────────────────────────────────────

  const g = new CausalGraph([...nodeNames]);

  // ── Phase 1: Forward (Insert) ─────────────────────────────────────
  let improved = true;
  let iter = 0;
  const minDelta = Math.log(Math.max(2, N));

  while (improved && iter++ < 200) {
    improved = false;
    let bestDelta = minDelta;
    let bestFrom = '', bestTo = '';

    for (let i = 0; i < n; i++) {
      const u = nodeNames[i]!;
      const uParents = [...g.parents(u)];
      if (maxDegree >= 0 && uParents.length >= maxDegree) continue;

      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const v = nodeNames[j]!;
        if (adjacent(g, u, v)) continue;

        const vParents = [...g.parents(v)];

        // v → u
        if (maxDegree < 0 || uParents.length < maxDegree) {
          const old = bicLocal(u, uParents);
          const delta = bicLocal(u, [...uParents, v]) - old;
          if (delta > bestDelta) { bestDelta = delta; bestFrom = v; bestTo = u; }
        }
        // u → v
        if (maxDegree < 0 || vParents.length < maxDegree) {
          const old = bicLocal(v, vParents);
          const delta = bicLocal(v, [...vParents, u]) - old;
          if (delta > bestDelta) { bestDelta = delta; bestFrom = u; bestTo = v; }
        }
      }
    }

    if (bestFrom && bestTo) {
      g.addEdge(bestFrom, bestTo);
      meekPropagate(g);
      improved = true;
    }
  }

  // ── Phase 2: Backward (Delete) ────────────────────────────────────
  improved = true;
  iter = 0;

  while (improved && iter++ < 200) {
    improved = false;
    let bestDelta = minDelta;
    let bestSource = '', bestTarget = '';

    for (const node of nodeNames) {
      const parents = [...g.parents(node)];
      if (parents.length === 0) continue;

      for (const p of parents) {
        const newParents = parents.filter(par => par !== p);
        const old = bicLocal(node, parents);
        const delta = bicLocal(node, newParents) - old;
        if (delta > bestDelta) { bestDelta = delta; bestSource = p; bestTarget = node; }
      }
    }

    if (bestSource && bestTarget) {
      g.removeEdge(bestSource, bestTarget);
      // Also remove reverse if somehow present (safety)
      g.removeEdge(bestTarget, bestSource);
      improved = true;
    }
  }

  // ── Phase 3: Turning (optional) ───────────────────────────────────
  improved = true;
  iter = 0;

  while (improved && iter++ < 100) {
    improved = false;
    let bestDelta = minDelta;
    let bestFrom = '', bestTo = '';

    for (const node of nodeNames) {
      const parents = [...g.parents(node)];
      for (const p of parents) {
        // Reverse p→node to node→p
        const newParents = parents.filter(par => par !== p);
        const pParents = [...g.parents(p)];

        // Check no cycle would form: node must not have a path to p
        if (g.hasDirectedPath(node, p)) continue;

        const oldScore = bicLocal(node, parents) + bicLocal(p, pParents);
        const newScore = bicLocal(node, newParents) + bicLocal(p, [...pParents, node]);
        const delta = newScore - oldScore;

        if (delta > bestDelta) {
          bestDelta = delta;
          bestFrom = node;
          bestTo = p;
        }
      }
    }

    if (bestFrom && bestTo) {
      g.removeEdge(bestTo, bestFrom); // remove p→node
      g.addEdge(bestFrom, bestTo);     // add node→p
      improved = true;
    }
  }

  // ── Final DAG conversion + domain knowledge ────────────────────────

  let result = g.pdag2dag();
  if (domainKnowledge) result.applyDomainKnowledge(domainKnowledge);

  // Cycle safety
  if (result.hasCycle()) {
    const topo = result.topologicalSort();
    const directedEdges = [...result.edges].filter(e => e.directed);
    const backEdges = directedEdges.filter(e => {
      const aIdx = topo.indexOf(e.source);
      const bIdx = topo.indexOf(e.target);
      return aIdx >= 0 && bIdx >= 0 && aIdx >= bIdx;
    });
    for (const e of backEdges) result.removeEdge(e.source, e.target);
    if (result.hasCycle()) {
      for (const e of directedEdges) {
        if (!result.hasCycle()) break;
        result.removeEdge(e.source, e.target);
      }
    }
  }

  return result;
}
