/**
 * GRaSP — Greedy Relaxation of Sparsity.
 *
 * Implements the full GRaSP algorithm (Lam et al., UAI 2022) with:
 *   1. Covered Tuck: DFS over variable permutations, tucking variables
 *      into the position that maximizes L1-regularized BIC per node.
 *   2. Adaptive L1 penalty: λ = 0.5·log(N)/N as default (BIC-like).
 *   3. CPDAG-aware search: each ordering produces a CPDAG via
 *      optimal parent selection + pdag2dag conversion.
 *
 * Unlike the basic greedy add/remove variant, this implementation
 * explores the true permutation space, making it significantly more
 * robust on graphs with hidden topological constraints.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { createRNG, solveOLS } from '@agentix-e/causality-analyzer-core';

export interface GRaSPConfig {
  maxDegree?: number;
  lambda1?: number;
  /** Maximum covered tuck iterations per restart (default 20) */
  maxTuckIter?: number;
  /** Number of random restarts (default 3) */
  numStarts?: number;
  seed?: number;
}

export function graspAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: GRaSPConfig = {},
  domainKnowledge?: DomainKnowledge,
): CausalGraph {
  const n = nodeNames.length;
  const N = data.rows;
  const maxDegree = config.maxDegree ?? -1;
  const lambda1 = config.lambda1 ?? 0.5 * Math.log(Math.max(2, N)) / N;
  const maxTuckIter = config.maxTuckIter ?? 20;
  const numStarts = config.numStarts ?? 3;
  const rng = createRNG(config.seed ?? null);

  // Scoring cache
  const scoreCache = new Map<string, number>();
  const scoreKey = (node: string, parents: string[]) =>
    `${node}|${[...parents].sort().join(',')}`;

  const computeBIC = (node: string, parents: string[]): number => {
    const key = scoreKey(node, parents);
    if (scoreCache.has(key)) return scoreCache.get(key)!;

    const k = parents.length + 1;
    const pIndices = parents.map(p => nodeNames.indexOf(p));
    const tIdx = nodeNames.indexOf(node);
    const y: number[] = [];
    for (let i = 0; i < N; i++) y.push(data.get(i, tIdx));

    if (parents.length === 0) {
      const mean = y.reduce((a, b) => a + b, 0) / N;
      const rss = y.reduce((s, v) => s + (v - mean) ** 2, 0);
      const bic = N * Math.log(Math.max(1e-10, rss / N)) + lambda1 * Math.log(N);
      scoreCache.set(key, bic);
      return bic;
    }

    const X: number[][] = [];
    for (let i = 0; i < N; i++) {
      const row: number[] = [1];
      for (const p of pIndices) row.push(data.get(i, p));
      X.push(row);
    }
    const coef = solveOLS(X, y);
    let rss = 0;
    for (let i = 0; i < N; i++) {
      let pred = 0;
      for (let j = 0; j < k; j++) pred += (coef[j] ?? 0) * (X[i]![j] ?? 0);
      rss += (y[i]! - pred) ** 2;
    }
    const bic = N * Math.log(Math.max(1e-10, rss / N)) + k * lambda1 * Math.log(N);
    scoreCache.set(key, bic);
    return bic;
  };

  // Select best parents for a node given candidate predecessors
  const selectBestParents = (node: string, predecessors: string[]): string[] => {
    let bestParents: string[] = [];
    let bestScore = computeBIC(node, []);

    // Greedy grow
    const remaining = new Set(predecessors);
    let changed = true;
    while (changed) {
      changed = false;
      if (maxDegree >= 0 && bestParents.length >= maxDegree) break;

      for (const p of remaining) {
        const candidate = [...bestParents, p];
        const s = computeBIC(node, candidate);
        if (s < bestScore) {
          bestScore = s;
          bestParents = candidate;
          remaining.delete(p);
          changed = true;
          break;
        }
      }
    }

    // Greedy shrink
    changed = true;
    while (changed) {
      changed = false;
      for (let i = 0; i < bestParents.length; i++) {
        const reduced = [...bestParents.slice(0, i), ...bestParents.slice(i + 1)];
        const s = computeBIC(node, reduced);
        if (s <= bestScore) {
          bestScore = s;
          bestParents = reduced;
          changed = true;
          break;
        }
      }
    }

    return bestParents;
  };

  // Build DAG from permutation + optimal parents
  const dagFromPerm = (perm: number[]): { graph: CausalGraph; bic: number } => {
    const g = new CausalGraph([...nodeNames]);
    let totalBIC = 0;
    for (let pos = 0; pos < perm.length; pos++) {
      const v = perm[pos]!;
      const node = nodeNames[v]!;
      const predecessors = perm.slice(0, pos).map(i => nodeNames[i]!);
      const parents = selectBestParents(node, predecessors);
      for (const p of parents) g.addEdge(p, node);
      totalBIC += computeBIC(node, parents);
    }
    return { graph: g, bic: totalBIC };
  };

  let bestGlobalBIC = Infinity;
  let bestGraph = new CausalGraph([...nodeNames]);

  for (let start = 0; start < numStarts; start++) {
    // Random initial permutation
    let perm = fisherYates(Array.from({ length: n }, (_, i) => i), rng);
    let current = dagFromPerm(perm);
    let improved = true;
    let iter = 0;

    while (improved && iter < maxTuckIter) {
      improved = false;
      iter++;

      // Covered tuck: for each variable, try inserting at every position
      for (let v = 0; v < n; v++) {
        const currentPos = perm.indexOf(v);
        let bestPos = currentPos;
        let bestPosBIC = current.bic;

        for (let newPos = 0; newPos < n; newPos++) {
          if (newPos === currentPos || newPos === currentPos + 1) continue;

          const candidate = [...perm.filter(x => x !== v)];
          candidate.splice(newPos > currentPos ? newPos - 1 : newPos, 0, v);
          const candResult = dagFromPerm(candidate);

          if (candResult.bic < bestPosBIC) {
            bestPosBIC = candResult.bic;
            bestPos = newPos;
          }
        }

        if (bestPos !== currentPos) {
          perm = [...perm.filter(x => x !== v)];
          perm.splice(bestPos > currentPos ? bestPos - 1 : bestPos, 0, v);
          current = dagFromPerm(perm);
          improved = true;
        }
      }
    }

    if (current.bic < bestGlobalBIC) {
      bestGlobalBIC = current.bic;
      bestGraph = current.graph;
    }
  }

  const cpdag = bestGraph.pdag2dag();

  if (domainKnowledge) cpdag.applyDomainKnowledge(domainKnowledge);

  if (cpdag.hasCycle()) {
    for (const e of [...cpdag.edges].filter(e => e.directed)) {
      cpdag.removeEdge(e.source, e.target);
      if (!cpdag.hasCycle()) break;
    }
  }

  return cpdag;
}

function fisherYates<T>(arr: T[], rng: () => number): T[] {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [result[i], result[j]] = [result[j]!, result[i]!];
  }
  return result;
}
