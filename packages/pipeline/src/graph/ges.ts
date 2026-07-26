/**
 * GES (Greedy Equivalence Search) — CPDAG-space causal discovery.
 *
 * Reference: Chickering (2002). "Optimal Structure Identification
 *   With Greedy Search." JMLR 3:507-554.
 *
 * Scoring: Covariance-matrix-based Gaussian BIC (ddof=0), matching
 * gCastle's BICScore._bic_by_scatter implementation. This preserves
 * asymmetric conditional covariances — crucial for direction identification
 * in linear Gaussian data.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';

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

  // ── Precompute covariance matrix (gCastle approach, ddof=0) ────────
  const means = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let r = 0; r < N; r++) sum += data.get(r, i);
    means[i] = sum / N;
  }

  // cov[i][j] = E[(xi - mx)(xj - mx)] with ddof=0
  const cov: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    cov[i] = new Array(n).fill(0);
    for (let j = 0; j <= i; j++) {
      let val = 0;
      for (let r = 0; r < N; r++) {
        val += (data.get(r, i) - means[i]!) * (data.get(r, j) - means[j]!);
      }
      cov[i]![j] = val / N;
      if (i !== j) cov[j]![i] = val / N;
    }
  }

  // ── BIC via covariance (gCastle BICScore._bic_by_scatter) ──────────

  const scoreCache = new Map<string, number>();
  const sk = (node: string, parents: string[]): string =>
    `${node}|${[...parents].sort().join(',')}`;

  const bicLocal = (yIdx: number, paIdx: number[]): number => {
    const k = paIdx.length;
    const key = `${yIdx}|${[...paIdx].sort().join(',')}`;
    if (scoreCache.has(key)) return scoreCache.get(key)!;

    // sigma = cov[y][y] (baseline variance)
    let sigma = cov[yIdx]![yIdx]!;

    if (k > 0) {
      // Extract sub-matrices
      // pa_cov = covariance among parents (k×k)
      // y_cov = covariance between y and parents (k×1)
      const paCov: number[][] = [];
      for (let i = 0; i < k; i++) {
        const row: number[] = [];
        for (let j = 0; j < k; j++) row.push(cov[paIdx[i]!]![paIdx[j]!]!);
        paCov.push(row);
      }
      const yCov: number[] = [];
      for (let i = 0; i < k; i++) yCov.push(cov[yIdx]![paIdx[i]!]!);

      // Solve paCov * coef = yCov using Gaussian elimination
      const coef = solveLinear(paCov, yCov);

      // sigma = cov(y,y) - sum(coef[i] * cov(y, pa[i]))
      for (let i = 0; i < k; i++) sigma -= (coef[i] ?? 0) * yCov[i]!;
    }

    // BIC = -(N * (1 + log(sigma)) + (k + 1) * log(N))
    // Matches gCastle: -(self.n * (1 + np.log(sigma)) + (k + 1) * np.log(self.n))
    const bic = -(N * (1 + Math.log(Math.max(1e-12, sigma))) + (k + 1) * Math.log(Math.max(2, N)));
    scoreCache.set(key, bic);
    return bic;
  };

  const nodeIdx = new Map(nodeNames.map((name, i) => [name, i]));

  // ── Meek Rules ────────────────────────────────────────────────────

  const adjacent = (g: CausalGraph, u: string, v: string): boolean =>
    g.hasEdge(u, v) || g.hasEdge(v, u);

  const meekPropagate = (graph: CausalGraph): void => {
    let changed = true;
    let safety = 0;
    while (changed && safety++ < 100) {
      changed = false;
      // R1: X→Y—Z, X⟂Z → Y→Z
      for (const y of graph.nodes) {
        const xList = [...graph.parents(y)];
        for (const z of graph.neighbors(y)) {
          if (graph.hasEdge(z, y)) continue;
          for (const x of xList) {
            if (!adjacent(graph, x, z)) {
              graph.removeEdge(z, y);
              changed = true;
              break;
            }
          }
        }
      }
      // R2: X→Y→Z and X—Z → X→Z
      if (!changed) {
        for (const y of graph.nodes) {
          for (const x of graph.parents(y)) {
            for (const z of graph.children(y)) {
              if (adjacent(graph, x, z) && !graph.hasEdge(x, z) && !graph.hasEdge(z, x)) {
                graph.removeEdge(z, x);
                graph.addEdge(x, z);
                changed = true;
              }
            }
          }
        }
      }
    }
  };

  // ── Working graph ──────────────────────────────────────────────────
  const g = new CausalGraph([...nodeNames]);
  const logN = Math.log(Math.max(2, N));
  const minDelta = logN;

  // ── Phase 1: Forward ──────────────────────────────────────────────
  let improved = true;
  let iter = 0;

  while (improved && iter++ < 200) {
    improved = false;
    let bestDelta = minDelta;
    let bestFrom = '', bestTo = '';

    for (let i = 0; i < n; i++) {
      const u = nodeNames[i]!;
      const ui = nodeIdx.get(u)!;
      const uParents = [...g.parents(u)];
      if (maxDegree >= 0 && uParents.length >= maxDegree) continue;

      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const v = nodeNames[j]!;
        const vi = nodeIdx.get(v)!;
        if (adjacent(g, u, v)) continue;

        const vParents = [...g.parents(v)];

        // v → u
        if (maxDegree < 0 || uParents.length < maxDegree) {
          const old = bicLocal(ui, uParents.map(p => nodeIdx.get(p)!));
          const delta = bicLocal(ui, [...uParents.map(p => nodeIdx.get(p)!), vi]) - old;
          if (delta > bestDelta) { bestDelta = delta; bestFrom = v; bestTo = u; }
        }
        // u → v
        if (maxDegree < 0 || vParents.length < maxDegree) {
          const old = bicLocal(vi, vParents.map(p => nodeIdx.get(p)!));
          const delta = bicLocal(vi, [...vParents.map(p => nodeIdx.get(p)!), ui]) - old;
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

  // ── Phase 2: Backward ─────────────────────────────────────────────
  improved = true;
  iter = 0;

  while (improved && iter++ < 200) {
    improved = false;
    let bestDelta = minDelta;
    let bestSource = '', bestTarget = '';

    for (const node of nodeNames) {
      const ni = nodeIdx.get(node)!;
      const parents = [...g.parents(node)];
      if (parents.length === 0) continue;

      for (const p of parents) {
        const newParents = parents.filter(par => par !== p);
        const old = bicLocal(ni, parents.map(par => nodeIdx.get(par)!));
        const delta = bicLocal(ni, newParents.map(par => nodeIdx.get(par)!)) - old;
        if (delta > bestDelta) { bestDelta = delta; bestSource = p; bestTarget = node; }
      }
    }

    if (bestSource && bestTarget) {
      g.removeEdge(bestSource, bestTarget);
      g.removeEdge(bestTarget, bestSource);
      improved = true;
    }
  }

  // ── Phase 3: Turning ──────────────────────────────────────────────
  improved = true;
  iter = 0;

  while (improved && iter++ < 100) {
    improved = false;
    let bestDelta = minDelta;
    let bestFrom = '', bestTo = '';

    for (const node of nodeNames) {
      const ni = nodeIdx.get(node)!;
      const parents = [...g.parents(node)];
      for (const p of parents) {
        const pi = nodeIdx.get(p)!;
        if (g.hasDirectedPath(node, p)) continue;

        const newParents = parents.filter(par => par !== p);
        const pParents = [...g.parents(p)];
        const oldScore = bicLocal(ni, parents.map(par => nodeIdx.get(par)!))
          + bicLocal(pi, pParents.map(par => nodeIdx.get(par)!));
        const newScore = bicLocal(ni, newParents.map(par => nodeIdx.get(par)!))
          + bicLocal(pi, [...pParents.map(par => nodeIdx.get(par)!), ni]);
        const delta = newScore - oldScore;

        if (delta > bestDelta) { bestDelta = delta; bestFrom = node; bestTo = p; }
      }
    }

    if (bestFrom && bestTo) {
      g.removeEdge(bestTo, bestFrom);
      g.addEdge(bestFrom, bestTo);
      improved = true;
    }
  }

  // ── Final ──────────────────────────────────────────────────────────
  let result = g.pdag2dag();
  if (domainKnowledge) result.applyDomainKnowledge(domainKnowledge);

  if (result.hasCycle()) {
    const topo = result.topologicalSort();
    const directedEdges = [...result.edges].filter(e => e.directed);
    for (const e of directedEdges) {
      if (!result.hasCycle()) break;
      const aIdx = topo.indexOf(e.source);
      const bIdx = topo.indexOf(e.target);
      if (aIdx >= 0 && bIdx >= 0 && aIdx >= bIdx) result.removeEdge(e.source, e.target);
    }
    if (result.hasCycle()) {
      for (const e of directedEdges) { if (!result.hasCycle()) break; result.removeEdge(e.source, e.target); }
    }
  }

  return result;
}

// Gaussian elimination with partial pivoting for small linear systems
function solveLinear(A: number[][], b: number[]): number[] {
  const n = A.length;
  const augmented: number[][] = A.map((row, i) => [...row, b[i]!]);

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(augmented[row]![col]!) > Math.abs(augmented[maxRow]![col]!)) maxRow = row;
    }
    [augmented[col], augmented[maxRow]] = [augmented[maxRow]!, augmented[col]!];

    const pivot = augmented[col]![col]!;
    if (Math.abs(pivot) < 1e-12) continue;

    for (let row = col + 1; row < n; row++) {
      const factor = augmented[row]![col]! / pivot;
      for (let j = col; j <= n; j++) augmented[row]![j] -= factor * augmented[col]![j]!;
    }
  }

  const x: number[] = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = augmented[i]![n]!;
    for (let j = i + 1; j < n; j++) sum -= augmented[i]![j]! * (x[j] ?? 0);
    x[i] = Math.abs(augmented[i]![i]!) < 1e-12 ? 0 : sum / augmented[i]![i]!;
  }
  return x;
}
