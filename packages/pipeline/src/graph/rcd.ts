/**
 * RCD — Reinforced Causal Discovery.
 *
 * Hybrid constraint-based + score-based causal discovery.
 * Phase 1: PC-style skeleton estimation (Fisher Z CI tests)
 * Phase 2: BIC-based reinforcement for edge orientation decisions.
 *
 * Unlike pure PC (which uses Meek's rules for orientation),
 * RCD uses a scoring approach to decide between competing
 * orientations, making it more robust with moderate sample sizes.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { erf, normalCDF, combinations } from '@agentix-e/causality-analyzer-core';

export interface RCDConfig {
  alpha?: number;
  maxDegree?: number;
}

export function rcdAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: RCDConfig = {},
  domainKnowledge?: DomainKnowledge,
): CausalGraph {
  const alpha = config.alpha ?? 0.05;
  const maxDegree = config.maxDegree ?? -1;
  const n = nodeNames.length;
  const N = data.rows;

  const g = new CausalGraph(nodeNames);

  // Phase 1: Skeleton (same as PC)
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      if (fisherZ(data, i, j, []) > alpha)
        g.undirectedEdge(nodeNames[i]!, nodeNames[j]!);

  let depth = 1;
  const maxDepth = maxDegree === -1 ? n : maxDegree;
  let changed = true;

  while (changed && depth <= maxDepth) {
    changed = false;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue;
        const neighbors = g.neighbors(nodeNames[i]!).filter(c => c !== nodeNames[j]);
        if (neighbors.length < depth) continue;
        const subsets = combinations(neighbors, depth);
        for (const S of subsets) {
          const p = fisherZ(data, i, j, S.map(s => nodeNames.indexOf(s)));
          if (p > alpha) {
            g.removeEdge(nodeNames[i]!, nodeNames[j]!);
            g.removeEdge(nodeNames[j]!, nodeNames[i]!);
            changed = true; break;
          }
        }
      }
    }
    depth++;
  }

  // Phase 2: BIC-based orientation (reinforcement)
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue;
      if (!g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue; // undirected
      const iName = nodeNames[i]!, jName = nodeNames[j]!;

      const bicIJ = bicScore(data, i, [j], N);
      const bicJI = bicScore(data, j, [i], N);

      if (bicIJ < bicJI) {
        g.toUndirected(iName, jName);
      } else if (bicJI < bicIJ) {
        g.toUndirected(jName, iName);
      }
    }
  }

  // V-structure detection
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (g.hasEdge(nodeNames[i]!, nodeNames[j]!) || g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;
      for (let k = 0; k < n; k++) {
        if (k === i || k === j) continue;
        if (!g.hasEdge(nodeNames[i]!, nodeNames[k]!) || !g.hasEdge(nodeNames[j]!, nodeNames[k]!)) continue;
        if (g.hasEdge(nodeNames[k]!, nodeNames[i]!) || g.hasEdge(nodeNames[k]!, nodeNames[j]!)) continue;
        const p = fisherZ(data, i, j, [k]);
        if (p <= alpha && g.hasEdge(nodeNames[i]!, nodeNames[k]!) && g.hasEdge(nodeNames[j]!, nodeNames[k]!)) {
          g.toUndirected(nodeNames[i]!, nodeNames[k]!);
          g.toUndirected(nodeNames[j]!, nodeNames[k]!);
        }
      }
    }
  }

  // Cycle safety
  if (g.hasCycle()) {
    for (const e of [...g.edges].filter(e => e.directed)) {
      g.removeEdge(e.source, e.target);
      if (!g.hasCycle()) break;
    }
  }

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);
  return g;
}

// ── Fisher Z test ──────────────────────────────────────────────────

function fisherZ(data: Matrix, i: number, j: number, condSet: number[]): number {
  const indices = [i, j, ...condSet];
  const k = condSet.length;
  const N = data.rows;

  const sub = data.subMatrixColumn(indices);
  const means = new Array(indices.length).fill(0);
  for (let c = 0; c < indices.length; c++) {
    let sum = 0; for (let r = 0; r < N; r++) sum += sub.get(r, c);
    means[c] = sum / N;
  }

  const cov = Array.from({length: indices.length}, () => new Array(indices.length).fill(0));
  for (let a = 0; a < indices.length; a++)
    for (let b = a; b < indices.length; b++) {
      let sum = 0;
      for (let r = 0; r < N; r++) sum += (sub.get(r, a) - means[a]!) * (sub.get(r, b) - means[b]!);
      cov[a]![b] = sum / (N - 1); cov[b]![a] = cov[a]![b]!;
    }

  const rho = partialCorr(cov, 0, 1);
  if (Math.abs(rho) >= 1) return 0;
  const z = 0.5 * Math.log((1 + rho) / (1 - rho)) * Math.sqrt(N - k - 3);
  return 2 * (1 - normalCDF(Math.abs(z)));
}

function partialCorr(cov: number[][], i: number, j: number): number {
  const m = cov.length;
  if (m === 2) return cov[i]![j]! / Math.sqrt(cov[i]![i]! * cov[j]![j]!);
  const prec = invert(cov);
  const r = -prec[i]![j]! / Math.sqrt(prec[i]![i]! * prec[j]![j]!);
  return Math.max(-1, Math.min(1, r));
}

function invert(m: number[][]): number[][] {
  const n = m.length;
  const aug = m.map((r, ri) => [...r, ...Array.from({length: n}, (_, ci) => ri === ci ? 1 : 0)]);
  for (let c = 0; c < n; c++) {
    let pivot = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(aug[r]![c]!) > Math.abs(aug[pivot]![c]!)) pivot = r;
    [aug[c], aug[pivot]] = [aug[pivot]!, aug[c]!];
    if (Math.abs(aug[c]![c]!) < 1e-12) continue;
    for (let j = c; j < 2 * n; j++) aug[c]![j]! /= aug[c]![c]!;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = aug[r]![c]!;
      for (let j = c; j < 2 * n; j++) aug[r]![j]! -= f * aug[c]![j]!;
    }
  }
  return aug.map(r => r.slice(n));
}

function bicScore(data: Matrix, target: number, parents: number[], N: number): number {
  const k = parents.length + 1;
  const y: number[] = [];
  for (let i = 0; i < N; i++) y.push(data.get(i, target));
  const yMean = y.reduce((a, b) => a + b, 0) / N;
  if (parents.length === 0) {
    const sst = y.reduce((s, v) => s + (v - yMean) ** 2, 0);
    return N * Math.log(Math.max(1e-10, sst / N)) + k * Math.log(N);
  }
  const X: number[][] = [];
  for (let i = 0; i < N; i++) {
    const row = [1];
    for (const p of parents) row.push(data.get(i, p));
    X.push(row);
  }
  const coef = solveOLS(X, y, k);
  const sse = y.reduce((s, v, i) => s + (v - coef.reduce((sc, c, j) => sc + c * (X[i]?.[j] ?? 0), 0)) ** 2, 0);
  return N * Math.log(Math.max(1e-10, sse / N)) + k * Math.log(N);
}

function solveOLS(A: number[][], b: number[], k: number): number[] {
  const aug = A.map((r, i) => [...r, b[i] ?? 0]);
  for (let c = 0; c < k; c++) {
    let pivot = c;
    for (let r = c + 1; r < k; r++) if (Math.abs(aug[r]![c]!) > Math.abs(aug[pivot]![c]!)) pivot = r;
    [aug[c], aug[pivot]] = [aug[pivot]!, aug[c]!];
    if (Math.abs(aug[c]![c]!) < 1e-12) continue;
    for (let j = c; j <= k; j++) aug[c]![j]! /= aug[c]![c]!;
    for (let r = 0; r < k; r++) {
      if (r === c) continue;
      const f = aug[r]![c]!;
      for (let j = c; j <= k; j++) aug[r]![j]! -= f * aug[c]![j]!;
    }
  }
  return aug.map(r => r[k]!);
}
