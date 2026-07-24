/**
 * TS-ICD — Time-Series Instantaneous Causal Discovery.
 *
 * Extends constraint-based causal discovery to time-series data
 * by constructing a time-windowed design matrix with lagged variables.
 * Detects both contemporaneous (same time-slice) and lagged causal
 * relationships using conditional independence tests.
 *
 * Core for AIOps: monitoring metrics are inherently time-series,
 * and root causes may manifest with time delays.
 *
 * Algorithm:
 *  1. Create lagged copies: X(t), X(t-1), ..., X(t-maxLag)
 *  2. Run PC-style skeleton on the expanded variable set
 *  3. Orient contemporaneous edges (v-structures + Meek's rules)
 *  4. Orient lagged edges (always forward in time, no cycles possible)
 *
 * Reference: Runge et al. (2019). "Detecting and quantifying causal
 * associations in large nonlinear time series datasets." Science Advances.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { normalCDF, combinations } from '@agentix-e/causality-analyzer-core';

export interface TSConfig {
  alpha?: number;
  maxDegree?: number;
  /** Maximum lag (number of time steps backwards) */
  maxLag?: number;
  /** Minimum window size for CI tests */
  minWindowSize?: number;
}

export interface TimeSeriesEdge {
  source: string;
  target: string;
  lag: number;       // 0 = contemporaneous, >0 = source leads target by `lag` steps
  weight: number;
}

export interface TSResult {
  /** Contemporaneous causal graph (lag=0 edges) */
  contemporaneous: CausalGraph;
  /** All edges including lagged */
  edges: TimeSeriesEdge[];
}

/**
 * Run time-series causal discovery.
 *
 * @param data — T×d matrix (rows = time, columns = variables)
 * @param nodeNames — variable names
 * @param config — algorithm parameters
 */
export function tsIcdAlgorithm(
  data: number[][],
  nodeNames: string[],
  config: TSConfig = {},
  domainKnowledge?: DomainKnowledge,
): TSResult {
  const alpha = config.alpha ?? 0.05;
  const maxDegree = config.maxDegree ?? -1;
  const maxLag = config.maxLag ?? 2;
  const d = nodeNames.length;
  const T = data.length;
  if (T < 10) return { contemporaneous: new CausalGraph(nodeNames), edges: [] };

  // ── Build time-windowed data matrix ───────────────────────────────
  // Columns: [var0(t), var1(t), ..., var0(t-1), var1(t-1), ...]
  const expandedNames: string[] = [];
  const effectiveT = T - maxLag;
  const expanded = new Matrix(effectiveT, d * (maxLag + 1));

  for (let j = 0; j < d; j++) {
    for (let lag = 0; lag <= maxLag; lag++) {
      const colIdx = j * (maxLag + 1) + lag;
      expandedNames.push(lag === 0 ? nodeNames[j]! : `${nodeNames[j]}(t-${lag})`);
      for (let t = 0; t < effectiveT; t++) {
        expanded.set(t, colIdx, data[t + maxLag - lag]?.[j] ?? NaN);
      }
    }
  }

  const N = effectiveT;
  if (N < 20) return { contemporaneous: new CausalGraph(nodeNames), edges: [] };

  const allNodes = expandedNames;
  const allN = allNodes.length;

  // ── Phase 1: Skeleton (PC on expanded variables) ─────────────────
  const sepSet = new Map<string, Set<string>>();
  const adj = new Set<string>();

  // Initialize complete graph among contemporaneous variables
  for (let i = 0; i < d; i++)
    for (let j = i + 1; j < d; j++)
      adj.add(`${i}-${j}`);

  // Contemporaneous ↔ lagged: add edges from lagged to contemporaneous only
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      for (let lag = 1; lag <= maxLag; lag++) {
        const lagIdx = j * (maxLag + 1) + lag;
        adj.add(`${i}-${lagIdx}`);
      }
    }
  }

  // Remove contemporaneous edges via CI tests
  let depth = 0;
  const maxDepth = maxDegree === -1 ? allN : maxDegree;
  let changed = true;

  while (changed && depth <= maxDepth) {
    changed = false;
    const toRemove: string[] = [];

    for (const key of adj) {
      const [ii, jj] = key.split('-').map(Number);
      const i = ii!, j = jj!;
      if (i >= d && j >= d) continue; // lag-lag pairs not tested

      const iNeighbors = getNeighbors(adj, i, allN);
      const jNeighbors = getNeighbors(adj, j, allN);
      const cands = [...new Set([...iNeighbors, ...jNeighbors])].filter(c => c !== i && c !== j);

      if (cands.length < depth) continue;
      const subsets = combinations(cands.map(c => allNodes[c]!), depth);

      for (const S of subsets) {
        const sIdx = S.map(s => allNodes.indexOf(s));
        const p = fisherZ(expanded, i, j, sIdx);
        if (p > alpha) {
          toRemove.push(key);
          sepSet.set(`${Math.min(i,j)}-${Math.max(i,j)}`, new Set(S));
          break;
        }
      }
    }

    for (const key of toRemove) { adj.delete(key); changed = true; }
    depth++;
  }

  // ── Phase 2: Orientation ────────────────────────────────────────
  const g = new CausalGraph(nodeNames);
  const edges: TimeSeriesEdge[] = [];

  // Contemporaneous v-structures
  for (let i = 0; i < d; i++) {
    for (let j = i + 1; j < d; j++) {
      if (adj.has(`${i}-${j}`)) continue; // i,j adjacent
      for (let k = 0; k < d; k++) {
        if (k === i || k === j) continue;
        if (!adj.has(`${i}-${k}`) || !adj.has(`${j}-${k}`)) continue;
        const key = `${Math.min(i,j)}-${Math.max(i,j)}`;
        const sep = sepSet.get(key);
        if (!sep || !sep.has(nodeNames[k]!)) {
          g.toUndirected(nodeNames[i]!, nodeNames[k]!);
          g.toUndirected(nodeNames[j]!, nodeNames[k]!);
        }
      }
    }
  }

  // Orient undirected contemporaneous edges by BIC
  for (let i = 0; i < d; i++) {
    for (let j = i + 1; j < d; j++) {
      if (!adj.has(`${i}-${j}`)) continue;
      const bicIJ = bicNodeScore(expanded, i, [j], N);
      const bicJI = bicNodeScore(expanded, j, [i], N);
      if (bicIJ < bicJI) {
        g.toUndirected(nodeNames[i]!, nodeNames[j]!);
        edges.push({ source: nodeNames[i]!, target: nodeNames[j]!, lag: 0, weight: 1 });
      } else {
        g.toUndirected(nodeNames[j]!, nodeNames[i]!);
        edges.push({ source: nodeNames[j]!, target: nodeNames[i]!, lag: 0, weight: 1 });
      }
    }
  }

  // Lagged edges: always source → target (past → present)
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      for (let lag = 1; lag <= maxLag; lag++) {
        const lagIdx = j * (maxLag + 1) + lag;
        const key = `${i}-${lagIdx}`;
        if (!adj.has(key)) continue;

        // Compute partial correlation to determine edge weight
        const rho = partialCorrCI(expanded, i, lagIdx, N);
        edges.push({
          source: nodeNames[j]!,
          target: nodeNames[i]!,
          lag,
          weight: Math.abs(rho),
        });
      }
    }
  }

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);
  return { contemporaneous: g, edges };
}

// ── Helpers ─────────────────────────────────────────────────────────

function getNeighbors(adj: Set<string>, node: number, n: number): number[] {
  const result: number[] = [];
  for (let k = 0; k < n; k++) {
    if (k === node) continue;
    const a = Math.min(node, k), b = Math.max(node, k);
    if (adj.has(`${a}-${b}`)) result.push(k);
  }
  return result;
}

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

  const rho = partialCorr2D(cov, 0, 1);
  if (Math.abs(rho) >= 1) return 0;
  const z = 0.5 * Math.log((1 + rho) / (1 - rho)) * Math.sqrt(N - k - 3);
  return 2 * (1 - normalCDF(Math.abs(z)));
}

function partialCorr2D(cov: number[][], i: number, j: number): number {
  const m = cov.length;
  if (m === 2) {
    const d = cov[i]![i]! * cov[j]![j]!;
    return d > 0 ? cov[i]![j]! / Math.sqrt(d) : 0;
  }
  const prec = invert2D(cov);
  const d = prec[i]![i]! * prec[j]![j]!;
  return d > 0 ? -prec[i]![j]! / Math.sqrt(d) : 0;
}

function partialCorrCI(data: Matrix, i: number, j: number, _N: number): number {
  const sub = data.subMatrixColumn([i, j]);
  const N = sub.rows;
  let m1 = 0, m2 = 0;
  for (let r = 0; r < N; r++) { m1 += sub.get(r, 0); m2 += sub.get(r, 1); }
  m1 /= N; m2 /= N;
  let cov = 0, v1 = 0, v2 = 0;
  for (let r = 0; r < N; r++) {
    const d1 = sub.get(r, 0) - m1, d2 = sub.get(r, 1) - m2;
    cov += d1 * d2; v1 += d1 * d1; v2 += d2 * d2;
  }
  cov /= (N - 1); v1 /= (N - 1); v2 /= (N - 1);
  const d = v1 * v2;
  return d > 0 ? cov / Math.sqrt(d) : 0;
}

function invert2D(m: number[][]): number[][] {
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

function bicNodeScore(data: Matrix, target: number, parents: number[], N: number): number {
  const y: number[] = [];
  for (let i = 0; i < N; i++) y.push(data.get(i, target));
  const yMean = y.reduce((a, b) => a + b, 0) / N;
  if (parents.length === 0) {
    const sst = y.reduce((s, v) => s + (v - yMean) ** 2, 0);
    return N * Math.log(Math.max(1e-10, sst / N)) + Math.log(N);
  }
  const X: number[][] = [];
  for (let i = 0; i < N; i++) {
    const row = [1];
    for (const p of parents) row.push(data.get(i, p));
    X.push(row);
  }
  const k = parents.length + 1;
  const coef = solveOLS2D(X, y, k);
  const sse = y.reduce((s, v, i) => s + (v - coef.reduce((sc, c, j) => sc + c * (X[i]?.[j] ?? 0), 0)) ** 2, 0);
  return N * Math.log(Math.max(1e-10, sse / N)) + k * Math.log(N);
}

function solveOLS2D(A: number[][], b: number[], k: number): number[] {
  const aug = A.map((r, i) => [...r, b[i] ?? 0]);
  for (let c = 0; c < k; c++) {
    let pivot = c;
    for (let r = c + 1; r < k; r++) if (Math.abs(aug[r]![c]!) > Math.abs(aug[pivot]![c]!)) pivot = r;
    [aug[c], aug[pivot]] = [aug[pivot]!, aug[c]!];
    if (Math.abs(aug[c]![c]!) < 1e-12) continue;
    for (let j = c; j <= k; j++) aug[c]![j]! /= aug[c]![c]!;
    for (let r = 0; r < k; r++) {
      if (r === c) continue;
      for (let j = c; j <= k; j++) aug[r]![j]! -= aug[r]![c]! * aug[c]![j]!;
    }
  }
  return aug.map(r => r[k]!);
}
