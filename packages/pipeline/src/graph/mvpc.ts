/**
 * MVPC — Missing Value PC Algorithm.
 *
 * Extends the PC algorithm to handle datasets with missing values.
 * Uses pairwise-complete observations for correlation estimation
 * and a modified Fisher Z test that accounts for variable sample sizes.
 *
 * This is the 10th causal discovery algorithm in Causality Analyzer,
 * matching the algorithm count of causal-js and exceeding it in
 * causal inference + RCA + SCM capabilities.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { normalCDF, combinations } from '@agentix-e/causality-analyzer-core';

export interface MVPCConfig {
  alpha?: number;
  maxDegree?: number;
  /** Threshold for minimum pairwise observations (fraction of N) */
  minObsRatio?: number;
}

export function mvpcAlgorithm(
  data: number[][],
  nodeNames: string[],
  config: MVPCConfig = {},
  domainKnowledge?: DomainKnowledge,
): CausalGraph {
  const alpha = config.alpha ?? 0.05;
  const maxDegree = config.maxDegree ?? -1;
  const minObsRatio = config.minObsRatio ?? 0.1;
  const n = nodeNames.length;
  const N = data.length;
  const minObs = Math.max(3, Math.floor(N * minObsRatio));

  const g = new CausalGraph(nodeNames);

  // ── Phase 1: Skeleton with pairwise-complete CI tests ─────────────
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      g.undirectedEdge(nodeNames[i]!, nodeNames[j]!);

  let depth = 0;
  const maxDepth = maxDegree === -1 ? n : maxDegree;
  let changed = true;

  while (changed && depth <= maxDepth) {
    changed = false;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue;
        const neighbors = g.neighbors(nodeNames[i]!).filter(c => c !== nodeNames[j]);
        if (neighbors.length < depth) continue;

        if (depth === 0) {
          // Unconditional independence with missing values
          const p = missingValueCI(data, i, j, [], minObs);
          if (p > alpha) {
            g.removeEdge(nodeNames[i]!, nodeNames[j]!);
            g.removeEdge(nodeNames[j]!, nodeNames[i]!);
            changed = true;
          }
          continue;
        }

        const subsets = combinations(neighbors, depth);
        for (const S of subsets) {
          const p = missingValueCI(data, i, j, S.map(s => nodeNames.indexOf(s)), minObs);
          if (p > alpha) {
            g.removeEdge(nodeNames[i]!, nodeNames[j]!);
            g.removeEdge(nodeNames[j]!, nodeNames[i]!);
            changed = true;
            break;
          }
        }
      }
    }
    depth++;
  }

  // Phase 2: V-structure orientation
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (g.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue;
      for (let k = 0; k < n; k++) {
        if (k === i || k === j) continue;
        if (!g.hasEdge(nodeNames[i]!, nodeNames[k]!) || !g.hasEdge(nodeNames[j]!, nodeNames[k]!)) continue;
        if (g.hasEdge(nodeNames[k]!, nodeNames[i]!) || g.hasEdge(nodeNames[k]!, nodeNames[j]!)) continue;

        const p = missingValueCI(data, i, j, [k], minObs);
        if (p <= alpha) {
          g.toUndirected(nodeNames[i]!, nodeNames[k]!);
          g.toUndirected(nodeNames[j]!, nodeNames[k]!);
        }
      }
    }
  }

  // Phase 3: Meek's rules R1-R3
  let rulesChanged = true;
  while (rulesChanged) {
    rulesChanged = false;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!) || g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;
        for (let k = 0; k < n; k++) {
          if (!g.hasEdge(nodeNames[j]!, nodeNames[k]!) || !g.hasEdge(nodeNames[k]!, nodeNames[j]!)) continue;
          if (g.hasEdge(nodeNames[i]!, nodeNames[k]!) || g.hasEdge(nodeNames[k]!, nodeNames[i]!)) continue;
          g.toUndirected(nodeNames[j]!, nodeNames[k]!);
          rulesChanged = true;
        }
      }
    }
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < n; k++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[k]!) || !g.hasEdge(nodeNames[k]!, nodeNames[i]!)) continue;
        for (let j = 0; j < n; j++) {
          if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!) || g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;
          if (!g.hasEdge(nodeNames[j]!, nodeNames[k]!) || g.hasEdge(nodeNames[k]!, nodeNames[j]!)) continue;
          g.toUndirected(nodeNames[i]!, nodeNames[k]!);
          rulesChanged = true;
        }
      }
    }
  }

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);
  return g;
}

// ── Missing Value CI Test ──────────────────────────────────────────

function missingValueCI(
  data: number[][], i: number, j: number, condSet: number[], minObs: number,
): number {
  const indices = [i, j, ...condSet];
  const m = indices.length;
  const N = data.length;

  // Pairwise-complete observation counts
  const pairCounts: number[][] = Array.from({length: m}, () => new Array(m).fill(0));
  const means = new Array(m).fill(0);
  const counts = new Array(m).fill(0);

  // Compute column means (using available values)
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < m; c++) {
      const idx = indices[c]!;
      const v = data[r]![idx];
      if (!Number.isNaN(v) && v !== null && v !== undefined) {
        means[c] += v;
        counts[c]!++;
      }
    }
  }
  for (let c = 0; c < m; c++) means[c] /= Math.max(1, counts[c]!);

  // Compute pairwise covariances
  const cov: number[][] = Array.from({length: m}, () => new Array(m).fill(0));
  for (let a = 0; a < m; a++) {
    for (let b = a; b < m; b++) {
      let sum = 0, cnt = 0;
      for (let r = 0; r < N; r++) {
        const va = data[r]?.[indices[a]!];
        const vb = data[r]?.[indices[b]!];
        if (!isValid(va) || !isValid(vb)) continue;
        sum += (va - means[a]!) * (vb - means[b]!);
        cnt++;
      }
      pairCounts[a]![b] = cnt;
      pairCounts[b]![a] = cnt;
      cov[a]![b] = cnt > 1 ? sum / (cnt - 1) : 0;
      cov[b]![a] = cov[a]![b]!;
    }
  }

  // Check minimum observations
  if (pairCounts[0]![1]! < minObs) return 0; // insufficient data → treat as dependent

  // Partial correlation via precision matrix
  const rho = partialCorrMV(cov, 0, 1);
  if (Math.abs(rho) >= 1) return 0;

  const k = condSet.length;
  const effN = pairCounts[0]![1]!;
  const z = 0.5 * Math.log((1 + rho) / (1 - rho)) * Math.sqrt(effN - k - 3);
  return 2 * (1 - normalCDF(Math.abs(z)));
}

function isValid(v: unknown): v is number {
  return v !== null && v !== undefined && !Number.isNaN(v);
}

function partialCorrMV(cov: number[][], i: number, j: number): number {
  const m = cov.length;
  if (m === 2) {
    const denom = Math.sqrt(cov[i]![i]! * cov[j]![j]!);
    return denom > 0 ? cov[i]![j]! / denom : 0;
  }
  const prec = invertMV(cov);
  const denom = Math.sqrt(prec[i]![i]! * prec[j]![j]!);
  return denom > 0 ? -prec[i]![j]! / denom : 0;
}

function invertMV(m: number[][]): number[][] {
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
