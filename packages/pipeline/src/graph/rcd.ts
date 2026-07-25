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
import {
  normalCDF,
  combinations,
  fisherZTest,
  partialCorrelationFromCov,
  invertMatrix,
  solveOLS as solveOLSCore,
} from '@agentix-e/causality-analyzer-core';

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

  // Phase 1: Skeleton
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      if (matrixFisherZ(data, i, j, []) > alpha)
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
          const p = matrixFisherZ(data, i, j, S.map(s => nodeNames.indexOf(s)));
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

  // Phase 2: BIC-based orientation
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue;
      if (!g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;
      const iName = nodeNames[i]!, jName = nodeNames[j]!;

      const bicIJ = rcdBicScore(data, i, [j], N);
      const bicJI = rcdBicScore(data, j, [i], N);

      if (bicIJ < bicJI) g.toUndirected(iName, jName);
      else if (bicJI < bicIJ) g.toUndirected(jName, iName);
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
        const p = matrixFisherZ(data, i, j, [k]);
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

// ── Adapters (ml-matrix → core) ─────────────────────────────────────

function matrixTo2D(data: Matrix): number[][] {
  const rows: number[][] = [];
  for (let r = 0; r < data.rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < data.columns; c++) row.push(data.get(r, c));
    rows.push(row);
  }
  return rows;
}

function matrixFisherZ(data: Matrix, i: number, j: number, condSet: number[]): number {
  return fisherZTest(matrixTo2D(data), i, j, condSet);
}

function rcdBicScore(data: Matrix, target: number, parents: number[], N: number): number {
  const k = parents.length + 1;
  const y: number[] = [];
  for (let r = 0; r < N; r++) y.push(data.get(r, target));
  const yMean = y.reduce((a, b) => a + b, 0) / N;

  if (parents.length === 0) {
    const sst = y.reduce((s, v) => s + (v - yMean) ** 2, 0);
    return N * Math.log(Math.max(1e-10, sst / N)) + k * Math.log(N);
  }

  const X: number[][] = [];
  for (let r = 0; r < N; r++) {
    const row = [1];
    for (const p of parents) row.push(data.get(r, p));
    X.push(row);
  }
  const coef = solveOLSCore(X, y);
  const sse = y.reduce((s, v, i) => s + (v - coef.reduce((sc, c, j) => sc + c * (X[i]?.[j] ?? 0), 0)) ** 2, 0);
  return N * Math.log(Math.max(1e-10, sse / N)) + k * Math.log(N);
}
