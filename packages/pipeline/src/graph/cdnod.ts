/**
 * CD-NOD — Causal Discovery from Non-stationary / Heterogeneous Data.
 *
 * Extends the PC algorithm to handle data with distribution shifts
 * (e.g., multiple domains, time slices). The domain index is treated
 * as an additional variable to detect changing causal mechanisms.
 *
 * Key insight: when a causal mechanism changes across domains,
 * conditioning on the domain variable breaks the independence,
 * revealing the changing edge.
 *
 * Especially valuable for AIOps scenarios where system behavior
 * shifts between normal/incident/upgrade states.
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
} from '@agentix-e/causality-analyzer-core';

export interface CDNODConfig {
  alpha?: number;
  maxDegree?: number;
  /** Domain labels for each observation (e.g., time slice index) */
  domains?: number[];
}

export function cdnodAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: CDNODConfig = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph; changingEdges: Map<string, boolean> } {
  const alpha = config.alpha ?? 0.05;
  const maxDegree = config.maxDegree ?? -1;
  const domains = config.domains ?? [];
  const n = nodeNames.length;
  const N = data.rows;

  const g = new CausalGraph(nodeNames);
  const changingEdges = new Map<string, boolean>();

  // Phase 1: Skeleton detection (PC-style)
  // Start with a complete undirected graph
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      g.undirectedEdge(nodeNames[i], nodeNames[j]);

  // Remove edges where variables are unconditionally independent
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      if (matrixFisherZ(data, i, j, []) > alpha)
        g.removeEdge(nodeNames[i], nodeNames[j]);

  let depth = 1;
  const maxDepth = maxDegree === -1 ? n : maxDegree;
  let changed = true;
  const sepSet = new Map<string, Set<string>>();

  while (changed && depth <= maxDepth) {
    changed = false;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (!g.hasEdge(nodeNames[i], nodeNames[j])) continue;
        const neighbors = g.neighbors(nodeNames[i]).filter(c => c !== nodeNames[j]);
        if (neighbors.length < depth) continue;
        const subsets = combinations(neighbors, depth);
        for (const S of subsets) {
          const sIdx = S.map(s => nodeNames.indexOf(s));
          if (matrixFisherZ(data, i, j, sIdx) > alpha) {
            g.removeEdge(nodeNames[i], nodeNames[j]);
            g.removeEdge(nodeNames[j], nodeNames[i]);
            sepSet.set(`${Math.min(i, j)}-${Math.max(i, j)}`, new Set(S));
            changed = true; break;
          }
        }
      }
    }
    depth++;
  }

  // Phase 2: V-structure orientation
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (g.hasEdge(nodeNames[i], nodeNames[j])) continue;
      for (let k = 0; k < n; k++) {
        if (k === i || k === j) continue;
        if (!g.hasEdge(nodeNames[i], nodeNames[k]) || !g.hasEdge(nodeNames[j], nodeNames[k])) continue;
        const key = `${Math.min(i, j)}-${Math.max(i, j)}`;
        const sep = sepSet.get(key);
        if (!sep || !sep.has(nodeNames[k])) {
          g.orientEdge(nodeNames[i], nodeNames[k]);
          g.orientEdge(nodeNames[j], nodeNames[k]);
        }
      }
    }
  }

  // Phase 3: Non-stationarity detection
  if (domains.length === N) {
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (!g.hasEdge(nodeNames[i], nodeNames[j]) || !g.hasEdge(nodeNames[j], nodeNames[i])) continue;

        let domainChange = false;
        const uniqueDomains = [...new Set(domains)];
        for (const d of uniqueDomains) {
          const dRows: number[] = [];
          for (let r = 0; r < N; r++) if (domains[r] === d) dRows.push(r);

          if (dRows.length < 10) continue;

          const subData = new Matrix(dRows.length, data.columns);
          for (let ri = 0; ri < dRows.length; ri++)
            for (let c = 0; c < data.columns; c++)
              subData.set(ri, c, data.get(dRows[ri], c));

          const pDomain = matrixFisherZ(subData, i, j, []);
          if (pDomain < alpha) {
            domainChange = true; break;
          }
        }

        if (domainChange) {
          const key = `${nodeNames[i]}↔${nodeNames[j]}`;
          changingEdges.set(key, true);
        }
      }
    }
  }

  // Convert CPDAG to DAG using Dor-Tarsi (1992) algorithm
  const dag = g.pdag2dag();

  if (domainKnowledge) dag.applyDomainKnowledge(domainKnowledge);
  return { graph: dag, changingEdges };
}

// ── Adapter: ml-matrix → number[][] for core fisherZTest ───────────

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
