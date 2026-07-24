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
import { erf, normalCDF, combinations } from '@agentix-e/causality-analyzer-core';

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
  // Start with a complete undirected graph — edges will be removed when
  // variables are conditionally independent (p > alpha means fail to reject
  // the null hypothesis of independence).
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      g.undirectedEdge(nodeNames[i]!, nodeNames[j]!);

  // Remove edges where variables are unconditionally independent
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      if (fisherZ(data, i, j, [], alpha) > alpha)
        g.removeEdge(nodeNames[i]!, nodeNames[j]!);

  let depth = 1;
  const maxDepth = maxDegree === -1 ? n : maxDegree;
  let changed = true;
  const sepSet = new Map<string, Set<string>>();

  while (changed && depth <= maxDepth) {
    changed = false;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue;
        const neighbors = g.neighbors(nodeNames[i]!).filter(c => c !== nodeNames[j]);
        if (neighbors.length < depth) continue;
        const subsets = combinations(neighbors, depth);
        for (const S of subsets) {
          const sIdx = S.map(s => nodeNames.indexOf(s));
          if (fisherZ(data, i, j, sIdx, alpha) > alpha) {
            g.removeEdge(nodeNames[i]!, nodeNames[j]!);
            g.removeEdge(nodeNames[j]!, nodeNames[i]!);
            sepSet.set(`${Math.min(i,j)}-${Math.max(i,j)}`, new Set(S));
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
      if (g.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue;
      for (let k = 0; k < n; k++) {
        if (k === i || k === j) continue;
        if (!g.hasEdge(nodeNames[i]!, nodeNames[k]!) || !g.hasEdge(nodeNames[j]!, nodeNames[k]!)) continue;
        const key = `${Math.min(i,j)}-${Math.max(i,j)}`;
        const sep = sepSet.get(key);
        if (!sep || !sep.has(nodeNames[k]!)) {
          g.toUndirected(nodeNames[i]!, nodeNames[k]!);
          g.toUndirected(nodeNames[j]!, nodeNames[k]!);
        }
      }
    }
  }

  // Phase 3: Non-stationarity detection
  // For each undirected edge, test if conditioning on domain changes independence
  if (domains.length === N) {
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!) || !g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;

        // Test: are i and j independent within each domain?
        let domainChange = false;
        const uniqueDomains = [...new Set(domains)];
        for (const d of uniqueDomains) {
          const dRows: number[] = [];
          for (let r = 0; r < N; r++) if (domains[r] === d) dRows.push(r);

          if (dRows.length < 10) continue;

          // Extract domain-specific data
          const subData = new Matrix(dRows.length, data.columns);
          for (let ri = 0; ri < dRows.length; ri++)
            for (let c = 0; c < data.columns; c++)
              subData.set(ri, c, data.get(dRows[ri]!, c));

          const pDomain = fisherZ(subData, i, j, [], alpha);
          if (pDomain < alpha) {
            // Within-domain dependence exists — check if it changes across domains
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

function fisherZ(data: Matrix, i: number, j: number, condSet: number[], _alpha?: number): number {
  const indices = [i, j, ...condSet];
  const K = condSet.length;
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
  const z = 0.5 * Math.log((1 + rho) / (1 - rho)) * Math.sqrt(N - K - 3);
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
