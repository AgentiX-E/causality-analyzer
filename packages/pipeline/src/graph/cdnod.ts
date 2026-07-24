/**
 * CD-NOD — Causal Discovery for Non-stationary/heterogeneous Data.
 *
 * Based on Huang et al. (AAAI 2020):
 * "Causal Discovery from Heterogeneous/Nonstationary Data"
 *
 * Extends PC algorithm with domain-index-aware conditional independence
 * testing. When data comes from multiple environments (domains), CD-NOD
 * uses domain-varying Fisher Z that accounts for distribution shifts
 * across domains, preventing spurious edges that PC would discover.
 *
 * Key insight: domain shift creates artificial correlations that
 * standard CI tests mistake for causal relationships. CD-NOD
 * separates these by testing CI conditional on the domain index.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { normalCDF } from '@agentix-e/causality-analyzer-core';
import { CausalGraph } from './causal-graph.js';

export interface CDNODConfig {
  /** Significance level */
  alpha?: number;
  /** Maximum conditioning set size (-1 = unlimited) */
  maxDegree?: number;
}

/**
 * Domain-varying Fisher Z test.
 *
 * Tests X ⟂ Y | Z, D where D is the domain index.
 * Domain-varying means we estimate partial correlations that are
 * sensitive to whether the relationship changes across domains.
 *
 * Strategy: Include domain index d as an additional conditioning variable.
 * A relationship that appears in pooled data but disappears when
 * conditioning on domain is likely a domain-shift artifact.
 */
function domainVaryingFisherZ(
  data: Matrix,
  i: number,
  j: number,
  condSet: number[],
  domainIdx: number,
): number {
  const n = data.rows;
  // Include domain index in conditioning set
  const allCond = [...condSet, domainIdx];
  const indices = [i, j, ...allCond];
  const k = allCond.length;

  // Extract sub-matrix
  const subColumns = indices.map(idx => {
    const col = new Float64Array(n);
    for (let r = 0; r < n; r++) col[r] = data.get(r, idx);
    return col;
  });

  // Compute column means
  const means = subColumns.map(col => col.reduce((a: number, b: number) => a + b, 0) / n);

  // Compute covariance matrix
  const p = indices.length;
  const cov = new Array(p).fill(0).map(() => new Array(p).fill(0));
  for (let a = 0; a < p; a++) {
    for (let b = a; b < p; b++) {
      let sum = 0;
      const colA = subColumns[a]!;
      const colB = subColumns[b]!;
      const mA = means[a]!;
      const mB = means[b]!;
      for (let r = 0; r < n; r++) {
        sum += (colA[r]! - mA) * (colB[r]! - mB);
      }
      cov[a]![b] = sum / (n - 1);
      cov[b]![a] = cov[a]![b]!;
    }
  }

  // Partial correlation using precision matrix
  const prec = invertWithPartial(cov);
  const rho = -prec[0]![1]! / Math.sqrt(Math.max(1e-12, prec[0]![0]! * prec[1]![1]!));
  const rhoClamped = Math.max(-1, Math.min(1, rho));

  if (Math.abs(rhoClamped) >= 1) return 0;
  const z = 0.5 * Math.log((1 + rhoClamped) / (1 - rhoClamped)) * Math.sqrt(n - k - 3);
  return 2 * (1 - normalCDF(Math.abs(z)));
}

/** Precision matrix via Gauss-Jordan full elimination */
function invertWithPartial(m: number[][]): number[][] {
  const p = m.length;
  const aug = m.map((row, ri) => [
    ...row,
    ...Array.from({ length: p }, (_, ci) => (ri === ci ? 1 : 0)),
  ]);

  for (let col = 0; col < p; col++) {
    let pivot = col;
    for (let row = col + 1; row < p; row++) {
      if (Math.abs(aug[row]![col]!) > Math.abs(aug[pivot]![col]!)) pivot = row;
    }
    [aug[col], aug[pivot]] = [aug[pivot]!, aug[col]!];
    const pv = aug[col]![col]!;
    if (Math.abs(pv) < 1e-12) continue;
    for (let j = col; j < 2 * p; j++) aug[col]![j]! /= pv;
    for (let row = 0; row < p; row++) {
      if (row === col) continue;
      const factor = aug[row]![col]!;
      for (let j = col; j < 2 * p; j++) aug[row]![j]! -= factor * aug[col]![j]!;
    }
  }
  return aug.map(row => row.slice(p));
}

/**
 * CD-NOD causal discovery algorithm.
 *
 * Extends PC by including a domain index as an implicit conditioning
 * variable in all CI tests, automatically detecting edges that exist
 * only due to domain-shift artifacts.
 *
 * @param data — observation matrix (rows × columns)
 * @param nodeNames — variable names
 * @param domainIndices — domain/segment index per row (length = data.rows)
 *   Use integer labels to distinguish environments.
 * @param config — algorithm configuration
 * @param domainKnowledge — optional domain constraints
 */
export function cdnodAlgorithm(
  data: Matrix,
  nodeNames: string[],
  domainIndices: number[],
  config: Partial<CDNODConfig> = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph } {
  const alpha = config.alpha ?? 0.05;
  const maxDegree = config.maxDegree ?? -1;
  const n = nodeNames.length;

  if (data.rows === 0 || n === 0) return { graph: new CausalGraph(nodeNames) };

  // Append domain index as an extra column for CI testing
  const extendedData = new Matrix(data.rows, data.columns + 1);
  for (let r = 0; r < data.rows; r++) {
    for (let c = 0; c < data.columns; c++) {
      extendedData.set(r, c, data.get(r, c));
    }
    extendedData.set(r, data.columns, domainIndices[r] ?? 0);
  }
  const domainIdx = data.columns;

  // Start with complete undirected graph
  const g = new CausalGraph(nodeNames);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      g.undirectedEdge(nodeNames[i]!, nodeNames[j]!);
    }
  }

  // Phase 1: Skeleton estimation with domain-varying CI
  let depth = 0;
  const maxDepth = maxDegree === -1 ? n : Math.min(maxDegree, n);

  for (depth = 0; depth <= maxDepth; depth++) {
    const toRemove: Array<[string, string]> = [];

    for (let i = 0; i < n; i++) {
      const neighbors = g.neighbors(nodeNames[i]!);
      if (neighbors.length <= depth) continue;

      for (const jName of neighbors) {
        if (jName <= nodeNames[i]!) continue;
        const otherNeighbors = neighbors.filter(nn => nn !== jName);
        if (otherNeighbors.length < depth) continue;

        // Enumerate subsets of size `depth` from otherNeighbors
        for (let subset = 0; subset < (1 << otherNeighbors.length); subset++) {
          const condSet: number[] = [];
          let bits = subset;
          for (let k = 0; k < otherNeighbors.length; k++) {
            if (bits & 1) condSet.push(nodeNames.indexOf(otherNeighbors[k]!));
            bits >>= 1;
          }
          if (condSet.length !== depth) continue;

          const p = domainVaryingFisherZ(
            extendedData, i, nodeNames.indexOf(jName), condSet, domainIdx,
          );
          if (p > alpha) {
            toRemove.push([nodeNames[i]!, jName]);
            break;
          }
        }
      }
    }

    for (const [a, b] of toRemove) {
      g.removeEdge(a, b);
      g.removeEdge(b, a);
    }
    if (toRemove.length === 0) break;
  }

  // Phase 2: Orient v-structures
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (g.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue;
      for (let k = 0; k < n; k++) {
        if (k === i || k === j) continue;
        if (!g.hasEdge(nodeNames[i]!, nodeNames[k]!) || !g.hasEdge(nodeNames[j]!, nodeNames[k]!)) continue;
        // Test: is k in separating set? Use domain-varying test
        const pWithoutK = domainVaryingFisherZ(extendedData, i, j, [], domainIdx);
        const pWithK = domainVaryingFisherZ(extendedData, i, j, [k], domainIdx);
        if (pWithoutK <= alpha && pWithK > alpha) {
          // k in separating set → NOT collider, skip
        } else {
          g.toUndirected(nodeNames[i]!, nodeNames[k]!);
          g.toUndirected(nodeNames[j]!, nodeNames[k]!);
        }
      }
    }
  }

  // Phase 3: Meek rules R1-R3
  let changed = true;
  while (changed) {
    changed = false;
    // R1
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!) || g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;
        for (let k = 0; k < n; k++) {
          if (!g.hasEdge(nodeNames[j]!, nodeNames[k]!) || !g.hasEdge(nodeNames[k]!, nodeNames[j]!)) continue;
          if (g.hasEdge(nodeNames[i]!, nodeNames[k]!) || g.hasEdge(nodeNames[k]!, nodeNames[i]!)) continue;
          g.toUndirected(nodeNames[j]!, nodeNames[k]!);
          changed = true;
        }
      }
    }
    // R2
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < n; k++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[k]!) || !g.hasEdge(nodeNames[k]!, nodeNames[i]!)) continue;
        for (let j = 0; j < n; j++) {
          if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!) || g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;
          if (!g.hasEdge(nodeNames[j]!, nodeNames[k]!) || g.hasEdge(nodeNames[k]!, nodeNames[j]!)) continue;
          g.toUndirected(nodeNames[i]!, nodeNames[k]!);
          changed = true;
        }
      }
    }
    // R3
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!) || !g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;
        for (let k = 0; k < n; k++) {
          if (!g.hasEdge(nodeNames[i]!, nodeNames[k]!) || !g.hasEdge(nodeNames[k]!, nodeNames[i]!)) continue;
          if (!g.hasEdge(nodeNames[k]!, nodeNames[j]!) || g.hasEdge(nodeNames[j]!, nodeNames[k]!)) continue;
          for (let l = 0; l < n; l++) {
            if (l === k) continue;
            if (!g.hasEdge(nodeNames[i]!, nodeNames[l]!) || !g.hasEdge(nodeNames[l]!, nodeNames[i]!)) continue;
            if (!g.hasEdge(nodeNames[l]!, nodeNames[j]!) || g.hasEdge(nodeNames[j]!, nodeNames[l]!)) continue;
            if (g.hasEdge(nodeNames[k]!, nodeNames[l]!) || g.hasEdge(nodeNames[l]!, nodeNames[k]!)) continue;
            g.toUndirected(nodeNames[i]!, nodeNames[j]!);
            changed = true;
            break;
          }
        }
      }
    }
  }

  const dag = g.pdag2dag();
  if (domainKnowledge) dag.applyDomainKnowledge(domainKnowledge);

  return { graph: dag };
}
