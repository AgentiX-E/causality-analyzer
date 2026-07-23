/**
 * PC Algorithm — constraint-based causal discovery.
 *
 * Based on Spirtes, Glymour & Scheines (2000). "Causation, Prediction, and Search."
 * Supports stable-PC variant (Colombo & Maathuis, 2014).
 */
import { Matrix } from 'ml-matrix';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { erf, normalCDF } from '@agentix-e/causality-analyzer-core';
import { combinations } from "@agentix-e/causality-analyzer-core";
import { CausalGraph } from './causal-graph.js';

export interface PCConfig {
  alpha: number;       // significance level (default 0.05)
  maxDegree: number;   // max conditioning set size (-1 = unlimited)
  stable: boolean;     // use stable-PC variant
}

/**
 * Fisher's Z conditional independence test.
 * Returns p-value for the null hypothesis X ⟂ Y | Z.
 */
export function fisherZTest(
  data: Matrix, i: number, j: number, condSet: number[],
): number {
  const n = data.rows;
  const indices = [i, j, ...condSet];
  const k = condSet.length;
  const sub = data.subMatrixColumn(indices);

  // Compute correlation matrix
  const means = new Array(indices.length).fill(0);
  for (let c = 0; c < indices.length; c++) {
    let sum = 0; for (let r = 0; r < n; r++) sum += sub.get(r, c);
    means[c] = sum / n;
  }
  const cov = new Array(indices.length).fill(0).map(() => new Array(indices.length).fill(0));
  for (let a = 0; a < indices.length; a++) {
    for (let b = a; b < indices.length; b++) {
      let sum = 0;
      for (let r = 0; r < n; r++) sum += (sub.get(r, a) - means[a]!) * (sub.get(r, b) - means[b]!);
      cov[a]![b] = sum / (n - 1);
      cov[b]![a] = cov[a]![b]!;
    }
  }

  // Partial correlation via precision matrix inversion
  const rho = partialCorrelation(cov, 0, 1);
  if (Math.abs(rho) >= 1) return 0;

  const z = 0.5 * Math.log((1 + rho) / (1 - rho)) * Math.sqrt(n - k - 3);
  return 2 * (1 - normalCDF(Math.abs(z)));
}

/** Compute partial correlation via precision matrix inversion */
function partialCorrelation(cov: number[][], i: number, j: number): number {
  const m = cov.length;
  if (m === 2) return cov[i]![j]! / Math.sqrt(cov[i]![i]! * cov[j]![j]!);
  const prec = invertMatrix(cov);
  const r = -prec[i]![j]! / Math.sqrt(prec[i]![i]! * prec[j]![j]!);
  return Math.max(-1, Math.min(1, r));
}

/** Specialized Gauss-Jordan matrix inversion (full elimination, not back-substitution) */
function invertMatrix(m: number[][]): number[][] {
  const n = m.length;
  const aug = m.map((row, ri) => [...row, ...Array.from({length: n}, (_, ci) => ri === ci ? 1 : 0)]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) if (Math.abs(aug[row]![col]!) > Math.abs(aug[pivot]![col]!)) pivot = row;
    [aug[col], aug[pivot]] = [aug[pivot]!, aug[col]!];
    const pv = aug[col]![col]!;
    if (Math.abs(pv) < 1e-12) continue;
    for (let j = col; j < 2 * n; j++) aug[col]![j]! /= pv;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row]![col]!;
      for (let j = col; j < 2 * n; j++) aug[row]![j]! -= factor * aug[col]![j]!;
    }
  }
  return aug.map(row => row.slice(n));
}

/**
 * PC algorithm: constraint-based causal discovery.
 */
export function pcAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: Partial<PCConfig> = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph; sepSet: Map<string, Set<string>> } {
  const cfg: PCConfig = { alpha: config.alpha ?? 0.05, maxDegree: config.maxDegree ?? -1, stable: config.stable ?? true };
  const n = nodeNames.length;
  if (data.rows === 0) return { graph: new CausalGraph(nodeNames), sepSet: new Map() };
  const sepSet = new Map<string, Set<string>>();

  // Start with complete undirected graph
  const g = new CausalGraph(nodeNames);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) g.undirectedEdge(nodeNames[i]!, nodeNames[j]!);

  // Phase 1: Skeleton estimation
  let depth = 0;
  let edgesRemoved = true;
  const maxDepth = cfg.maxDegree === -1 ? n : cfg.maxDegree;

  while (edgesRemoved && depth <= maxDepth) {
    edgesRemoved = false;
    const edgesToRemove: Array<[string, string, number[]]> = [];

    for (let i = 0; i < n; i++) {
      const neighbors = g.neighbors(nodeNames[i]!);
      if (neighbors.length - 1 < depth) continue;

      for (const jName of neighbors) {
        if (jName <= nodeNames[i]!) continue;
        const j = nodeNames.indexOf(jName);
        // Find conditioning sets of size depth
        const otherNeighbors = neighbors.filter(n => n !== jName);
        const subsets = combinations(otherNeighbors, depth);

        for (const S of subsets) {
          const sIndices = S.map(s => nodeNames.indexOf(s));
          const p = fisherZTest(data, i, j, sIndices);
          if (p > cfg.alpha) {
            edgesToRemove.push([nodeNames[i]!, jName, sIndices]);
            const key = `${Math.min(i, j)}-${Math.max(i, j)}`;
            sepSet.set(key, new Set(S));
            break;
          }
        }
      }
    }

    // Stable PC (Colombo & Maathuis, 2014): collect all qualifying edges
    // at each depth level, then remove them all at once.
    // Classic PC removes edges immediately — this is order-dependent and
    // not recommended. We always use stable PC for deterministic results.
    for (const [a, b, _] of edgesToRemove) {
      g.removeEdge(a, b); g.removeEdge(b, a);
      edgesRemoved = true;
    }
    depth++;
  }

  // Phase 2: Orient v-structures (colliders)
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (g.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue; // i and j not adjacent
      for (let k = 0; k < n; k++) {
        if (k === i || k === j) continue;
        if (!g.hasEdge(nodeNames[i]!, nodeNames[k]!) || !g.hasEdge(nodeNames[j]!, nodeNames[k]!)) continue;
        // i-k-j is an unshielded triple
        const key = `${Math.min(i, j)}-${Math.max(i, j)}`;
        const sep = sepSet.get(key);
        if (!sep || !sep.has(nodeNames[k]!)) {
          // k is NOT in separating set → orient i→k←j
          g.toUndirected(nodeNames[i]!, nodeNames[k]!);
          g.toUndirected(nodeNames[j]!, nodeNames[k]!);
        }
      }
    }
  }

  // Phase 3: Meek's rules R1-R3
  let changed = true;
  while (changed) {
    changed = false;

    // R1: i→j—k with i,k non-adjacent → orient j→k
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
    // R2: i→j→k and i—k → orient i→k
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
    // R3: i—k→j, i—l→j, k and l non-adjacent → orient i→j
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

  // Convert PDAG to DAG
  const dag = g.pdag2dag();

  if (domainKnowledge) dag.applyDomainKnowledge(domainKnowledge);

  return { graph: dag, sepSet };
}

/** Generate all combinations of size k from an array */
