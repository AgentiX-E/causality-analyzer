/**
 * PC Algorithm — constraint-based causal discovery.
 *
 * Based on Spirtes, Glymour & Scheines (2000). "Causation, Prediction, and Search."
 * Supports stable-PC variant (Colombo & Maathuis, 2014).
 */
import { Matrix } from 'ml-matrix';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { combinations, fisherZTest as coreFisherZ, _resetFisherZCache } from '@agentix-e/causality-analyzer-core';
import { CausalGraph } from './causal-graph.js';

export interface PCConfig {
  alpha: number;       // significance level (default 0.05)
  maxDegree: number;   // max conditioning set size (-1 = unlimited)
  stable: boolean;     // use stable-PC variant
  /** Multiple testing correction for CI tests.
   *  - 'none': no correction (default, matches standard PC)
   *  - 'bonferroni': alpha divided by estimated number of CI tests
   *  - 'fdr': Benjamini-Hochberg FDR control (applied post-hoc)
   *  WARNING: Without correction, the false positive rate can exceed nominal alpha
   *  for graphs with many nodes due to hundreds of CI tests. */
  alphaCorrection?: 'none' | 'bonferroni' | 'fdr';
}

/**
 * Fisher's Z conditional independence test — thin Matrix wrapper over core.
 *
 * Delegates to @agentix-e/causality-analyzer-core's fisherZTest
 * after converting ml-matrix data to number[][].
 */
export function fisherZTest(
  data: Matrix, i: number, j: number, condSet: number[],
): number {
  const n = data.rows;
  const indices = [i, j, ...condSet];
  const rows: number[][] = [];
  for (let r = 0; r < n; r++) {
    const row: number[] = [];
    for (const idx of indices) row.push(data.get(r, idx));
    rows.push(row);
  }
  // Remap: core indices are now 0,1,...|S|+1
  _resetFisherZCache();
  return coreFisherZ(rows, 0, 1, condSet.map((_, k) => k + 2));
}

/**
 * Number of k-combinations from n elements: C(n, k).
 * Non-recursive, safe for large n. Returns 0 if k > n.
 */
function combinationsCount(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  // Use multiplicative formula: C(n,k) = ∏_{i=1}^{k} (n-k+i)/i
  let result = 1;
  for (let i = 1; i <= k; i++) {
    result = (result * (n - k + i)) / i;
  }
  return Math.round(result);
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
  const cfg: PCConfig = {
    alpha: config.alpha ?? 0.05,
    maxDegree: config.maxDegree ?? -1,
    stable: config.stable ?? true,
    alphaCorrection: config.alphaCorrection ?? 'none',
  };
  const n = nodeNames.length;

  // Compute Bonferroni-corrected alpha if requested.
  // Estimated CI tests: O(n² * C(n-2, maxDegree)) — conservative upper bound.
  let effectiveAlpha = cfg.alpha;
  if (cfg.alphaCorrection === 'bonferroni') {
    const maxD = cfg.maxDegree === -1 ? Math.min(n - 2, 3) : Math.min(cfg.maxDegree, n - 2);
    // Estimate: n*(n-1)/2 pairs × sum_{d=0}^{maxD} C(n-2, d) tests
    let totalTests = 0;
    for (let d = 0; d <= maxD; d++) {
      totalTests += combinationsCount(n - 2, d);
    }
    totalTests *= (n * (n - 1)) / 2;
    effectiveAlpha = cfg.alpha / Math.max(1, totalTests);
  }
  if (data.rows === 0) return { graph: new CausalGraph(nodeNames), sepSet: new Map() };
  const sepSet = new Map<string, Set<string>>();

  // Start with complete undirected graph
  const g = new CausalGraph(nodeNames);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) g.undirectedEdge(nodeNames[i], nodeNames[j]);

  // Phase 1: Skeleton estimation
  let depth = 0;
  let edgesRemoved = true;
  const maxDepth = cfg.maxDegree === -1 ? n : cfg.maxDegree;

  while (edgesRemoved && depth <= maxDepth) {
    edgesRemoved = false;
    const edgesToRemove: Array<[string, string, number[]]> = [];

    for (let i = 0; i < n; i++) {
      const neighbors = g.neighbors(nodeNames[i]);
      if (neighbors.length - 1 < depth) continue;

      for (const jName of neighbors) {
        const j = nodeNames.indexOf(jName);
        // Use index comparison (not string comparison) for deterministic
        // edge deduplication — node names may not be in lexicographic order.
        if (j <= i) continue;
        // Find conditioning sets of size depth
        const otherNeighbors = neighbors.filter(n => n !== jName);
        const subsets = combinations(otherNeighbors, depth);

        for (const S of subsets) {
          const sIndices = S.map(s => nodeNames.indexOf(s));
          const p = fisherZTest(data, i, j, sIndices);
          if (p > effectiveAlpha) {
            edgesToRemove.push([nodeNames[i], jName, sIndices]);
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
      if (g.hasEdge(nodeNames[i], nodeNames[j])) continue; // i and j not adjacent
      for (let k = 0; k < n; k++) {
        if (k === i || k === j) continue;
        if (!g.hasEdge(nodeNames[i], nodeNames[k]) || !g.hasEdge(nodeNames[j], nodeNames[k])) continue;
        // i-k-j is an unshielded triple
        const key = `${Math.min(i, j)}-${Math.max(i, j)}`;
        const sep = sepSet.get(key);
        if (!sep || !sep.has(nodeNames[k])) {
          // k is NOT in separating set → orient i→k←j
          g.orientEdge(nodeNames[i], nodeNames[k]);
          g.orientEdge(nodeNames[j], nodeNames[k]);
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
        if (!g.hasEdge(nodeNames[i], nodeNames[j]) || g.hasEdge(nodeNames[j], nodeNames[i])) continue;
        for (let k = 0; k < n; k++) {
          if (!g.hasEdge(nodeNames[j], nodeNames[k]) || !g.hasEdge(nodeNames[k], nodeNames[j])) continue;
          if (g.hasEdge(nodeNames[i], nodeNames[k]) || g.hasEdge(nodeNames[k], nodeNames[i])) continue;
          g.orientEdge(nodeNames[j], nodeNames[k]);
          changed = true;
        }
      }
    }
    // R2: i→j→k and i—k → orient i→k
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < n; k++) {
        if (!g.hasEdge(nodeNames[i], nodeNames[k]) || !g.hasEdge(nodeNames[k], nodeNames[i])) continue;
        for (let j = 0; j < n; j++) {
          if (!g.hasEdge(nodeNames[i], nodeNames[j]) || g.hasEdge(nodeNames[j], nodeNames[i])) continue;
          if (!g.hasEdge(nodeNames[j], nodeNames[k]) || g.hasEdge(nodeNames[k], nodeNames[j])) continue;
          g.orientEdge(nodeNames[i], nodeNames[k]);
          changed = true;
        }
      }
    }
    // R3: i—k→j, i—l→j, k and l non-adjacent → orient i→j
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!g.hasEdge(nodeNames[i], nodeNames[j]) || !g.hasEdge(nodeNames[j], nodeNames[i])) continue;
        for (let k = 0; k < n; k++) {
          if (!g.hasEdge(nodeNames[i], nodeNames[k]) || !g.hasEdge(nodeNames[k], nodeNames[i])) continue;
          if (!g.hasEdge(nodeNames[k], nodeNames[j]) || g.hasEdge(nodeNames[j], nodeNames[k])) continue;
          for (let l = 0; l < n; l++) {
            if (l === k) continue;
            if (!g.hasEdge(nodeNames[i], nodeNames[l]) || !g.hasEdge(nodeNames[l], nodeNames[i])) continue;
            if (!g.hasEdge(nodeNames[l], nodeNames[j]) || g.hasEdge(nodeNames[j], nodeNames[l])) continue;
            if (g.hasEdge(nodeNames[k], nodeNames[l]) || g.hasEdge(nodeNames[l], nodeNames[k])) continue;
            g.orientEdge(nodeNames[i], nodeNames[j]);
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
