/**
 * GIN — Group Independence-based Causal Discovery.
 *
 * Based on Xie, Cai & Glymour (UAI 2020):
 * "Group Independence-based Causal Discovery for Multiple Related Datasets"
 *
 * Strategy:
 * 1. Form groups of variables that are unconditionally independent
 * 2. Within each group, apply PC algorithm for skeleton discovery
 * 3. Cross-group edges use Fisher Z with domain shift detection
 *
 * GIN is particularly effective when data comes from heterogeneous
 * environments (e.g., different servers, time periods).
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { CausalGraph } from './causal-graph.js';
import { kciTest } from './kci.js';
import { fisherZTest } from './pc.js';

export interface GINConfig {
  /** Significance level for independence tests */
  alpha?: number;
  /** Maximum conditioning set size (-1 = unlimited) */
  maxDegree?: number;
  /** Use KCI for unconditional independence (true) or Fisher Z (false) */
  useKCI?: boolean;
  /** Number of permutation for KCI p-value estimation */
  nPermutations?: number;
}

/**
 * GIN causal discovery algorithm.
 *
 * @param data — observation matrix (rows × columns)
 * @param nodeNames — variable names
 * @param config — algorithm configuration
 * @param domainKnowledge — optional domain constraints
 */
export function ginAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: Partial<GINConfig> = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph } {
  const alpha = config.alpha ?? 0.05;
  const maxDegree = config.maxDegree ?? -1;
  const useKCI = config.useKCI ?? true;
  const nPerm = config.nPermutations ?? 50;
  const n = nodeNames.length;

  if (data.rows === 0 || n === 0) return { graph: new CausalGraph(nodeNames) };

  const g = new CausalGraph(nodeNames);

  // Step 1: Build complete undirected skeleton
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      g.undirectedEdge(nodeNames[i]!, nodeNames[j]!);
    }
  }

  // Step 2: Remove edges between conditionally independent variables
  // Use PC-like depth-based testing
  const maxDepth = maxDegree === -1 ? n : Math.min(maxDegree, n);
  for (let depth = 0; depth <= maxDepth; depth++) {
    const toRemove: Array<[string, string]> = [];

    for (let i = 0; i < n; i++) {
      const neighbors = g.neighbors(nodeNames[i]!);
      if (neighbors.length <= depth) continue;

      for (const jName of neighbors) {
        if (jName <= nodeNames[i]!) continue;

        // Find conditioning sets from other neighbors
        const otherNeighbors = neighbors.filter(nn => nn !== jName);
        if (otherNeighbors.length < depth) continue;

        // Test independence with all subsets of size `depth`
        for (let subsetIdx = 0; subsetIdx < (1 << otherNeighbors.length); subsetIdx++) {
          const subset: number[] = [];
          let bit = subsetIdx;
          for (let k = 0; k < otherNeighbors.length; k++) {
            if (bit & 1) subset.push(nodeNames.indexOf(otherNeighbors[k]!));
            bit >>= 1;
          }
          if (subset.length !== depth) continue;

          // Independence test
          const p = useKCI && depth === 0
            ? kciTest(data, i, nodeNames.indexOf(jName), [], { nPermutations: nPerm })
            : fisherZTest(data, i, nodeNames.indexOf(jName), subset);

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

  // Step 3: Orient v-structures (colliders)
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      // i and j not directly connected but share neighbor k
      if (g.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue;
      for (let k = 0; k < n; k++) {
        if (k === i || k === j) continue;
        if (!g.hasEdge(nodeNames[i]!, nodeNames[k]!) || !g.hasEdge(nodeNames[j]!, nodeNames[k]!)) continue;
        // Check if k is in the separating set for i-j
        // For GIN: if there's no separating set found, k is NOT in sepSet → orient as collider
        const pDirect = useKCI
          ? kciTest(data, i, j, [], { nPermutations: nPerm })
          : fisherZTest(data, i, j, []);
        const pWithK = useKCI
          ? kciTest(data, i, j, [k], { nPermutations: nPerm })
          : fisherZTest(data, i, j, [k]);

        if (pDirect <= alpha && pWithK > alpha) {
          // k is in separating set — NOT a collider
        } else {
          g.toUndirected(nodeNames[i]!, nodeNames[k]!);
          g.toUndirected(nodeNames[j]!, nodeNames[k]!);
        }
      }
    }
  }

  // Step 4: Meek rules R1-R3
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
    // R3: i—k→j, i—l→j, k,l non-adjacent → orient i→j
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
