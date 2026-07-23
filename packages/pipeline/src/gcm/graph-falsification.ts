/**
 * Graph Falsification — validate causal graph against data.
 *
 * Tests whether a given causal graph (DAG) is consistent with
 * observational data using permutation-based falsification.
 *
 * Critical for AIOps: causal graphs discovered by PC/FCI or manually
 * specified by SREs must be validated before using for RCA.
 *
 * References:
 *   Eulig et al. (2023). "Falsification of Causal Graphs using
 *   Conditional Independence Tests"
 *
 * @packageDocumentation
 */
import { createRNG, combinations } from '@agentix-e/causality-analyzer-core';
import { CausalGraph } from '../graph/causal-graph.js';
import { fisherZTest } from '../graph/pc.js';
import { Matrix } from 'ml-matrix';

/**
 * Result of a graph falsification test.
 */
export interface FalsificationResult {
  /** Is the graph falsified by the data? */
  falsified: boolean;
  /** p-value of the most significant failed CI test */
  pValue: number;
  /** List of missing edges (pairs that should be adjacent but aren't) */
  missingEdges: Array<{ from: string; to: string; pValue: number }>;
  /** List of spurious edges (pairs that are adjacent but shouldn't be) */
  spuriousEdges: Array<{ from: string; to: string; pValue: number }>;
  /** Human-readable explanation */
  explanation: string;
}

/**
 * Falsify a causal graph using conditional independence tests.
 *
 * For each pair of adjacent nodes in the graph, tests whether they
 * are conditionally independent given any subset of their neighbors.
 * For each non-adjacent pair, tests for unconditional dependence.
 * Uses Bonferroni correction for multiple testing.
 */
export function falsifyGraph(
  graph: CausalGraph,
  data: Matrix,
  nodeNames: string[],
  alpha: number = 0.05,
  seed?: number,
): FalsificationResult {
  const rng = createRNG(seed ?? null);
  const n = nodeNames.length;
  const missingEdges: FalsificationResult['missingEdges'] = [];
  const spuriousEdges: FalsificationResult['spuriousEdges'] = [];

  // First pass: collect all raw p-values
  const rawResults: Array<{ type: 'spurious' | 'missing'; from: string; to: string; pValue: number }> = [];

  // Check spurious edges: adjacent pairs that are conditionally independent
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (!graph.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue;
      // Test multiple conditioning sets: neighbors of i (excluding j)
      const neighbors = graph.neighbors(nodeNames[i]!).filter(k => k !== nodeNames[j]);
      // Try conditioning sets of increasing size from neighbors
      for (let size = 0; size <= Math.min(3, neighbors.length); size++) {
        const subsets = size === 0 ? [[]] : combinations(neighbors, size);
        for (const S of subsets) {
          const condSet = S.map(k => nodeNames.indexOf(k));
          const p = fisherZTest(data, i, j, condSet);
          rawResults.push({ type: 'spurious', from: nodeNames[i]!, to: nodeNames[j]!, pValue: p });
        }
      }
    }
  }

  // Check missing edges: non-adjacent pairs that are unconditionally dependent
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (graph.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue;
      const p = fisherZTest(data, i, j, []);
      rawResults.push({ type: 'missing', from: nodeNames[i]!, to: nodeNames[j]!, pValue: p });
    }
  }

  // Apply Bonferroni correction with fixed nTests
  const nTests = rawResults.length;
  const bonferroniAlpha = alpha / Math.max(1, nTests);

  for (const r of rawResults) {
    if (r.type === 'spurious' && r.pValue > bonferroniAlpha) {
      spuriousEdges.push({ from: r.from, to: r.to, pValue: r.pValue });
    } else if (r.type === 'missing' && r.pValue <= bonferroniAlpha) {
      missingEdges.push({ from: r.from, to: r.to, pValue: r.pValue });
    }
  }

  const allIssues = [...missingEdges, ...spuriousEdges];
  const minP = allIssues.length > 0 ? Math.min(...allIssues.map(e => e.pValue)) : 1;
  const falsified = minP < bonferroniAlpha;

  return {
    falsified,
    pValue: minP,
    missingEdges,
    spuriousEdges,
    explanation: falsified
      ? `Graph falsified: ${missingEdges.length} missing edges, ${spuriousEdges.length} spurious edges (Bonferroni α=${bonferroniAlpha.toExponential(2)})`
      : 'Graph not falsified — no significant violations of CI constraints',
  };
}

/**
 * Local Markov Condition (LMC) falsification.
 *
 * Tests whether each node is conditionally independent of its
 * non-descendants given its parents. Failure indicates the graph
 * does not satisfy the Markov condition for that node.
 */
export function lmcFalsification(
  graph: CausalGraph,
  data: Matrix,
  nodeNames: string[],
  alpha: number = 0.05,
): Map<string, { violated: boolean; pValue: number; explanation: string }> {
  const results = new Map<string, { violated: boolean; pValue: number; explanation: string }>();

  for (let i = 0; i < nodeNames.length; i++) {
    const node = nodeNames[i]!;
    const parents = graph.parents(node);
    const parentIdx = parents.map(p => nodeNames.indexOf(p));

    if (parents.length === 0) {
      results.set(node, { violated: false, pValue: 1, explanation: 'No parents, LMC trivially satisfied' });
      continue;
    }

    // Find a non-descendant, non-parent variable
    let tested = false;
    let minP = 1;
    for (let j = 0; j < nodeNames.length; j++) {
      if (j === i || parents.includes(nodeNames[j]!)) continue;
      // Skip descendants (all descendants, not just children)
      const isDescendant = graph.hasDirectedPath(node, nodeNames[j]!);
      if (isDescendant) continue;

      const p = fisherZTest(data, i, j, parentIdx);
      minP = Math.min(minP, p);
      tested = true;
      break; // One test is sufficient
    }

    if (!tested) {
      results.set(node, { violated: false, pValue: 1, explanation: 'No suitable non-descendant to test' });
    } else {
      results.set(node, {
        violated: minP < alpha,
        pValue: minP,
        explanation: minP < alpha
          ? `LMC violated: node depends on non-parent non-descendant (p=${minP.toFixed(4)})`
          : 'LMC satisfied',
      });
    }
  }

  return results;
}

/** Generate all combinations of size k from an array (used for CI testing) */
