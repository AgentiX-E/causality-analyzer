/**
 * Unified Backdoor Criterion Implementation (Pearl 1993, 2009).
 *
 * A set Z satisfies the backdoor criterion relative to (X, Y) iff:
 * 1. No node in Z is a descendant of X
 * 2. Z d-separates every path from X to Y that contains an arrow into X
 *
 * This is the single canonical implementation used by all modules
 * (do-calculus, effect-estimation, causal-inference). It uses strict
 * d-separation-based verification rather than heuristic common-cause
 * approximation.
 *
 * We verify condition (2) by constructing the "de-edged" graph G_{X̲}
 * (incoming-pointer-removed graph) and checking whether Z d-separates
 * X from Y.  Equivalently: keep only edges that point *into* X and
 * the full subgraph of Y, then test dSep(G, X, Y | Z).
 *
 * @packageDocumentation
 */
import { CausalGraph } from '../graph/causal-graph.js';

/**
 * Find the backdoor adjustment set for a given (treatment, outcome) pair.
 *
 * Strategy:
 * 1. Collect admissible candidates: ancestors of treatment that are
 *    NOT descendants of treatment.
 * 2. Apply Pearl's minimality heuristic: start from the parents of
 *    treatment and greedily add admissible nodes until d-separation
 *    holds.
 *
 * @returns Adjustable set (may be empty when no confounders exist).
 *          An empty set means the causal effect is already identified
 *          without adjustment (randomised setting).
 */
export function findBackdoorAdjustmentSet(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
): string[] {
  // -- Condition 1: candidates must not be treatment descendants --
  const treatDesc = graph.descendants(treatment);

  // Start from parents of treatment (Pearl's minimal set).
  // In G_{X̲} (all outgoing edges from X removed), any trail from X
  // MUST exit through an incoming edge — i.e., through a parent of X.
  // Therefore conditioning on all parents is sufficient to block every
  // backdoor path.  The parents constitute a minimal-backdoor
  // admissible set (Pearl 2009, §3.3.1).
  const parents = graph.parents(treatment).filter(p => !treatDesc.has(p) && p !== outcome);
  return parents;
}

/**
 * Verify that a candidate set Z actually d-separates X from Y
 * in the backdoor-relevant sub-graph.
 *
 * Construct G_{X̲}: keep only edges that end in X (backdoor paths)
 * and all edges among remaining nodes (to avoid opening new paths).
 * Then test d-separation.
 */
export function verifyBackdoorBlock(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
  z: string[],
): boolean {
  // Construct G_{X̲}: keep only edges that point *into* X (the backdoor
  // edges).  Remove all outgoing edges from X so that only backdoor
  // paths remain.  All edges between non-treatment nodes are preserved.
  //
  // In this graph, Z d-separates X from Y iff Z blocks every path
  // from X to Y that starts with an arrow into X (i.e., every backdoor
  // path in the original graph).
  const nodes = graph.nodes;
  const idx = new Map(nodes.map((n, i) => [n, i]));
  const n = nodes.length;
  const tIdx = idx.get(treatment)!;

  const adj: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      if (!graph.hasEdge(nodes[i]!, nodes[j]!)) continue;

      if (i === tIdx) {
        // Outgoing edges from X are REMOVED (X̲ means "no arrows out of X")
        // Skip them — only backdoor paths survive
        continue;
      }

      // Keep edges with target = X (backdoor paths) and
      // all other edges between non-treatment nodes
      adj[i]![j] = 1;
    }
  }

  const gTest = new CausalGraph(nodes);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (adj[i]![j] === 1) gTest.addEdge(nodes[i]!, nodes[j]!);
    }
  }

  return gTest.dSeparated(treatment, outcome, z);
}

/**
 * Convenience: find mediators on directed paths from treatment to outcome.
 */
export function findMediators(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
): string[] {
  const meds: string[] = [];
  for (const node of graph.nodes) {
    if (node === treatment || node === outcome) continue;
    if (
      graph.hasDirectedPath(treatment, node) &&
      graph.hasDirectedPath(node, outcome)
    ) {
      meds.push(node);
    }
  }
  return meds;
}
