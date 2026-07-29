/**
 * CPDAG — Completed Partially Directed Acyclic Graph orientation.
 *
 * Orients contemporaneous edges from a discovered skeleton into a CPDAG
 * by applying v-structure detection followed by Meek's rules R1-R3.
 *
 * Only used for the contemporaneous (tau=0) layer of PCMCI+ output.
 * Lagged edges are always fully directed (arrow target) because the
 * past cannot depend on the future.
 *
 * References:
 *   - Meek, C. (1995). "Causal inference and causal explanation with
 *     background knowledge." UAI 1995.
 *   - Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation,
 *     Prediction, and Search*, §5.4 (Meek rules).
 *
 * @packageDocumentation
 */

import type { CPDAGInput, EdgeMark } from '@agentix-e/causality-analyzer-core';

/** Internal representation of an oriented edge pair */
interface OrientedEdge {
  /** Source node */
  from: string;
  /** Target node */
  to: string;
  /** Mark at source endpoint */
  sourceMark: EdgeMark;
  /** Mark at target endpoint */
  targetMark: EdgeMark;
}

/** Key format for undirected adjacency map: "A|B" (lexicographically sorted) */
type AdjKey = string;

/**
 * Orient contemporaneous edges into a CPDAG.
 *
 * Input: undirected skeleton with separation sets from the PC₁ phase.
 * Output: a Map from adjacency key to { sourceMark, targetMark }, where
 * the key is the lexicographically sorted pair "source|target".
 *
 * Orientation steps:
 *   1. Mark all edges as undirected (tail—tail).
 *   2. Detect v-structures: i → k ← j where i-j is not adjacent and
 *      k is NOT in SepSet(i, j).
 *   3. Apply Meek rules R1–R3 iteratively until convergence.
 *
 * @param input - adjacencies, separation sets, and node names
 * @returns map from sorted-pair key to orientation marks
 */
export function orientCPDAG(
  input: CPDAGInput,
): Map<string, { sourceMark: EdgeMark; targetMark: EdgeMark }> {
  const { adjacencies, sepSets, nodes } = input;
  const nodeSet = new Set(nodes);

  // Build adjacency set for O(1) lookups
  const adjSet = new Set<AdjKey>();
  for (const [a, b] of adjacencies) {
    adjSet.add(edgeKey(a, b));
  }

  // Initialize all edges as undirected: A — B (tail—tail)
  const oriented = new Map<AdjKey, OrientedEdge>();
  for (const [a, b] of adjacencies) {
    const key = edgeKey(a, b);
    oriented.set(key, { from: a, to: b, sourceMark: 'tail', targetMark: 'tail' });
  }

  // ── Step 1: Find v-structures ────────────────────────────────────────
  // For every triple (i, k, j) where i—k—j and i,j are not adjacent:
  //   if k ∉ SepSet(i, j) → orient i → k ← j
  for (const nA of nodes) {
    for (const nB of nodes) {
      if (nA >= nB) continue; // Only process each unordered pair once
      const abKey = edgeKey(nA, nB);
      if (!adjSet.has(abKey)) continue; // i—k edge must exist

      for (const nC of nodes) {
        if (nC === nA || nC === nB) continue;
        // nB—nC must be adjacent (middle—wing2 edge)
        const bcKey = edgeKey(nB, nC);
        if (!adjSet.has(bcKey)) continue;

        // nA—nC must NOT be adjacent (wings are unshielded)
        const acKey = edgeKey(nA, nC);
        if (adjSet.has(acKey)) continue;

        // nB should NOT be in SepSet(nA, nC) for it to be a collider
        if (isInSepSet(sepSets, nB, nA, nC)) continue;

        // Orient as v-structure: nA → nB ← nC
        orientEdge(oriented, nA, nB, 'arrow');
        orientEdge(oriented, nC, nB, 'arrow');
      }
    }
  }

  // ── Step 2: Apply Meek rules iteratively ──────────────────────────────
  let changed = true;
  let iterations = 0;
  const maxIterations = 100; // Safety against infinite loops

  while (changed && iterations < maxIterations) {
    changed = false;
    iterations++;

    // R1: i → j — k, i,k non-adjacent → j → k
    if (applyMeekR1(oriented, adjSet, nodes)) changed = true;

    // R2: i → k → j, i — j → i → j
    if (applyMeekR2(oriented, adjSet, nodes)) changed = true;

    // R3: i — k → j, i — l → j, k,l non-adjacent → i → j
    if (applyMeekR3(oriented, adjSet, nodes)) changed = true;
  }

  // Convert to output format
  const result = new Map<string, { sourceMark: EdgeMark; targetMark: EdgeMark }>();
  for (const [key, edge] of oriented) {
    // Lexicographically consistent: always store from the lexicographically
    // smaller node. The marks describe "from → to" relation.
    const [a, b] = key.split('|') as [string, string];
    if (a! < b!) {
      result.set(key, { sourceMark: edge.sourceMark, targetMark: edge.targetMark });
    } else {
      // Swap: the mark at the "from" end is the targetMark of the stored edge
      result.set(key, { sourceMark: edge.targetMark, targetMark: edge.sourceMark });
    }
  }

  return result;
}

// ── Meek Rule Implementations ───────────────────────────────────────────

/**
 * R1: If i → j — k and i,k are NOT adjacent, orient j → k.
 *
 * The existing i→j prevents the orientation j←k (would create a cycle).
 *
 * @returns true if any edge was oriented
 */
function applyMeekR1(
  oriented: Map<AdjKey, OrientedEdge>,
  adjSet: Set<AdjKey>,
  nodes: readonly string[],
): boolean {
  let changed = false;

  for (const i of nodes) {
    for (const j of nodes) {
      if (i === j) continue;
      const ijKey = edgeKey(i, j);
      if (!adjSet.has(ijKey)) continue;

      // Check if i → j (i has tail, j has arrow)
      if (!hasEdgeDirected(oriented, i, j)) continue;

      // Find k such that j — k (undirected)
      for (const k of nodes) {
        if (k === i || k === j) continue;
        const jkKey = edgeKey(j, k);
        if (!adjSet.has(jkKey)) continue;
        if (isEdgeDirected(oriented, j, k)) continue;

        // i and k must NOT be adjacent
        const ikKey = edgeKey(i, k);
        if (adjSet.has(ikKey)) continue;

        // Orient j → k
        orientEdge(oriented, j, k, 'arrow');
        changed = true;
      }
    }
  }

  return changed;
}

/**
 * R2: If i → k → j and i — j, orient i → j.
 *
 * The path i → k → j means there's a directed path, preventing j → i
 * (would create a cycle). So i — j must be i → j.
 *
 * @returns true if any edge was oriented
 */
function applyMeekR2(
  oriented: Map<AdjKey, OrientedEdge>,
  adjSet: Set<AdjKey>,
  nodes: readonly string[],
): boolean {
  let changed = false;

  for (const i of nodes) {
    for (const j of nodes) {
      if (i === j) continue;
      const ijKey = edgeKey(i, j);
      if (!adjSet.has(ijKey)) continue;
      if (isEdgeDirected(oriented, i, j)) continue; // Already oriented

      // Check if there exists k: i → k → j
      for (const k of nodes) {
        if (k === i || k === j) continue;
        if (!hasEdgeDirected(oriented, i, k)) continue;
        if (!hasEdgeDirected(oriented, k, j)) continue;

        // Found i → k → j and i — j → orient i → j
        orientEdge(oriented, i, j, 'arrow');
        changed = true;
        break; // Edge oriented, move to next ij pair
      }
    }
  }

  return changed;
}

/**
 * R3: If i — k → j, i — l → j, and k,l are NOT adjacent, orient i → j.
 *
 * Two competing paths: k → j and l → j. If i → j were reversed (j → i),
 * it would create a cycle either through k or l. Since we want a DAG,
 * orient i → j.
 *
 * @returns true if any edge was oriented
 */
function applyMeekR3(
  oriented: Map<AdjKey, OrientedEdge>,
  adjSet: Set<AdjKey>,
  nodes: readonly string[],
): boolean {
  let changed = false;

  for (const i of nodes) {
    for (const j of nodes) {
      if (i === j) continue;
      const ijKey = edgeKey(i, j);
      if (!adjSet.has(ijKey)) continue;
      if (isEdgeDirected(oriented, i, j)) continue; // Already oriented

      // Find two distinct nodes k, l such that i — k → j and i — l → j
      for (const k of nodes) {
        if (k === i || k === j) continue;
        const ikKey = edgeKey(i, k);
        if (!adjSet.has(ikKey)) continue;
        if (isEdgeDirected(oriented, i, k)) continue; // Must be undirected i — k
        if (!hasEdgeDirected(oriented, k, j)) continue; // Must be k → j

        for (const l of nodes) {
          if (l === i || l === j || l === k) continue;
          const ilKey = edgeKey(i, l);
          if (!adjSet.has(ilKey)) continue;
          if (isEdgeDirected(oriented, i, l)) continue; // Must be undirected i — l
          if (!hasEdgeDirected(oriented, l, j)) continue; // Must be l → j

          // k and l must NOT be adjacent
          const klKey = edgeKey(k, l);
          if (adjSet.has(klKey)) continue;

          // Orient i → j
          orientEdge(oriented, i, j, 'arrow');
          changed = true;
        }
        if (changed) break;
      }
      if (changed) break;
    }
  }

  return changed;
}

// ── Utility Functions ───────────────────────────────────────────────────

/** Generate a deterministic, lexicographically sorted key for an edge pair. */
function edgeKey(a: string, b: string): AdjKey {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}

/** Check if n is in SepSet(a, b) — used for v-structure detection. */
function isInSepSet(
  sepSets: Readonly<Map<string, ReadonlySet<string>>>,
  n: string,
  a: string,
  b: string,
): boolean {
  const key = edgeKey(a, b);
  const sep = sepSets.get(key);
  if (!sep) return false;
  return sep.has(n);
}

/** Check if edge (a, b) is directed — at least one endpoint has an arrow. */
function isEdgeDirected(oriented: Map<AdjKey, OrientedEdge>, a: string, b: string): boolean {
  const key = edgeKey(a, b);
  const entry = oriented.get(key);
  if (!entry) return false;
  // Check if arrow at either end
  if (entry.from === a && entry.targetMark === 'arrow') return true;
  if (entry.from === b && entry.targetMark === 'arrow') return true;
  if (entry.from === a && entry.sourceMark === 'arrow') return true;
  if (entry.from === b && entry.sourceMark === 'arrow') return true;
  return false;
}

/** Check if edge is specifically a → b (tail at from=a, arrow at to=b). */
function hasEdgeDirected(oriented: Map<AdjKey, OrientedEdge>, from: string, to: string): boolean {
  const key = edgeKey(from, to);
  const entry = oriented.get(key);
  if (!entry) return false;
  // Case 1: stored as from→to with tail at source, arrow at target
  if (entry.from === from && entry.to === to && entry.sourceMark === 'tail' && entry.targetMark === 'arrow') return true;
  // Case 2: stored as to→from (lexicographic) with tail at source (=to), arrow at target (=from)
  if (entry.from === to && entry.to === from && entry.targetMark === 'arrow' && entry.sourceMark === 'tail') return true;
  return false;
}

/**
 * Set the orientation mark of the edge (from → to).
 * Adds arrow at target endpoint, tail at source endpoint.
 */
function orientEdge(
  oriented: Map<AdjKey, OrientedEdge>,
  from: string,
  to: string,
  mark: EdgeMark,
): void {
  const key = edgeKey(from, to);
  let entry = oriented.get(key);
  if (!entry) return;

  // Ensure the edge is stored with predictable orientation
  if (entry.from === from && entry.to === to) {
    // Natural direction: set targetMark to arrow, sourceMark to tail
    if (mark === 'arrow') {
      entry.targetMark = 'arrow';
      entry.sourceMark = 'tail';
    }
  } else {
    // Reverse direction: swap marks
    if (mark === 'arrow') {
      entry.sourceMark = 'arrow';
      entry.targetMark = 'tail';
    }
  }

  oriented.set(key, entry);
}
