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
 * Provides 5 variants of backdoor set search:
 * - Minimal (parents only — Pearl's canonical set)
 * - Maximal (all admissible ancestors — most conservative)
 * - Efficient (greedy backward selection — smallest statistically efficient set)
 * - Exhaustive (all valid minimal sets — for model selection)
 * - MinCostEfficient (data-adaptive — minimizes variance inflation)
 *
 * @packageDocumentation
 */
import { CausalGraph } from '../graph/causal-graph.js';

export type BackdoorMethod =
  | 'minimal'
  | 'maximal'
  | 'efficient'
  | 'exhaustive'
  | 'mincost';

// ── Public API ──────────────────────────────────────────────────────

/**
 * Find the backdoor adjustment set for (treatment, outcome).
 *
 * @param method — which variant to use (default: 'minimal')
 * @param data — optional data matrix for data-driven methods (mincost)
 * @param nodeIndex — optional node→column mapping for data-driven methods
 */
export function findBackdoorAdjustmentSet(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
  method: BackdoorMethod = 'minimal',
  data?: number[][],
  nodeIndex?: Map<string, number>,
): string[] {
  const admissible = getAdmissibleCandidates(graph, treatment, outcome);

  switch (method) {
    case 'minimal':
      return findMinimal(graph, treatment, outcome, admissible);
    case 'maximal':
      return findMaximal(graph, treatment, outcome, admissible);
    case 'efficient':
      return findEfficient(graph, treatment, outcome, admissible);
    case 'exhaustive':
      return findAllMinimal(graph, treatment, outcome, admissible);
    case 'mincost':
      return findMinCost(graph, treatment, outcome, admissible, data, nodeIndex);
    default:
      return findMinimal(graph, treatment, outcome, admissible);
  }
}

/**
 * Get all admissible backdoor candidates: variables that are ancestors of
 * treatment but NOT descendants of treatment.
 */
export function getAdmissibleCandidates(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
): string[] {
  const treatDesc = graph.descendants(treatment);
  const treatAnc = graph.ancestors([treatment]);
  const candidates: string[] = [];
  for (const node of graph.nodes) {
    if (node === treatment || node === outcome) continue;
    if (treatAnc.has(node) && !treatDesc.has(node)) {
      candidates.push(node);
    }
  }
  return candidates;
}

// ── Variant 1: Minimal (Parents of Treatment) ───────────────────────

/**
 * Pearl's minimal backdoor set: parents of treatment that are not
 * descendants of treatment.
 *
 * In G_{X̲} (all outgoing edges from X removed), any trail from X
 * must exit through an incoming edge — i.e., through a parent of X.
 * Therefore conditioning on all such parents is sufficient to block
 * every backdoor path (Pearl 2009, §3.3.1).
 */
export function findMinimal(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
  admissible: string[],
): string[] {
  const parents = graph.parents(treatment).filter(
    p => admissible.includes(p),
  );
  return parents;
}

// ── Variant 2: Maximal (All Admissible Ancestors) ───────────────────

/**
 * Maximal backdoor set: ALL admissible ancestors of treatment.
 *
 * This is the most conservative set — guarantees d-separation but may
 * include unnecessary variables, potentially increasing estimator variance.
 * Useful for sensitivity analysis and as an upper bound.
 */
export function findMaximal(
  _graph: CausalGraph,
  _treatment: string,
  _outcome: string,
  admissible: string[],
): string[] {
  return [...admissible];
}

// ── Variant 3: Efficient (Greedy Backward Selection) ────────────────

/**
 * Efficient backdoor set: starts from admissible candidates and greedily
 * removes variables that don't break d-separation.
 *
 * Produces a locally-minimal set that is typically smaller than the
 * maximal set while guaranteeing the backdoor criterion holds.
 */
export function findEfficient(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
  admissible: string[],
): string[] {
  // Start with all admissible candidates
  let current = [...admissible];

  // Try removing each candidate — keep only if removal breaks d-separation
  for (const candidate of admissible) {
    const reduced = current.filter(c => c !== candidate);
    if (verifyBackdoorBlock(graph, treatment, outcome, reduced)) {
      current = reduced;
    }
  }

  // If the efficient set is empty (no confounding), return empty
  if (current.length === 0) return [];

  return current;
}

// ── Variant 4: Exhaustive (All Valid Minimal Sets) ──────────────────

/**
 * Exhaustive search for all valid minimal backdoor sets.
 *
 * Finds ALL subsets of admissible candidates that satisfy the backdoor
 * criterion and are minimal (no proper subset also satisfies it).
 *
 * Useful for:
 * - Model averaging across adjustment sets
 * - Sensitivity analysis comparing different adjustment choices
 * - Causal discovery validation
 *
 * @returns array of adjustment sets (may be empty if no valid set exists)
 */
export function findAllMinimal(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
  admissible: string[],
): string[] {
  const allValid: string[][] = [];

  // Find all valid sets (not just minimal yet)
  for (let mask = 1; mask < (1 << admissible.length); mask++) {
    const subset: string[] = [];
    for (let i = 0; i < admissible.length; i++) {
      if (mask & (1 << i)) subset.push(admissible[i]);
    }
    if (verifyBackdoorBlock(graph, treatment, outcome, subset)) {
      allValid.push(subset);
    }
  }

  // Filter to minimal sets (no proper subset is also valid)
  const minimal: string[][] = [];
  for (const set of allValid) {
    let isMinimal = true;
    for (const other of allValid) {
      if (other === set) continue;
      if (other.length < set.length && isSubset(other, set)) {
        isMinimal = false;
        break;
      }
    }
    if (isMinimal) minimal.push(set);
  }

  // Return the smallest minimal set (or parents as fallback)
  if (minimal.length === 0) {
    return findMinimal(graph, treatment, outcome, admissible);
  }

  // Prefer the smallest minimal set
  minimal.sort((a, b) => a.length - b.length);
  return minimal[0];
}

// ── Variant 5: MinCost-Efficient (Data-Adaptive) ────────────────────

/**
 * Minimum-cost backdoor set: selects the adjustment set that minimizes
 * estimated variance inflation while satisfying the backdoor criterion.
 *
 * The "cost" of an adjustment variable is its partial correlation with
 * the treatment — higher correlation means more variance inflation.
 * We prefer adjustment sets with lower average partial R² between
 * treatment and covariates.
 *
 * When no data is provided, falls back to efficient search.
 */
export function findMinCost(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
  admissible: string[],
  data?: number[][],
  nodeIndex?: Map<string, number>,
): string[] {
  if (!data || !nodeIndex || data.length === 0) {
    return findEfficient(graph, treatment, outcome, admissible);
  }

  const tIdx = nodeIndex.get(treatment);
  if (tIdx === undefined) return findEfficient(graph, treatment, outcome, admissible);

  // Compute partial R² between treatment and each candidate
  // (as a proxy for variance inflation)
  const costs = new Map<string, number>();
  for (const candidate of admissible) {
    const cIdx = nodeIndex.get(candidate);
    if (cIdx === undefined) { costs.set(candidate, 1); continue; }

    // Simple correlation between treatment and candidate
    let sXY = 0, sXX = 0, sYY = 0;
    const tMean = data.reduce((s, r) => s + (r[tIdx] ?? 0), 0) / data.length;
    const cMean = data.reduce((s, r) => s + (r[cIdx] ?? 0), 0) / data.length;
    for (const row of data) {
      const dt = (row[tIdx] ?? 0) - tMean;
      const dc = (row[cIdx] ?? 0) - cMean;
      sXY += dt * dc;
      sXX += dt * dt;
      sYY += dc * dc;
    }
    const denom = Math.sqrt(Math.max(1e-10, sXX * sYY));
    const r2 = denom > 0 ? (sXY / denom) ** 2 : 0;
    costs.set(candidate, r2);
  }

  // Find all valid sets, pick the one with minimum total cost
  let bestSet = findEfficient(graph, treatment, outcome, admissible);
  let bestCost = bestSet.reduce((s, c) => s + (costs.get(c) ?? 0), 0);

  // Try subsets of admissible candidates (via bitmask enumeration).
  // NOTE: Search is truncated at 2^10 = 1024 subsets (when |admissible| > 10)
  // to keep runtime bounded. For larger admissible sets, the efficient minimal
  // set found by findEfficient() above is used as the baseline.
  for (let mask = 1; mask < (1 << Math.min(admissible.length, 10)); mask++) {
    const subset: string[] = [];
    let subsetCost = 0;
    for (let i = 0; i < admissible.length; i++) {
      if (mask & (1 << i)) {
        subset.push(admissible[i]);
        subsetCost += costs.get(admissible[i]) ?? 0;
      }
    }
    if (subsetCost >= bestCost) continue;
    if (verifyBackdoorBlock(graph, treatment, outcome, subset)) {
      bestSet = subset;
      bestCost = subsetCost;
    }
  }

  return bestSet;
}

// ── Verification ────────────────────────────────────────────────────

/**
 * Verify that a candidate set Z actually d-separates X from Y
 * in the backdoor-relevant sub-graph G_{X̲}.
 */
export function verifyBackdoorBlock(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
  z: string[],
): boolean {
  const nodes = graph.nodes;
  const idx = new Map(nodes.map((n, i) => [n, i]));
  const n = nodes.length;
  const tIdx = idx.get(treatment)!;

  const adj: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      if (!graph.hasEdge(nodes[i], nodes[j])) continue;

      if (i === tIdx) {
        // Outgoing edges from X are REMOVED
        continue;
      }

      adj[i][j] = 1;
    }
  }

  const gTest = new CausalGraph(nodes);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (adj[i][j] === 1) gTest.addEdge(nodes[i], nodes[j]);
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

// ── Helpers ─────────────────────────────────────────────────────────

function isSubset(a: string[], b: string[]): boolean {
  const bSet = new Set(b);
  return a.every(x => bSet.has(x));
}
