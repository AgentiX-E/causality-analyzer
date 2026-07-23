/**
 * Collider Bias Detection — diagnostic for causal reasoning errors.
 *
 * Conditioning on a collider (or its descendant) introduces spurious
 * association between its parents — a common pitfall in causal analysis.
 * This module detects potential collider bias in user-specified
 * conditioning sets.
 *
 * Reference: Elwert & Winship (2014). "Endogenous Selection Bias."
 *
 * @packageDocumentation
 */
import { CausalGraph } from '../graph/causal-graph.js';

export interface ColliderBiasWarning {
  /** The collider node that would introduce bias if conditioned on */
  collider: string;
  /** The parent nodes that become spuriously associated */
  parents: string[];
  /** Whether the collider is in the conditioning set */
  isConditioned: boolean;
  /** Whether a descendant of the collider is in the conditioning set */
  descendantConditioned: string | null;
  /** Human-readable explanation */
  explanation: string;
}

/**
 * Check a conditioning set for collider bias.
 *
 * Returns warnings for each collider that is conditioned on
 * (or has a conditioned descendant), which would introduce
 * spurious association between its parents.
 *
 * @param graph — causal DAG
 * @param conditioningSet — nodes being conditioned on
 * @returns list of collider bias warnings
 */
export function detectColliderBias(
  graph: CausalGraph,
  conditioningSet: string[],
): ColliderBiasWarning[] {
  const zSet = new Set(conditioningSet);
  const warnings: ColliderBiasWarning[] = [];

  for (const node of graph.nodes) {
    const parents = graph.parents(node);
    if (parents.length < 2) continue; // need at least 2 parents to be a collider

    // Check if node itself is conditioned
    if (zSet.has(node)) {
      warnings.push({
        collider: node,
        parents,
        isConditioned: true,
        descendantConditioned: null,
        explanation: `${node} is conditioned on, but it is a collider (${parents.join(', ')} are parents). This introduces spurious association between ${parents.join(' and ')}.`,
      });
      continue;
    }

    // Check if any descendant of this collider is conditioned
    const desc = graph.descendants(node);
    for (const d of desc) {
      if (d !== node && zSet.has(d)) {
        warnings.push({
          collider: node,
          parents,
          isConditioned: false,
          descendantConditioned: d,
          explanation: `${d} (a descendant of collider ${node}) is conditioned on. This partially activates the collider and introduces spurious association between ${parents.join(' and ')}.`,
        });
        break; // one warning per collider is sufficient
      }
    }
  }

  return warnings;
}

/**
 * Find all collider nodes in the graph.
 *
 * A collider is a node with at least two incoming edges (parents)
 * that converge at the node: parent1 → node ← parent2.
 */
export function findColliders(graph: CausalGraph): string[] {
  return [...graph.nodes].filter(n => graph.parents(n).length >= 2);
}

/**
 * Check whether adjusting for a specific variable introduces collider bias.
 *
 * @returns true if the variable is a collider or has a conditioned descendant
 */
export function isColliderBias(
  graph: CausalGraph,
  variable: string,
  conditioningSet: string[],
): boolean {
  const zSet = new Set(conditioningSet);

  // Direct: is the variable a collider that's conditioned?
  if (graph.parents(variable).length >= 2 && zSet.has(variable)) return true;

  // Indirect: is the variable an ancestor of a conditioned collider?
  const desc = graph.descendants(variable);
  for (const d of desc) {
    if (d !== variable && graph.parents(d).length >= 2 && zSet.has(d)) return true;
  }

  return false;
}

/**
 * Suggest a safe adjustment set by removing collider-biased variables.
 *
 * @returns filtered adjustment set with collider-biased variables removed
 */
export function removeColliderBiasedAdjustments(
  graph: CausalGraph,
  adjustmentSet: string[],
): string[] {
  const warnings = detectColliderBias(graph, adjustmentSet);
  const biasedNodes = new Set(warnings.flatMap(w => [w.collider, w.descendantConditioned].filter(Boolean) as string[]));
  return adjustmentSet.filter(v => !biasedNodes.has(v));
}
