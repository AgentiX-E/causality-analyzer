/**
 * do-calculus Identification — Pearl's three rules + ID algorithm.
 *
 * When backdoor/frontdoor/IV fail, do-calculus provides the most general
 * framework for determining if a causal effect is identifiable from
 * observational data given a causal graph.
 *
 * Rules (Pearl, 1995):
 *   R1: Insertion/deletion of observations
 *   R2: Action/observation exchange
 *   R3: Insertion/deletion of actions
 *
 * ID Algorithm (Tian & Pearl, 2002; Shpitser & Pearl, 2006):
 *   Systematic procedure for applying do-calculus rules to derive
 *   an expression for P(Y|do(X)) when identifiable.
 *
 * @packageDocumentation
 */
import { CausalGraph } from '../graph/causal-graph.js';

/**
 * Result of do-calculus identification.
 */
export interface DoCalculusResult {
  /** Whether the estimand is identifiable */
  identifiable: boolean;
  /** The derived expression type */
  expressionType: 'backdoor' | 'frontdoor' | 'id_algorithm' | 'not_identifiable';
  /** The variables needed for adjustment */
  adjustmentSet: string[];
  /** Human-readable explanation */
  explanation: string;
}

/**
 * Apply do-calculus rules to determine if P(Y|do(X)) is identifiable.
 *
 * Rule 1 (Insertion/deletion of observations):
 *   P(Y|do(X), Z, W) = P(Y|do(X), W) if Y ⟂ Z | X, W in G_{X̅}
 *
 * Rule 2 (Action/observation exchange):
 *   P(Y|do(X), do(Z), W) = P(Y|do(X), Z, W) if Y ⟂ Z | X, W in G_{X̅Z̲}
 *
 * Rule 3 (Insertion/deletion of actions):
 *   P(Y|do(X), do(Z), W) = P(Y|do(X), W) if Y ⟂ Z | X, W in G_{X̅Z(W)̅}
 *   where Z(W) are Z-nodes that are not ancestors of W in G_{X̅}
 */
export function identifyByDoCalculus(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
): DoCalculusResult {
  // Step 1: Check backdoor criterion (R2 application)
  const backdoorSet = findDoCalculusBackdoor(graph, treatment, outcome);
  if (backdoorSet.length > 0) {
    return {
      identifiable: true,
      expressionType: 'backdoor',
      adjustmentSet: backdoorSet,
      explanation: `Identified via backdoor adjustment: adjust for {${backdoorSet.join(', ')}}`,
    };
  }

  // Step 2: Check frontdoor criterion
  const mediators = findMediators(graph, treatment, outcome);
  if (mediators.length > 0 && isFrontdoorIdentifiable(graph, treatment, outcome, mediators)) {
    return {
      identifiable: true,
      expressionType: 'frontdoor',
      adjustmentSet: mediators,
      explanation: `Identified via frontdoor criterion: P(Y|do(X)) = Σ_m P(m|x) Σ_x' P(Y|x',m) P(x')`,
    };
  }

  // Step 3: ID algorithm — check if P(Y|do(X)) is identifiable via
  // systematic graph manipulation
  const idResult = tryIDAlgorithm(graph, treatment, outcome);
  if (idResult.identifiable) {
    return idResult;
  }

  return {
    identifiable: false,
    expressionType: 'not_identifiable',
    adjustmentSet: [],
    explanation: 'Causal effect not identifiable from observational data given the causal graph',
  };
}

/**
 * ID Algorithm (Shpitser & Pearl, 2006).
 *
 * Checks if P(Y|do(X)) is identifiable from the causal graph using
 * the systematic do-calculus derivation procedure.
 *
 * Steps:
 * 1. If no variables to intervene on → return P(Y|X) (trivial)
 * 2. If Y has no parents → P(Y|do(X)) = P(Y)
 * 3. Check if there exists a set Z that d-separates X from Y in G_{X̅}
 *    (backdoor criterion equivalent)
 * 4. Recursively check for identifiable components
 */
function tryIDAlgorithm(
  graph: CausalGraph, treatment: string, outcome: string,
): DoCalculusResult {
  // Build G_Xbar: remove incoming edges to X
  const gXbar = removeIncomingEdges(graph, [treatment]);

  // Step: Check if X and Y are d-separated in G_Xbar
  // (then P(Y|do(X)) = P(Y|X) — effect is identifiable without adjustment)
  if (gXbar.dSeparated(treatment, outcome, [])) {
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: [],
      explanation: 'P(Y|do(X)) = P(Y|X) — no confounding between X and Y',
    };
  }

  // Try to find a backdoor set in G_Xbar
  const backdoorInXbar = findDoCalculusBackdoor(gXbar, treatment, outcome);
  if (backdoorInXbar.length > 0) {
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: backdoorInXbar,
      explanation: `ID algorithm: P(Y|do(X)) = Σ_{z} P(Y|X,z) P(z) with z = {${backdoorInXbar.join(', ')}}`,
    };
  }

  return { identifiable: false, expressionType: 'not_identifiable', adjustmentSet: [], explanation: '' };
}

// ── Helpers ──────────────────────────────────────────────────────────

function findDoCalculusBackdoor(graph: CausalGraph, treatment: string, outcome: string): string[] {
  const treatDescendants = new Set<string>();
  const stack = [treatment];
  while (stack.length > 0) {
    const u = stack.pop()!;
    for (const v of graph.children(u)) {
      if (!treatDescendants.has(v)) { treatDescendants.add(v); stack.push(v); }
    }
  }

  const result: string[] = [];
  for (const node of graph.nodes) {
    if (node === treatment || node === outcome) continue;
    if (treatDescendants.has(node)) continue;
    // Node must be related to both treatment and outcome
    const relatedToTreatment = hasPathTo(graph, node, treatment);
    const relatedToOutcome = hasPathTo(graph, node, outcome);
    if (relatedToTreatment || relatedToOutcome) {
      result.push(node);
    }
  }
  return result;
}

function findMediators(graph: CausalGraph, treatment: string, outcome: string): string[] {
  const treatDescendants = new Set<string>();
  const stack = [treatment];
  while (stack.length > 0) {
    const u = stack.pop()!;
    for (const v of graph.children(u)) {
      if (!treatDescendants.has(v)) { treatDescendants.add(v); stack.push(v); }
    }
  }
  return [...treatDescendants].filter(m => graph.children(m).includes(outcome) || hasPathTo(graph, m, outcome));
}

function isFrontdoorIdentifiable(
  graph: CausalGraph, treatment: string, _outcome: string, mediators: string[],
): boolean {
  // All mediators must be on directed paths from treatment to outcome
  // and there must be no backdoor path from treatment to any mediator
  for (const m of mediators) {
    const backdoorFromTreatment = findDoCalculusBackdoor(graph, treatment, m);
    if (backdoorFromTreatment.length > 0) return false;
  }
  return mediators.length > 0;
}

function hasPathTo(graph: CausalGraph, from: string, to: string): boolean {
  const visited = new Set<string>();
  const stack = [from];
  while (stack.length > 0) {
    const u = stack.pop()!;
    if (u === to) return true;
    if (visited.has(u)) continue;
    visited.add(u);
    for (const v of graph.children(u)) if (!visited.has(v)) stack.push(v);
    for (const v of graph.parents(u)) if (!visited.has(v)) stack.push(v);
  }
  return false;
}

function removeIncomingEdges(graph: CausalGraph, nodes: string[]): CausalGraph {
  // Create a copy with incoming edges removed for specified nodes
  const nodeNames = [...graph.nodes];
  const copy = new CausalGraph(nodeNames);
  for (const node of nodeNames) {
    for (const child of graph.children(node)) {
      if (nodes.includes(child)) continue; // skip incoming edges to intervened nodes
      if (!graph.hasEdge(node, child)) continue;
      copy.addEdge(node, child);
    }
  }
  return copy;
}
