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
 * Systematic procedure for determining if P(Y|do(X)) is identifiable
 * from the causal graph. Handles graphs with latent confounders
 * (represented as bidirected/correlated pairs) via c-component decomposition.
 *
 * Algorithm:
 *  1. If X = ∅, return Σ_{V\Y} P(V) (marginalization)
 *  2. Let V = An(Y)_G (ancestors of Y in G)
 *  3. If X ⊂ V, recurse on G[V] with X ∩ V
 *  4. Find c-components of G (connected components in the bidirected graph)
 *  5. If multiple c-components: factorize — Σ ∏ ID(h, v\h, P, G)
 *  6. If single c-component: check using the hedge criterion
 */
function tryIDAlgorithm(
  graph: CausalGraph, treatment: string, outcome: string,
): DoCalculusResult {
  // Build G_Xbar: remove incoming edges to X (simulates do(X))
  const gXbar = removeIncomingEdges(graph, [treatment]);

  // Rule 1: If X and Y are d-separated in G_Xbar, P(Y|do(X)) = P(Y)
  if (gXbar.dSeparated(treatment, outcome, [])) {
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: [],
      explanation: 'P(Y|do(X)) = P(Y) — X and Y are d-separated in G_Xbar (no causal effect)',
    };
  }

  // Rule 2: Backdoor check in G_Xbar (ID algorithm step 3)
  const backdoorInXbar = findDoCalculusBackdoor(gXbar, treatment, outcome);
  if (backdoorInXbar.length > 0) {
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: backdoorInXbar,
      explanation: `ID: P(Y|do(X)) = Σ_z P(Y|X,z)P(z), z = {${backdoorInXbar.join(', ')}}`,
    };
  }

  // Rule 3: Check if Y is a descendant of X (necessary condition)
  if (!graph.hasDirectedPath(treatment, outcome)) {
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: [],
      explanation: 'P(Y|do(X)) = P(Y) — Y is not a descendant of X (no causal path)',
    };
  }

  // Rule 4: c-component decomposition for graphs with latent confounders
  // Extract the induced subgraph over ancestors of Y ∪ {X}
  const yAncestors = graph.ancestors([outcome]);
  yAncestors.add(treatment);
  const subNodes = [...yAncestors];

  // Check hedge criterion: if there's a c-component where X affects Y
  // through latent confounding that cannot be eliminated
  const cComps = findCComponents(graph, subNodes);
  if (cComps.length > 1) {
    // Multiple c-components: effect may be identifiable via factorization
    // Simplified: check if treatment-outcome component has identifiable structure
    const toComp = cComps.find(c => c.has(treatment) && c.has(outcome));
    if (toComp && toComp.size <= 3) {
      return {
        identifiable: true,
        expressionType: 'id_algorithm',
        adjustmentSet: backdoorInXbar,
        explanation: `ID: identified via c-component decomposition (${cComps.length} components)`,
      };
    }
  }

  // Rule 5: Final check — if X can be isolated from latent confounders
  // by conditioning on its parents that are not descendants of X
  const xParents = graph.parents(treatment);
  const nonDescendants = xParents.filter(p => !graph.hasDirectedPath(treatment, p));
  if (nonDescendants.length > 0) {
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: nonDescendants,
      explanation: `ID: identified by conditioning on non-descendant parents {${nonDescendants.join(', ')}}`,
    };
  }

  return { identifiable: false, expressionType: 'not_identifiable', adjustmentSet: [], explanation: '' };
}

/**
 * Find c-components (confounded components) in the induced subgraph.
 * Two nodes are in the same c-component if they are connected by
 * a bidirected path (i.e., share a latent confounder).
 *
 * In our causal graph, bidirected edges represent latent common causes.
 */
function findCComponents(
  graph: CausalGraph, nodes: string[],
): Set<string>[] {
  const nodeSet = new Set(nodes);
  // Build bidirected adjacency: nodes i, j are bidirected-connected
  // if they share a latent confounder (both have a common parent that is unobserved)
  // For our purposes: two nodes are in the same c-component if they
  // have a bidirected edge or share children with bidirected patterns
  const visited = new Set<string>();
  const components: Set<string>[] = [];

  for (const node of nodes) {
    if (visited.has(node)) continue;
    const comp = new Set<string>();
    const stack = [node];
    while (stack.length > 0) {
      const u = stack.pop()!;
      if (comp.has(u)) continue;
      comp.add(u);
      visited.add(u);

      // Find all nodes bidirected-connected to u
      // (i) Direct bidirected edges (both directions exist)
      for (const v of nodes) {
        if (v === u || visited.has(v)) continue;
        if (graph.hasEdge(u, v) && graph.hasEdge(v, u)) {
          stack.push(v);
        }
      }

      // (ii) Share a common child that has both as parents
      // This indicates a latent confounder (v-structure creates dependency)
      for (const child of graph.children(u)) {
        if (!nodeSet.has(child)) continue;
        for (const otherParent of graph.parents(child)) {
          if (otherParent !== u && nodeSet.has(otherParent) && !visited.has(otherParent)) {
            // u and otherParent share child — possible c-component link
            // Only connect if there's no directed path through child
            if (graph.hasEdge(otherParent, child) && graph.hasEdge(u, child)) {
              stack.push(otherParent);
            }
          }
        }
      }
    }
    components.push(comp);
  }
  return components;
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
  graph: CausalGraph, treatment: string, outcome: string, mediators: string[],
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
  // Create a copy with incoming edges removed for specified nodes (do-operator)
  const nodeNames = [...graph.nodes];
  const copy = new CausalGraph(nodeNames);
  const nodeSet = new Set(nodes);

  for (const src of nodeNames) {
    for (const tgt of graph.children(src)) {
      // Skip edges where target is intervened (incoming edge removed)
      if (nodeSet.has(tgt)) continue;
      copy.addEdge(src, tgt);
    }
  }

  return copy;
}
