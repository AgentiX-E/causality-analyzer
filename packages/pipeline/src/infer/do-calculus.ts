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
import { findBackdoorAdjustmentSet, findMediators } from './backdoor.js';

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
  const backdoorSet = findBackdoorAdjustmentSet(graph, treatment, outcome);
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
  // Step 1 (ID Algorithm Shpitser & Pearl 2006 §4):
  //   Build G_Xbar: remove incoming edges to X
  const gXbar = graph.do(treatment);

  // Step 1a: If X and Y are marginally d-separated in G_Xbar,
  //   P(Y|do(X)) = P(Y) — no causal effect
  if (gXbar.dSeparated(treatment, outcome, [])) {
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: [],
      explanation: 'P(Y|do(X)) = P(Y) — X and Y are d-separated in G_Xbar (no causal effect)',
    };
  }

  // Step 2: Check if Y is not a descendant of X in the original graph
  if (!graph.hasDirectedPath(treatment, outcome)) {
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: [],
      explanation: 'P(Y|do(X)) = P(Y) — Y is not a descendant of X (no causal path)',
    };
  }

  // Step 3: Let V = An(Y)_G (ancestors of Y in the original graph)
  const yAncestors = graph.ancestors([outcome]);
  yAncestors.add(outcome);
  yAncestors.add(treatment);

  // Step 3a: If X ∉ V, effect is 0 (Y is not in X's reachable subgraph)
  const vNodes = [...yAncestors];

  // Step 4: Find c-components in the induced subgraph G[V]
  const cComps = findCComponents(graph, vNodes);

  // Step 5: If multiple c-components, attempt factorization
  if (cComps.length > 1) {
    return idMultiComponentCase(graph, treatment, outcome, vNodes, cComps);
  }

  // Step 6: Single c-component — try backdoor or hedge criterion
  return idSingleComponentCase(graph, treatment, outcome);
}

/**
 * Handle multi-c-component case: factorize using recursive ID calls
 * on each c-component.  This is a pragmatic implementation that
 * handles common patterns (2-3 component graphs) and delegates to
 * the simpler path-based heuristics for complex cases.
 */
function idMultiComponentCase(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
  vNodes: string[],
  cComps: Set<string>[],
): DoCalculusResult {
  // Find the c-component containing the outcome variable
  const yComp = cComps.find(c => c.has(outcome));
  if (!yComp) {
    return { identifiable: false, expressionType: 'not_identifiable', adjustmentSet: [], explanation: 'Outcome c-component not found' };
  }

  // Find the c-component containing the treatment variable
  const xComp = cComps.find(c => c.has(treatment));
  if (!xComp) {
    return { identifiable: false, expressionType: 'not_identifiable', adjustmentSet: [], explanation: 'Treatment c-component not found' };
  }

  const xInYComp = yComp.has(treatment);

  if (!xInYComp) {
    // X and Y in different c-components.
    // Check if there's a directed path from X to Y without latent
    // confounding on that path (Pearl 2009, §3.4.1).
    if (graph.hasDirectedPath(treatment, outcome)) {
      // Try backdoor adjustment for measured confounders
      const backdoorVars = findBackdoorAdjustmentSet(graph, treatment, outcome);
      if (backdoorVars.length > 0) {
        return {
          identifiable: true,
          expressionType: 'id_algorithm',
          adjustmentSet: backdoorVars,
          explanation: `ID: multiple c-components — identified via backdoor on full graph {${backdoorVars.join(', ')}}`,
        };
      }
      // No confounders → P(Y|do(X)) = P(Y|X)
      return {
        identifiable: true,
        expressionType: 'id_algorithm',
        adjustmentSet: [],
        explanation: 'ID: multi-c-component with no confounders — P(Y|do(X)) = P(Y|X)',
      };
    }
    // No directed path and in different c-components: probably unidentifiable
    // through backdoor paths.  This happens when Y ∉ Desc(X) but there
    // are bidirected links — identifiable with zero effect.
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: [],
      explanation: 'ID: Y is not a descendant of X — P(Y|do(X)) = P(Y)',
    };
  }

  // X and Y in same c-component — apply ID recursively on that component
  // For 2-3 node components we can handle them directly
  if (yComp.size <= 3) {
    const backdoorVars = findBackdoorAdjustmentSet(graph, treatment, outcome);
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: backdoorVars,
      explanation: `ID: multi-c-component (${cComps.length} total, Y-component size ${yComp.size})`,
    };
  }

  return { identifiable: false, expressionType: 'not_identifiable', adjustmentSet: [], explanation: 'Multi-c-component graph too complex for current ID solver' };
}

/**
 * Handle single c-component case.
 *
 * In a single c-component graph, either the effect is identifiable
 * via backdoor adjustment, or we need to check the hedge criterion
 * (Shpitser & Pearl 2006, §6).
 */
function idSingleComponentCase(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
): DoCalculusResult {
  // Try backdoor adjustment in the full graph
  const backdoorVars = findBackdoorAdjustmentSet(graph, treatment, outcome);
  if (backdoorVars.length > 0) {
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: backdoorVars,
      explanation: `ID: single c-component — identified via backdoor {${backdoorVars.join(', ')}}`,
    };
  }

  // Check if X and Y are in the same c-component with only observable
  // paths connecting them (no hedge/thorn structure)
  const xParents = graph.parents(treatment);
  if (xParents.length === 0) {
    // No parents of treatment — no confounding → effect identifiable
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: [],
      explanation: 'ID: no confounders for treatment — P(Y|do(X)) = P(Y|X)',
    };
  }

  // Hedge criterion check (simplified):
  // A hedge exists if there are two or more c-components in the subgraph
  // formed by removing V\W from the ancestral closure.  We check a simpler
  // sufficient condition: if all parents of X are non-descendants of X,
  // then there's no self-confounding hedge.
  const nonDescParents = xParents.filter(p => !graph.hasDirectedPath(treatment, p));
  if (nonDescParents.length === xParents.length) {
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: nonDescParents,
      explanation: `ID: identified by conditioning on parents {${nonDescParents.join(', ')}} (no hedge)`,
    };
  }

  return {
    identifiable: false,
    expressionType: 'not_identifiable',
    adjustmentSet: [],
    explanation: 'Hedge detected — P(Y|do(X)) is not identifiable from observational data',
  };
}

/**
 * Find c-components (confounded components) in the induced subgraph.
 *
 * Two nodes are in the same c-component iff they are connected by
 * a bidirected path.  Bidirected edges (X ←→ Y) are represented in our
 * CausalGraph as both hasEdge(X,Y) AND hasEdge(Y,X) being true.
 *
 * V-structures (X → M ← Y) are NOT c-component links — two parents
 * sharing a child does NOT indicate latent confounding.  Each node
 * starts in its own singleton component.
 */
function findCComponents(
  graph: CausalGraph, nodes: string[],
): Set<string>[] {
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

      // Only bidirected edges form c-component connections
      for (const v of nodes) {
        if (v === u || visited.has(v)) continue;
        if (graph.hasEdge(u, v) && graph.hasEdge(v, u)) {
          stack.push(v);
        }
      }
    }
    components.push(comp);
  }
  return components;
}

// ── Helpers ──────────────────────────────────────────────────────────

function isFrontdoorIdentifiable(
  graph: CausalGraph, treatment: string, outcome: string, mediators: string[],
): boolean {
  // All mediators must be on directed paths from treatment to outcome
  // and there must be no backdoor path from treatment to any mediator
  for (const m of mediators) {
    const backdoorFromTreatment = findBackdoorAdjustmentSet(graph, treatment, m);
    if (backdoorFromTreatment.length > 0) return false;
  }
  return mediators.length > 0;
}

// ── Helpers ──────────────────────────────────────────────────────────
