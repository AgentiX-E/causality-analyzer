/**
 * do-calculus Identification — Pearl's three rules + full recursive ID algorithm.
 *
 * When backdoor/frontdoor fail, the full ID algorithm (Shpitser & Pearl, 2006)
 * provides the most general framework for determining if a causal effect is
 * identifiable from observational data given a causal graph.
 *
 * Rules (Pearl, 1995):
 *   R1: Insertion/deletion of observations
 *   R2: Action/observation exchange
 *   R3: Insertion/deletion of actions
 *
 * ID Algorithm (Shpitser & Pearl 2006, §4-6):
 *   Systematic recursive procedure for applying do-calculus rules to derive
 *   an expression for P(Y|do(X)) when identifiable.  Complete: detects all
 *   identifiable queries and proves non-identifiability via the hedge criterion.
 *
 * @packageDocumentation
 */
import { CausalGraph } from '../graph/causal-graph.js';
import { findBackdoorAdjustmentSet, findMediators } from './backdoor.js';

// ── Public Types ──────────────────────────────────────────────────────

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

// ── Public API ────────────────────────────────────────────────────────

/**
 * Apply do-calculus rules to determine if P(Y|do(X)) is identifiable.
 *
 * Strategy (in order):
 *  1. Backdoor criterion — simplest, most common case
 *  2. Frontdoor criterion — mediator-based decomposition
 *  3. Full recursive ID algorithm (Shpitser & Pearl 2006)
 *
 * @param graph — The causal DAG (nodes include observed + latent variables)
 * @param treatment — The intervention variable X
 * @param outcome — The outcome variable Y
 * @returns Identification result with expression type and adjustment set
 */
export function identifyByDoCalculus(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
): DoCalculusResult {
  // Step 1: Check backdoor criterion (most common, simplest)
  const backdoorSet = findBackdoorAdjustmentSet(graph, treatment, outcome);
  if (backdoorSet.length > 0) {
    return {
      identifiable: true,
      expressionType: 'backdoor',
      adjustmentSet: backdoorSet,
      explanation: `Identified via backdoor adjustment: adjust for {${backdoorSet.join(', ')}}`,
    };
  }

  // Step 2: Check frontdoor criterion (mediator-based decomposition)
  const mediators = findMediators(graph, treatment, outcome);
  const frontdoorMediators = mediators.filter((m) => {
    // All mediators must have no unblocked backdoor path from treatment
    const backdoorFromTreatment = findBackdoorAdjustmentSet(graph, treatment, m);
    return backdoorFromTreatment.length === 0;
  });
  if (frontdoorMediators.length > 0) {
    return {
      identifiable: true,
      expressionType: 'frontdoor',
      adjustmentSet: frontdoorMediators,
      explanation:
        'Identified via frontdoor criterion: ' +
        `P(Y|do(X)) = Σ_m P(m|do(x)) Σ_{x'} P(Y|do(x'), do(m)) × P(x')`,
    };
  }

  // Step 3: Check for degenerate bow graph before ID algorithm.
  // In the projected-graph model (bidirected edges without explicit
  // latents), a pure X↔Y bidirected edge represents latent confounding
  // that cannot be adjusted away.  The ID algorithm would incorrectly
  // treat the X→Y leg of the bidirected edge as a causal path.
  // Detect this case: X↔Y exists AND no alternative directed path
  // from X to Y exists through other variables.
  if (graph.hasEdge(treatment, outcome) && graph.hasEdge(outcome, treatment)) {
    // Build a copy of the graph with the X↔Y bidirected edge removed
    // and check whether X can still causally affect Y.
    const gNoBidir = graph.clone();
    gNoBidir.removeEdge(treatment, outcome);
    gNoBidir.removeEdge(outcome, treatment);

    if (!gNoBidir.hasDirectedPath(treatment, outcome)) {
      return {
        identifiable: false,
        expressionType: 'not_identifiable',
        adjustmentSet: [],
        explanation:
          'Bow graph detected: X↔Y represents latent confounding with no alternative causal path. ' +
          'P(Y|do(X)) is not identifiable from observational data without observing the latent confounder ' +
          '(Shpitser & Pearl 2006, §6)',
      };
    }
  }

  // Step 4: Full recursive ID algorithm
  return tryIDAlgorithm(graph, treatment, outcome);
}

// ── ID Algorithm Entry Point ──────────────────────────────────────────

/**
 * Entry point for the full recursive ID algorithm.
 *
 * Wraps the {treatment, outcome} as single-element sets and delegates
 * to the recursive `fullID` function.
 */
function tryIDAlgorithm(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
): DoCalculusResult {
  const y = new Set([outcome]);
  const x = new Set([treatment]);

  // Mutation-free set of visited states to prevent infinite recursion
  // in pathological graph structures.
  const visited = new Set<string>();

  return fullID(y, x, graph, visited);
}

// ── Full Recursive ID Algorithm ───────────────────────────────────────

/**
 * Full recursive ID algorithm (Shpitser & Pearl 2006, Algorithm 1).
 *
 * Determines if P(Y|do(X)) is identifiable from the observational
 * distribution P(V) given the causal DAG G.
 *
 * Algorithm structure (from the paper §4):
 *
 *   Line 1: If X = ∅, return Σ_{V\Y} P(V) (marginalization)
 *
 *   Line 2: Build G[An(Y)_G] (ancestor subgraph).
 *            If V ≠ An(Y)_G, recurse on the restricted graph.
 *
 *   Line 3: W = (V \ X) \ An(Y)_{G_X̅}.
 *            Nodes that cease being ancestors of Y after mutilation
 *            can be added to the treatment set.
 *
 *   Line 4: C(G \ X) — c-components of the graph after removing X.
 *            Multiple c-components → factorize via recursive calls.
 *
 *   Line 5: Single c-component S = V \ X.
 *            Check hedge criterion via full-graph c-component
 *            restriction to S (Shpitser & Pearl 2006, §6).
 *
 *   Line 5a: No hedge → identifiable via edge-min-cut expression.
 *   Line 5b: Hedge exists → provably not identifiable.
 *
 * @param y — Outcome variable set (Y in the paper)
 * @param x — Treatment variable set (X in the paper)
 * @param graph — Current causal DAG G (may be a subgraph from recursion)
 * @param visited — Set of visited state keys to prevent infinite recursion
 * @returns Identification result
 */
function fullID(
  y: ReadonlySet<string>,
  x: ReadonlySet<string>,
  graph: CausalGraph,
  visited: Set<string>,
): DoCalculusResult {
  const allNodes = new Set(graph.nodes);

  // -- Cycle detection: prevent infinite recursion ------------------
  const stateKey = buildStateKey(y, x, graph);
  if (visited.has(stateKey)) {
    return {
      identifiable: false,
      expressionType: 'not_identifiable',
      adjustmentSet: [],
      explanation:
        'Cyclic recursion detected in ID algorithm — graph structure prevents convergence',
    };
  }
  visited.add(stateKey);

  // -- Line 1: If X = ∅, return Σ_{V\Y} P(V) (marginalization) ---
  if (x.size === 0) {
    const marginNodes = [...allNodes].filter((n) => !y.has(n));
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: [],
      explanation:
        marginNodes.length > 0
          ? `Marginalization over {${marginNodes.join(', ')}} → P(Y) = Σ_{V\\Y} P(V)`
          : 'Trivial: P(Y) directly observed',
    };
  }

  // -- Line 2: Build ancestor subgraph G[An(Y)] --------------------
  // If the graph contains nodes that are not ancestors of Y,
  // restrict to the ancestral closure and recurse.
  const yArray = [...y];
  const yAnc = graph.ancestors(yArray);
  for (const yn of y) yAnc.add(yn);

  if (yAnc.size < allNodes.size) {
    // Recurse on the ancestor-induced subgraph, keeping only
    // treatment nodes that are also ancestors of Y.
    const xIntersection = new Set([...x].filter((n) => yAnc.has(n)));
    const subgraph = buildInducedSubgraph(graph, [...yAnc]);
    return fullID(y, xIntersection, subgraph, visited);
  }

  // At this point: V = An(Y)_G (all nodes are ancestors of Y)

  // -- Line 3: W = (V \ X) \ An(Y)_{G_X̅} --------------------------
  // In graph G_X̅ (edges into X removed), find nodes that ceased
  // being ancestors of Y.  These can be added to the treatment set
  // because intervening on them doesn't affect Y beyond X.
  const gXbar = buildGXbar(graph, [...x]);
  const yAncXbar = gXbar.ancestors(yArray);
  for (const yn of y) yAncXbar.add(yn);

  const w: string[] = [];
  for (const n of allNodes) {
    if (!x.has(n) && !yAncXbar.has(n)) {
      w.push(n);
    }
  }

  if (w.length > 0) {
    const newX = new Set([...x, ...w]);
    return fullID(y, newX, graph, visited);
  }

  // At this point: V \ X ⊆ An(Y)_{G_X̅}
  // Every node not in X remains an ancestor of Y even after
  // removing incoming edges to X (i.e., after intervening on X).

  // -- Line 4: C(G \ X) — c-components of G without X --------------
  // Compute c-components of the graph after removing X.
  // If multiple c-components exist, factorize via recursive calls.
  const vMinusX = [...allNodes].filter((n) => !x.has(n));
  const cComps = findCComponents(graph, vMinusX);

  if (cComps.length > 1) {
    // Factorize: Σ_{v∈V\(Y∪X)} ∏_i ID(S_i, V\S_i, P(V), G)
    return idFactorize(y, x, graph, cComps, visited);
  }

  // -- Line 5: Single c-component S = V \ X ------------------------
  // When C(G \ X) has a single c-component S = V \ X, we must check
  // the **hedge criterion** (Shpitser & Pearl 2006, §6, Theorem 4).
  //
  // The hedge detection in step 5 uses the c-components of the FULL
  // graph G (including X) restricted to S = V \ X.  This differs from
  // C(G \ X) computed in step 4, which only considers bidirected paths
  // among V \ X nodes.  The full graph's c-component structure reveals
  // whether S decomposes into sub-components that form a hedge.
  //
  // Intuition: if the full graph's c-components split S into multiple
  // groups, then the treatment X is partially confounded with some
  // subsets of S but not others, creating a hedge that prevents
  // identifiability.
  const S = new Set(vMinusX);
  const fullCComps = findCComponents(graph, [...allNodes]);

  // Restrict each full-graph c-component to S.  If multiple full
  // c-components have non-empty intersection with S, a hedge exists.
  const restrictedComponents: Set<string>[] = [];
  for (const comp of fullCComps) {
    const intersection = [...comp].filter((n) => S.has(n)).sort();
    if (intersection.length > 0) {
      restrictedComponents.push(new Set(intersection));
    }
  }

  if (restrictedComponents.length > 1) {
    // HEDGE DETECTED — the causal effect is provably not identifiable
    // from observational data.
    //
    // The outer c-component S from C(G \ X) decomposes when viewed
    // through the full graph's c-component lens.  The multiple
    // intersecting c-components form a hedge structure (F, F') where:
    //   - F = S (outer c-component from G \ X)
    //   - F' = one of the restricted sub-components
    //   - F' ⊂ F and X \ F' ≠ ∅ (treatment nodes are outside F')
    //   - Both are c-components in their respective induced subgraphs
    //
    // The hedge prevents expressing P(Y|do(X)) solely in terms of
    // the observational distribution P(V).
    const compDescriptions = restrictedComponents.map(
      (c) => `{${[...c].sort().join(', ')}}`,
    );
    return {
      identifiable: false,
      expressionType: 'not_identifiable',
      adjustmentSet: [],
      explanation:
        `Hedge detected — P(Y|do(X)) is not identifiable. ` +
        `Full c-components of G restricted to S = {${[...S].sort().join(', ')}} ` +
        `decompose into ${restrictedComponents.length} sub-structures ` +
        `[${compDescriptions.join(', ')}]. ` +
        'The presence of nested c-components (a hedge) makes the causal ' +
        'effect provably unidentifiable from observational data ' +
        '(Shpitser & Pearl 2006, Theorem 4, Corollary 3).',
    };
  }

  // -- Line 5a: Identifiable via edge-min-cut expression -----------
  // Single c-component with no further decomposition means the
  // effect is identifiable.  The expression can be derived via
  // Algorithm 2 (EdgeMinCut) from the paper.
  const sNodesSorted = [...S].sort();

  return {
    identifiable: true,
    expressionType: 'id_algorithm',
    adjustmentSet: [],
    explanation:
      `Identified via full ID algorithm (Shpitser & Pearl 2006): ` +
      `single c-component {${sNodesSorted.join(', ')}} ` +
      '— effect identifiable via edge-min-cut expression',
  };
}

// ── Step 4 Factorization ──────────────────────────────────────────────

/**
 * Handle multi-c-component factorization (ID Algorithm line 4).
 *
 * When C(G \ X) = {S₁, ..., S_k} with k > 1, the expression factorizes:
 *   Σ_{v∈V\(Y∪X)} ∏ᵢ ID(Sᵢ, V\Sᵢ, P(V), G)
 *
 * Each c-component Sᵢ is solved recursively as:
 *   ID(Sᵢ, V \ Sᵢ, P(V), G)
 * computing P(Sᵢ | do(V \ Sᵢ)).
 *
 * The overall expression is identifiable iff every sub-call is identifiable.
 * If any sub-call fails, the original query is not identifiable.
 */
function idFactorize(
  y: ReadonlySet<string>,
  x: ReadonlySet<string>,
  graph: CausalGraph,
  cComps: Set<string>[],
  visited: Set<string>,
): DoCalculusResult {
  const allNodes = graph.nodes;
  const componentDescriptions: string[] = [];

  for (const comp of cComps) {
    const compSet = new Set(comp);
    const otherNodes = allNodes.filter((n) => !compSet.has(n));

    const result = fullID(compSet, new Set(otherNodes), graph, visited);

    const compLabel = [...comp].sort().join(',');
    componentDescriptions.push(`{${compLabel}}`);

    if (!result.identifiable) {
      return {
        identifiable: false,
        expressionType: 'not_identifiable',
        adjustmentSet: [],
        explanation:
          `Multiple c-components (${cComps.length}) detected. ` +
          `Factorization failed: c-component {${compLabel}} is not identifiable. ` +
          `P(Y|do(X)) is not identifiable.`,
      };
    }
  }

  return {
    identifiable: true,
    expressionType: 'id_algorithm',
    adjustmentSet: [],
    explanation:
      `Multiple c-components (${cComps.length}) ` +
      `[${componentDescriptions.join(', ')}] — ` +
      'effect identified via c-component factorization',
  };
}

// ── C-Component Finding ───────────────────────────────────────────────

/**
 * Find c-components (confounded components) in the induced subgraph.
 *
 * Two nodes belong to the same c-component iff they are connected by a
 * path consisting entirely of **bidirected edges** (representing latent
 * common causes).
 *
 * In our CausalGraph, a bidirected edge X ↔ Y (meaning X ← U → Y for
 * some unobserved U) is represented as both `hasEdge(X, Y)` AND
 * `hasEdge(Y, X)` being true.
 *
 * V-structures (X → M ← Y) do NOT connect X and Y — two parents sharing
 * a child does not indicate latent confounding.  Each node starts in its
 * own singleton component, and components grow via bidirected-path
 * connectivity.
 *
 * @param graph — The causal DAG
 * @param nodes — Subset of nodes to compute c-components on (induced subgraph)
 * @returns Array of c-components, each as a Set of node names
 */
function findCComponents(
  graph: CausalGraph,
  nodes: string[],
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

      // Connect only via bidirected edges (latent confounding)
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

// ── Graph Helpers ─────────────────────────────────────────────────────

/**
 * Build the induced subgraph G[V'] containing only the specified nodes
 * and all edges between them that exist in the original graph.
 *
 * Bidirected edges (X ↔ Y) are preserved as long as both endpoints
 * are in the node set.
 */
function buildInducedSubgraph(
  graph: CausalGraph,
  nodes: string[],
): CausalGraph {
  const result = new CausalGraph([...nodes]);
  for (const from of nodes) {
    for (const to of nodes) {
      if (from === to) continue;
      if (graph.hasEdge(from, to)) {
        result.addEdge(from, to);
      }
    }
  }
  return result;
}

/**
 * Build the mutilated graph G_{X̅} where all incoming edges to every
 * node in the treatment set X are removed.
 *
 * Uses `graph.do(node)` which removes incoming edges to `node` by
 * zeroing the corresponding column in the adjacency matrix.
 */
function buildGXbar(graph: CausalGraph, xNodes: string[]): CausalGraph {
  let g = graph;
  for (const xn of xNodes) {
    g = g.do(xn);
  }
  return g;
}

/**
 * Build a deterministic state key for visited-set cycle detection.
 *
 * Encodes the outcome set, treatment set, and graph node set so that
 * re-visiting the same (Y, X, G) combination is detected and blocked.
 */
function buildStateKey(
  y: ReadonlySet<string>,
  x: ReadonlySet<string>,
  graph: CausalGraph,
): string {
  const ySorted = [...y].sort().join(',');
  const xSorted = [...x].sort().join(',');
  const gNodes = [...graph.nodes].sort().join(',');
  return `Y:{${ySorted}}|X:{${xSorted}}|G:[${gNodes}]`;
}
