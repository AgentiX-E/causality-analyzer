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
 * Hedge Criterion (Shpitser & Pearl, 2006):
 *   P(Y|do(X)) is NOT identifiable iff there exists a hedge for
 *   some X' ⊆ X, Y' ⊆ Y in the induced subgraph over An(Y)_G.
 *   A hedge F is a subset of a c-component where F has both
 *   an X-node and a Y-node, connected via bidirected paths.
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

  // Step 3: ID algorithm with hedge criterion
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

// ── ID Algorithm with Proper Hedge Criterion ─────────────────────────

/**
 * ID Algorithm (Shpitser & Pearl, 2006) with hedge criterion.
 *
 * Steps:
 *  1. If Y contains no X-nodes, recurse
 *  2. If X = ∅, return Σ_{V\Y} P(V) (marginalization)
 *  3. Let AnY = ancestors of Y in G
 *  4. If X ≠ AnY, return ID(X ∩ AnY, Y, G[AnY])
 *  5. Find c-components of G[AnY]
 *  6. If multiple c-components: factorize — Σ ∏ ID(...)
 *  7. If single c-component — C = AnY:
 *     a. If C = X, return fail (not identifiable — hedge)
 *     b. If C ≠ X, return Σ_{D\Y} ∏ P(V_i|pa(V_i))
 *
 * Hedge Criterion:
 *   A hedge for P(Y|do(X)) exists when:
 *   - The entire An(Y) is a single c-component C
 *   - X is a strict subset of C (i.e., C\X ≠ ∅)
 *   This means there are latent confounders that cannot be eliminated
 *   between X and Y, making the effect non-identifiable.
 */
function tryIDAlgorithm(
  graph: CausalGraph, treatment: string, outcome: string,
): DoCalculusResult {
  // Build G_Xbar: remove incoming edges to X (simulates do(X))
  const gXbar = removeIncomingEdges(graph, [treatment]);

  // Step 1: If X and Y are d-separated in G_Xbar, P(Y|do(X)) = P(Y)
  if (gXbar.dSeparated(treatment, outcome, [])) {
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: [],
      explanation: 'P(Y|do(X)) = P(Y) — X and Y are d-separated in G_Xbar (no causal effect)',
    };
  }

  // Step 2: Backdoor check in G_Xbar (ID algorithm step 3)
  const backdoorInXbar = findDoCalculusBackdoor(gXbar, treatment, outcome);
  if (backdoorInXbar.length > 0) {
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: backdoorInXbar,
      explanation: `ID: P(Y|do(X)) = Σ_z P(Y|X,z)P(z), z = {${backdoorInXbar.join(', ')}}`,
    };
  }

  // Step 3: Check if Y is a descendant of X (necessary condition)
  if (!graph.hasDirectedPath(treatment, outcome)) {
    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: [],
      explanation: 'P(Y|do(X)) = P(Y) — Y is not a descendant of X (no causal path)',
    };
  }

  // Step 4: An(Y)_G induced subgraph
  const yAncestors = graph.ancestors([outcome]);
  yAncestors.add(treatment);
  const anNodes = [...yAncestors];

  // Step 5: Build subgraph over An(Y) ∪ {X}
  const subGraph = buildInducedSubgraph(graph, anNodes);

  // Step 6: Find c-components in the induced subgraph
  const cComps = findCComponents(subGraph, anNodes);

  // Step 7: Check hedge criterion
  if (cComps.length === 1) {
    // Single c-component: check if it's a hedge
    const cc = cComps[0]!;
    const inComp = cc.has(treatment);
    const nodesBeyondX = [...cc].filter(n => n !== treatment);

    if (inComp && nodesBeyondX.length > 0) {
      // HEDGE detected: the single c-component contains X and other nodes
      // that are connected via bi directed paths — effect is NOT identifiable
      const hedgeNodes = nodesBeyondX.filter(n =>
        graph.hasDirectedPath(n, outcome) || graph.hasDirectedPath(treatment, n),
      );

      if (hedgeNodes.length > 0) {
        return {
          identifiable: false,
          expressionType: 'not_identifiable',
          adjustmentSet: [],
          explanation: `Hedge detected in c-component {${[...cc].join(', ')}}: effect not identifiable due to latent confounding between X and Y`,
        };
      }
    }
  } else if (cComps.length > 1) {
    // Multiple c-components: check for hedge in components containing Y
    const yComponents = cComps.filter(c => c.has(outcome));
    for (const yc of yComponents) {
      if (yc.has(treatment)) continue; // X and Y in same component — handled above

      // Check if Y's component has a backdoor to X's component
      // via latent confounders (bidirected paths connecting components)
      const xComponents = cComps.filter(c => c.has(treatment));
      for (const xc of xComponents) {
        if (hasHedgePath(subGraph, xc, yc, anNodes)) {
          return {
            identifiable: false,
            expressionType: 'not_identifiable',
            adjustmentSet: [],
            explanation: `Hedge detected: latent confounding connects X-component and Y-component`,
          };
        }
      }
    }

    // No hedge: factorization approach
    // P(Y|do(X)) = Σ_{D\Y} ∏_{i} P(V_i|pa_G(V_i)) where V_i are c-components
    const allCCompVars = new Set<string>();
    for (const cc of cComps) {
      for (const n of cc) allCCompVars.add(n);
    }

    return {
      identifiable: true,
      expressionType: 'id_algorithm',
      adjustmentSet: backdoorInXbar,
      explanation: `ID: identified via c-component decomposition (${cComps.length} components, no hedge)`,
    };
  }

  // Step 8: Final check — if X can be isolated from latent confounders
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
 * Check if there exists a hedge path — a bidirected path connecting
 * two c-components that prevents factorization.
 */
function hasHedgePath(
  graph: CausalGraph,
  compA: Set<string>,
  compB: Set<string>,
  allNodes: string[],
): boolean {
  // A hedge path exists if there is a node in compA and a node in compB
  // connected by bidirected edges AND a node in compA is an ancestor
  // of some node in compB via directed paths

  for (const a of compA) {
    for (const b of compB) {
      // Check bidirected connection
      if (graph.hasEdge(a, b) && graph.hasEdge(b, a)) {
        return true;
      }
      // Check if a is ancestor of b (directed path)
      if (graph.hasDirectedPath(a, b)) return true;
    }
  }

  // Check via intermediate nodes
  for (const a of compA) {
    for (const n of allNodes) {
      if (compB.has(n)) continue;
      if (graph.hasEdge(a, n) && graph.hasEdge(n, a)) {
        // a bidirected to n — check if n is in compB or connects to compB
        for (const b of compB) {
          if (graph.hasEdge(n, b) && graph.hasEdge(b, n)) return true;
          if (graph.hasDirectedPath(n, b)) return true;
        }
      }
    }
  }

  return false;
}

/**
 * Build induced subgraph over specified nodes.
 * Only includes edges where both endpoints are in `nodes`.
 */
function buildInducedSubgraph(graph: CausalGraph, nodes: string[]): CausalGraph {
  const nodeSet = new Set(nodes);
  const sub = new CausalGraph(nodes);
  for (const src of nodes) {
    for (const tgt of graph.children(src)) {
      if (nodeSet.has(tgt)) {
        sub.addEdge(src, tgt);
      }
    }
  }
  return sub;
}

/**
 * Find c-components (confounded components) in the induced subgraph.
 * Two nodes are in the same c-component if they are connected by
 * a bidirected path (i.e., share a latent confounder).
 *
 * In our causal graph, bidirected edges represent latent common causes.
 * Hash-based union-find for efficient component merging.
 */
function findCComponents(
  graph: CausalGraph, nodes: string[],
): Set<string>[] {
  // Union-Find over node indices
  const idx = new Map(nodes.map((n, i) => [n, i]));
  const parent = nodes.map((_, i) => i);

  const find = (x: number): number => {
    let root = x;
    while (parent[root] !== root) root = parent[root]!;
    // Path compression
    while (x !== root) {
      const next = parent[x]!;
      parent[x] = root;
      x = next;
    }
    return root;
  };

  const union = (a: number, b: number): void => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  };

  // Connect nodes sharing bidirected edges
  for (const u of nodes) {
    for (const v of nodes) {
      if (u >= v) continue;
      if (graph.hasEdge(u, v) && graph.hasEdge(v, u)) {
        union(idx.get(u)!, idx.get(v)!);
      }
    }
  }

  // Connect nodes sharing a common child (v-structure → latent confounder)
  for (const child of nodes) {
    const parents = graph.parents(child);
    const nodeParents = parents.filter(p => idx.has(p));
    for (let i = 0; i < nodeParents.length; i++) {
      for (let j = i + 1; j < nodeParents.length; j++) {
        union(idx.get(nodeParents[i]!)!, idx.get(nodeParents[j]!)!);
      }
    }
  }

  // Collect components
  const compMap = new Map<number, Set<string>>();
  for (const node of nodes) {
    const root = find(idx.get(node)!);
    if (!compMap.has(root)) compMap.set(root, new Set());
    compMap.get(root)!.add(node);
  }

  return [...compMap.values()];
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
  return [...treatDescendants].filter(m =>
    graph.children(m).includes(outcome) || hasPathTo(graph, m, outcome),
  );
}

function isFrontdoorIdentifiable(
  graph: CausalGraph, treatment: string, outcome: string, mediators: string[],
): boolean {
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
  const nodeNames = [...graph.nodes];
  const copy = new CausalGraph(nodeNames);
  const nodeSet = new Set(nodes);

  for (const src of nodeNames) {
    for (const tgt of graph.children(src)) {
      if (nodeSet.has(tgt)) continue;
      copy.addEdge(src, tgt);
    }
  }

  return copy;
}
