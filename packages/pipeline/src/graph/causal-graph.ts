/**
 * CausalGraph — DAG representation with causal semantics.
 *
 * Supports strict d-separation (Pearl 2009, pp. 16–17), do-surgery,
 * cycle detection, PDAG→DAG conversion (Dor & Tarsi 1992),
 * and adjacency matrix operations backed by ml-matrix.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import type { CausalEdge, DomainKnowledge } from '@agentix-e/causality-analyzer-core';

export class CausalGraph {
  readonly nodes: readonly string[];
  private adj: Matrix; // adjacency matrix (n×n, entry[i][j]=1 means i→j)
  private _edges: CausalEdge[] | null = null;

  constructor(nodes: readonly string[], adjacency?: Matrix) {
    this.nodes = [...nodes];
    this.adj = adjacency ?? Matrix.zeros(nodes.length, nodes.length);
  }

  get nodeCount(): number {
    return this.nodes.length;
  }

  /** Resolve a node name to its matrix index; throws if unknown. */
  nodeIndex(name: string): number {
    const idx = this.nodes.indexOf(name);
    if (idx === -1) throw new Error(`Node "${name}" not found`);
    return idx;
  }

  // ── Edge operations ────────────────────────────────────────────────

  hasEdge(from: string, to: string): boolean {
    const i = this.nodes.indexOf(from),
      j = this.nodes.indexOf(to);
    if (i === -1 || j === -1) return false;
    return this.adj.get(i, j) === 1;
  }

  addEdge(from: string, to: string): void {
    const i = this.nodeIndex(from),
      j = this.nodeIndex(to);
    this.adj.set(i, j, 1);
    this._edges = null;
  }

  removeEdge(from: string, to: string): void {
    const i = this.nodeIndex(from),
      j = this.nodeIndex(to);
    this.adj.set(i, j, 0);
    this._edges = null;
  }

  /** Add an undirected edge (both directions). */
  undirectedEdge(a: string, b: string): void {
    this.addEdge(a, b);
    this.addEdge(b, a);
  }

  /** Convert an undirected edge to directed a→b. */
  toUndirected(a: string, b: string): void {
    this.removeEdge(b, a);
    this.addEdge(a, b);
  }

  // ── Graph traversal ────────────────────────────────────────────────

  get edges(): CausalEdge[] {
    if (this._edges) return this._edges;
    const result: CausalEdge[] = [];
    for (let i = 0; i < this.nodeCount; i++) {
      for (let j = 0; j < this.nodeCount; j++) {
        if (this.adj.get(i, j) === 1 && i !== j) {
          const isBidirected = this.adj.get(j, i) === 1;
          result.push({
            source: this.nodes[i]!,
            target: this.nodes[j]!,
            weight: 1,
            directed: !isBidirected,
          });
          if (isBidirected && i < j) continue; // deduplicate undirected
        }
      }
    }
    this._edges = result;
    return result;
  }

  /** Direct parents of `node` (nodes with an edge → node). */
  parents(node: string): string[] {
    const j = this.nodeIndex(node);
    const result: string[] = [];
    for (let i = 0; i < this.nodeCount; i++) {
      if (i !== j && this.adj.get(i, j) === 1) result.push(this.nodes[i]!);
    }
    return result;
  }

  /** Direct children of `node` (nodes with an edge node →). */
  children(node: string): string[] {
    const i = this.nodeIndex(node);
    const result: string[] = [];
    for (let j = 0; j < this.nodeCount; j++) {
      if (i !== j && this.adj.get(i, j) === 1) result.push(this.nodes[j]!);
    }
    return result;
  }

  /** Undirected neighbors of `node` (both directions exist). */
  neighbors(node: string): string[] {
    const idx = this.nodeIndex(node);
    const result: string[] = [];
    for (let k = 0; k < this.nodeCount; k++) {
      if (k !== idx && this.adj.get(idx, k) === 1 && this.adj.get(k, idx) === 1)
        result.push(this.nodes[k]!);
    }
    return result;
  }

  // ── Ancestors / Descendants ────────────────────────────────────────

  /**
   * All ancestors of the given nodes (including the nodes themselves).
   * Uses iterative DFS on the transpose graph for stack-safety.
   */
  ancestors(nodes: string[]): Set<string> {
    const anc = new Set<string>();
    const stack = nodes.map(n => this.nodeIndex(n));
    for (const v of stack) anc.add(this.nodes[v]!);

    while (stack.length > 0) {
      const v = stack.pop()!;
      for (let u = 0; u < this.nodeCount; u++) {
        if (this.adj.get(u, v) === 1) {
          const name = this.nodes[u]!;
          if (!anc.has(name)) {
            anc.add(name);
            stack.push(u);
          }
        }
      }
    }
    return anc;
  }

  /**
   * All descendants of `node` (including the node itself).
   * Uses iterative DFS for stack-safety.
   */
  descendants(node: string): Set<string> {
    const desc = new Set<string>();
    const vi = this.nodeIndex(node);
    desc.add(node);
    const stack = [vi];
    while (stack.length > 0) {
      const v = stack.pop()!;
      for (let w = 0; w < this.nodeCount; w++) {
        if (this.adj.get(v, w) === 1) {
          const name = this.nodes[w]!;
          if (!desc.has(name)) {
            desc.add(name);
            stack.push(w);
          }
        }
      }
    }
    return desc;
  }

  /**
   * Whether there exists a directed path from `from` to `to`.
   */
  hasDirectedPath(from: string, to: string): boolean {
    return this.descendants(from).has(to);
  }

  // ── Graph properties ───────────────────────────────────────────────

  isDAG(): boolean {
    return !this.hasCycle();
  }

  hasCycle(): boolean {
    const n = this.nodeCount;
    const visited = new Array(n).fill(0); // 0=unvisited, 1=visiting, 2=done
    const dfs = (v: number): boolean => {
      visited[v] = 1;
      for (let w = 0; w < n; w++) {
        if (this.adj.get(v, w) === 1) {
          if (visited[w] === 1) return true;
          if (visited[w] === 0 && dfs(w)) return true;
        }
      }
      visited[v] = 2;
      return false;
    };
    for (let v = 0; v < n; v++) if (visited[v] === 0 && dfs(v)) return true;
    return false;
  }

  /** Do-surgery: return a new graph with all incoming edges to `node` removed. */
  do(node: string): CausalGraph {
    const j = this.nodeIndex(node);
    const newAdj = this.adj.clone();
    for (let i = 0; i < this.nodeCount; i++) newAdj.set(i, j, 0);
    return new CausalGraph(this.nodes, newAdj);
  }

  // ── d-Separation (Pearl 2009, pp. 16–17) ───────────────────────────

  /**
   * Strict d-separation test: are X and Y conditionally independent given Z?
   *
   * A trail (path treating edges as undirected) is **d-connecting** iff:
   * 1. For every non-collider node on the trail: it is NOT in Z.
   * 2. For every collider node on the trail: it IS in Z, or some descendant is in Z.
   *
   * Two nodes are **d-separated** by Z iff no d-connecting trail exists between them.
   *
   * @param x — first node name
   * @param y — second node name
   * @param z — conditioning set (node names)
   * @returns true if X and Y are d-separated given Z
   */
  dSeparated(x: string, y: string, z: string[]): boolean {
    const ix = this.nodeIndex(x);
    const iy = this.nodeIndex(y);
    const zIdx = new Set(z.map(n => this.nodeIndex(n)));
    const zn = new Set(z);

    // A collider is activated iff the collider itself is in Z,
    // or some node in Z is a descendant of the collider.
    // Equivalent to: collider ∈ Z OR collider ∈ ancestors(Z).
    // So we precompute all ancestors of every node in Z.
    const zAncestors = new Set<string>();
    for (const n of zn) {
      for (const a of this.ancestors([n])) zAncestors.add(a);
    }

    // Edge case: self-dependence. X is never d-separated from itself.
    if (x === y) return false;

    // Check if any d-connecting trail exists via DFS on trails
    // A trail is a sequence of nodes where each consecutive pair shares an edge
    // (in either direction). We track the edge directions to classify colliders.
    const visitedTrails = new Set<string>();

    /**
     * Determine if, after arriving at node `v` from `prev`, node `v`
     * is a **collider** on the current trail.
     *
     * On a trail ... — u — v — w — ...
     *   If edges are u→v and w→v: v is a collider.
     *   If edges are u←v and v→w: v is NOT a collider (chain).
     *   If edges are u←v and w←v: v is NOT a collider (fork).
     *   If edges are u→v and v→w: v is NOT a collider (chain).
     */
    const isCollider = (u: number, v: number, w: number): boolean => {
      const uToV = this.adj.get(u, v) === 1;
      const wToV = this.adj.get(w, v) === 1;
      // A node is a collider if both edges point into it
      return uToV && wToV;
    };

    /**
     * Trail DFS. A trail is a simple path where edges are treated
     * as undirected (we can traverse either direction).
     *
     * @param current — current node index
     * @param prev — previous node index (-1 for start)
     * @param pp — node before previous, for collider detection (-1 when not available)
     * @param trail — visited nodes on current trail (for cycle prevention)
     * @param trailKey — string key for deduplication
     * @param open — whether the trail is currently open (not blocked)
     */
    const dfsTrail = (
      current: number,
      prev: number,
      pp: number,
      trail: number[],
      trailKey: string,
      open: boolean,
    ): boolean => {
      // If we reached Y and the trail is open, we found a d-connecting trail
      if (current === iy && open && trail.length > 1) return true;

      for (let w = 0; w < this.nodeCount; w++) {
        if (w === current || w === prev) continue;
        // Follow any edge in either direction
        const hasEdge = this.adj.get(current, w) === 1 || this.adj.get(w, current) === 1;
        if (!hasEdge) continue;
        if (trail.includes(w)) continue; // no cycles in trail

        const newTrail = [...trail, w];
        const newKey = trailKey + '|' + w;
        if (visitedTrails.has(newKey)) continue;
        visitedTrails.add(newKey);

        // Determine if `current` acts as a collider on the current trail segment
        let newOpen = open;
        if (trail.length >= 2) {
          // We're at `current`, coming from `prev`, going to `w`
          // pp is the node before prev (for collider detection at `prev` we already handled it)
          // Now we check if `current` is a collider in the segment pp-prev-current-w
          // Wait — collider is determined by the node in the middle:
          // For trail segment A — B — C, B is the node being evaluated.
          // We should evaluate `current` (which is B) using `prev` (A) and `w` (C).

          if (isCollider(prev, current, w)) {
            // current is a collider: trail opens if current or any descendant is in Z
            const cName = this.nodes[current]!;
            // Collider activation: collider ∈ Z OR ∃ node in Z that descends from collider
            // i.e., collider ∈ Z OR collider is an ancestor of some node in Z
            if (zn.has(cName) || zAncestors.has(cName)) {
              // collider activated — trail opens (or stays open)
              newOpen = true;
            } else {
              // collider not activated — trail is BLOCKED, skip this branch
              continue;
            }
          } else {
            // current is NOT a collider: trail is blocked if current is in Z
            if (zn.has(this.nodes[current]!)) continue;
            // otherwise trail stays in current state
          }
        } else {
        // First step from X: can't determine collider/non-collider yet
        // (the node w may be a collider on trails X-w-...).
        // Defer Z-checking to the next step where trail length ≥ 2.
        }

        if (dfsTrail(w, current, prev, newTrail, newKey, newOpen)) return true;
      }
      return false;
    };

    // Start DFS from X
    const startTrail = [ix];
    return !dfsTrail(ix, -1, -1, startTrail, `${ix}`, true);
  }

  // ── PDAG → DAG conversion (Dor & Tarsi 1992) ──────────────────────

  /**
   * Convert a PDAG (partially directed acyclic graph) to a DAG.
   *
   * Algorithm (Dor & Tarsi 1992):
   * 1. Collect undirected edges (i—j where both i→j and j→i exist).
   * 2. While undirected edges remain:
   *    a. Find a **sink** among the undirected-component nodes:
   *       a node that has undirected edges but NO outgoing *directed* edges.
   *    b. Orient all undirected edges incident to the sink **into** the sink.
   *    c. Remove these edges from the undirected set.
   *
   * @returns a fully directed DAG
   */
  pdag2dag(): CausalGraph {
    const n = this.nodeCount;
    const adj = this.adj.clone();

    // Collect current undirected edges
    const undirected = new Set<string>();
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (adj.get(i, j) === 1 && adj.get(j, i) === 1) {
          undirected.add(`${i}-${j}`);
        }
      }
    }

    // Sink-finding loop
    let changed = true;
    while (changed && undirected.size > 0) {
      changed = false;

      for (let i = 0; i < n; i++) {
        // Check if node i is incident to any undirected edges
        let hasUndirected = false;
        for (let j = 0; j < n; j++) {
          if (i !== j && undirected.has(`${Math.min(i, j)}-${Math.max(i, j)}`)) {
            hasUndirected = true;
            break;
          }
        }
        if (!hasUndirected) continue; // node not involved in undirected edges

        // Check if node i has any outgoing *directed* edges (strictly one-way)
        let hasOutgoing = false;
        for (let j = 0; j < n; j++) {
          if (i !== j && adj.get(i, j) === 1 && adj.get(j, i) === 0) {
            hasOutgoing = true;
            break;
          }
        }
        if (hasOutgoing) continue; // not a sink — has outgoing directed edges

        // Node i is a sink: orient all its undirected edges into i
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          const key = `${Math.min(i, j)}-${Math.max(i, j)}`;
          if (undirected.has(key)) {
            // Remove i→j edge, keep j→i (orient into sink i)
            adj.set(i, j, 0); // i no longer points to j
            undirected.delete(key);
            changed = true;
          }
        }

        if (changed) break; // restart scan after removing a sink
      }
    }

    return new CausalGraph(this.nodes, adj);
  }

  // ── Topological sort ───────────────────────────────────────────────

  /** Kahn's algorithm for topological ordering. */
  topologicalSort(): string[] {
    const n = this.nodeCount;
    const inDegree = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (this.adj.get(i, j) === 1) inDegree[j]++;
    const queue: number[] = [];
    for (let i = 0; i < n; i++) if (inDegree[i] === 0) queue.push(i);
    const result: string[] = [];
    while (queue.length > 0) {
      const v = queue.shift()!;
      result.push(this.nodes[v]!);
      for (let w = 0; w < n; w++) {
        if (this.adj.get(v, w) === 1 && --inDegree[w] === 0) queue.push(w);
      }
    }
    return result;
  }

  // ── Utility ────────────────────────────────────────────────────────

  get adjacencyMatrix(): Matrix {
    return this.adj.clone();
  }

  /** Structural Hamming Distance to another graph. */
  shd(other: CausalGraph): number {
    let dist = 0;
    for (let i = 0; i < this.nodeCount; i++) {
      for (let j = 0; j < this.nodeCount; j++) {
        if (i !== j && this.adj.get(i, j) !== other.adj.get(i, j)) dist++;
      }
    }
    return dist;
  }

  applyDomainKnowledge(dk: DomainKnowledge): void {
    if (dk.forbids) for (const [f, t] of dk.forbids) { this.removeEdge(f, t); this.removeEdge(t, f); }
    if (dk.requires) for (const [f, t] of dk.requires) { this.addEdge(f, t); this.removeEdge(t, f); }
    if (dk.rootNodes) for (const r of dk.rootNodes) {
      for (const n of this.nodes) { if (n !== r) this.removeEdge(n, r); }
    }
    if (dk.leafNodes) for (const l of dk.leafNodes) {
      for (const n of this.nodes) { if (n !== l) this.removeEdge(l, n); }
    }
  }

  clone(): CausalGraph {
    return new CausalGraph([...this.nodes], this.adj.clone());
  }

  toJSON() {
    return { nodes: this.nodes, adjacency: this.adj.to2DArray(), edges: this.edges };
  }

  /** Reconstruct a CausalGraph from its JSON representation. */
  static fromJSON(json: { nodes: string[]; adjacency: number[][]; edges?: CausalEdge[] }): CausalGraph {
    const n = json.nodes.length;

    // Prefer adjacency matrix for exact reconstruction
    if (json.adjacency && json.adjacency.length === n) {
      const adj = new Matrix(json.adjacency);
      return new CausalGraph([...json.nodes], adj);
    }

    // Fallback: reconstruct from edge list
    if (json.edges) return CausalGraph.fromEdges(json.nodes, json.edges);

    return new CausalGraph(json.nodes);
  }

  static fromEdges(nodes: string[], edges: CausalEdge[]): CausalGraph {
    const g = new CausalGraph(nodes);
    for (const e of edges) {
      g.addEdge(e.source, e.target);
      if (!e.directed) g.addEdge(e.target, e.source);
    }
    return g;
  }
}
