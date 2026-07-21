/**
 * CausalGraph — DAG representation with causal semantics.
 *
 * Supports d-separation, do-surgery, cycle detection, PDAG→DAG conversion,
 * and adjacency matrix operations backed by ml-matrix.
 */
import { Matrix } from 'ml-matrix';
import type { CausalEdge, DomainKnowledge } from '@agentix-e/causality-analyzer-core';

export class CausalGraph {
  readonly nodes: readonly string[];
  private adj: Matrix;  // adjacency matrix (n×n, entry[i][j]=1 means i→j)
  private _edges: CausalEdge[] | null = null;

  constructor(nodes: readonly string[], adjacency?: Matrix) {
    this.nodes = [...nodes];
    this.adj = adjacency ?? Matrix.zeros(nodes.length, nodes.length);
  }

  get nodeCount(): number { return this.nodes.length; }

  nodeIndex(name: string): number {
    const idx = this.nodes.indexOf(name);
    if (idx === -1) throw new Error(`Node "${name}" not found`);
    return idx;
  }

  hasEdge(from: string, to: string): boolean {
    const i = this.nodes.indexOf(from), j = this.nodes.indexOf(to);
    if (i === -1 || j === -1) return false;
    return this.adj.get(i, j) === 1;
  }

  addEdge(from: string, to: string): void {
    const i = this.nodeIndex(from), j = this.nodeIndex(to);
    this.adj.set(i, j, 1);
    this._edges = null;
  }

  removeEdge(from: string, to: string): void {
    const i = this.nodeIndex(from), j = this.nodeIndex(to);
    this.adj.set(i, j, 0);
    this._edges = null;
  }

  undirectedEdge(a: string, b: string): void {
    this.addEdge(a, b); this.addEdge(b, a);
  }

  toUndirected(a: string, b: string): void {
    this.removeEdge(b, a); this.addEdge(a, b);
  }

  get edges(): CausalEdge[] {
    if (this._edges) return this._edges;
    const result: CausalEdge[] = [];
    for (let i = 0; i < this.nodeCount; i++) {
      for (let j = 0; j < this.nodeCount; j++) {
        if (this.adj.get(i, j) === 1 && i !== j) {
          const isBidirected = this.adj.get(j, i) === 1;
          result.push({ source: this.nodes[i]!, target: this.nodes[j]!, weight: 1, directed: !isBidirected });
          if (isBidirected && i < j) continue;
        }
      }
    }
    this._edges = result;
    return result;
  }

  parents(node: string): string[] {
    const j = this.nodeIndex(node);
    const result: string[] = [];
    for (let i = 0; i < this.nodeCount; i++) {
      if (i !== j && this.adj.get(i, j) === 1) result.push(this.nodes[i]!);
    }
    return result;
  }

  children(node: string): string[] {
    const i = this.nodeIndex(node);
    const result: string[] = [];
    for (let j = 0; j < this.nodeCount; j++) {
      if (i !== j && this.adj.get(i, j) === 1) result.push(this.nodes[j]!);
    }
    return result;
  }

  neighbors(node: string): string[] {
    const idx = this.nodeIndex(node);
    const result: string[] = [];
    for (let k = 0; k < this.nodeCount; k++) {
      if (k !== idx && this.adj.get(idx, k) === 1 && this.adj.get(k, idx) === 1) result.push(this.nodes[k]!);
    }
    return result;
  }

  isDAG(): boolean { return !this.hasCycle(); }

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

  /** Do-surgery: remove all incoming edges to the specified node */
  do(node: string): CausalGraph {
    const j = this.nodeIndex(node);
    const newAdj = this.adj.clone();
    for (let i = 0; i < this.nodeCount; i++) newAdj.set(i, j, 0);
    return new CausalGraph(this.nodes, newAdj);
  }

  /** d-separation check: are X and Y conditionally independent given Z? */
  dSeparated(x: string, y: string, z: string[]): boolean {
    const ix = this.nodeIndex(x), iy = this.nodeIndex(y);
    const zSet = new Set(z.map(n => this.nodeIndex(n)));
    const n = this.nodeCount;

    // Build moralized ancestral graph and check m-separation
    const ancestors = new Set<number>();
    const collectAncestors = (v: number) => {
      if (ancestors.has(v)) return;
      ancestors.add(v);
      for (let u = 0; u < n; u++) if (this.adj.get(u, v) === 1) collectAncestors(u);
    };
    collectAncestors(ix); collectAncestors(iy);
    for (const zi of zSet) collectAncestors(zi);

    // Check via BFS reachability in moralized graph
    const moralAdj = new Array(n).fill(null).map(() => new Array(n).fill(false));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (this.adj.get(i, j) === 1 || this.adj.get(j, i) === 1) {
          moralAdj[i]![j] = true; moralAdj[j]![i] = true;
        }
      }
    }
    // Moralize: connect non-adjacent parents of each node
    for (let v = 0; v < n; v++) {
      const pa: number[] = [];
      for (let u = 0; u < n; u++) if (this.adj.get(u, v) === 1) pa.push(u);
      for (let a = 0; a < pa.length; a++) {
        for (let b = a + 1; b < pa.length; b++) {
          moralAdj[pa[a]!]![pa[b]!] = true;
          moralAdj[pa[b]!]![pa[a]!] = true;
        }
      }
    }

    // BFS reachability in moralized graph, blocked by Z
    const visited = new Array(n).fill(false);
    const queue = [ix];
    visited[ix] = true;
    while (queue.length > 0) {
      const v = queue.shift()!;
      if (v === iy) return false;
      for (let w = 0; w < n; w++) {
        if (!moralAdj[v]![w] || visited[w] || !ancestors.has(w)) continue;
        if (zSet.has(w)) continue; // blocked
        visited[w] = true;
        queue.push(w);
      }
    }
    return true;
  }

  /** PDAG → DAG conversion via sink-finding algorithm */
  pdag2dag(): CausalGraph {
    const n = this.nodeCount;
    const adj = this.adj.clone();
    // Track undirected edges (both directions exist)
    const undirected = new Set<string>();
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      if (adj.get(i, j) === 1 && adj.get(j, i) === 1) undirected.add(`${i}-${j}`);
    }
    let changed = true;
    while (changed) {
      changed = false;
      for (let i = 0; i < n; i++) {
        let isSink = true;
        for (let j = 0; j < n; j++) {
          if (undirected.has(`${Math.min(i, j)}-${Math.max(i, j)}`)) { isSink = false; break; }
        }
        if (isSink) {
          for (let j = 0; j < n; j++) {
            const key = `${Math.min(i, j)}-${Math.max(i, j)}`;
            if (undirected.has(key) && adj.get(j, i) === 1 && adj.get(i, j) === 1) {
              adj.set(i, j, 0); undirected.delete(key); changed = true;
            }
          }
        }
      }
    }
    return new CausalGraph(this.nodes, adj);
  }

  /** Topological order (Kahn's algorithm) */
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

  get adjacencyMatrix(): Matrix { return this.adj.clone(); }

  /** Structural Hamming Distance to another graph */
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

  clone(): CausalGraph { return new CausalGraph([...this.nodes], this.adj.clone()); }

  toJSON() {
    return { nodes: this.nodes, adjacency: this.adj.to2DArray(), edges: this.edges };
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
