/**
 * Causal Graph Fingerprint & Similarity.
 *
 * Computes a fixed-size structural fingerprint vector from a causal DAG's
 * adjacency matrix, then compares fingerprints via cosine similarity.
 *
 * Fingerprint features (13 dimensions):
 *   0: nodeCount (normalized)
 *   1: edgeCount / maxPossible (density proxy)
 *   2: rootRatio (exogenous variables)
 *   3: leafRatio (terminal nodes)
 *   4: vStructureRatio (collider signatures — uniquely causal)
 *   5-9: out-degree distribution bins [0,1,2,3,4+]
 *   10-12: maxDepth, avgDepth, graphDiameter
 *
 * Cosine similarity ∈ [0, 1]: 1 = structurally identical fingerprint.
 * Deterministic (no randomness), O(n²) compute, O(d) compare (d=13).
 *
 * @packageDocumentation
 */

/** A 13-dimensional structural fingerprint of a causal graph. */
export type CausalFingerprint = Float64Array;

// ── BFS helper ─────────────────────────────────────────────────────

function bfsDepth(
  adj: { get(i: number, j: number): number },
  n: number,
  startNodes: number[],
): { maxDepth: number; totalDepth: number; reachable: number } {
  const visited = new Array<boolean>(n).fill(false);
  const depth = new Array<number>(n).fill(0);
  const queue: number[] = [];
  for (const s of startNodes) {
    visited[s] = true;
    queue.push(s);
  }

  let head = 0;
  while (head < queue.length) {
    const u = queue[head++]!;
    for (let v = 0; v < n; v++) {
      if (adj.get(u, v) !== 0 && !visited[v]) {
        visited[v] = true;
        depth[v] = depth[u]! + 1;
        queue.push(v);
      }
    }
  }

  let maxD = 0, totalD = 0, r = 0;
  for (let i = 0; i < n; i++) {
    if (visited[i]) {
      r++;
      if (depth[i]! > maxD) maxD = depth[i]!;
      totalD += depth[i]!;
    }
  }
  return { maxDepth: maxD, totalDepth: totalD, reachable: r };
}

// ── Fingerprint Computation ─────────────────────────────────────────

/**
 * Compute a causal structural fingerprint from a graph.
 *
 * Accepts any object with `get(i,j): number` adjacency access
 * and a `nodes: readonly string[]` property.
 */
export function computeFingerprint(graph: {
  readonly nodes: readonly string[];
  readonly edges?: readonly { source: string; target: string }[];
  adjacencyMatrix?: { get(i: number, j: number): number };
  hasEdge?(from: string, to: string): boolean;
  parents?(node: string): string[];
}): CausalFingerprint {
  const n = graph.nodes.length;
  const fp = new Float64Array(13);

  if (n === 0) return fp;

  const adj = graph.adjacencyMatrix;
  const edges = graph.edges;

  // Build adjacency lookup if no matrix but edges available
  let edgeSet: Set<string> | undefined;
  let parentsMap: Map<string, string[]> | undefined;
  if (!adj && edges) {
    edgeSet = new Set(edges.map(e => `${e.source}|${e.target}`));
    parentsMap = new Map<string, string[]>();
    for (const e of edges) {
      const list = parentsMap.get(e.target) ?? [];
      list.push(e.source);
      parentsMap.set(e.target, list);
    }
  }

  let edges2 = 0, roots = 0, leaves = 0, vstructs = 0;
  const outDeg = new Array(5).fill(0);
  const rootIndices: number[] = [];

  for (let i = 0; i < n; i++) {
    let out = 0, inn = 0;
    for (let j = 0; j < n; j++) {
      if (adj) {
        if (adj.get(i, j) !== 0) { out++; edges2++; }
        if (adj.get(j, i) !== 0) inn++;
      } else if (edgeSet) {
        const ni = graph.nodes[i]!, nj = graph.nodes[j]!;
        if (edgeSet.has(`${ni}|${nj}`)) { out++; edges2++; }
        if (edgeSet.has(`${nj}|${ni}`)) inn++;
      }
    }
    outDeg[Math.min(out, 4)]!++;
    if (out === 0) leaves++;
    if (inn === 0) { roots++; rootIndices.push(i); }
  }

  // V-structures
  for (let k = 0; k < n; k++) {
    const pars: number[] = [];
    for (let i = 0; i < n; i++) {
      const isParent = adj
        ? adj.get(i, k) !== 0
        : edgeSet
          ? edgeSet.has(`${graph.nodes[i]}|${graph.nodes[k]}`)
          : false;
      if (isParent) pars.push(i);
    }
    for (let a = 0; a < pars.length; a++) {
      for (let b = a + 1; b < pars.length; b++) {
        const i = pars[a]!, j = pars[b]!;
        const ij = adj ? adj.get(i, j) !== 0 : edgeSet ? edgeSet.has(`${graph.nodes[i]}|${graph.nodes[j]}`) : false;
        const ji = adj ? adj.get(j, i) !== 0 : edgeSet ? edgeSet.has(`${graph.nodes[j]}|${graph.nodes[i]}`) : false;
        if (!ij && !ji) vstructs++;
      }
    }
  }

  // BFS from roots
  const { maxDepth, totalDepth, reachable } = bfsDepth(
    adj ?? (edgeSet
      ? { get: (i: number, j: number) => edgeSet.has(`${graph.nodes[i]}|${graph.nodes[j]}`) ? 1 : 0 }
      : { get: () => 0 }),
    n,
    rootIndices.length > 0 ? rootIndices : n > 0 ? [0] : [],
  );

  const maxEdges = n * (n - 1);
  const maxDepthNorm = Math.min(maxDepth, 50) / 50;
  const avgDepthNorm = reachable > 0 ? (totalDepth / reachable) / 20 : 0;
  const diameterNorm = reachable > 0 ? Math.min(maxDepth / Math.max(1, n - 1), 1) : 0;

  fp[0] = n <= 1 ? 0 : Math.log2(n) / Math.log2(200);  // log-scaled node count
  fp[1] = maxEdges > 0 ? edges2 / maxEdges : 0;
  fp[2] = n > 0 ? roots / n : 0;
  fp[3] = n > 0 ? leaves / n : 0;
  fp[4] = maxEdges > 0 ? Math.min(vstructs / Math.max(1, n), 1) : 0;
  for (let i = 0; i < 5; i++) fp[5 + i] = n > 0 ? (outDeg[i] ?? 0) / n : 0;
  fp[10] = maxDepthNorm;
  fp[11] = avgDepthNorm;
  fp[12] = diameterNorm;

  return fp;
}

// ── Cosine Similarity ────────────────────────────────────────────────

/**
 * Compute cosine similarity between two fingerprint vectors.
 *
 * cos(a, b) = (a·b) / (|a|·|b|)
 *
 * @returns similarity ∈ [0, 1] (1 = identical structure)
 */
export function cosineSimilarity(a: CausalFingerprint, b: CausalFingerprint): number {
  const len = Math.min(a.length, b.length);
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < len; i++) {
    dot += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom < 1e-12 ? 0 : Math.max(0, Math.min(1, dot / denom));
}

/**
 * Full pipeline: compute fingerprint → cosine similarity.
 */
export function graphSimilarity(
  a: {
    readonly nodes: readonly string[];
    adjacencyMatrix?: { get(i: number, j: number): number };
    hasEdge?(from: string, to: string): boolean;
    parents?(node: string): string[];
  },
  b: {
    readonly nodes: readonly string[];
    adjacencyMatrix?: { get(i: number, j: number): number };
    hasEdge?(from: string, to: string): boolean;
    parents?(node: string): string[];
  },
): number {
  return cosineSimilarity(computeFingerprint(a), computeFingerprint(b));
}
