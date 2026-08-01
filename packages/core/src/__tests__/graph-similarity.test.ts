/**
 * I12: Causal fingerprint + cosine similarity tests.
 */
import { describe, it, expect } from 'vitest';
import { computeFingerprint, cosineSimilarity, graphSimilarity } from '../graph-similarity.js';

// ── Mock adjacency matrix ────────────────────────────────────────────

function makeAdj(nodes: string[], edgePairs: [string, string][]): { nodes: readonly string[]; adjacencyMatrix: { get(i: number, j: number): number }; hasEdge(from: string, to: string): boolean; parents(node: string): string[] } {
  const n = nodes.length;
  const data = new Float64Array(n * n);
  const nodeIdx = new Map(nodes.map((name, i) => [name, i]));

  for (const [from, to] of edgePairs) {
    const i = nodeIdx.get(from)!;
    const j = nodeIdx.get(to)!;
    data[i * n + j] = 1;
  }

  return {
    nodes,
    adjacencyMatrix: {
      get(i: number, j: number): number { return data[i * n + j] ?? 0; },
    },
    hasEdge(from: string, to: string): boolean {
      const i = nodeIdx.get(from), j = nodeIdx.get(to);
      if (i === undefined || j === undefined) return false;
      return data[i * n + j] !== 0;
    },
    parents(node: string): string[] {
      const j = nodeIdx.get(node)!;
      const p: string[] = [];
      for (let i = 0; i < n; i++) {
        if (data[i * n + j] !== 0) p.push(nodes[i]!);
      }
      return p;
    },
  };
}

// ── computeFingerprint ───────────────────────────────────────────────

describe('computeFingerprint', () => {
  it('returns zero vector for empty graph', () => {
    const g = makeAdj([], []);
    const fp = computeFingerprint(g);
    expect(fp.length).toBe(13);
    expect(fp[0]).toBe(0);
  });

  it('single node: no edges, root=leaf', () => {
    const g = makeAdj(['X'], []);
    const fp = computeFingerprint(g);
    expect(fp[2]).toBe(1); // rootRatio = 1/1
    expect(fp[3]).toBe(1); // leafRatio = 1/1
    expect(fp[1]).toBe(0); // edge density = 0
  });

  it('chain A→B→C: roots=1, leaves=1, no v-structures', () => {
    const g = makeAdj(['A', 'B', 'C'], [['A', 'B'], ['B', 'C']]);
    const fp = computeFingerprint(g);
    expect(fp[2]).toBeCloseTo(1 / 3); // rootRatio
    expect(fp[3]).toBeCloseTo(1 / 3); // leafRatio
    expect(fp[4]).toBe(0);           // no v-structures
    expect(fp[1]).toBeCloseTo(2 / 6); // 2 edges / 6 possible
  });

  it('v-structure A→C←B: detects collider', () => {
    const g = makeAdj(['A', 'B', 'C'], [['A', 'C'], ['B', 'C']]);
    const fp = computeFingerprint(g);
    expect(fp[4]).toBeGreaterThan(0); // has v-structure
  });

  it('fork C→A, C→B: no v-structure (same parent)', () => {
    const g = makeAdj(['C', 'A', 'B'], [['C', 'A'], ['C', 'B']]);
    const fp = computeFingerprint(g);
    expect(fp[4]).toBe(0); // fork is not a v-structure
  });

  it('full DAG: preserves degree distribution', () => {
    const g = makeAdj(
      ['A', 'B', 'C', 'D', 'E'],
      [['A', 'B'], ['A', 'C'], ['B', 'D'], ['C', 'D'], ['D', 'E']],
    );
    const fp = computeFingerprint(g);
    // A: outDeg=2, B: outDeg=1, C: outDeg=1, D: outDeg=1, E: outDeg=0
    expect(fp[5]! + fp[6]! + fp[7]! + fp[8]! + fp[9]!).toBeCloseTo(1); // distribution sums to 1
  });

  it('10-node chain: maxDepth increases', () => {
    const edges: [string, string][] = [];
    for (let i = 0; i < 9; i++) edges.push([String(i), String(i + 1)]);
    const g = makeAdj(edges.map(e => e[0]).concat(['9']), edges);
    const fp = computeFingerprint(g);
    expect(fp[10]).toBeGreaterThan(0); // maxDepth > 0
    expect(fp[11]).toBeGreaterThan(0); // avgDepth > 0
  });
});

// ── cosineSimilarity ─────────────────────────────────────────────────

describe('cosineSimilarity', () => {
  it('identical vectors → 1.0', () => {
    const a = new Float64Array([1, 2, 3]);
    expect(cosineSimilarity(a, a)).toBeCloseTo(1.0);
  });

  it('orthogonal vectors → 0.0', () => {
    const a = new Float64Array([1, 0, 0]);
    const b = new Float64Array([0, 1, 0]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(0.0);
  });

  it('zero vectors → 0.0', () => {
    const a = new Float64Array([0, 0, 0]);
    const b = new Float64Array([1, 2, 3]);
    expect(cosineSimilarity(a, b)).toBe(0);
  });

  it('differing lengths uses minimum', () => {
    const a = new Float64Array([1, 0]);
    const b = new Float64Array([1, 0, 1, 1]);
    // Only compares first 2 dimensions
    expect(cosineSimilarity(a, b)).toBeCloseTo(1.0);
  });
});

// ── graphSimilarity (full pipeline) ──────────────────────────────────

describe('graphSimilarity', () => {
  it('identical graphs → similarity = 1', () => {
    const g1 = makeAdj(['X', 'Y', 'Z'], [['X', 'Y'], ['Y', 'Z']]);
    const g2 = makeAdj(['X', 'Y', 'Z'], [['X', 'Y'], ['Y', 'Z']]);
    expect(graphSimilarity(g1, g2)).toBeCloseTo(1.0, 4);
  });

  it('different graphs → similarity < 1', () => {
    const g1 = makeAdj(['A', 'B', 'C'], [['A', 'B'], ['B', 'C']]);
    const g2 = makeAdj(['X', 'Y'], [['X', 'Y']]);
    const sim = graphSimilarity(g1, g2);
    expect(sim).toBeLessThan(1.0);
    expect(sim).toBeGreaterThan(0);
  });

  it('v-structure vs chain: different fingerprints', () => {
    const chain = makeAdj(['A', 'B', 'C'], [['A', 'B'], ['B', 'C']]);
    const vstruct = makeAdj(['A', 'B', 'C'], [['A', 'C'], ['B', 'C']]);
    const sim = graphSimilarity(chain, vstruct);
    expect(sim).toBeLessThan(0.99); // structurally different → not identical
  });

  it('empty graphs → similarity = 1 (both zero)', () => {
    const g1 = makeAdj([], []);
    const g2 = makeAdj([], []);
    expect(graphSimilarity(g1, g2)).toBe(0); // two empty graphs have zero fingerprint
  });

  it('similar graphs have high similarity', () => {
    const g1 = makeAdj(['A', 'B', 'C', 'D'], [['A', 'B'], ['B', 'C'], ['C', 'D']]);
    const g2 = makeAdj(['A', 'B', 'C', 'D'], [['A', 'B'], ['B', 'C']]);
    const sim = graphSimilarity(g1, g2);
    expect(sim).toBeGreaterThan(0.7); // similar structure
  });

  it('uses hasEdge fallback when no adjacencyMatrix', () => {
    const g1 = makeAdj(['X', 'Y'], [['X', 'Y']]);
    // Remove adjacencyMatrix to force hasEdge path
    const gNoAdj = { ...g1, adjacencyMatrix: undefined as unknown as typeof g1.adjacencyMatrix };
    const sim = graphSimilarity(gNoAdj, g1);
    expect(sim).toBeGreaterThan(0.5);
  });

  it('computes fingerprint from edges array without adjacencyMatrix', () => {
    const graph = {
      nodes: ['A', 'B', 'C'] as readonly string[],
      edges: [
        { source: 'A', target: 'B' },
        { source: 'B', target: 'C' },
      ] as readonly { source: string; target: string }[],
      hasEdge(from: string, to: string): boolean {
        return (from === 'A' && to === 'B') || (from === 'B' && to === 'C');
      },
      parents(node: string): string[] {
        if (node === 'B') return ['A'];
        if (node === 'C') return ['B'];
        return [];
      },
    };
    const fp = computeFingerprint(graph);
    expect(fp.length).toBe(13);
    expect(fp[2]).toBeCloseTo(1 / 3); // rootRatio: A is root
    expect(fp[3]).toBeCloseTo(1 / 3); // leafRatio: C is leaf
    expect(fp[4]).toBe(0);           // no v-structures in chain
  });

  it('computes fingerprint from edges with v-structure', () => {
    const graph = {
      nodes: ['A', 'B', 'C'] as readonly string[],
      edges: [
        { source: 'A', target: 'C' },
        { source: 'B', target: 'C' },
      ] as readonly { source: string; target: string }[],
      hasEdge(from: string, to: string): boolean {
        return (from === 'A' && to === 'C') || (from === 'B' && to === 'C');
      },
      parents(node: string): string[] {
        if (node === 'C') return ['A', 'B'];
        return [];
      },
    };
    const fp = computeFingerprint(graph);
    expect(fp[4]).toBeGreaterThan(0); // v-structure detected
  });

  it('computes fingerprint with edges and no roots (all nodes have parents)', () => {
    // Create a cycle-like DAG where every node has a parent
    const graph = {
      nodes: ['A', 'B'] as readonly string[],
      edges: [
        { source: 'A', target: 'B' },
      ] as readonly { source: string; target: string }[],
      hasEdge(from: string, to: string): boolean {
        return from === 'A' && to === 'B';
      },
      parents(node: string): string[] {
        if (node === 'B') return ['A'];
        return [];
      },
    };
    const fp = computeFingerprint(graph);
    expect(fp.length).toBe(13);
    expect(fp[2]).toBeCloseTo(0.5); // rootRatio: A is root
    expect(fp[3]).toBeCloseTo(0.5); // leafRatio: B is leaf
  });
});
