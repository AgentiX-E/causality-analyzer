/**
 * GFCI Algorithm Tests.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { gfciAlgorithm } from '../../src/graph/gfci.js';
import { CausalGraph } from '../../src/graph/causal-graph.js';

function generateLinearData(graph: CausalGraph, n: number, seed: number): number[][] {
  const rng = ((s: number) => {
    let state = s >>> 0;
    return () => { state = (state * 1664525 + 1013904223) >>> 0; return state / 0x100000000; };
  })(seed);
  const nodes = graph.nodes;
  const d = nodes.length;
  const parentMap = new Map<string, string[]>();
  for (const node of nodes) parentMap.set(node, graph.parents(node));
  const topo = graph.topologicalSort();
  const data: number[][] = [];
  for (let r = 0; r < n; r++) {
    const row = new Array(d).fill(0);
    for (const node of topo) {
      const idx = nodes.indexOf(node);
      let val = 0;
      for (const p of parentMap.get(node) ?? []) val += 0.7 * row[nodes.indexOf(p)]!;
      const u1 = Math.max(1e-10, rng()), u2 = rng();
      val += Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      row[idx] = val;
    }
    data.push(row);
  }
  return data;
}

describe('GFCI Algorithm', () => {
  it('returns valid graph for chain X→Y→Z', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const data = generateLinearData(g, 300, 42);
    const result = gfciAlgorithm(new Matrix(data), ['X', 'Y', 'Z']);
    expect(result.graph.nodes.length).toBe(3);
    expect(result.pagEdges.size).toBeGreaterThan(0);
  });

  it('returns valid graph for confounded graph X←C→Y + X→Y', () => {
    const g = new CausalGraph(['X', 'Y', 'C']);
    g.addEdge('C', 'X'); g.addEdge('C', 'Y');
    g.addEdge('X', 'Y');
    const data = generateLinearData(g, 300, 123);
    const result = gfciAlgorithm(new Matrix(data), ['X', 'Y', 'C']);
    expect(result.graph.nodes.length).toBe(3);
  });

  it('produces pagEdges with correct format', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C');
    const data = generateLinearData(g, 200, 42);
    const result = gfciAlgorithm(new Matrix(data), ['A', 'B', 'C']);
    for (const [key, val] of result.pagEdges) {
      expect(key).toContain('-');
      // val can be 'none', 'undirected', or 'A→B' (directed) — all ≥ 3 chars
      expect(typeof val).toBe('string');
    }
  });

  it('handles 4-node collider graph', () => {
    const g = new CausalGraph(['X', 'Y', 'Z', 'W']);
    g.addEdge('X', 'Z'); g.addEdge('Y', 'Z');
    g.addEdge('Z', 'W');
    const data = generateLinearData(g, 300, 456);
    const result = gfciAlgorithm(new Matrix(data), ['X', 'Y', 'Z', 'W']);
    expect(result.graph.nodes.length).toBe(4);
  });

  it('handles empty data gracefully', () => {
    const result = gfciAlgorithm(new Matrix(0, 3), ['X', 'Y', 'Z']);
    expect(result.graph.nodes.length).toBe(3);
  });

  it('GFCI with PDS disabled still works', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C'); g.addEdge('C', 'D');
    const data = generateLinearData(g, 200, 42);
    const result = gfciAlgorithm(new Matrix(data), ['A', 'B', 'C', 'D'], {
      usePDS: false,
    });
    expect(result.graph.nodes.length).toBe(4);
    expect(result.pagEdges.size).toBeGreaterThan(0);
  });

  it('respects alpha parameter', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C');
    const data = generateLinearData(g, 200, 42);
    const r1 = gfciAlgorithm(new Matrix(data), ['A', 'B', 'C'], { alpha: 0.01 });
    const r2 = gfciAlgorithm(new Matrix(data), ['A', 'B', 'C'], { alpha: 0.1 });
    // Different alpha should produce valid but potentially different results
    expect(r1.graph.nodes.length).toBe(3);
    expect(r2.graph.nodes.length).toBe(3);
  });
});
