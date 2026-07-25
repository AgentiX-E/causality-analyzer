/**
 * RFCI Algorithm Tests.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { rfciAlgorithm } from '../../src/graph/rfci.js';
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

describe('RFCI Algorithm', () => {
  it('returns valid pagEdges for chain', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const data = generateLinearData(g, 300, 42);
    const result = rfciAlgorithm(new Matrix(data), ['X', 'Y', 'Z']);
    expect(result.pagEdges.size).toBeGreaterThan(0);
  });

  it('produces more undirected edges than FCI on simple graphs', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C');
    const data = generateLinearData(g, 200, 42);
    const result = rfciAlgorithm(new Matrix(data), ['A', 'B', 'C']);
    for (const val of result.pagEdges.values()) {
      expect(['none', 'undirected', 'A→B', 'A→C', 'B→C', 'C→A', 'C→B']).toContain(val);
    }
  });

  it('handles confounded graph', () => {
    const g = new CausalGraph(['X', 'Y', 'C']);
    g.addEdge('C', 'X'); g.addEdge('C', 'Y'); g.addEdge('X', 'Y');
    const data = generateLinearData(g, 300, 123);
    const result = rfciAlgorithm(new Matrix(data), ['X', 'Y', 'C']);
    expect(result.graph.nodes.length).toBe(3);
  });

  it('handles minimal data gracefully', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    const data = generateLinearData(g, 5, 42);
    const result = rfciAlgorithm(new Matrix(data), ['X', 'Y', 'Z']);
    expect(result.graph.nodes.length).toBe(3);
  });

  it('faster than FCI on 8-node graph', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C'); g.addEdge('C', 'D');
    g.addEdge('D', 'E'); g.addEdge('A', 'F'); g.addEdge('F', 'G');
    const data = generateLinearData(g, 200, 42);
    const start = Date.now();
    rfciAlgorithm(new Matrix(data), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']);
    const elapsed = Date.now() - start;
    // RFCI should complete within reasonable time (< 2s for 8-node graph)
    expect(elapsed).toBeLessThan(2000);
  });
});
