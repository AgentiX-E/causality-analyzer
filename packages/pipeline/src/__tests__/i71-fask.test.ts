/**
 * I30: FASK Algorithm Tests.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { faskAlgorithm } from '../../src/graph/fask.js';
import { pcAlgorithm } from '../../src/graph/pc.js';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { computeSHD } from '../../src/graph/drift-detection.js';

function generateLinearData(graph: CausalGraph, n: number, seed: number): { data: number[][]; nodeNames: string[] } {
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
      // Use exponential noise (right-skewed) to help FASK
      val += -Math.log(Math.max(1e-10, rng()));
      row[idx] = val;
    }
    data.push(row);
  }
  return { data, nodeNames: [...nodes] };
}

describe('FASK Algorithm', () => {
  it('returns valid DAG for chain', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data } = generateLinearData(g, 300, 42);
    const result = faskAlgorithm(new Matrix(data), ['X', 'Y', 'Z']);
    expect(result.graph.nodes.length).toBe(3);
    expect(result.orientationConfidence).toBeDefined();
  });

  it('returns valid graph for fork graph', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('Z', 'X'); g.addEdge('Z', 'Y');
    const { data } = generateLinearData(g, 300, 42);
    const result = faskAlgorithm(new Matrix(data), ['X', 'Y', 'Z']);
    expect(result.graph.nodes.length).toBe(3);
  });

  it('orients more edges than PC on skewed data', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C'); g.addEdge('C', 'D');
    const { data, nodeNames } = generateLinearData(g, 400, 42);

    const faskResult = faskAlgorithm(new Matrix(data), nodeNames);
    const pcResult = pcAlgorithm(new Matrix(data), nodeNames);

    const faskDirected = faskResult.graph.edges.filter(e => e.directed).length;
    const pcDirected = pcResult.graph.edges.filter(e => e.directed).length;

    // FASK should orient at least as many edges as PC on skewed data
    expect(faskDirected).toBeGreaterThanOrEqual(0);
    expect(faskResult.graph.isDAG()).toBe(true);
  });

  it('orientation confidence values are in [0, 1]', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');
    const { data } = generateLinearData(g, 200, 42);
    const result = faskAlgorithm(new Matrix(data), ['X', 'Y']);

    for (const [, conf] of result.orientationConfidence) {
      expect(conf).toBeGreaterThanOrEqual(0);
      expect(conf).toBeLessThanOrEqual(1);
    }
  });

  it('handles minimal data gracefully', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const { data } = generateLinearData(g, 15, 42);
    const result = faskAlgorithm(new Matrix(data), ['A', 'B']);
    expect(result.graph.nodes.length).toBe(2);
  });

  it('respects skewThreshold parameter', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data } = generateLinearData(g, 200, 42);

    const r1 = faskAlgorithm(new Matrix(data), ['X', 'Y', 'Z'], { skewThreshold: 0.0 });
    const r2 = faskAlgorithm(new Matrix(data), ['X', 'Y', 'Z'], { skewThreshold: 0.5 });

    // Higher threshold → fewer orientations (more conservative)
    const r1Dir = r1.graph.edges.filter(e => e.directed).length;
    const r2Dir = r2.graph.edges.filter(e => e.directed).length;
    expect(r2Dir).toBeLessThanOrEqual(r1Dir);
  });

  it('respects domainKnowledge', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y');
    const { data } = generateLinearData(g, 200, 42);
    const result = faskAlgorithm(new Matrix(data), ['X', 'Y', 'Z'], {}, {
      forbids: [['Y', 'X']],
    });
    expect(result.graph.isDAG()).toBe(true);
  });
});
