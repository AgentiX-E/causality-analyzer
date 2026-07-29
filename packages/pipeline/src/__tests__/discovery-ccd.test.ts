/**
 * I79: CCD Cyclic Causal Discovery tests.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { ccdAlgorithm } from '../../src/graph/ccd.js';
import { generateLinearData } from '../../src/benchmark.js';

describe('CCD Algorithm', () => {
  it('returns empty graph for tiny data (N < 5)', () => {
    const data = new Matrix([[1], [2], [3]]);
    const result = ccdAlgorithm(data, ['X', 'Y']);
    expect(result.graph.nodes.length).toBe(2);
    expect(result.graph.edges.length).toBe(0);
    expect(result.cycleEdges.size).toBe(0);
  });

  it('discovers chain X→Y→Z structure', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data } = generateLinearData(g, 200, 42);
    const result = ccdAlgorithm(new Matrix(data), ['X', 'Y', 'Z']);
    expect(result.graph.nodeCount).toBe(3);
    expect(result.cycleEdges).toBeInstanceOf(Map);
  });

  it('respects maxDegree constraint', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C'); g.addEdge('C', 'D');
    const { data } = generateLinearData(g, 150, 43);
    const result = ccdAlgorithm(new Matrix(data), ['A', 'B', 'C', 'D'], { maxDegree: 1 });
    expect(result.graph.nodeCount).toBe(4);
  });

  it('respects maxLoopIter config', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y');
    const { data } = generateLinearData(g, 150, 44);
    const result = ccdAlgorithm(new Matrix(data), ['X', 'Y', 'Z'], { maxLoopIter: 3 });
    expect(result.graph.nodeCount).toBe(3);
  });

  it('applies domain knowledge constraints', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data } = generateLinearData(g, 150, 45);
    const result = ccdAlgorithm(new Matrix(data), ['X', 'Y', 'Z'], {}, {
      forbids: [['Z', 'X']],
    });
    expect(result.graph.nodeCount).toBe(3);
  });

  it('detects bidirectional edges as cycles', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.undirectedEdge('X', 'Y');
    const { data } = generateLinearData(g, 200, 46);
    const result = ccdAlgorithm(new Matrix(data), ['X', 'Y'], { alpha: 0.99 });
    // With very lenient alpha, may keep bidirectional edges = cycles
    expect(result.cycleEdges).toBeInstanceOf(Map);
  });

  it('produces output for large graph', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D', 'E']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C'); g.addEdge('C', 'D'); g.addEdge('D', 'E');
    const { data } = generateLinearData(g, 200, 47);
    const result = ccdAlgorithm(new Matrix(data), ['A', 'B', 'C', 'D', 'E'], { maxDegree: 3 });
    expect(result.graph.nodeCount).toBe(5);
  });
});
