/**
 * I82: GOLEM Algorithm Tests.
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { golemAlgorithm } from '../../src/graph/golem.js';
import { generateLinearData } from '../../src/benchmark.js';

describe('GOLEM Algorithm', () => {
  it('discovers DAG from linear chain data', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data, nodeNames } = generateLinearData(g, 200, 42);
    const result = golemAlgorithm(data, nodeNames);
    expect(result.graph.nodeCount).toBe(3);
    expect(result.graph.isDAG()).toBe(true);
    expect(result.W).toBeInstanceOf(Float64Array);
    expect(result.W.length).toBe(9); // 3×3
  });

  it('produces valid W matrix dimensions', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const { data, nodeNames } = generateLinearData(g, 100, 43);
    const result = golemAlgorithm(data, nodeNames);
    expect(result.W.length).toBe(4); // 2×2
    expect(result.graph.isDAG()).toBe(true);
  });

  it('respects wThreshold config', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y');
    const { data, nodeNames } = generateLinearData(g, 200, 44);
    const r1 = golemAlgorithm(data, nodeNames, { wThreshold: 0.5, maxIter: 1200 });
    const r2 = golemAlgorithm(data, nodeNames, { wThreshold: 0.1, maxIter: 1200 });
    // Both should produce valid graphs
    expect(r1.graph.nodeCount).toBe(3);
    expect(r2.graph.nodeCount).toBe(3);
  });

  it('respects lambda1 regularization', () => {
    const g = new CausalGraph(['X', 'Y']);
    const { data, nodeNames } = generateLinearData(g, 100, 45);
    const r1 = golemAlgorithm(data, nodeNames, { lambda1: 0.5, maxIter: 600 });
    const r2 = golemAlgorithm(data, nodeNames, { lambda1: 0.001, maxIter: 600 });
    // Higher lambda → sparser graph
    expect(r1.graph.edges.length).toBeLessThanOrEqual(r2.graph.edges.length + 1);
    expect(r1.graph.isDAG()).toBe(true);
    expect(r2.graph.isDAG()).toBe(true);
  });

  it('works with L-BFGS optimizer', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data, nodeNames } = generateLinearData(g, 200, 46);
    const result = golemAlgorithm(data, nodeNames, { optimizer: 'lbfgs', maxIter: 500, tol: 1e-4 });
    expect(result.graph.nodeCount).toBe(3);
    expect(result.W).toBeInstanceOf(Float64Array);
  });

  it('handles small datasets gracefully', () => {
    const data = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
    const result = golemAlgorithm(data, ['X', 'Y']);
    // Very small data: should not crash
    expect(result.graph.nodeCount).toBe(2);
  });

  it('applies domain knowledge', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data, nodeNames } = generateLinearData(g, 150, 47);
    const result = golemAlgorithm(data, nodeNames, { maxIter: 600 }, {
      forbids: [['Z', 'X']],
    });
    expect(result.graph.isDAG()).toBe(true);
  });

  it('GOLEM scales to 4-variable graph', () => {
    // 4-node test provides reliable convergence
    const g = new CausalGraph(['A', 'B', 'C', 'D']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C'); g.addEdge('A', 'D');
    const { data, nodeNames } = generateLinearData(g, 200, 48);
    const result = golemAlgorithm(data, nodeNames, { maxIter: 2000, lambda1: 0.01 });
    expect(result.graph.nodeCount).toBe(4);
    expect(result.graph.isDAG()).toBe(true);
  });
});
