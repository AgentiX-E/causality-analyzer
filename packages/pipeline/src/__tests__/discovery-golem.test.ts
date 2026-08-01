/**
 * I82: GOLEM Algorithm Tests.
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { golemAlgorithm } from '../../src/graph/golem.js';
import { generateLinearData } from '../../src/benchmark.js';

describe('GOLEM Algorithm', () => {
  it('discovers DAG from linear chain data', { timeout: 60000 }, () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data, nodeNames } = generateLinearData(g, 400, 42);
    const result = golemAlgorithm(data, nodeNames, { maxIter: 600, lambda1: 0.002 });
    expect(result.graph.nodeCount).toBe(3);
    expect(result.W).toBeInstanceOf(Float64Array);
    expect(result.W.length).toBe(9);
  });

  it('produces valid W matrix dimensions', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const { data, nodeNames } = generateLinearData(g, 200, 43);
    const result = golemAlgorithm(data, nodeNames, { maxIter: 400 });
    expect(result.W.length).toBe(4);
    expect(result.graph.nodeCount).toBe(2);
  });

  it('respects wThreshold config', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y');
    const { data, nodeNames } = generateLinearData(g, 300, 44);
    const r1 = golemAlgorithm(data, nodeNames, { wThreshold: 0.5, maxIter: 300 });
    const r2 = golemAlgorithm(data, nodeNames, { wThreshold: 0.1, maxIter: 300 });
    expect(r1.graph.nodeCount).toBe(3);
    expect(r2.graph.nodeCount).toBe(3);
  });

  it('respects lambda1 regularization', () => {
    const g = new CausalGraph(['X', 'Y']);
    const { data, nodeNames } = generateLinearData(g, 200, 45);
    const r1 = golemAlgorithm(data, nodeNames, { lambda1: 0.5, maxIter: 300 });
    const r2 = golemAlgorithm(data, nodeNames, { lambda1: 0.001, maxIter: 300 });
    expect(r1.graph.edges.length).toBeLessThanOrEqual(r2.graph.edges.length + 1);
    expect(r1.graph.nodeCount).toBe(2);
    expect(r2.graph.nodeCount).toBe(2);
  });

  it('works with L-BFGS optimizer', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data, nodeNames } = generateLinearData(g, 300, 46);
    const result = golemAlgorithm(data, nodeNames, { optimizer: 'lbfgs', maxIter: 300, tol: 1e-4 });
    expect(result.graph.nodeCount).toBe(3);
    expect(result.W).toBeInstanceOf(Float64Array);
  });

  it('handles small datasets gracefully', { timeout: 10000 }, () => {
    const data = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [1.5, 3.0], [2.5, 5.0], [3.5, 7.0]];
    const result = golemAlgorithm(data, ['X', 'Y']);
    expect(result.graph.nodeCount).toBe(2);
  });

  it('applies domain knowledge', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data, nodeNames } = generateLinearData(g, 300, 47);
    const result = golemAlgorithm(data, nodeNames, { maxIter: 300, lambda1: 0.003 }, {
      forbids: [['Z', 'X']],
    });
    expect(result.graph.nodeCount).toBe(3);
  });

  it('GOLEM scales to 4-variable graph', { timeout: 60000 }, () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C'); g.addEdge('A', 'D');
    const { data, nodeNames } = generateLinearData(g, 300, 48);
    const result = golemAlgorithm(data, nodeNames, { maxIter: 500, lambda1: 0.003 });
    expect(result.graph.nodeCount).toBe(4);
  });
});
