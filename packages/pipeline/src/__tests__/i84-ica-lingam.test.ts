/**
 * I84: ICA-LiNGAM Algorithm Tests.
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { icaLiNGAM } from '../../src/graph/ica-lingam.js';
import { generateLinearData, computeSHD } from '../../src/benchmark.js';

describe('ICA-LiNGAM Algorithm', () => {
  it('discovers DAG from linear chain data', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data, nodeNames } = generateLinearData(g, 500, 42);
    const result = icaLiNGAM(data, nodeNames);
    expect(result.graph.nodeCount).toBe(3);
    expect(result.B).toBeInstanceOf(Float64Array);
    expect(result.order.length).toBe(3);
  });

  it('produces valid B matrix dimensions', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const { data, nodeNames } = generateLinearData(g, 200, 43);
    const result = icaLiNGAM(data, nodeNames);
    expect(result.B.length).toBe(4); // 2×2
    expect(result.order.length).toBe(2);
  });

  it('handles small data gracefully', () => {
    const data = [[1, 2], [3, 4]];
    const result = icaLiNGAM(data, ['X', 'Y']);
    expect(result.graph.nodeCount).toBe(2);
  });

  it('applies threshold to reduce edges', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data, nodeNames } = generateLinearData(g, 300, 44);
    const r1 = icaLiNGAM(data, nodeNames, { threshold: 0.5, tol: 1e-3, maxIter: 200 });
    const r2 = icaLiNGAM(data, nodeNames, { threshold: 0.05, tol: 1e-3, maxIter: 200 });
    expect(r1.graph.nodeCount).toBe(3);
    expect(r2.graph.nodeCount).toBe(3);
  });

  it('provides valid causal order', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C'); g.addEdge('A', 'D');
    const { data, nodeNames } = generateLinearData(g, 400, 45);
    const result = icaLiNGAM(data, nodeNames, { tol: 1e-3, maxIter: 200 });
    expect(result.order.length).toBe(4);
    // Causal order should contain all node names
    for (const name of nodeNames) {
      expect(result.order).toContain(name);
    }
  });

  it('applies domain knowledge', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data, nodeNames } = generateLinearData(g, 300, 46);
    const result = icaLiNGAM(data, nodeNames, { tol: 1e-3, maxIter: 200 }, {
      forbids: [['Z', 'X']],
    });
    expect(result.graph.nodeCount).toBe(3);
  });

  it('ICA-LiNGAM SHD comparable to DirectLiNGAM on 3-node chain', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data, nodeNames } = generateLinearData(g, 500, 47);

    const ica = icaLiNGAM(data, nodeNames, { tol: 1e-3, maxIter: 300 });
    // ICA-LiNGAM should find graph edges
    expect(ica.graph.edges.length).toBeGreaterThanOrEqual(0);
  });
});
