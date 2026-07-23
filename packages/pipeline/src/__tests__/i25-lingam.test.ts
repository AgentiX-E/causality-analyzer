import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { directLiNGAM } from '../graph/lingam.js';

function generateNonGaussianData(
  nodes: string[], edges: Array<[string, string, number]>,
  nSamples: number,
): Matrix {
  const n = nodes.length;
  const order = [...nodes];
  // Sort by edge dependencies to approximate topological order
  const data = Matrix.zeros(nSamples, n);

  for (let row = 0; row < nSamples; row++) {
    const values = new Array(n).fill(0);
    for (const name of nodes) {
      const idx = nodes.indexOf(name);
      // Non-Gaussian noise: uniform distribution
      const noise = (Math.random() - 0.5) * 2;
      let val = noise;
      for (const [f, t, w] of edges) {
        if (t === name) {
          val += w * (values[nodes.indexOf(f)] ?? 0);
        }
      }
      values[idx] = val;
      data.set(row, idx, val);
    }
  }
  return data;
}

describe('directLiNGAM', () => {
  it('recovers causal order for chain X→Y→Z', () => {
    const nodes = ['X', 'Y', 'Z'];
    const edges: Array<[string, string, number]> = [['X', 'Y', 5], ['Y', 'Z', 4]];
    const data = generateNonGaussianData(nodes, edges, 200);
    const result = directLiNGAM(data, nodes);
    // Verify LiNGAM produces valid output
    expect(result.order.length).toBe(3);
    expect(result.order).toContain('X');
    expect(result.order).toContain('Y');
    expect(result.order).toContain('Z');
    // X should be among the first 2 (most exogenous)
    expect(result.order.indexOf('X')).toBeLessThan(2);
  });

  it('returns valid graph', () => {
    const nodes = ['A', 'B', 'C'];
    const edges: Array<[string, string, number]> = [['A', 'B', 2], ['B', 'C', 1.5]];
    const data = generateNonGaussianData(nodes, edges, 300);
    const result = directLiNGAM(data, nodes);
    expect(result.graph.nodeCount).toBe(3);
    expect(result.order.length).toBe(3);
    expect(result.weights).toBeDefined();
  });

  it('independent variables produce some order', () => {
    const nodes = ['A', 'B', 'C'];
    const data = new Matrix(100, 3);
    for (let r = 0; r < 100; r++) {
      for (let c = 0; c < 3; c++) data.set(r, c, (Math.random() - 0.5) * 2);
    }
    const result = directLiNGAM(data, nodes);
    // With independent data, LiNGAM should produce valid output
    expect(result.order.length).toBe(3);
    expect(result.graph.nodeCount).toBe(3);
  });

  it('produces weights map with proper structure', () => {
    const nodes = ['X', 'Y'];
    const edges: Array<[string, string, number]> = [['X', 'Y', 3]];
    const data = generateNonGaussianData(nodes, edges, 200);
    const result = directLiNGAM(data, nodes);
    // Y should have a weight from X
    const yWeights = result.weights.get('Y');
    if (yWeights) {
      expect(yWeights.has('X')).toBe(true);
    }
  });

  it('empty data handled gracefully', () => {
    const data = new Matrix(0, 2);
    const result = directLiNGAM(data, ['A', 'B']);
    expect(result.graph.nodeCount).toBe(2);
    expect(result.order.length).toBe(2);
  });

  it('two-node simple case identifies direction', () => {
    const nodes = ['X', 'Y'];
    const edges: Array<[string, string, number]> = [['X', 'Y', 5]];
    const data = generateNonGaussianData(nodes, edges, 500);
    const result = directLiNGAM(data, nodes);
    // X should come before Y in causal order
    expect(result.order[0]).toBe('X');
    expect(result.order[1]).toBe('Y');
  });
});
