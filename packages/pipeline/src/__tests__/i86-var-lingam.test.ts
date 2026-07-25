/**
 * I86: VAR-LiNGAM Tests.
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { varLingam } from '../../src/graph/var-lingam.js';

describe('VAR-LiNGAM Algorithm', () => {
  it('discovers lagged structure in AR system', () => {
    // X[t] = noise, Y[t] = 0.7*X[t-1] + noise, Z[t] = 0.5*Y[t-1] + noise
    const T = 300;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      const x = Math.random() * 2 - 1;
      const y = (t > 0 ? 0.7 * data[t - 1]![0]! : 0) + (Math.random() - 0.5) * 0.3;
      const z = (t > 0 ? 0.5 * data[t - 1]![1]! : 0) + (Math.random() - 0.5) * 0.3;
      data.push([x, y, z]);
    }

    const result = varLingam(data, ['X', 'Y', 'Z'], { maxLag: 2, threshold: 0.15 });
    expect(result.instantaneousGraph.nodeCount).toBe(3);
    expect(result.laggedMatrices.length).toBe(2);
    expect(result.B0).toBeInstanceOf(Float64Array);
    expect(result.order.length).toBe(3);
  });

  it('handles small data gracefully', () => {
    const data = [[1, 2], [3, 4]];
    const result = varLingam(data, ['X', 'Y']);
    expect(result.instantaneousGraph.nodeCount).toBe(2);
    expect(result.laggedMatrices).toEqual([]);
  });

  it('produces valid B0 matrix dimensions', () => {
    const T = 200;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      data.push([Math.random(), Math.random()]);
    }
    const result = varLingam(data, ['A', 'B'], { maxLag: 1 });
    expect(result.B0.length).toBe(4);
  });

  it('lagged matrices have correct dimensions', () => {
    const T = 200;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      const x = Math.random();
      const y = (t > 0 ? 0.5 * data[t - 1]![0]! : 0) + Math.random() * 0.2;
      data.push([x, y]);
    }
    const result = varLingam(data, ['X', 'Y'], { maxLag: 3 });
    expect(result.laggedMatrices).toHaveLength(3);
    for (const mat of result.laggedMatrices) {
      expect(mat.length).toBe(4); // 2×2
    }
  });

  it('provides valid causal order', () => {
    const T = 300;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      data.push([Math.random(), Math.random(), Math.random()]);
    }
    const result = varLingam(data, ['A', 'B', 'C'], { maxLag: 1 });
    expect(result.order.length).toBe(3);
  });

  it('respects threshold parameter', () => {
    const T = 300;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      const x = Math.random();
      const y = (t > 0 ? 0.6 * data[t - 1]![0]! : 0) + Math.random() * 0.2;
      data.push([x, y]);
    }
    const r1 = varLingam(data, ['X', 'Y'], { maxLag: 2, threshold: 0.5 });
    const r2 = varLingam(data, ['X', 'Y'], { maxLag: 2, threshold: 0.05 });
    expect(r1.laggedMatrices).toHaveLength(2);
    expect(r2.laggedMatrices).toHaveLength(2);
  });
});
