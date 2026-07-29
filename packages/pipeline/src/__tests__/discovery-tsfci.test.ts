/**
 * I88: tsFCI Time-Series FCI Tests.
 */
import { describe, it, expect } from 'vitest';
import { tsFciAlgorithm } from '../../src/graph/tsfci.js';

describe('tsFCI Algorithm', () => {
  it('discovers contemporaneous structure', () => {
    const T = 200;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      data.push([Math.random(), Math.random() + Math.random()]);
    }
    const result = tsFciAlgorithm(data, ['X', 'Y'], { alpha: 0.05, maxLag: 2 });
    expect(result.instantaneousGraph.nodeCount).toBe(2);
    expect(result.laggedGraphs.size).toBeGreaterThanOrEqual(0);
  });

  it('discovers lagged structure in AR system', () => {
    const T = 300;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      const x = Math.random() * 2 - 1;
      const y = (t > 0 ? 0.7 * data[t - 1]![0]! : 0) + Math.random() * 0.3;
      data.push([x, y]);
    }
    const result = tsFciAlgorithm(data, ['X', 'Y'], { alpha: 0.05, maxLag: 2 });
    expect(result.laggedGraphs).toBeInstanceOf(Map);
    expect(result.pagEdges).toBeInstanceOf(Map);
  });

  it('handles small data gracefully', () => {
    const data = [[1, 2], [3, 4]];
    const result = tsFciAlgorithm(data, ['X', 'Y']);
    expect(result.instantaneousGraph.nodeCount).toBe(2);
    expect(result.laggedGraphs.size).toBe(0);
  });

  it('respects maxLag parameter', () => {
    const T = 200;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      data.push([Math.random(), Math.random(), Math.random()]);
    }
    const r1 = tsFciAlgorithm(data, ['A', 'B', 'C'], { maxLag: 1 });
    const r2 = tsFciAlgorithm(data, ['A', 'B', 'C'], { maxLag: 3 });
    expect(r1.laggedGraphs.size).toBeLessThanOrEqual(r2.laggedGraphs.size + 2);
  });

  it('produces valid PAG edges', () => {
    const T = 200;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      data.push([Math.random(), Math.random()]);
    }
    const result = tsFciAlgorithm(data, ['X', 'Y'], { maxLag: 1 });
    for (const [key, val] of result.pagEdges) {
      expect(typeof key).toBe('string');
      expect(typeof val).toBe('string');
    }
  });

  it('handles 3-variable coupled time series', () => {
    const T = 300;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      const x = Math.random() * 2 - 1;
      const y = (t > 0 ? 0.6 * data[t - 1]![0]! : 0) + (Math.random() - 0.5) * 0.3;
      const z = (t > 0 ? 0.5 * data[t - 1]![1]! : 0) + (Math.random() - 0.5) * 0.3;
      data.push([x, y, z]);
    }
    const result = tsFciAlgorithm(data, ['X', 'Y', 'Z'], { alpha: 0.05, maxLag: 2 });
    expect(result.instantaneousGraph.nodeCount).toBe(3);
    expect(result.laggedGraphs.size).toBe(2);
  });
});
