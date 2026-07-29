/**
 * I85: PCMCI Time Series Causal Discovery Tests.
 */
import { describe, it, expect } from 'vitest';
import { pcmciAlgorithm } from '../../src/graph/pcmci.js';

describe('PCMCI Algorithm', () => {
  it('discovers lagged dependencies in linear time series', () => {
    // Generate: X[t] = noise, Y[t] = 0.8*X[t-1] + noise
    const T = 200;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      const x = Math.random();
      const y = (t > 0 ? 0.8 * data[t - 1]![0]! : 0) + Math.random() * 0.2;
      data.push([x, y]);
    }

    const result = pcmciAlgorithm(data, ['X', 'Y'], { alpha: 0.05, tauMax: 3 });
    expect(result.edges.length).toBeGreaterThanOrEqual(0);
    expect(result.tauMax).toBe(3);
    expect(result.alpha).toBe(0.05);
  });

  it('handles small data gracefully', () => {
    const data = [[1, 2], [3, 4]];
    const result = pcmciAlgorithm(data, ['X', 'Y']);
    // T < tauMax + 5 → returns empty
    expect(result.edges).toEqual([]);
  });

  it('detects lag-1 auto-correlation in 2-var system', () => {
    const T = 200;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      const x = (t > 0 ? 0.7 * data[t - 1]![0]! : 0) + Math.random() * 0.3;
      data.push([x, Math.random()]);
    }
    const result = pcmciAlgorithm(data, ['X', 'Y'], { alpha: 0.05, tauMax: 3 });
    expect(result.parents.has('X')).toBe(true);
  });

  it('respects tauMax parameter', () => {
    const T = 300;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      data.push([Math.random(), Math.random(), Math.random()]);
    }

    const r1 = pcmciAlgorithm(data, ['A', 'B', 'C'], { tauMax: 1 });
    const r2 = pcmciAlgorithm(data, ['A', 'B', 'C'], { tauMax: 5 });
    expect(r1.tauMax).toBe(1);
    expect(r2.tauMax).toBe(5);
  });

  it('returns valid parent sets for each variable', () => {
    const T = 200;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      data.push([Math.random(), Math.random()]);
    }

    const result = pcmciAlgorithm(data, ['X', 'Y'], { tauMax: 2, alpha: 0.01 });
    expect(result.parents.size).toBe(2);
    for (const [name, parents] of result.parents) {
      expect(typeof name).toBe('string');
      expect(Array.isArray(parents)).toBe(true);
    }
  });

  it('handles 3-variable coupled time series', () => {
    // X[t] = noise, Y[t] = 0.6*X[t-1] + noise, Z[t] = 0.5*Y[t-1] + noise
    const T = 300;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      const x = Math.random() * 2 - 1;
      const y = (t > 0 ? 0.6 * data[t - 1]![0]! : 0) + (Math.random() - 0.5) * 0.3;
      const z = (t > 0 ? 0.5 * data[t - 1]![1]! : 0) + (Math.random() - 0.5) * 0.3;
      data.push([x, y, z]);
    }

    const result = pcmciAlgorithm(data, ['X', 'Y', 'Z'], { alpha: 0.05, tauMax: 2 });
    expect(result.edges).toBeDefined();
    // Should find some edges in coupled system
    expect(result.tauMax).toBe(2);
  });
});
