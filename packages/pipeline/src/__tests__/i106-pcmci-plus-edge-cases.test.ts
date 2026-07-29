/**
 * I106: PCMCI+ Edge Case and Regression Tests
 *
 * Tests extreme boundary conditions:
 *   - All-zero data, NaN, Infinity values
 *   - Minimum/maximum parameter values
 *   - Dimension extremes (very small d, very large T)
 *   - Unusual graph topologies
 */

import { describe, it, expect } from 'vitest';
import { pcmciPlusAlgorithm } from '../graph/pcmci-plus.js';
import {
  generateVARTimeSeries,
  simpleTestTimeSeries,
} from '../graph/ts-data-generators.js';

/** Generate all-zero data */
function allZeroData(T: number, d: number): number[][] {
  return Array.from({ length: T }, () => new Array(d).fill(0));
}

/** Generate constant (non-zero) data */
function constantData(T: number, d: number): number[][] {
  return Array.from({ length: T }, () => new Array(d).fill(3.14));
}

// ── Degenerate Data ────────────────────────────────────────────────────

describe('PCMCI+ edge cases — degenerate data', () => {
  it('handles all-zero data gracefully', () => {
    const data = allZeroData(100, 3);
    const result = pcmciPlusAlgorithm(data, ['A', 'B', 'C']);
    expect(result.graph).toBeDefined();
    expect(result.graph.edges.length).toBeGreaterThanOrEqual(0);
    // Zero data should not crash; may produce empty graph
  });

  it('handles constant (non-zero) data gracefully', () => {
    const data = constantData(100, 3);
    const result = pcmciPlusAlgorithm(data, ['A', 'B', 'C']);
    expect(result.graph).toBeDefined();
  });

  it('handles NaN values without throwing', () => {
    const data: number[][] = Array.from({ length: 100 }, (_, i) => [i * 0.1, i * 0.2]);
    // Inject NaN
    data[50] = [NaN, 0.5];
    expect(() => pcmciPlusAlgorithm(data, ['A', 'B'], { tauMax: 1 })).not.toThrow();
  });

  it('handles Infinite values without throwing', () => {
    const data: number[][] = Array.from({ length: 100 }, (_, i) => [i * 0.1, i * 0.2]);
    data[50] = [Infinity, 0.5];
    expect(() => pcmciPlusAlgorithm(data, ['A', 'B'], { tauMax: 1 })).not.toThrow();
  });

  it('handles negative values normally', () => {
    const data: number[][] = Array.from({ length: 100 }, () => [
      Math.random() - 0.5,
      Math.random() - 0.5,
    ]);
    const result = pcmciPlusAlgorithm(data, ['A', 'B'], { tauMax: 1 });
    expect(result.graph).toBeDefined();
  });

  it('handles very small data (T=10)', () => {
    const data = Array.from({ length: 10 }, () => [
      Math.random(), Math.random(),
    ]);
    const result = pcmciPlusAlgorithm(data, ['A', 'B'], { tauMax: 2 });
    expect(result.graph).toBeDefined();
  });

  it('handles empty data array', () => {
    const result = pcmciPlusAlgorithm([], ['A', 'B']);
    expect(result.graph.edges).toHaveLength(0);
  });
});

// ── Parameter Extremes ─────────────────────────────────────────────────

describe('PCMCI+ edge cases — parameter extremes', () => {
  it('handles alpha = 0.0 (never reject)', () => {
    const { data } = generateVARTimeSeries(['A', 'B'], {
      T: 200, d: 2, maxLag: 1,
      coeffMatrices: [[[0, 0], [0.5, 0]]],
      noiseStd: 0.3, seed: 42,
    });
    const result = pcmciPlusAlgorithm(data, ['A', 'B'], { tauMax: 2, alpha: 0.0 });
    // Alpha=0 means we never reject H0 → zero edges
    expect(result.graph.edges.length).toBe(0);
  });

  it('handles alpha = 0.5 result shape validation', () => {
    const ts = simpleTestTimeSeries(200);
    const result = pcmciPlusAlgorithm(ts.data, ts.nodeNames, { tauMax: 2, alpha: 0.5 });
    // Any result is valid — edges may or may not be found
    expect(result.graph).toBeDefined();
    expect(result.summary).toBeDefined();
  });

  it('handles maxCondVars = 10 (large conditioning)', () => {
    const { data } = generateVARTimeSeries(['A', 'B', 'C', 'D', 'E', 'F'], {
      T: 300, d: 6, maxLag: 1,
      coeffMatrices: [[
        [0.6, 0, 0, 0, 0, 0],
        [0.5, 0, 0, 0, 0, 0],
        [0, 0.5, 0, 0, 0, 0],
        [0, 0, 0.5, 0, 0, 0],
        [0, 0, 0, 0.5, 0, 0],
        [0, 0, 0, 0, 0.5, 0],
      ]],
      noiseStd: 0.3, seed: 42,
    });
    const result = pcmciPlusAlgorithm(data, ['A', 'B', 'C', 'D', 'E', 'F'], {
      tauMax: 2, maxCondVars: 10, alpha: 0.05,
    });
    expect(result.graph).toBeDefined();
  });

  it('handles tauMax = 0 (no lagged edges at all)', () => {
    const ts = simpleTestTimeSeries(200);
    const result = pcmciPlusAlgorithm(ts.data, ['X0', 'X1', 'X2'], { tauMax: 0 });
    // All edges must be contemporaneous (lag=0)
    for (const edge of result.graph.edges) {
      expect(edge.lag).toBe(0);
    }
  });
});

// ── Dimension Extremes ─────────────────────────────────────────────────

describe('PCMCI+ edge cases — dimension extremes', () => {
  it('handles d > T scenario (more variables than time steps)', () => {
    // PCMCI+ gracefully handles this — returns empty result
    const data = Array.from({ length: 50 }, () =>
      Array.from({ length: 5 }, () => Math.random())
    );
    const result = pcmciPlusAlgorithm(data,
      ['A', 'B', 'C', 'D', 'E'],
      { tauMax: 2 }
    );
    expect(result.graph).toBeDefined();
  });

  it('handles very large T (T=2000) without error', () => {
    const { data } = generateVARTimeSeries(['A', 'B'], {
      T: 2000, d: 2, maxLag: 1,
      coeffMatrices: [[[0, 0], [0.5, 0]]],
      noiseStd: 0.3, seed: 42,
    });
    const result = pcmciPlusAlgorithm(data, ['A', 'B'], { tauMax: 2, alpha: 0.10 });
    // Large T with strong signal: graph should be well-formed
    expect(result.graph).toBeDefined();
    expect(result.graph.nodes).toEqual(['A', 'B']);
    expect(result.runtimeMs).toBeGreaterThanOrEqual(0);
  });
});

// ── Array Descriptor Validity ──────────────────────────────────────────

describe('PCMCI+ edge cases — output validity', () => {
  it('all p-values in result edges are in [0, 1]', () => {
    const ts = simpleTestTimeSeries(200);
    const result = pcmciPlusAlgorithm(ts.data, ts.nodeNames, { tauMax: 2, alpha: 0.05 });
    for (const edge of result.graph.edges) {
      expect(edge.pValue).toBeGreaterThanOrEqual(0);
      expect(edge.pValue).toBeLessThanOrEqual(1);
    }
  });

  it('all strength values in [-1, 1]', () => {
    const ts = simpleTestTimeSeries(200);
    const result = pcmciPlusAlgorithm(ts.data, ts.nodeNames, { tauMax: 2, alpha: 0.05 });
    for (const edge of result.graph.edges) {
      expect(edge.strength).toBeGreaterThanOrEqual(-1);
      expect(edge.strength).toBeLessThanOrEqual(1);
    }
  });

  it('edge phase is always pc1 or mci', () => {
    const ts = simpleTestTimeSeries(200);
    const result = pcmciPlusAlgorithm(ts.data, ['X0', 'X1', 'X2'], { tauMax: 2, alpha: 0.05 });
    for (const edge of result.graph.edges) {
      expect(['pc1', 'mci']).toContain(edge.phase);
    }
  });

  it('summary counts are self-consistent', () => {
    const ts = simpleTestTimeSeries(200);
    const result = pcmciPlusAlgorithm(ts.data, ['X0', 'X1', 'X2'], { tauMax: 2, alpha: 0.05 });
    const s = result.summary;
    expect(s.totalEdges).toBe(result.graph.edges.length);
    expect(s.laggedEdges + s.contemporaneousEdges).toBe(s.totalEdges);
    expect(s.directedEdges + s.partiallyDirectedEdges).toBe(s.contemporaneousEdges);
  });
});

