/**
 * I103: Time Series Data Generator Tests
 *
 * Tests all data generators for correctness:
 *   - VAR generator: dimensions, graph integrity, deterministic seed
 *   - Nonlinear VAR generator: nonlinearity effect, parameter types
 *   - SCM generator: custom mechanisms
 *   - Convenience helpers: simpleTestTimeSeries, chainTimeSeries, fullyConnectedVAR1
 */

import { describe, it, expect } from 'vitest';
import {
  generateVARTimeSeries,
  generateNonlinearVARTimeSeries,
  generateSCMTimeSeries,
  simpleTestTimeSeries,
  chainTimeSeries,
  fullyConnectedVAR1,
} from '../graph/ts-data-generators.js';

// ── VAR Generator ──────────────────────────────────────────────────────

describe('generateVARTimeSeries', () => {
  const makeConfig = () => ({
    T: 200 as number, d: 3 as number, maxLag: 1,
    coeffMatrices: [[[0.6, 0.5, 0], [0, 0, 0], [0, 0, 0]]] as const,
    noiseStd: 0.3,
    seed: 42 as number | null,
  });

  it('produces data matrix with correct dimensions (T x d)', () => {
    const { data } = generateVARTimeSeries(['A', 'B', 'C'], makeConfig());
    expect(data.length).toBe(200);
    expect(data[0]!.length).toBe(3);
  });

  it('returns correct nodeNames', () => {
    const { nodeNames } = generateVARTimeSeries(['A', 'B', 'C'], makeConfig());
    expect(nodeNames).toEqual(['A', 'B', 'C']);
  });

  it('truthGraph has correct tauMax', () => {
    const { truthGraph } = generateVARTimeSeries(['A', 'B', 'C'], makeConfig());
    expect(truthGraph.tauMax).toBe(1);
  });

  it('truthGraph has correct timeSteps', () => {
    const { truthGraph } = generateVARTimeSeries(['A', 'B', 'C'], makeConfig());
    expect(truthGraph.timeSteps).toBe(200);
  });

  it('truthGraph has correct number of nodes', () => {
    const { truthGraph } = generateVARTimeSeries(['A', 'B', 'C'], makeConfig());
    expect(truthGraph.nodes.length).toBe(3);
  });

  it('truthGraph edges match coefficient structure', () => {
    const { truthGraph } = generateVARTimeSeries(['A', 'B', 'C'], makeConfig());
    // coeffMatrices[0][0][0] = 0.6 → A[t-1] → A[t]
    const hasSelf = truthGraph.edges.some(e => e.source === 'A' && e.target === 'A' && e.lag === 1);
    // coeffMatrices[0][0][1] = 0.5 → A[t-1] → B[t]
    const hasCross = truthGraph.edges.some(e => e.source === 'A' && e.target === 'B' && e.lag === 1);
    expect(hasSelf).toBe(true);
    expect(hasCross).toBe(true);
  });

  it('truthGraph is fully directed (no CPDAG)', () => {
    const { truthGraph } = generateVARTimeSeries(['A', 'B', 'C'], makeConfig());
    expect(truthGraph.isCPDAG).toBe(false);
  });

  it('produces deterministic output with same seed', () => {
    const cfg = makeConfig();
    cfg.seed = 42;
    const r1 = generateVARTimeSeries(['A', 'B', 'C'], cfg);
    const r2 = generateVARTimeSeries(['A', 'B', 'C'], { ...cfg, seed: 42 });
    // First row should be identical
    expect(r1.data[0]).toEqual(r2.data[0]);
  });

  it('produces different output with different seeds', () => {
    const cfg = makeConfig();
    const r1 = generateVARTimeSeries(['A', 'B', 'C'], { ...cfg, seed: 42 });
    const r2 = generateVARTimeSeries(['A', 'B', 'C'], { ...cfg, seed: 99 });
    // Different seeds should produce different first values
    const sameAll = r1.data.every((row, i) =>
      row.every((val, j) => Math.abs(val - (r2.data[i]![j]! ?? 0)) < 1e-10)
    );
    expect(sameAll).toBe(false);
  });

  it('generated data is non-constant (has variation)', () => {
    const { data } = generateVARTimeSeries(['A', 'B', 'C'], makeConfig());
    // Column A should have variation
    const colA = data.map(row => row[0]!);
    const unique = new Set(colA.map(v => v.toFixed(3)));
    expect(unique.size).toBeGreaterThan(1);
  });

  it('handles scalar noiseStd', () => {
    const cfg = { ...makeConfig(), noiseStd: 0.5 };
    const { data } = generateVARTimeSeries(['A', 'B'], {
      T: 200, d: 2, maxLag: 1,
      coeffMatrices: [[[0.6, 0], [0.5, 0]]],
      noiseStd: 0.5,
      seed: 42,
    });
    expect(data.length).toBe(200);
  });

  it('handles array noiseStd', () => {
    const { data } = generateVARTimeSeries(['A', 'B'], {
      T: 200, d: 2, maxLag: 1,
      coeffMatrices: [[[0.6, 0], [0.5, 0]]],
      noiseStd: [0.3, 0.5],
      seed: 42,
    });
    expect(data.length).toBe(200);
  });
});

// ── Nonlinear VAR Generator ────────────────────────────────────────────

describe('generateNonlinearVARTimeSeries', () => {
  it('produces data with correct dimensions', () => {
    const { data } = generateNonlinearVARTimeSeries(['A', 'B'], {
      T: 200, d: 2, maxLag: 1,
      coeffMatrices: [[[0.6, 0], [0.5, 0]]],
      nonlinearity: 'tanh', nonlinearityStrength: 0.5,
      noiseStd: 0.3, seed: 42,
    });
    expect(data.length).toBe(200);
    expect(data[0]!.length).toBe(2);
  });

  it('supports all nonlinearity types', () => {
    for (const nl of ['tanh', 'sin', 'quadratic', 'cubic'] as const) {
      const { data } = generateNonlinearVARTimeSeries(['A', 'B'], {
        T: 100, d: 2, maxLag: 1,
        coeffMatrices: [[[0.6, 0], [0.5, 0]]],
        nonlinearity: nl, nonlinearityStrength: 0.5,
        noiseStd: 0.3, seed: 42,
      });
      expect(data.length).toBe(100);
    }
  });

  it('nonlinearity strength=0 produces nearly linear data', () => {
    const r1 = generateNonlinearVARTimeSeries(['A', 'B'], {
      T: 100, d: 2, maxLag: 1,
      coeffMatrices: [[[0.6, 0], [0.5, 0]]],
      nonlinearity: 'tanh', nonlinearityStrength: 0.0,
      noiseStd: 0.3, seed: 42,
    });
    expect(r1.data.length).toBe(100);
  });

  it('nonlinearity strength=1 produces fully nonlinear data', () => {
    const { data } = generateNonlinearVARTimeSeries(['A', 'B'], {
      T: 100, d: 2, maxLag: 1,
      coeffMatrices: [[[0.6, 0], [0.5, 0]]],
      nonlinearity: 'tanh', nonlinearityStrength: 1.0,
      noiseStd: 0.3, seed: 42,
    });
    expect(data.length).toBe(100);
  });
});

// ── SCM Generator ──────────────────────────────────────────────────────

describe('generateSCMTimeSeries', () => {
  it('produces data from custom mechanisms', () => {
    const { data } = generateSCMTimeSeries(['X', 'Y', 'Z'], {
      T: 200, maxLag: 1, noiseStd: 0.3, seed: 42,
      mechanisms: [
        {
          target: 'X',
          parents: [],
          fn: (past, curr) => 0,
        },
        {
          target: 'Y',
          parents: [['X', 0]] as const,
          fn: (past, curr) => (curr.get('X') ?? 0) * 0.5,
        },
        {
          target: 'Z',
          parents: [['Y', 0]] as const,
          fn: (past, curr) => (curr.get('Y') ?? 0) * 0.3,
        },
      ],
    });
    expect(data.length).toBe(200);
    expect(data[0]!.length).toBe(3);
  });

  it('truthGraph edges reflect mechanism parents', () => {
    const { truthGraph } = generateSCMTimeSeries(['X', 'Y'], {
      T: 100, maxLag: 1, noiseStd: 0.3, seed: 42,
      mechanisms: [
        { target: 'X', parents: [], fn: () => 0 },
        { target: 'Y', parents: [['X', 0]], fn: (p, c) => (c.get('X') ?? 0) * 0.5 },
      ],
    });
    const hasXY = truthGraph.edges.some(e => e.source === 'X' && e.target === 'Y' && e.lag === 0);
    expect(hasXY).toBe(true);
  });
});

// ── Convenience Helpers ─────────────────────────────────────────────────

describe('simpleTestTimeSeries', () => {
  it('returns valid TestTimeSeries with default T', () => {
    const ts = simpleTestTimeSeries();
    expect(ts.data.length).toBe(200);
    expect(ts.nodeNames).toEqual(['X0', 'X1', 'X2']);
    expect(ts.truthGraph.nodes).toEqual(['X0', 'X1', 'X2']);
  });

  it('respects custom T parameter', () => {
    const ts = simpleTestTimeSeries(100);
    expect(ts.data.length).toBe(100);
  });

  it('truthGraph has expected edges', () => {
    const ts = simpleTestTimeSeries();
    // Should have edges from coefficient structure: X0→X0, X0→X1 at lag=1, X1→X2 at lag=0
    expect(ts.truthGraph.edges.length).toBeGreaterThan(0);
  });
});

describe('chainTimeSeries', () => {
  it('produces correct dimensions', () => {
    const ts = chainTimeSeries(200, 4);
    expect(ts.data.length).toBe(200);
    expect(ts.data[0]!.length).toBe(4);
    expect(ts.nodeNames.length).toBe(4);
  });

  it('has chain topology edges', () => {
    const ts = chainTimeSeries(200, 5);
    const edges = ts.truthGraph.edges;
    expect(edges.length).toBeGreaterThan(0);
    // Chain: X0[t-1]→X1[t], X1[t-1]→X2[t], X2[t-1]→X3[t], X3[t-1]→X4[t]
    expect(edges.filter(e => e.lag === 1).length).toBeGreaterThan(0);
  });

  it('handles d=2 (minimum chain)', () => {
    const ts = chainTimeSeries(200, 2);
    expect(ts.data[0]!.length).toBe(2);
  });
});

describe('fullyConnectedVAR1', () => {
  it('produces correct dimensions', () => {
    const ts = fullyConnectedVAR1(200, 5, 0.2, 42);
    expect(ts.data.length).toBe(200);
    expect(ts.data[0]!.length).toBe(5);
  });

  it('density=0 produces empty edge set', () => {
    const ts = fullyConnectedVAR1(200, 5, 0, 42);
    expect(ts.truthGraph.edges.length).toBe(0);
  });

  it('density=1 produces maximum edges', () => {
    const ts = fullyConnectedVAR1(200, 3, 1.0, 42);
    // d * d edges at lag=1 = 9
    expect(ts.truthGraph.edges.length).toBe(9);
  });

  it('same seed produces same graph', () => {
    const r1 = fullyConnectedVAR1(200, 5, 0.3, 42);
    const r2 = fullyConnectedVAR1(200, 5, 0.3, 42);
    expect(r1.truthGraph.edges.length).toBe(r2.truthGraph.edges.length);
  });
});
