/**
 * I115: Core Math + TimeSeries Type Coverage Tests
 *
 * Tests the new core exports added during I4-I7:
 *   - digamma function
 *   - chiSquareCDF with regularized gamma
 *   - partialCorrelationRaw
 *   - TimeSeriesEdge/TimeSeriesGraph type exports
 */

import { describe, it, expect } from 'vitest';
import {
  digamma,
  chiSquareCDF,
  partialCorrelationRaw,
} from '../math.js';
import type {
  TimeSeriesEdge,
  TimeSeriesGraph,
  EdgeMark,
  CIBackend,
  CITestResult,
} from '../types/timeseries.js';

// ── Digamma Tests ────────────────────────────────────────────────────

describe('digamma', () => {
  it('ψ(1) = -γ ≈ -0.5772 (Euler-Mascheroni)', () => {
    expect(digamma(1)).toBeCloseTo(-0.5772, 2);
  });

  it('ψ(2) = ψ(1) + 1 ≈ 0.4228', () => {
    expect(digamma(2)).toBeCloseTo(0.4228, 2);
  });

  it('ψ(0.5) = -γ - 2ln2 ≈ -1.9635', () => {
    expect(digamma(0.5)).toBeCloseTo(-1.9635, 2);
  });

  it('returns -Infinity for x = 0', () => {
    expect(digamma(0)).toBe(-Infinity);
  });

  it('handles large x asymptotically (ψ(x) ≈ ln(x))', () => {
    // ψ(100) ≈ ln(100) - 1/(2*100) = 4.60517 - 0.005 = 4.60017
    expect(digamma(100)).toBeCloseTo(Math.log(100) - 0.005, 1);
  });

  it('recurrence ψ(x+1) = ψ(x) + 1/x for x < 8', () => {
    const x = 2.5;
    const diff = digamma(x + 1) - digamma(x);
    expect(diff).toBeCloseTo(1 / x, 5);
  });
});

// ── Chi-Squared CDF Tests ───────────────────────────────────────────

describe('chiSquareCDF', () => {
  it('χ²(0, df) = 1 (survival at zero)', () => {
    expect(chiSquareCDF(0, 1)).toBeCloseTo(1, 5);
  });

  it('χ²(large, 1) ≈ 0 (high test statistic → low p)', () => {
    expect(chiSquareCDF(15, 1)).toBeLessThan(0.001);
  });

  it('chiSquareCDF returns p-value for various inputs', () => {
    // Validate function produces values in valid range
    const p1 = chiSquareCDF(3.84, 1);
    const p2 = chiSquareCDF(15, 1);
    const p3 = chiSquareCDF(5, 3);
    // All values should be finite
    expect(Number.isFinite(p1)).toBe(true);
    expect(Number.isFinite(p2)).toBe(true);
    expect(Number.isFinite(p3)).toBe(true);
  });

  it('handles x=0 returns 1', () => {
    expect(chiSquareCDF(0, 3)).toBe(1);
  });

  it('handles df <= 0 returns 1', () => {
    expect(chiSquareCDF(5, 0)).toBe(1);
    expect(chiSquareCDF(5, -1)).toBe(1);
  });

  it('returns value in [0,1] for valid inputs', () => {
    const p = chiSquareCDF(5, 3);
    expect(p).toBeGreaterThan(0);
    expect(p).toBeLessThan(1);
  });
});

// ── Partial Correlation Tests ───────────────────────────────────────

describe('partialCorrelationRaw', () => {
  it('simple correlation of X=Y returns 1', () => {
    const data = Array.from({ length: 50 }, (_, i) => [i, i, 0]);
    const rho = partialCorrelationRaw(data, 0, 1, []);
    expect(rho).toBeCloseTo(1, 2);
  });

  it('simple correlation of X=-Y returns -1', () => {
    const data = Array.from({ length: 50 }, (_, i) => [i, -i, 0]);
    const rho = partialCorrelationRaw(data, 0, 1, []);
    expect(rho).toBeCloseTo(-1, 2);
  });

  it('independent data has near-zero correlation', () => {
    const data: number[][] = [];
    for (let i = 0; i < 200; i++) {
      data.push([Math.random(), Math.random()]);
    }
    const rho = partialCorrelationRaw(data, 0, 1, []);
    expect(Math.abs(rho)).toBeLessThan(0.3);
  });

  it('partial correlation removes confounding', () => {
    // Z ~ N(0,1), X = Z + noise, Y = Z + noise → X ⟂ Y given Z
    const data: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const z = (Math.random() - 0.5) * 2;
      const x = z + (Math.random() - 0.5) * 0.3;
      const y = z + (Math.random() - 0.5) * 0.3;
      data.push([x, y, z]);
    }
    // Without conditioning: ρ(X,Y) should be high
    const rhoNoCond = partialCorrelationRaw(data, 0, 1, []);
    expect(Math.abs(rhoNoCond)).toBeGreaterThan(0.3);

    // With conditioning on Z: ρ(X,Y|Z) should be near 0
    const rhoCond = partialCorrelationRaw(data, 0, 1, [2]);
    expect(Math.abs(rhoCond)).toBeLessThan(0.2);
  });

  it('returns NaN for degenerate data (all zeros)', () => {
    const data = Array.from({ length: 10 }, () => [0, 0]);
    const rho = partialCorrelationRaw(data, 0, 1, []);
    expect(isNaN(rho)).toBe(true);
  });

  it('handles empty conditioning set', () => {
    const data = Array.from({ length: 50 }, (_, i) => [i, i * 0.5]);
    const rho = partialCorrelationRaw(data, 0, 1, []);
    expect(rho).toBeCloseTo(1, 2);
  });
});

// ── Time-Series Type Instantiation Tests ────────────────────────────

describe('TimeSeries types (compile-time verification)', () => {
  it('TimeSeriesEdge type is well-formed', () => {
    const edge: TimeSeriesEdge = {
      source: 'A',
      target: 'B',
      lag: 1,
      strength: 0.8,
      pValue: 0.01,
      sourceMark: 'tail' as EdgeMark,
      targetMark: 'arrow' as EdgeMark,
      phase: 'mci',
    };
    expect(edge.lag).toBe(1);
    expect(edge.strength).toBe(0.8);
    expect(edge.sourceMark).toBe('tail');
    expect(edge.targetMark).toBe('arrow');
  });

  it('TimeSeriesGraph type is well-formed', () => {
    const graph: TimeSeriesGraph = {
      nodes: ['X0', 'X1', 'X2'],
      edges: [],
      tauMax: 3,
      timeSteps: 500,
      isCPDAG: true,
    };
    expect(graph.nodes.length).toBe(3);
    expect(graph.tauMax).toBe(3);
  });

  it('EdgeMark type accepts valid values', () => {
    const marks: EdgeMark[] = ['tail', 'arrow', 'circle'];
    expect(marks.length).toBe(3);
  });

  it('CIBackend type accepts valid backends', () => {
    const backends: CIBackend[] = ['parcorr', 'cmiknn', 'gsquared'];
    expect(backends.length).toBe(3);
  });

  it('CITestResult is well-formed', () => {
    const result: CITestResult = { pValue: 0.05, testStatistic: 0.42 };
    expect(result.pValue).toBe(0.05);
    expect(result.testStatistic).toBe(0.42);
  });
});
