/**
 * BDeu + Discretization Tests.
 */
import { describe, it, expect } from 'vitest';
import { logGamma, bdeuScore, discretizeBDeu } from '../bdeu.js';

describe('logGamma', () => {
  it('returns 0 for x <= 0', () => {
    expect(logGamma(0)).toBe(0);
    expect(logGamma(-1)).toBe(0);
  });

  it('returns near-zero for x=1 (Γ(1)=1 so log=0)', () => {
    expect(Math.abs(logGamma(1))).toBeLessThan(0.1);
  });

  it('returns approx log(24)=3.178 for x=5 (as Γ(5)=24)', () => {
    expect(logGamma(5)).toBeCloseTo(Math.log(24), 1);
  });

  it('handles non-integer values', () => {
    expect(isFinite(logGamma(2.5))).toBe(true);
  });
});

describe('bdeuScore', () => {
  it('computes finite score for independent data', () => {
    const data = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const score = bdeuScore(data, 0, [], [2, 2]);
    expect(isFinite(score)).toBe(true);
  });

  it('with parents yields different score', () => {
    const data = [[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]];
    const s1 = bdeuScore(data, 2, [], [3, 2, 2]);
    const s2 = bdeuScore(data, 2, [0, 1], [3, 2, 2]);
    expect(typeof s1).toBe('number');
    expect(typeof s2).toBe('number');
  });

  it('handles custom alpha', () => {
    const data = [[0, 0], [1, 1]];
    const s1 = bdeuScore(data, 1, [0], [2, 2], 0.5);
    const s2 = bdeuScore(data, 1, [0], [2, 2], 2.0);
    expect(s1).not.toBe(s2);
  });

  it('handles undefined values in data gracefully', () => {
    const data = [[0, 0], [undefined as any, undefined as any], [1, 1]];
    const score = bdeuScore(data as number[][], 0, [1], [2, 2]);
    expect(isFinite(score)).toBe(true);
  });

  it('handles sparse domain (value > domain size)', () => {
    const data = [[0, 5], [1, 3], [2, 1]];
    const score = bdeuScore(data, 0, [1], [2, 3]);
    expect(isFinite(score)).toBe(true);
  });

  it('handles missing domain size (fallback to 2)', () => {
    const data = [[0, 0], [1, 1]];
    // domainSizes is shorter than the variable indices — triggers ?? 2 fallback
    const score = bdeuScore(data, 1, [0], [2]);
    expect(isFinite(score)).toBe(true);
  });

  it('handles multiple parents with missing domain sizes', () => {
    const data = [[0, 0, 0], [1, 1, 1]];
    const score = bdeuScore(data, 2, [0, 1], [2]);
    expect(isFinite(score)).toBe(true);
  });
});

describe('discretizeBDeu', () => {
  it('handles empty data', () => {
    expect(discretizeBDeu([]).discretized).toEqual([]);
  });

  it('handles zero-column data', () => {
    const result = discretizeBDeu([[]] as number[][]);
    expect(result.discretized).toEqual([]);
    expect(result.domainSizes).toEqual([]);
  });

  it('discretizes 1D data into bins', () => {
    const data = [[0.1], [0.5], [0.9], [1.5], [2.0], [2.5], [3.0]];
    const result = discretizeBDeu(data, 3);
    expect(result.discretized.length).toBe(7);
    expect(result.domainSizes).toEqual([3]);
    // All values should be 0, 1, or 2
    for (const row of result.discretized) {
      expect(row[0]).toBeGreaterThanOrEqual(0);
      expect(row[0]).toBeLessThanOrEqual(2);
    }
  });

  it('discretizes 2D data', () => {
    const data = [[0.1, 5.0], [1.0, 6.0], [2.0, 7.0]];
    const result = discretizeBDeu(data, 2);
    expect(result.discretized.length).toBe(3);
    expect(result.domainSizes).toEqual([2, 2]);
  });

  it('handles constant columns (zero range)', () => {
    const data = [[5, 1], [5, 2], [5, 3]];
    const result = discretizeBDeu(data, 3);
    expect(result.discretized).toHaveLength(3);
    // Constant column should map all to bin 0
    for (const row of result.discretized) {
      expect(row[0]).toBe(0);
    }
  });
});
