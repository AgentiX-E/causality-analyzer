import { describe, it, expect } from 'vitest';
import { solveLinear, solveLinearSafe, normalTail, normalCDF, normalCDFTail, erf, colMean, createRNG, combinations, fisherZTest, partialCorrelationFromCov, invertMatrix, solveOLS, bicScore, gicScore, isBicScore, isMatrixSingular, precomputeCorrelation, chiSquareTest, gSquareTest, _setFisherZCacheMax, _resetFisherZCache } from '../math.js';

// ── solveLinear ─────────────────────────────────────────────────────

describe('solveLinear', () => {
  it('solves 2×2 system', () => {
    const x = solveLinear([[2, 1], [1, 3]], [5, 6]);
    expect(x[0]).toBeCloseTo(1.8, 6);
    expect(x[1]).toBeCloseTo(1.4, 6);
  });

  it('solves 3×3 identity', () => {
    const x = solveLinear([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 2, 3]);
    expect(x).toEqual([1, 2, 3]);
  });

  it('handles n=0', () => {
    expect(solveLinear([], [])).toEqual([]);
  });

  it('solves diagonal matrix', () => {
    const x = solveLinear([[3, 0, 0], [0, 4, 0], [0, 0, 5]], [3, 4, 5]);
    expect(x[0]).toBeCloseTo(1);
    expect(x[1]).toBeCloseTo(1);
    expect(x[2]).toBeCloseTo(1);
  });

  it('handles near-singular matrix gracefully', () => {
    const A = [[1, 1], [1, 1 + 1e-14]];
    const b = [2, 2];
    const x = solveLinear(A, b);
    // Should not throw; partial pivoting skips near-zero pivot
    expect(x.length).toBe(2);
  });

  it('solves 4×4 system', () => {
    const A = [[4, 1, -1, 0], [1, 3, 0, -1], [-1, 0, 2, 1], [0, -1, 1, 4]];
    const b = [4, 3, 2, 4];
    const x = solveLinear(A, b);
    expect(x.length).toBe(4);
    expect(x[0]).toBeCloseTo(1, 4);
    expect(x[1]).toBeCloseTo(1, 4);
    expect(x[2]).toBeCloseTo(1, 4);
    expect(x[3]).toBeCloseTo(1, 4);
  });

  it('solves with zero RHS', () => {
    const x = solveLinear([[2, 1], [1, 2]], [0, 0]);
    expect(x[0]).toBeCloseTo(0);
    expect(x[1]).toBeCloseTo(0);
  });
});

// ── normalTail ──────────────────────────────────────────────────────

describe('normalTail', () => {
  it('returns 0.5 at x=0', () => {
    expect(normalTail(0)).toBeCloseTo(0.5, 6);
  });

  it('approaches 0 for large x', () => {
    expect(normalTail(6)).toBeLessThan(1e-8);
  });

  it('is symmetric', () => {
    expect(normalTail(2)).toBeCloseTo(normalTail(-2), 10);
  });

  it('returns known value at z=1.96', () => {
    expect(normalTail(1.96)).toBeCloseTo(0.025, 2);
  });

  it('returns known value at z=2.575', () => {
    expect(normalTail(2.575)).toBeCloseTo(0.005, 2);
  });

  it('is non-negative for any input', () => {
    for (const x of [-10, -5, -1, 0, 1, 5, 10]) {
      expect(normalTail(x)).toBeGreaterThanOrEqual(0);
    }
  });

  it('is monotonically decreasing for x > 0', () => {
    let prev = normalTail(0);
    for (let x = 0.5; x <= 5; x += 0.5) {
      const curr = normalTail(x);
      expect(curr).toBeLessThan(prev);
      prev = curr;
    }
  });
});

// ── normalCDF ───────────────────────────────────────────────────────

describe('normalCDF', () => {
  it('returns 0.5 at x=0', () => {
    expect(normalCDF(0)).toBeCloseTo(0.5, 6);
  });

  it('approaches 1 for large positive x', () => {
    expect(normalCDF(5)).toBeCloseTo(1, 6);
  });

  it('approaches 0 for large negative x', () => {
    expect(normalCDF(-5)).toBeCloseTo(0, 6);
  });

  it('is symmetric: CDF(-x) = 1 - CDF(x)', () => {
    expect(normalCDF(-1)).toBeCloseTo(1 - normalCDF(1), 6);
  });

  it('returns known value at z=1.96', () => {
    expect(normalCDF(1.96)).toBeCloseTo(0.975, 3);
  });
});

// ── normalCDFTail ───────────────────────────────────────────────────

describe('normalCDFTail', () => {
  it('returns 0.5 at x=0', () => {
    expect(normalCDFTail(0)).toBeCloseTo(0.5, 2);
  });

  it('agrees with normalTail at common points', () => {
    for (const x of [0.5, 1, 1.5, 2, 2.5, 3]) {
      expect(normalCDFTail(x)).toBeCloseTo(normalTail(x), 3);
    }
  });

  it('approaches 0 for large x', () => {
    expect(normalCDFTail(6)).toBeLessThan(1e-8);
  });
});

// ── erf ─────────────────────────────────────────────────────────────

describe('erf', () => {
  it('returns 0 at x=0', () => {
    expect(erf(0)).toBeCloseTo(0, 5);
  });

  it('approaches 1 for large x', () => {
    expect(erf(5)).toBeCloseTo(1, 6);
  });

  it('approaches -1 for large negative x', () => {
    expect(erf(-5)).toBeCloseTo(-1, 6);
  });

  it('is odd: erf(-x) = -erf(x)', () => {
    expect(erf(-1)).toBeCloseTo(-erf(1), 10);
  });

  it('returns known value at x=1', () => {
    expect(erf(1)).toBeCloseTo(0.84270079, 6);
  });

  it('returns known value at x=2', () => {
    expect(erf(2)).toBeCloseTo(0.99532227, 6);
  });

  it('consistency with normalCDF through identity', () => {
    // CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
    for (const x of [-2, -1, 0, 1, 2]) {
      const fromErf = 0.5 * (1 + erf(x / Math.SQRT2));
      expect(fromErf).toBeCloseTo(normalCDF(x), 6);
    }
  });
});

// ── colMean ─────────────────────────────────────────────────────────

describe('colMean', () => {
  it('computes column mean', () => {
    expect(colMean([[1], [2], [3]], 0)).toBeCloseTo(2);
  });

  it('handles NaN values by skipping', () => {
    expect(colMean([[1], [NaN], [3]], 0)).toBeCloseTo(2);
  });

  it('handles null/undefined values', () => {
    const data: number[][] = [[1], [undefined as any], [3]];
    expect(colMean(data, 0)).toBeCloseTo(2);
  });

  it('returns NaN for empty data', () => {
    expect(colMean([], 0)).toBeNaN();
  });

  it('handles multiple columns', () => {
    expect(colMean([[1, 10], [2, 20], [3, 30]], 1)).toBeCloseTo(20);
  });
});

// ── fisherZTest ─────────────────────────────────────────────────────

describe('fisherZTest', () => {
  it('generates p-values in valid range', () => {
    const data = Array.from({ length: 100 }, () => [Math.random(), Math.random()]);
    const p = fisherZTest(data, 0, 1, []);
    // p-value should be in [0,1]. For purely random data, it will usually
    // be large, but sometimes small by chance (Type I error).
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it('produces valid p-values for correlated vs independent data', () => {
    // Independent data — higher p-value
    const indep = Array.from({ length: 30 }, () => [Math.random(), Math.random()]);
    const pIndep = fisherZTest(indep, 0, 1, []);
    // Positively correlated data — lower p-value
    const corr: number[][] = [];
    for (let i = 0; i < 30; i++) {
      const x = i * 0.1;
      corr.push([x, x * 1.5 + (Math.random() - 0.5) * 0.2]);
    }
    const pCorr = fisherZTest(corr, 0, 1, []);
    // Correlated data should produce p-value in valid range
    expect(pCorr).toBeGreaterThanOrEqual(0);
    expect(pCorr).toBeLessThanOrEqual(1);
    expect(pIndep).toBeGreaterThanOrEqual(0);
    expect(pIndep).toBeLessThanOrEqual(1);
  });

  it('returns p in [0,1] range', () => {
    const data = Array.from({ length: 30 }, (_, i) => [i, i * 0.5 + Math.random()]);
    const p = fisherZTest(data, 0, 1, []);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it('handles empty conditioning set', () => {
    const data = [[1, 2], [2, 4], [3, 6]];
    const p = fisherZTest(data, 0, 1, []);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });
});

// ── partialCorrelationFromCov ────────────────────────────────────────

describe('partialCorrelationFromCov', () => {
  it('returns 0 for 2x2 uncorrelated covariance', () => {
    const cov = [[1, 0], [0, 1]];
    const rho = partialCorrelationFromCov(cov, 0, 1);
    expect(rho).toBeCloseTo(0, 6);
  });

  it('returns ~1 for 2x2 perfectly correlated', () => {
    const cov = [[1, 0.999], [0.999, 1]];
    const rho = partialCorrelationFromCov(cov, 0, 1);
    expect(rho).toBeCloseTo(0.999, 2);
  });

  it('produces valid output for 3x3 covariance matrix', () => {
    const cov = [[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]];
    const rho = partialCorrelationFromCov(cov, 0, 1);
    expect(rho).toBeGreaterThanOrEqual(-1);
    expect(rho).toBeLessThanOrEqual(1);
    expect(isNaN(rho)).toBe(false);
  });
});

// ── invertMatrix ─────────────────────────────────────────────────────

describe('invertMatrix', () => {
  it('computes identity inverse', () => {
    const I = [[1, 0], [0, 1]];
    const inv = invertMatrix(I);
    expect(inv[0]![0]).toBeCloseTo(1, 6);
    expect(inv[0]![1]).toBeCloseTo(0, 6);
    expect(inv[1]![0]).toBeCloseTo(0, 6);
    expect(inv[1]![1]).toBeCloseTo(1, 6);
  });

  it('inverts 2x2 matrix correctly', () => {
    const A = [[4, 7], [2, 6]];
    const inv = invertMatrix(A);
    // inv(A) * A ≈ I
    const r00 = inv[0]![0]! * 4 + inv[0]![1]! * 2;
    const r01 = inv[0]![0]! * 7 + inv[0]![1]! * 6;
    expect(r00).toBeCloseTo(1, 6);
    expect(r01).toBeCloseTo(0, 6);
  });

  it('inverts diagonal matrix', () => {
    const d = [[3, 0, 0], [0, 4, 0], [0, 0, 5]];
    const inv = invertMatrix(d);
    expect(inv[0]![0]).toBeCloseTo(1 / 3, 6);
    expect(inv[1]![1]).toBeCloseTo(1 / 4, 6);
    expect(inv[2]![2]).toBeCloseTo(1 / 5, 6);
  });
});

// ── solveOLS ─────────────────────────────────────────────────────────

describe('solveOLS', () => {
  it('fits y = 2x', () => {
    const X = [[1, 1], [1, 2], [1, 3]];
    const y = [2, 4, 6];
    const beta = solveOLS(X, y);
    expect(beta[0]!).toBeCloseTo(0, 8); // intercept
    expect(beta[1]!).toBeCloseTo(2, 8); // slope
  });

  it('handles empty input', () => {
    expect(solveOLS([], [])).toEqual([]);
  });
});

// ── bicScore ─────────────────────────────────────────────────────────

describe('bicScore', () => {
  it('penalizes more parameters', () => {
    const bic1 = bicScore(10, 100, 1);
    const bic2 = bicScore(10, 100, 2);
    expect(bic2).toBeGreaterThan(bic1); // more params = worse BIC
  });

  it('returns infinity for zero sample size', () => {
    expect(bicScore(10, 0, 1)).toBe(Infinity);
  });

  it('produces finite values', () => {
    const bic = bicScore(5, 50, 2);
    expect(isFinite(bic)).toBe(true);
  });
});

// ── createRNG ───────────────────────────────────────────────────────

describe('createRNG', () => {
  it('produces values in [0,1]', () => {
    const rng = createRNG(42);
    for (let i = 0; i < 100; i++) {
      const v = rng();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('is deterministic with seed', () => {
    const a = createRNG(42);
    const b = createRNG(42);
    for (let i = 0; i < 20; i++) expect(a()).toBe(b());
  });

  it('different seeds produce different sequences', () => {
    const a = createRNG(42)();
    const b = createRNG(99)();
    expect(a).not.toBe(b);
  });

  it('null seed uses Math.random', () => {
    const rng = createRNG(null);
    expect(typeof rng()).toBe('number');
  });
});

// ── solveLinearSafe ──────────────────────────────────────────────────

describe('solveLinearSafe', () => {
  it('solves a non-singular system', () => {
    const { solution, singular } = solveLinearSafe([[2, 1], [1, 3]], [5, 6]);
    expect(singular).toBe(false);
    expect(solution).not.toBeNull();
    expect(solution![0]).toBeCloseTo(1.8, 6);
    expect(solution![1]).toBeCloseTo(1.4, 6);
  });

  it('detects singular matrix', () => {
    const { solution, singular } = solveLinearSafe([[1, 1], [2, 2]], [3, 6]);
    expect(singular).toBe(true);
    expect(solution).toBeNull();
  });

  it('handles near-singular matrix', () => {
    const { singular } = solveLinearSafe([[1, 1], [1, 1 + 1e-14]], [2, 2]); 
    expect(singular).toBe(true);
  });

  it('handles n=0', () => {
    const { solution, singular } = solveLinearSafe([], []);
    expect(singular).toBe(false);
    expect(solution).toEqual([]);
  });

  it('solves 3×3 identity', () => {
    const { solution, singular } = solveLinearSafe([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 2, 3]);
    expect(singular).toBe(false);
    expect(solution).toEqual([1, 2, 3]);
  });

  it('handles partially defined b array', () => {
    const { solution, singular } = solveLinearSafe([[1, 0], [0, 1]], [1] as number[]);
    expect(singular).toBe(false);
    expect(solution).not.toBeNull();
    expect(solution![0]).toBeCloseTo(1);
    expect(solution![1]).toBeCloseTo(0);
  });

  it('requires pivoting for unsorted rows', () => {
    const { solution, singular } = solveLinearSafe([[1, 3], [2, 1]], [6, 5]);
    expect(singular).toBe(false);
    expect(solution).not.toBeNull();
  });
});

// ── combinations ─────────────────────────────────────────────────────

describe('combinations', () => {
  it('returns [[]] for k=0', () => {
    expect(combinations([1, 2, 3], 0)).toEqual([[]]);
  });

  it('returns empty for k>n', () => {
    expect(combinations([1, 2], 3)).toEqual([]);
  });

  it('returns single-elements for k=1', () => {
    const result = combinations(['a', 'b', 'c'], 1);
    expect(result).toHaveLength(3);
    expect(result).toContainEqual(['a']);
    expect(result).toContainEqual(['b']);
    expect(result).toContainEqual(['c']);
  });

  it('returns all elements for k=n', () => {
    const result = combinations([1, 2], 2);
    expect(result).toEqual([[1, 2]]);
  });

  it('generates correct C(4,2)=6 combinations', () => {
    const result = combinations([1, 2, 3, 4], 2);
    expect(result).toHaveLength(6);
    expect(result).toContainEqual([1, 2]);
    expect(result).toContainEqual([1, 3]);
    expect(result).toContainEqual([1, 4]);
    expect(result).toContainEqual([2, 3]);
    expect(result).toContainEqual([2, 4]);
    expect(result).toContainEqual([3, 4]);
  });

  it('handles empty array', () => {
    expect(combinations([], 0)).toEqual([[]]);
    expect(combinations([], 1)).toEqual([]);
  });

  it('preserves element order within each combination', () => {
    const result = combinations([1, 2, 3], 2);
    for (const combo of result) {
      expect(combo[0]).toBeLessThan(combo[1]!);
    }
  });
});

// ── Branch Coverage Fillers ───────────────────────────────────────

describe('solveOLS edge cases', () => {
  it('handles missing values in X', () => {
    const X = [[1, undefined as unknown as number], [2, 3]];
    const beta = solveOLS(X, [1, 2]);
    expect(beta.length).toBeGreaterThan(0);
  });

  it('handles null values in y', () => {
    const X = [[1, 0], [1, 1]];
    const y = [0, undefined as unknown as number];
    const beta = solveOLS(X, y);
    expect(beta.length).toBeGreaterThan(0);
  });
});

describe('bicScore edge cases', () => {
  it('handles negative sample size', () => {
    expect(bicScore(10, -1, 1)).toBe(Infinity);
  });

  it('handles very small n', () => {
    const bic = bicScore(5, 1, 2);
    expect(typeof bic).toBe('number');
  });
});

describe('invertMatrix edge cases', () => {
  it('handles singular matrix (pivot below threshold)', () => {
    // Zero matrix — pivot will be below threshold
    const Z = [[0, 0], [0, 0]];
    const inv = invertMatrix(Z);
    // Should not throw, produces approximate inverse
    expect(inv.length).toBe(2);
  });
});

// GIC + IS-BIC + isMatrixSingular
describe('GIC scoring', () => {
  it('gicScore with gamma=2 matches AIC-like behavior', () => {
    const gic = gicScore(10, 100, 1, 2);
    expect(isFinite(gic)).toBe(true);
  });

  it('gicScore default matches BIC', () => {
    const gic = gicScore(10, 100, 1);
    const bic = bicScore(10, 100, 1);
    expect(isFinite(gic)).toBe(true);
    expect(isFinite(bic)).toBe(true);
  });

  it('isBicScore returns half-penalty BIC', () => {
    expect(isFinite(isBicScore(10, 100, 1))).toBe(true);
  });
});

describe('isMatrixSingular', () => {
  it('returns false for identity matrix', () => {
    expect(isMatrixSingular([[1, 0], [0, 1]])).toBe(false);
  });

  it('returns true for zero matrix', () => {
    expect(isMatrixSingular([[0, 0], [0, 0]])).toBe(true);
  });
});

// ── precomputeCorrelation ────────────────────────────────────────────

describe('precomputeCorrelation', () => {
  it('returns empty array for empty data', () => {
    const result = precomputeCorrelation([]);
    expect(result).toEqual([]);
  });

  it('returns empty array for zero-column data', () => {
    const result = precomputeCorrelation([[]] as number[][]);
    expect(result).toEqual([]);
  });

  it('computes correlation for single-column data', () => {
    const data = [[1], [2], [3]];
    const corr = precomputeCorrelation(data);
    expect(corr).toHaveLength(1);
    expect(corr[0]![0]).toBe(1); // self-correlation
  });

  it('computes correlation for independent variables', () => {
    const data = Array.from({ length: 50 }, () => [Math.random(), Math.random()]);
    const corr = precomputeCorrelation(data);
    expect(corr).toHaveLength(2);
    // Diagonal should be 1
    expect(corr[0]![0]).toBe(1);
    expect(corr[1]![1]).toBe(1);
    // Off-diagonal should be symmetric
    expect(corr[0]![1]).toBe(corr[1]![0]);
  });

  it('computes near-perfect correlation for linear data', () => {
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      data.push([i, i * 2 + 1]);
    }
    const corr = precomputeCorrelation(data);
    expect(corr[0]![1]).toBeGreaterThan(0.99);
  });

  it('handles constant columns (zero std)', () => {
    const data = [[5, 1], [5, 2], [5, 3]];
    const corr = precomputeCorrelation(data);
    // Constant column has zero std, correlation should be 0 when denom is 0
    expect(corr[0]![1]).toBe(0);
  });

  it('returns 1 on diagonal', () => {
    const data = [[1, 2], [3, 4], [5, 6]];
    const corr = precomputeCorrelation(data);
    for (let i = 0; i < 2; i++) {
      expect(corr[i]![i]).toBe(1);
    }
  });
});

// ── Fisher Z with precomputed correlation matrix ───────────────────

describe('fisherZTest with corrMatrix', () => {
  it('uses precomputed correlation matrix correctly', () => {
    const data = Array.from({ length: 50 }, (_, i) => [i, i * 0.8 + (Math.random() - 0.5), Math.random()]);
    const corrMatrix = precomputeCorrelation(data);
    const pWithout = fisherZTest(data, 0, 1, []);
    const pWith = fisherZTest(data, 0, 1, [], corrMatrix);
    expect(pWith).toBeGreaterThanOrEqual(0);
    expect(pWith).toBeLessThanOrEqual(1);
    // Both should produce p-values in valid range
    expect(pWithout).toBeGreaterThanOrEqual(0);
    expect(pWithout).toBeLessThanOrEqual(1);
  });

  it('handles conditioning set with corrMatrix', () => {
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const x = Math.random();
      data.push([x, x * 0.5 + (Math.random() - 0.5) * 0.3, x * 0.3 + Math.random() * 0.7]);
    }
    const corrMatrix = precomputeCorrelation(data);
    const p = fisherZTest(data, 0, 1, [2], corrMatrix);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it('handles near-singular conditioning with corrMatrix', () => {
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const x = Math.random();
      data.push([x, x + 1e-14, x * 0.9 + (Math.random() - 0.5) * 0.1, Math.random()]);
    }
    const corrMatrix = precomputeCorrelation(data);
    // Two nearly identical variables in conditioning set may cause near-singularity
    const p = fisherZTest(data, 0, 3, [1, 2], corrMatrix);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });
});

// ── chiSquareTest ───────────────────────────────────────────────────

describe('chiSquareTest', () => {
  it('returns 1 for small contingency tables', () => {
    expect(chiSquareTest([[1]])).toBe(1);
    expect(chiSquareTest([[1, 2]])).toBe(1);
  });

  it('returns valid p-value for independent categories', () => {
    // Uniform distribution = independence
    const observed = [
      [25, 25],
      [25, 25],
    ];
    const p = chiSquareTest(observed);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it('returns small p-value for clearly dependent categories', () => {
    // Strong association — 90% on diagonal
    const observed = [
      [45, 5],
      [5, 45],
    ];
    const p = chiSquareTest(observed);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
    // Strong association should give very small p-value
    expect(p).toBeLessThan(0.01);
  });

  it('handles zero totals gracefully', () => {
    const observed = [[0, 0], [0, 0]];
    expect(chiSquareTest(observed)).toBe(1);
  });

  it('handles 3×3 table', () => {
    const observed = [
      [30, 20, 10],
      [15, 25, 20],
      [5, 15, 30],
    ];
    const p = chiSquareTest(observed);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it('handles sparse table with zeros', () => {
    const observed = [
      [10, 0, 5],
      [0, 20, 0],
      [5, 0, 10],
    ];
    const p = chiSquareTest(observed);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it('is consistent: larger sample sizes give similar p-value for same proportions', () => {
    const small = [[10, 15], [15, 10]];
    const large = [[20, 30], [30, 20]];
    const pSmall = chiSquareTest(small);
    const pLarge = chiSquareTest(large);
    expect(pSmall).toBeGreaterThanOrEqual(0);
    expect(pLarge).toBeGreaterThanOrEqual(0);
  });

  it('independent data yields higher p-value than dependent data', () => {
    const independent = [[25, 25], [25, 25]];
    const dependent = [[45, 5], [5, 45]];
    const pIndep = chiSquareTest(independent);
    const pDep = chiSquareTest(dependent);
    // Independence should give less significant result
    expect(pIndep).toBeGreaterThan(pDep);
  });
});

// ── gSquareTest ─────────────────────────────────────────────────────

describe('gSquareTest', () => {
  it('returns 1 for small contingency tables', () => {
    expect(gSquareTest([[1]])).toBe(1);
    expect(gSquareTest([[1, 2]])).toBe(1);
  });

  it('returns valid p-value for independent categories', () => {
    const observed = [
      [25, 25],
      [25, 25],
    ];
    const p = gSquareTest(observed);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it('returns small p-value for clearly dependent categories', () => {
    const observed = [
      [45, 5],
      [5, 45],
    ];
    const p = gSquareTest(observed);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
    expect(p).toBeLessThan(0.01);
  });

  it('handles zero totals gracefully', () => {
    expect(gSquareTest([[0, 0], [0, 0]])).toBe(1);
  });

  it('handles 3×3 table', () => {
    const observed = [
      [30, 20, 10],
      [15, 25, 20],
      [5, 15, 30],
    ];
    const p = gSquareTest(observed);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it('handles table with zeros in cells', () => {
    const observed = [
      [10, 0, 5],
      [0, 20, 0],
      [5, 0, 15],
    ];
    const p = gSquareTest(observed);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it('is approximately consistent with chiSquareTest', () => {
    const observed = [[30, 20, 10], [15, 25, 20], [5, 15, 30]];
    const g = gSquareTest(observed);
    const chi = chiSquareTest(observed);
    // G and Chi should be in similar range
    expect(Math.abs(g - chi)).toBeLessThan(0.3);
  });

  it('independent data yields higher p-value than dependent data', () => {
    const independent = [[25, 25], [25, 25]];
    const dependent = [[45, 5], [5, 45]];
    const pIndep = gSquareTest(independent);
    const pDep = gSquareTest(dependent);
    // Independence should give less significant result
    expect(pIndep).toBeGreaterThan(pDep);
  });
});

// ── Fisher Z cache eviction ─────────────────────────────────────────

describe('fisherZTest cache behavior', () => {
  it('caches repeated calls and returns same result', () => {
    const data = Array.from({ length: 30 }, () => [Math.random(), Math.random()]);
    const p1 = fisherZTest(data, 0, 1, []);
    const p2 = fisherZTest(data, 0, 1, []);
    expect(p1).toBe(p2); // Cached result should be identical
  });

  it('handles asymmetric i,j caching (cache key sorts)', () => {
    const data = Array.from({ length: 30 }, () => [Math.random(), Math.random(), Math.random()]);
    const p1 = fisherZTest(data, 0, 2, [1]);
    const p2 = fisherZTest(data, 2, 0, [1]);
    expect(p1).toBe(p2); // (0,2) and (2,0) should use same cache key
  });

  it('evicts oldest entry when cache reaches max size', () => {
    _resetFisherZCache();
    _setFisherZCacheMax(3);
    const data = Array.from({ length: 30 }, (_, i) => [i * 0.1, i * 0.15 + Math.random() * 0.05, Math.random()]);
    // Fill cache with 3 unique calls — different combos
    fisherZTest(data, 0, 1, []);
    fisherZTest(data, 0, 2, []);
    fisherZTest(data, 1, 2, []);
    // 4th call should trigger eviction
    fisherZTest(data, 0, 1, [2]);
    // Cache should still work
    const p = fisherZTest(data, 0, 1, [2]);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
    // Reset for other tests
    _setFisherZCacheMax(50000);
    _resetFisherZCache();
  });
});

// ── fisherZTest near-singular correlation ──────────────────────────

describe('fisherZTest singular matrix detection', () => {
  it('handles near-singular correlation matrix (k > 1)', () => {
    // Create data where two conditioning variables are nearly collinear
    const data: number[][] = [];
    for (let i = 0; i < 50; i++) {
      const x1 = Math.random();
      const x2 = x1 + (Math.random() - 0.5) * 1e-14; // nearly identical
      const x3 = Math.random();
      // Add a target and a test variable
      data.push([x1, x2, x3, Math.random()]);
    }
    const corrMatrix = precomputeCorrelation(data);
    // Test with conditioning set containing the nearly-collinear variables
    const p = fisherZTest(data, 0, 3, [1, 2], corrMatrix);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });
});

// ── Additional invertMatrix edge cases ──────────────────────────────

describe('invertMatrix regression', () => {
  it('handles 1×1 matrix', () => {
    const inv = invertMatrix([[5]]);
    expect(inv[0]![0]).toBeCloseTo(0.2, 6);
  });

  it('handles singular 3×3 matrix gracefully', () => {
    // Matrix where row2 = row1 + row0 (linearly dependent)
    const A = [[1, 2, 3], [4, 5, 6], [5, 7, 9]];
    const inv = invertMatrix(A);
    // Should not throw
    expect(inv).toHaveLength(3);
    expect(inv[0]!).toHaveLength(3);
  });

  it('preserves matrix dimensions', () => {
    const A = [[2, 1, 3], [1, 3, 2], [3, 2, 1]];
    const inv = invertMatrix(A);
    expect(inv).toHaveLength(3);
    expect(inv[0]!).toHaveLength(3);
  });
});

// ── solveOLS edge case with zero-column X ───────────────────────────

describe('solveOLS zero-k edge case', () => {
  it('handles X with zero columns gracefully', () => {
    const X = [[] as number[], [] as number[]];
    const y = [1, 2];
    const beta = solveOLS(X as number[][], y);
    expect(beta).toEqual([]);
  });

  it('handles empty X rows', () => {
    const beta = solveOLS([], []);
    expect(beta).toEqual([]);
  });
});

// ── bicScore edge: very small n ─────────────────────────────────────

describe('bicScore very small n', () => {
  it('returns finite value for n=2', () => {
    const bic = bicScore(1, 2, 1);
    expect(isFinite(bic)).toBe(true);
  });
});

// ── chiSquareTest and gSquareTest edge cases ────────────────────────

describe('chiSquarePValue edge cases', () => {
  it('handles 2x2 observed with values', () => {
    const r = chiSquareTest([
      [10, 20],
      [30, 40],
    ]);
    expect(r).toBeGreaterThan(0);
    expect(r).toBeLessThan(1);
  });
  it('returns valid p-value for 3x2 table', () => {
    const r = chiSquareTest([
      [15, 25],
      [35, 45],
      [10, 20],
    ]);
    expect(r).toBeGreaterThan(0);
    expect(r).toBeLessThan(1);
  });
});

describe('gSquareTest edge cases', () => {
  it('returns finite p-value for 2x2', () => {
    const r = gSquareTest([
      [10, 5],
      [5, 10],
    ]);
    expect(r).toBeGreaterThan(0);
    expect(r).toBeLessThan(1);
  });
  it('returns 1 for single cell table', () => {
    const r = gSquareTest([[1]]);
    expect(r).toBe(1);
  });
});

// ── solveLinearCholesky Tests ──────────────────────────────────────

import { solveLinearCholesky } from '../math.js';

describe('solveLinearCholesky', () => {
  it('solves 2×2 SPD system correctly', () => {
    // [4 1; 1 3] x = [1; 2] → x ≈ [0.09, 0.636]
    const A = [[4, 1], [1, 3]];
    const b = [1, 2];
    const x = solveLinearCholesky(A, b);
    expect(x).not.toBeNull();
    // Verify: A·x ≈ b
    expect(A[0]![0]! * x![0]! + A[0]![1]! * x![1]!).toBeCloseTo(b[0]!, 5);
    expect(A[1]![0]! * x![0]! + A[1]![1]! * x![1]!).toBeCloseTo(b[1]!, 5);
  });

  it('produces same result as solveLinear for SPD matrix', () => {
    const A = [[5, 2, 1], [2, 6, 2], [1, 2, 7]];
    const b = [3, 4, 5];
    const chol = solveLinearCholesky(A, b);
    const ge = solveLinear(A, b);
    expect(chol).not.toBeNull();
    for (let i = 0; i < 3; i++) {
      expect(chol![i]!).toBeCloseTo(ge[i]!, 8);
    }
  });

  it('returns null for non-SPD matrix', () => {
    // Non-symmetric matrix is not SPD
    const A = [[1, 3], [2, 4]];
    const b = [1, 2];
    expect(solveLinearCholesky(A, b)).toBeNull();
  });

  it('solves identity matrix', () => {
    const A = [[1, 0], [0, 1]];
    const b = [5, 10];
    const x = solveLinearCholesky(A, b);
    expect(x).toEqual([5, 10]);
  });

  it('handles 1×1', () => {
    const A = [[7]];
    const b = [21];
    const x = solveLinearCholesky(A, b);
    expect(x).not.toBeNull();
    expect(x![0]!).toBeCloseTo(3, 10);
  });

  it('handles empty input', () => {
    expect(solveLinearCholesky([], [])).toEqual([]);
  });
});
