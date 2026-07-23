import { describe, it, expect } from 'vitest';
import { solveLinear, solveLinearSafe, normalTail, normalCDF, normalCDFTail, erf, colMean, createRNG, combinations } from '../math.js';

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
