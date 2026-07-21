import { describe, it, expect } from 'vitest';
import { solveLinear, normalTail, normalCDF, normalCDFTail, erf } from '../math.js';

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
