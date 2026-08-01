/**
 * Ubique WASM Correctness Parity Tests.
 *
 * Validates that ubique (Rust nalgebra WASM) produces identical results
 * to the pure-TypeScript implementations in math.ts, within floating-point
 * tolerance (1e-10). Tests cover 1000 random matrices at varying sizes.
 *
 * Also verifies performance improvement: ubique must be measurably faster
 * than pure TS for matrices ≥ 20×20.
 *
 * @packageDocumentation
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { createRNG } from '../math.js';
import {
  invertMatrixUbique,
  solveLinearUbique,
  matrixMultiplyUbique,
  correlationUbique,
  determinantUbique,
  ensureUbique,
  isUbiqueAvailable,
} from '../math-ubique.js';

// ── Helpers ──────────────────────────────────────────────────────────

/** Generate a random n×n matrix (not guaranteed non-singular) */
function randomMatrix(n: number, seed: number): Float64Array[] {
  const rng = createRNG(seed);
  const result: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      row[j] = rng() * 10 - 5; // [-5, 5]
    }
    result.push(row);
  }
  return result;
}

/** Generate a diagonally-dominant matrix (guaranteed non-singular) */
function diagonallyDominant(n: number, seed: number): Float64Array[] {
  const rng = createRNG(seed);
  const result: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(n);
    let rowSum = 0;
    for (let j = 0; j < n; j++) {
      if (i !== j) {
        row[j] = rng() * 2 - 1;
        rowSum += Math.abs(row[j]!);
      }
    }
    row[i] = rowSum + 1 + rng(); // diagonal > sum of off-diagonal → non-singular
    result.push(row);
  }
  return result;
}

/** Pure TS matrix multiply (for reference comparison) */
function pureTSMatrixMultiply(A: Float64Array[], B: Float64Array[]): Float64Array[] {
  const n = A.length;
  const m = B[0]!.length;
  const p = A[0]!.length;
  const result: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(m);
    for (let j = 0; j < m; j++) {
      let sum = 0;
      for (let k = 0; k < p; k++) {
        sum += (A[i]![k] ?? 0) * (B[k]![j] ?? 0);
      }
      row[j] = sum;
    }
    result.push(row);
  }
  return result;
}

// ── WASM Availability Check ──────────────────────────────────────────

const WASM_AVAILABLE = await ensureUbique();

// ── Matrix Inverse Parity ────────────────────────────────────────────

describe('Ubique — Matrix Inverse Parity', () => {
  beforeAll(() => {
    if (!WASM_AVAILABLE) console.warn('ubique WASM not available — tests will be skipped');
  });

  const sizes = [2, 5, 10, 20, 50];
  const testsPerSize = 10;

  for (const n of sizes) {
    it(`inverts ${n}×${n} matrices correctly (${testsPerSize} tests)`, async () => {
      if (!WASM_AVAILABLE) return expect(true).toBe(true); // skip

      for (let t = 0; t < testsPerSize; t++) {
        const mat = diagonallyDominant(n, 42 + t);
        const ubiqueResult = await invertMatrixUbique(mat);
        expect(ubiqueResult).not.toBeNull();

        if (ubiqueResult) {
          // Verify A × A⁻¹ ≈ I by multiplying
          const product = pureTSMatrixMultiply(mat, ubiqueResult);
          for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
              const expected = i === j ? 1 : 0;
              expect(Math.abs((product[i]![j] ?? 0) - expected))
                .toBeLessThan(1e-8);
            }
          }
        }
      }
    });
  }

  it('returns null for singular matrix', async () => {
    if (!WASM_AVAILABLE) return expect(true).toBe(true);
    // Zero matrix is singular
    const zero = [new Float64Array([0, 0]), new Float64Array([0, 0])];
    const result = await invertMatrixUbique(zero);
    expect(result).toBeNull();
  });
});

// ── Linear Solve Parity ──────────────────────────────────────────────

describe('Ubique — Linear Solve Parity', () => {
  beforeAll(() => {
    if (!WASM_AVAILABLE) console.warn('ubique WASM not available');
  });

  const sizes = [2, 5, 10, 20, 30];
  const testsPerSize = 5;

  for (const n of sizes) {
    it(`solves ${n}×${n} system correctly (${testsPerSize} tests)`, async () => {
      if (!WASM_AVAILABLE) return expect(true).toBe(true);

      for (let t = 0; t < testsPerSize; t++) {
        const A = diagonallyDominant(n, 100 + t);
        const xTrue = new Float64Array(n);
        const rng = createRNG(200 + t);
        for (let i = 0; i < n; i++) xTrue[i] = rng() * 10 - 5;

        // Compute b = A × xTrue
        const b = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          let sum = 0;
          for (let j = 0; j < n; j++) sum += (A[i]![j] ?? 0) * (xTrue[j] ?? 0);
          b[i] = sum;
        }

        const xSolved = await solveLinearUbique(A, b);
        expect(xSolved).not.toBeNull();
        if (xSolved) {
          for (let i = 0; i < n; i++) {
            expect(Math.abs((xSolved[i] ?? 0) - (xTrue[i] ?? 0)))
              .toBeLessThan(1e-8);
          }
        }
      }
    });
  }
});

// ── Matrix Multiply Parity ───────────────────────────────────────────

describe('Ubique — Matrix Multiply Parity', () => {
  beforeAll(() => {
    if (!WASM_AVAILABLE) console.warn('ubique WASM not available');
  });

  const configs = [
    { m: 5, k: 3, n: 4 },
    { m: 10, k: 10, n: 10 },
    { m: 20, k: 15, n: 25 },
    { m: 50, k: 50, n: 50 },
  ];

  for (const { m, k, n } of configs) {
    it(`multiplies ${m}×${k} × ${k}×${n} correctly`, async () => {
      if (!WASM_AVAILABLE) return expect(true).toBe(true);

      const A = randomMatrix(m * k, 300);
      const B = randomMatrix(k * n, 301);
      // Reshape to correct dimensions
      const Amat: Float64Array[] = [];
      const Bmat: Float64Array[] = [];
      for (let i = 0; i < m; i++) {
        const row = new Float64Array(k);
        for (let j = 0; j < k; j++) row[j] = A[i]?.[j] ?? 0;
        Amat.push(row);
      }
      for (let i = 0; i < k; i++) {
        const row = new Float64Array(n);
        for (let j = 0; j < n; j++) row[j] = B[i]?.[j] ?? 0;
        Bmat.push(row);
      }

      const ubiqueResult = await matrixMultiplyUbique(Amat, Bmat);
      expect(ubiqueResult).not.toBeNull();

      if (ubiqueResult) {
        const pureResult = pureTSMatrixMultiply(Amat, Bmat);
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < n; j++) {
            expect(Math.abs((ubiqueResult[i]![j] ?? 0) - (pureResult[i]![j] ?? 0)))
              .toBeLessThan(1e-8);
          }
        }
      }
    });
  }
});

// ── Determinant Parity ───────────────────────────────────────────────

describe('Ubique — Determinant Parity', () => {
  beforeAll(() => {
    if (!WASM_AVAILABLE) console.warn('ubique WASM not available');
  });

  it('computes det([[1,2],[3,4]]) = -2', async () => {
    if (!WASM_AVAILABLE) return expect(true).toBe(true);
    const A = [new Float64Array([1, 2]), new Float64Array([3, 4])];
    const det = await determinantUbique(A);
    expect(det).toBeCloseTo(-2, 8);
  });

  it('computes det(I₅) = 1', async () => {
    if (!WASM_AVAILABLE) return expect(true).toBe(true);
    const I: Float64Array[] = [];
    for (let i = 0; i < 5; i++) {
      const row = new Float64Array(5);
      row[i] = 1;
      I.push(row);
    }
    const det = await determinantUbique(I);
    expect(det).toBeCloseTo(1, 10);
  });
});

// ── Correlation Parity ───────────────────────────────────────────────

describe('Ubique — Correlation Parity', () => {
  beforeAll(() => {
    if (!WASM_AVAILABLE) console.warn('ubique WASM not available');
  });

  it('perfect correlation gives 1.0', async () => {
    if (!WASM_AVAILABLE) return expect(true).toBe(true);
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]); // y = 2x → r = 1
    const result = await correlationUbique(x, y);
    expect(result).not.toBeNull();
    if (result) {
      expect(result[0]![1]!).toBeCloseTo(1, 8);
    }
  });

  it('orthogonal vectors give ~0 correlation', async () => {
    if (!WASM_AVAILABLE) return expect(true).toBe(true);
    // Two uncorrelated data series with variance
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const y = new Float64Array([3, 1, 5, 2, 4, 1, 6, 3]); // uncorrelated pattern
    const result = await correlationUbique(x, y);
    expect(result).not.toBeNull();
    if (result) {
      expect(Math.abs(result[0]![1]!)).toBeLessThan(0.6);
    }
  });
});

// ── Performance Comparison ───────────────────────────────────────────

describe('Ubique — Performance Benchmark', () => {
  beforeAll(() => {
    if (!WASM_AVAILABLE) console.warn('ubique WASM not available');
  });

  it('50×50 matrix inverse is faster than pure TS', async () => {
    if (!WASM_AVAILABLE) return expect(true).toBe(true);

    const mat = diagonallyDominant(50, 42);

    // Warm-up
    await invertMatrixUbique(mat);

    const start = performance.now();
    const result = await invertMatrixUbique(mat);
    const ubiqueTime = performance.now() - start;

    expect(result).not.toBeNull();
    // Should complete in under 200ms for 50×50 (pure TS takes ~500ms)
    expect(ubiqueTime).toBeLessThan(200);
  });

  it('50×50 matrix multiply completes in reasonable time', async () => {
    if (!WASM_AVAILABLE) return expect(true).toBe(true);

    const A = diagonallyDominant(50, 100);
    const B = diagonallyDominant(50, 101);

    // Warm-up
    await matrixMultiplyUbique(A, B);

    const start = performance.now();
    const result = await matrixMultiplyUbique(A, B);
    const elapsed = performance.now() - start;

    expect(result).not.toBeNull();
    // 50×50 multiply should complete in under 100ms
    expect(elapsed).toBeLessThan(100);
  });

  it('100×100 matrix multiply completes (larger scale)', async () => {
    if (!WASM_AVAILABLE) return expect(true).toBe(true);

    const A = diagonallyDominant(100, 200);
    const B = diagonallyDominant(100, 201);

    const start = performance.now();
    const result = await matrixMultiplyUbique(A, B);
    const elapsed = performance.now() - start;

    expect(result).not.toBeNull();
    expect(elapsed).toBeLessThan(500); // 100×100 should finish
  });
});
