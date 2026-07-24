/**
 * Shared mathematical utilities.
 *
 * Eliminates 5x solveLinear and 3x normalTail duplication across pipeline.
 * All implementations are battle-tested from the original code and
 * unified with full type safety.
 *
 * @packageDocumentation
 */

/**
 * Gaussian elimination with partial pivoting.
 *
 * Solves Ax = b for x. Handles n=0 (returns []).
 *
 * **WARNING**: Near-singular pivots (< 1e-12) are skipped, which can produce
 * unreliable results for ill-conditioned matrices. Prefer `solveLinearSafe`
 * for production use, which detects singularity explicitly.
 *
 * Complexity: O(n³) worst case.
 */
export function solveLinear(A: number[][], b: number[]): number[] {
  const n = A.length;
  if (n === 0) return [];
  // Build augmented matrix [A|b]
  const aug = A.map((row, i) => [...row, b[i] ?? 0]);
  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row]![col]!) > Math.abs(aug[pivot]![col]!)) pivot = row;
    }
    [aug[col], aug[pivot]] = [aug[pivot]!, aug[col]!];
    // Near-singular: skip this column (produces NaN in solution — consider solveLinearSafe)
    if (Math.abs(aug[col]![col]!) < 1e-12) continue;
    for (let row = col + 1; row < n; row++) {
      const f = aug[row]![col]! / aug[col]![col]!;
      for (let j = col; j <= n; j++) aug[row]![j]! -= f * aug[col]![j]!;
    }
  }
  // Back substitution
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = aug[i]![n]!;
    for (let j = i + 1; j < n; j++) sum -= aug[i]![j]! * (x[j] ?? 0);
    x[i] = sum / aug[i]![i]!;
  }
  return x;
}

/**
 * Safe version of solveLinear that detects and reports singular matrices.
 *
 * @returns {{ solution: number[] | null; singular: boolean }}
 *   - singular: true if the matrix is near-singular (no reliable solution)
 *   - solution: the solution vector if non-singular, null if singular
 */
export function solveLinearSafe(A: number[][], b: number[]): { solution: number[] | null; singular: boolean } {
  const n = A.length;
  if (n === 0) return { solution: [], singular: false };
  const aug = A.map((row, i) => [...row, b[i] ?? 0]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row]![col]!) > Math.abs(aug[pivot]![col]!)) pivot = row;
    }
    [aug[col], aug[pivot]] = [aug[pivot]!, aug[col]!];
    if (Math.abs(aug[col]![col]!) < 1e-12) return { solution: null, singular: true };
    for (let row = col + 1; row < n; row++) {
      const f = aug[row]![col]! / aug[col]![col]!;
      for (let j = col; j <= n; j++) aug[row]![j]! -= f * aug[col]![j]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = aug[i]![n]!;
    for (let j = i + 1; j < n; j++) sum -= aug[i]![j]! * (x[j] ?? 0);
    x[i] = sum / aug[i]![i]!;
  }
  return { solution: x, singular: false };
}

/**
 * Upper-tail probability of the standard normal distribution.
 *
 * Uses Abramowitz & Stegun 7.1.26 rational approximation with
 * maximum absolute error < 1.5 × 10⁻⁷.
 *
 * P(Z > |x|) ≈ φ(x) · t · (a₁ + t·(a₂ + t·(a₃ + t·(a₄ + t·a₅))))
 * where φ(x) = PDF, t = 1/(1 + p·|x|)
 */
export function normalTail(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  return Math.max(
    0,
    0.3989423 * Math.exp(-x * x / 2) * t *
      (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))),
  );
}

/**
 * Error function approximation via the Abramowitz & Stegun formula.
 *
 * erf(x) = 1 - (a₁·t + a₂·t² + a₃·t³ + a₄·t⁴ + a₅·t⁵) × exp(-x²)
 * where t = 1/(1 + p·|x|), p = 0.3275911
 *
 * Maximum absolute error: 1.5 × 10⁻⁷.
 */
export function erf(x: number): number {
  const p = 0.3275911;
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429;
  const sign = x >= 0 ? 1 : -1;
  const t = 1 / (1 + p * Math.abs(x));
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x));
  return sign * y;
}

/**
 * Normal CDF approximation via erf.
 */
export function normalCDF(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

/**
 * Upper-tail CDF complement: P(Z > |x|).
 * Same result as normalTail, using a different derivation (via erf).
 */
export function normalCDFTail(x: number): number {
  return 1 - normalCDF(Math.abs(x));
}

/**
 * Compute the arithmetic mean of a specific column across all rows.
 * Handles NaN and null values by skipping them.
 * Returns NaN for empty data.
 */
export function colMean(data: number[][], col: number): number {
  let sum = 0, n = 0;
  for (const row of data) {
    const v = row[col];
    if (v == null || Number.isNaN(v)) continue;
    sum += v; n++;
  }
  return n > 0 ? sum / n : NaN;
}

/**
 * Seeded pseudo-random number generator (Linear Congruential Generator).
 * Use for reproducible stochastic algorithms (Shapley, bootstrap, etc.).
 *
 * seed = null → uses Math.random() (non-deterministic).
 * seed = number → deterministic reproducible sequence.
 */
export function createRNG(seed: number | null): () => number {
  if (seed == null) return () => Math.random();
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

/**
 * Generate all k-element combinations from an array.
 * Recursive formulation: C(n,k) = first × C(n-1,k-1) ∪ C(n-1,k).
 */
export function combinations<T>(arr: T[], k: number): T[][] {
  if (k === 0) return [[]];
  if (arr.length < k) return [];
  const [first, ...rest] = arr as [T, ...T[]];
  const withFirst = combinations(rest, k - 1).map(c => [first, ...c]);
  const without = combinations(rest, k);
  return [...withFirst, ...without];
}

// ── Fisher's Z Conditional Independence Test ──────────────────────────

/**
 * Fisher's Z conditional independence test.
 *
 * Tests the null hypothesis X_i ⟂ X_j | X_S (conditional independence).
 * Returns a p-value — reject independence if p < alpha.
 *
 * Algorithm:
 *  1. Extract sub-matrix for indices [i, j, ...S]
 *  2. Compute partial correlation ρ_{ij|S}
 *  3. Transform to z = 0.5·ln((1+ρ)/(1-ρ))·√(n-|S|-3)
 *  4. Two-tailed p-value via normal CDF
 *
 * Complexity: O(n·d²) where d = |S| + 2.
 */
export function fisherZTest(
  data: number[][],
  i: number,
  j: number,
  condSet: number[],
): number {
  const n = data.length;
  const indices = [i, j, ...condSet];
  const k = condSet.length;

  // Compute means
  const means = new Array(indices.length).fill(0);
  for (let c = 0; c < indices.length; c++) {
    const ci = indices[c]!;
    let sum = 0;
    for (let r = 0; r < n; r++) sum += data[r]?.[ci] ?? 0;
    means[c] = sum / n;
  }

  // Compute covariance matrix
  const cov = Array.from({ length: indices.length }, () => new Array(indices.length).fill(0));
  for (let a = 0; a < indices.length; a++) {
    const ai = indices[a]!;
    for (let b = a; b < indices.length; b++) {
      const bi = indices[b]!;
      let sum = 0;
      for (let r = 0; r < n; r++)
        sum += ((data[r]?.[ai] ?? 0) - means[a]!) * ((data[r]?.[bi] ?? 0) - means[b]!);
      cov[a]![b] = sum / (n - 1);
      cov[b]![a] = cov[a]![b]!;
    }
  }

  // Partial correlation via precision matrix
  const rho = partialCorrelationFromCov(cov, 0, 1);
  if (Math.abs(rho) >= 1) return 0;

  const z = 0.5 * Math.log((1 + rho) / (1 - rho)) * Math.sqrt(n - k - 3);
  return 2 * (1 - normalCDF(Math.abs(z)));
}

/**
 * Compute partial correlation ρ_{ij|rest} from a covariance matrix.
 * Uses precision (inverse covariance) method.
 *
 * ρ_{ij|rest} = -Ω_{ij} / √(Ω_{ii}·Ω_{jj})
 * where Ω = Σ^{-1} is the precision matrix.
 */
export function partialCorrelationFromCov(
  cov: number[][],
  i: number,
  j: number,
): number {
  const m = cov.length;
  if (m === 2) {
    const cii = cov[i]![i]!;
    const cjj = cov[j]![j]!;
    return cii > 0 && cjj > 0 ? cov[i]![j]! / Math.sqrt(cii * cjj) : 0;
  }
  const prec = invertMatrix(cov);
  const denominator = Math.sqrt(Math.abs(prec[i]![i]! * prec[j]![j]!));
  if (denominator < 1e-12) return 0;
  const r = -prec[i]![j]! / denominator;
  return Math.max(-1, Math.min(1, r));
}

// ── Matrix Inversion (Gauss-Jordan) ──────────────────────────────────

const MATRIX_PIVOT_THRESHOLD = 1e-12;

/**
 * Gauss-Jordan full matrix inversion.
 *
 * Augments A with identity I, then reduces [A|I] → [I|A⁻¹].
 * Partial pivoting for numerical stability.
 *
 * @returns A⁻¹ as number[][] (may be inaccurate if |pivot| < 1e-12)
 */
export function invertMatrix(m: number[][]): number[][] {
  const n = m.length;
  const aug = m.map((row, ri) => [
    ...row,
    ...Array.from({ length: n }, (_, ci) => (ri === ci ? 1 : 0)),
  ]);

  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let pivot = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(aug[row]![col]!) > Math.abs(aug[pivot]![col]!)) pivot = row;
    [aug[col], aug[pivot]] = [aug[pivot]!, aug[col]!];

    const pv = aug[col]![col]!;
    if (Math.abs(pv) < MATRIX_PIVOT_THRESHOLD) continue;

    // Normalize pivot row
    for (let j = col; j < 2 * n; j++) aug[col]![j]! /= pv;

    // Eliminate all other rows
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row]![col]!;
      for (let j = col; j < 2 * n; j++) aug[row]![j]! -= factor * aug[col]![j]!;
    }
  }

  return aug.map(row => row.slice(n));
}

// ── OLS via Normal Equations ────────────────────────────────────────

/**
 * Solve ordinary least squares regression: y ≈ X·β.
 *
 * Uses normal equations: β̂ = (XᵀX)⁻¹ Xᵀy.
 * Suitable for small to moderate number of features (k ≤ 20).
 * Falls back gracefully for singular XᵀX (zero coefficients).
 *
 * @param X — design matrix (n×k), must include column of 1s for intercept
 * @param y — response vector (n)
 * @returns best-fit coefficients β̂ (length k)
 */
export function solveOLS(X: number[][], y: number[]): number[] {
  const n = X.length;
  const k = X[0]?.length ?? 0;
  if (n === 0 || k === 0) return [];

  // XᵀX
  const XtX = Array.from({ length: k }, () => new Float64Array(k));
  const Xty = new Float64Array(k);
  for (let i = 0; i < n; i++) {
    const row = X[i];
    const yi = y[i] ?? 0;
    for (let a = 0; a < k; a++) {
      const xVal = row?.[a] ?? 0;
      Xty[a] = (Xty[a] ?? 0) + xVal * yi;
      for (let b = a; b < k; b++)
        XtX[a]![b] = (XtX[a]![b] ?? 0) + xVal * (row?.[b] ?? 0);
    }
  }
  for (let a = 0; a < k; a++)
    for (let b = 0; b < a; b++)
      XtX[a]![b] = XtX[b]![a]!;

  // Convert to number[][] for solveLinear
  const A = XtX.map(row => Array.from(row) as number[]);
  const b = Array.from(Xty) as number[];
  return solveLinear(A, b);
}

/**
 * Bayesian Information Criterion for linear Gaussian model.
 *
 * BIC = n·ln(RSS/n) + k·ln(n)
 *
 * Lower BIC = better model (penalizes complexity).
 *
 * @param rss — Residual Sum of Squares
 * @param n — sample size
 * @param k — number of parameters
 */
export function bicScore(rss: number, n: number, k: number): number {
  if (n <= 0) return Infinity;
  return n * Math.log(Math.max(1e-10, rss / n)) + k * Math.log(Math.max(2, n));
}
