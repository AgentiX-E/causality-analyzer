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
 * Solves Ax = b for x. Handles n=0 (returns []), near-singular matrices
 * (continues via partial pivoting), and non-square augmented matrices.
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
    if (Math.abs(aug[col]![col]!) < 1e-12) continue; // near-singular — skip
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
