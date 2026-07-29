/**
 * GsquaredCI — G-squared log-likelihood ratio conditional independence test.
 *
 * Tests H₀: X ⟂ Y | Z using a likelihood ratio approach:
 *   1. Fit restricted model: X ~ Z, Y ~ Z
 *   2. Fit full model: X ~ Y + Z
 *   3. G² = N · log(RSS₀ / RSS₁)
 *   4. Under H₀ (linear-Gaussian): G² ~ χ²(1)
 *
 * For non-Gaussian data, p-values are estimated via permutation testing.
 *
 * @packageDocumentation
 */

import {
  chiSquareCDF,
  createRNG,
  type CITestResult,
} from '@agentix-e/causality-analyzer-core';

/** Default number of permutations for non-parametric p-value */
const DEFAULT_N_PERMUTATIONS = 500;

/**
 * Configuration for the G-squared CI test.
 */
export interface GsquaredConfig {
  /** Number of permutations for p-value estimation (default: 500, 0 = use χ² approximation) */
  nPermutations?: number;
  /** Random seed for permutation reproducibility */
  seed?: number;
}

/**
 * Run a G-squared conditional independence test.
 *
 * Tests H₀: X ⟂ Y | Z using the G² likelihood ratio statistic.
 *
 * For the linear-Gaussian case, p-values are computed from the χ²(1)
 * distribution. For permutation mode (nPermutations > 0), the null
 * distribution is built by shuffling Y.
 *
 * @param data - (n × totalCols) design matrix
 * @param xCol - column index for X
 * @param yCol - column index for Y
 * @param condCols - column indices for Z
 * @param config - permutation and seed settings
 * @returns CITestResult with p-value and G² statistic
 */
export function gsquaredCITest(
  data: number[][],
  xCol: number,
  yCol: number,
  condCols: number[],
  config: GsquaredConfig = {},
): CITestResult {
  const n = data.length;
  if (n < condCols.length + 5) {
    return { pValue: 1, testStatistic: 0 };
  }

  const nPermutations = config.nPermutations ?? DEFAULT_N_PERMUTATIONS;

  // Compute observed G²
  const g2 = computeG2(data, xCol, yCol, condCols);
  if (!isFinite(g2) || g2 <= 0) {
    return { pValue: 1, testStatistic: 0 };
  }

  let pValue: number;

  if (nPermutations > 0) {
    // Permutation-based p-value
    const rng = createRNG(config.seed ?? Date.now());
    const yValues = data.map(row => row[yCol]);
    let countGreater = 0;

    for (let p = 0; p < nPermutations; p++) {
      // Shuffle Y
      const shuffled = [...yValues];
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }

      const permData = data.map((row, idx) => {
        const nr = [...row];
        nr[yCol] = shuffled[idx]!;
        return nr;
      });

      const permG2 = computeG2(permData, xCol, yCol, condCols);
      if (permG2 >= g2) countGreater++;
    }

    pValue = (countGreater + 1) / (nPermutations + 1);
  } else {
    // χ²(1) approximation (linear-Gaussian assumption)
    pValue = chiSquareCDF(g2, 1);
  }

  return { pValue, testStatistic: g2 };
}

/**
 * Compute G² = N · log(RSS₀ / RSS₁) for testing X ⟂ Y | Z.
 *
 * M₀: X ~ Z, Y ~ Z  → RSS₀ = RSS(X ~ Z) + RSS(Y ~ Z)
 * M₁: X ~ Y + Z     → RSS₁ = RSS(X ~ Y + Z) + RSS(Y ~ Z)
 *
 * Note: We keep RSS(Y ~ Z) in both models, so:
 *   G² = N · log(RSS(X ~ Z) / RSS(X ~ Y + Z))
 *
 * @internal
 */
function computeG2(
  data: number[][],
  xCol: number,
  yCol: number,
  condCols: number[],
): number {
  const n = data.length;

  // RSS₀: X ~ Z
  const rss0 = computeRSS(data, xCol, condCols);

  // RSS₁: X ~ Y + Z
  const fullCond = [yCol, ...condCols];
  const rss1 = computeRSS(data, xCol, fullCond);

  if (rss1 <= 0 || rss0 <= 0) return 0;
  if (rss1 >= rss0) return 0; // Model didn't improve

  return n * Math.log(rss0 / rss1);
}

/**
 * Compute residual sum of squares for regression of targetCol on condCols.
 *
 * Uses normal equations (X'X)β = X'y with Gauss-Jordan solve.
 *
 * @internal
 */
function computeRSS(
  data: number[][],
  targetCol: number,
  condCols: number[],
): number {
  const n = data.length;
  const k = condCols.length;

  if (k === 0) {
    // No predictors: RSS = total sum of squares around the mean
    let sum = 0, sum2 = 0;
    for (let i = 0; i < n; i++) {
      const v = data[i][targetCol];
      sum += v;
      sum2 += v * v;
    }
    const mean = sum / n;
    return sum2 - n * mean * mean;
  }

  // X'X and X'y
  const xtx: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  const xty: number[] = new Array(k).fill(0) as number[];

  for (let row = 0; row < n; row++) {
    const rd = data[row];
    for (let ci = 0; ci < k; ci++) {
      const xv = rd[condCols[ci]];
      xty[ci] += xv * rd[targetCol];
      for (let cj = 0; cj <= ci; cj++) {
        xtx[ci][cj] += xv * rd[condCols[cj]];
      }
    }
  }
  for (let ci = 0; ci < k; ci++) {
    for (let cj = ci + 1; cj < k; cj++) {
      xtx[cj][ci] = xtx[ci][cj]!;
    }
  }

  // Gauss-Jordan solve
  const aug: number[][] = Array.from({ length: k }, (_, r) => [...xtx[r], xty[r]]);
  for (let col = 0; col < k; col++) {
    let maxRow = col;
    let maxV = Math.abs(aug[col][col]);
    for (let r = col + 1; r < k; r++) {
      const v = Math.abs(aug[r][col]);
      if (v > maxV) { maxV = v; maxRow = r; }
    }
    if (maxV < 1e-12) continue;
    if (maxRow !== col) [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    const piv = aug[col][col];
    for (let c = col; c <= k; c++) aug[col][c] /= piv;
    for (let r = 0; r < k; r++) {
      if (r === col) continue;
      const f = aug[r][col];
      if (f === 0) continue;
      for (let c = col; c <= k; c++) aug[r][c] -= f * aug[col][c];
    }
  }

  const beta: number[] = new Array(k).fill(0) as number[];
  for (let ci = 0; ci < k; ci++) beta[ci] = aug[ci][k]!;

  // Compute RSS = Σ(y - ŷ)²
  let rss = 0;
  for (let row = 0; row < n; row++) {
    const rd = data[row];
    let yh = 0;
    for (let ci = 0; ci < k; ci++) yh += beta[ci] * rd[condCols[ci]];
    const e = rd[targetCol] - yh;
    rss += e * e;
  }
  return rss;
}
