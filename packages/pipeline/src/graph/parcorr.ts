/**
 * ParCorr — Partial correlation conditional independence test.
 *
 * Wraps the Fisher Z transform test from the core package, providing
 * a standardized CITestResult interface for use with PCMCI+.
 *
 * This is the fastest CI backend (O(n) per test with cached correlation
 * matrix) and the default choice for linear-Gaussian data.
 *
 * @packageDocumentation
 */

import { fisherZTest, type CITestResult } from '@agentix-e/causality-analyzer-core';

/**
 * Run a partial correlation conditional independence test.
 *
 * Tests H₀: X ⟂ Y | Z using Fisher's Z transform.
 *
 * @param data - (n × totalCols) design matrix as a flat number[][]
 * @param xCol - column index for variable X
 * @param yCol - column index for variable Y
 * @param condCols - column indices for conditioning set Z (empty for unconditional)
 * @param corrMatrix - precomputed correlation matrix for performance (optional)
 * @returns CITestResult with p-value and |partial correlation|
 */
export function parCorrTest(
  data: number[][],
  xCol: number,
  yCol: number,
  condCols: number[],
  corrMatrix?: number[][],
): CITestResult {
  // fisherZTest from core already handles the full computation
  const pValue = fisherZTest(
    data,
    xCol,
    yCol,
    condCols,
    corrMatrix,
  );

  // Compute the absolute partial correlation as the test statistic
  // (Fisher Z transform is Z = 0.5 * ln((1+r)/(1-r)), invert to get |r|)
  // For simplicity, we approximate: r = tanh(Z), but since fisherZTest
  // returns the p-value directly, we compute |partial rho| separately.
  const rho = Math.abs(partialCorrelationEstimate(data, xCol, yCol, condCols));

  return { pValue, testStatistic: rho };
}

/**
 * Estimate the absolute partial correlation coefficient for the test
 * statistic. Uses OLS residual correlation (same method as core's
 * partialCorrelationRaw, but returns absolute value for display).
 *
 * @internal
 */
function partialCorrelationEstimate(
  data: number[][],
  xCol: number,
  yCol: number,
  condCols: number[],
): number {
  if (condCols.length === 0) {
    return Math.abs(simpleCorrelation(data, xCol, yCol));
  }
  return Math.abs(partialViaResiduals(data, xCol, yCol, condCols));
}

/**
 * Simple Pearson correlation between two columns.
 *
 * @internal
 */
function simpleCorrelation(data: number[][], a: number, b: number): number {
  const n = data.length;
  let sumA = 0, sumB = 0, sumAB = 0, sumA2 = 0, sumB2 = 0;
  for (let i = 0; i < n; i++) {
    const va = data[i][a];
    const vb = data[i][b];
    sumA += va; sumB += vb;
    sumAB += va * vb; sumA2 += va * va; sumB2 += vb * vb;
  }
  const denom = Math.sqrt((n * sumA2 - sumA * sumA) * (n * sumB2 - sumB * sumB));
  return denom === 0 ? 0 : (n * sumAB - sumA * sumB) / denom;
}

/**
 * Partial correlation via OLS residual correlation.
 *
 * Regresses both X and Y on Z, then correlates the residuals.
 *
 * @internal
 */
function partialViaResiduals(
  data: number[][],
  xCol: number,
  yCol: number,
  condCols: number[],
): number {
  const n = data.length;
  const k = condCols.length;
  if (k === 0) return simpleCorrelation(data, xCol, yCol);

  // Build OLS residuals for X ~ Z
  const rX = olsResiduals(data, xCol, condCols);
  const rY = olsResiduals(data, yCol, condCols);

  // Correlation of residuals
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
  for (let i = 0; i < n; i++) {
    const x = rX[i];
    const y = rY[i];
    sumX += x; sumY += y;
    sumXY += x * y; sumX2 += x * x; sumY2 += y * y;
  }
  const denom = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
  return denom === 0 ? 0 : (n * sumXY - sumX * sumY) / denom;
}

/**
 * Compute OLS residuals by regressing targetCol on condCols.
 *
 * @internal
 */
function olsResiduals(
  data: number[][],
  targetCol: number,
  condCols: number[],
): number[] {
  const n = data.length;
  const k = condCols.length;

  // X'X and X'y via normal equations
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

  const res: number[] = new Array(n);
  for (let row = 0; row < n; row++) {
    const rd = data[row];
    let yh = 0;
    for (let ci = 0; ci < k; ci++) yh += beta[ci] * rd[condCols[ci]];
    res[row] = rd[targetCol] - yh;
  }
  return res;
}
