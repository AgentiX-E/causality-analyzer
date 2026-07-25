/**
 * RESIT — REgression with Subsequent Independence Test.
 *
 * Pairwise causal direction inference for nonlinear relationships.
 * Tests whether X causes Y or Y causes X by comparing the
 * independence of residuals from regressions in both directions.
 *
 * Algorithm (Peters et al., 2014 / Mooij et al., 2016):
 *  1. Direction X→Y: fit Y ≈ f(X) via polynomial regression,
 *     compute residuals ε_X→Y. Test ε_X→Y ⟂ X.
 *  2. Direction Y→X: fit X ≈ g(Y) via polynomial regression,
 *     compute residuals ε_Y→X. Test ε_Y→X ⟂ Y.
 *  3. Choose direction with larger p-value (more independent).
 *
 * References:
 *  - Peters, Mooij, Janzing & Schölkopf (2014).
 *    "Causal discovery with continuous additive noise models."
 *    JMLR 15:2009-2053.
 *  - Mooij, Peters, Janzing, Zscheischler & Schölkopf (2016).
 *    "Distinguishing cause from effect using observational data."
 *    JMLR 17(32):1-102.
 *
 * @packageDocumentation
 */
import { fisherZTest } from '@agentix-e/causality-analyzer-core';

export interface RESITResult {
  /** Inferred causal direction: "X→Y" or "Y→X" or "uncertain" */
  direction: 'X→Y' | 'Y→X' | 'uncertain';
  /** p-value for X→Y (independence of residuals) */
  pValueXY: number;
  /** p-value for Y→X */
  pValueYX: number;
  /** Confidence score: |log10(pYX) - log10(pXY)| */
  confidence: number;
}

export interface RESITConfig {
  /** Polynomial degree for regression */
  degree?: number;
}

/**
 * Test causal direction between two variables using RESIT.
 *
 * @param X — first variable
 * @param Y — second variable
 * @returns direction, p-values, and confidence
 */
export function resitTest(
  X: number[],
  Y: number[],
  config: RESITConfig = {},
): RESITResult {
  const degree = config.degree ?? 3;
  const n = X.length;

  if (n < 10) return { direction: 'uncertain', pValueXY: 1, pValueYX: 1, confidence: 0 };

  // Direction 1: X → Y — fit polynomial Y = f(X) + ε
  const pValueXY = testDirection(X, Y, degree, n);

  // Direction 2: Y → X — fit polynomial X = g(Y) + ε
  const pValueYX = testDirection(Y, X, degree, n);

  let direction: 'X→Y' | 'Y→X' | 'uncertain';
  if (pValueXY > pValueYX) direction = 'X→Y';
  else if (pValueYX > pValueXY) direction = 'Y→X';
  else direction = 'uncertain';

  const confidence = Math.abs(Math.log10(Math.max(1e-10, pValueYX)) - Math.log10(Math.max(1e-10, pValueXY)));

  return { direction, pValueXY, pValueYX, confidence };
}

/**
 * Test whether Y can be explained by polynomial of X.
 * Returns p-value of independence test between residuals and X.
 * Higher p-value = more independent = more likely X causes Y.
 */
function testDirection(X: number[], Y: number[], degree: number, n: number): number {
  // Polynomial basis: [1, x, x², ..., x^degree]
  const Xpoly: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [1]; // intercept
    let pow = 1;
    for (let d = 0; d < degree; d++) {
      pow *= X[i]!;
      row.push(pow);
    }
    Xpoly.push(row);
  }

  // OLS: minimize ||Y - X·β||²
  const k = degree + 1; // number of basis functions
  const XtX = Array.from({ length: k }, () => new Float64Array(k));
  const Xty = new Float64Array(k);

  for (let i = 0; i < n; i++) {
    for (let a = 0; a < k; a++) {
      Xty[a] += (Xpoly[i]![a] ?? 0) * Y[i]!;
      for (let b = 0; b < k; b++) {
        XtX[a]![b] += (Xpoly[i]![a] ?? 0) * (Xpoly[i]![b] ?? 0);
      }
    }
  }

  const beta = solveOLS(XtX, Xty, k);

  // Compute residuals
  const residuals: number[] = [];
  for (let i = 0; i < n; i++) {
    let pred = 0;
    for (let b = 0; b < k; b++) pred += (beta[b] ?? 0) * (Xpoly[i]![b] ?? 0);
    residuals.push(Y[i]! - pred);
  }

  // Independence test: residuals ⟂ X via Fisher Z on (X, residuals)
  const dataMatrix = X.map((x, i) => [x, residuals[i]!]);
  return fisherZTest(dataMatrix, 0, 1, []);
}

// ── OLS Solver ─────────────────────────────────────────────────────

function solveOLS(XtX: Float64Array[], Xty: Float64Array, n: number): Float64Array {
  const XtXArr = XtX.map(r => Array.from(r));
  const XtyArr = Array.from(Xty);

  const aug = XtXArr.map((row, i) => [...row, XtyArr[i] ?? 0]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(aug[row]![col]!) > Math.abs(aug[pivot]![col]!)) pivot = row;
    if (Math.abs(aug[pivot]![col]!) < 1e-14) continue;
    [aug[col], aug[pivot]] = [aug[pivot]!, aug[col]!];
    const pv = aug[col]![col]!;
    for (let j = col; j <= n; j++) aug[col]![j]! /= pv;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const f = aug[row]![col]!;
      for (let j = col; j <= n; j++) aug[row]![j]! -= f * aug[col]![j]!;
    }
  }
  return new Float64Array(aug.map(r => r[n]!));
}
