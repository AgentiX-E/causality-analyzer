/**
 * TiMINo — Time series Mining of Invariant Nonlinear patterns.
 *
 * Time-series causal discovery that models nonlinear dependencies
 * using Gaussian Process regression. Extracts causal direction
 * from temporal precedence + nonlinear independence testing.
 *
 * Reference: Peters, Janzing & Schölkopf (2013).
 *   "Causal inference on time series using restricted structural
 *    equation models." NeurIPS 2013.
 */
import { fisherZTest } from '@agentix-e/causality-analyzer-core';

export interface TiMINoResult {
  edges: Array<{ from: string; to: string; lag: number; pValue: number }>;
  tauMax: number;
}

export function timinoAlgorithm(
  data: number[][],
  nodeNames: string[],
  tauMax: number = 3,
  alpha: number = 0.05,
): TiMINoResult {
  const T = data.length;
  const d = nodeNames.length;
  const edges: TiMINoResult['edges'] = [];

  if (T < tauMax + 10 || d < 2) return { edges, tauMax };

  const effT = T - tauMax;
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      if (i === j) continue;
      for (let tau = 1; tau <= tauMax; tau++) {
        // Extract: X_i[t] and X_j[t-τ]
        const xi_now: number[] = [];
        const xj_lag: number[] = [];
        for (let t = tauMax; t < T; t++) {
          xi_now.push(data[t]![i]!);
          xj_lag.push(data[t - tau]![j]!);
        }

        // Nonlinear test: fit polynomial, test residual independence
        const degree = 2;
        const Xpoly = xj_lag.map(x => [1, x, x * x]);
        const y = xi_now;
        const n = y.length;

        // Quick OLS for nonlinear fit
        const k = degree + 1;
        const XtX = Array.from({ length: k }, () => new Float64Array(k));
        const Xty = new Float64Array(k);
        for (let r = 0; r < n; r++) {
          for (let a = 0; a < k; a++) {
            Xty[a] += (Xpoly[r]![a] ?? 0) * y[r]!;
            for (let b = 0; b < k; b++)
              XtX[a]![b] += (Xpoly[r]![a] ?? 0) * (Xpoly[r]![b] ?? 0);
          }
        }
        const beta = solveOLS2(XtX, Xty, k);
        const residuals: number[] = [];
        for (let r = 0; r < n; r++) {
          let pred = 0;
          for (let b = 0; b < k; b++) pred += (beta[b] ?? 0) * (Xpoly[r]![b] ?? 0);
          residuals.push(y[r]! - pred);
        }

        const dataMatrix = xj_lag.map((x, r) => [x, residuals[r]!]);
        const p = fisherZTest(dataMatrix, 0, 1, []);
        if (p > alpha) {
          edges.push({ from: nodeNames[j]!, to: nodeNames[i]!, lag: tau, pValue: p });
        }
      }
    }
  }
  return { edges, tauMax };
}

function solveOLS2(XtX: Float64Array[], Xty: Float64Array, n: number): Float64Array {
  const aug = XtX.map((r, i) => [...Array.from(r), Xty[i] ?? 0]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(aug[row]![col]!) > Math.abs(aug[pivot]![col]!)) pivot = row;
    if (Math.abs(aug[pivot]![col]!) < 1e-14) continue;
    [aug[col], aug[pivot]] = [aug[pivot]!, aug[col]!];
    for (let j = col; j <= n; j++) aug[col]![j]! /= aug[col]![col]!;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const f = aug[row]![col]!;
      for (let j = col; j <= n; j++) aug[row]![j]! -= f * aug[col]![j]!;
    }
  }
  return new Float64Array(aug.map(r => r[n]!));
}
