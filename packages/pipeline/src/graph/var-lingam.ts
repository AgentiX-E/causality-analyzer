/**
 * VAR-LiNGAM — Vector Autoregressive LiNGAM for time-series causal discovery.
 *
 * Combines a VAR (Vector Autoregressive) model for lagged effects
 * with LiNGAM on the residuals for instantaneous causal structure.
 * This captures both temporal (lagged) and contemporaneous (same-time)
 * causal relationships.
 *
 * Algorithm (Hyvärinen et al., 2010):
 *  1. Fit VAR(p) to estimate lagged coefficient matrices B_τ
 *  2. Extract residuals e[t] = X[t] - Σ B_τ · X[t-τ]
 *  3. Apply DirectLiNGAM to residuals for instantaneous graph B₀
 *  4. Return combined causal structure
 *
 * References:
 *  - Hyvärinen, Zhang, Shimizu & Hoyer (2010).
 *    "Estimation of a structural vector autoregression model using
 *     non-Gaussianity." JMLR 11:1709-1731.
 *
 * @packageDocumentation
 */
import { CausalGraph } from './causal-graph.js';
import { directLiNGAM } from './lingam.js';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { Matrix } from 'ml-matrix';

export interface VARLiNGAMConfig {
  /** AR order (number of lags) */
  maxLag?: number;
  /** Significance threshold for edge selection */
  threshold?: number;
}

export interface VARLiNGAMResult {
  /** Instantaneous causal graph (same-time) */
  instantaneousGraph: CausalGraph;
  /** Instantaneous weight matrix B₀ */
  B0: Float64Array;
  /** Lagged coefficient matrices: B_τ[i,j] = effect from j[t-τ] to i[t] */
  laggedMatrices: Float64Array[];
  /** Causal order from LiNGAM */
  order: string[];
}

/**
 * Run VAR-LiNGAM on time-series data.
 *
 * @param data — (T × d) matrix
 * @param nodeNames — variable names
 */
export function varLingam(
  data: number[][],
  nodeNames: string[],
  config: VARLiNGAMConfig = {},
  domainKnowledge?: DomainKnowledge,
): VARLiNGAMResult {
  const maxLag = config.maxLag ?? Math.min(5, Math.floor(data.length / 20));
  const threshold = config.threshold ?? 0.1;
  const T = data.length;
  const d = nodeNames.length;

  if (T < maxLag + 5 || d < 2) {
    return {
      instantaneousGraph: new CausalGraph([...nodeNames]),
      B0: new Float64Array(d * d),
      laggedMatrices: [],
      order: [...nodeNames],
    };
  }

  // ── Phase 1: Fit VAR by OLS per variable ──
  // For each variable j, fit: X[t,j] = Σ_{i,τ} B_τ[j,i] · X[t-τ,i] + e_j[t]
  const laggedMatrices: Float64Array[] = [];
  for (let tau = 0; tau < maxLag; tau++) {
    laggedMatrices.push(new Float64Array(d * d));
  }

  // Build design matrix: rows (T - maxLag), cols (d * maxLag + 1 for intercept)
  const effT = T - maxLag;
  const X = new Matrix(effT, d * maxLag + 1);
  for (let t = 0; t < effT; t++) {
    X.set(t, 0, 1); // intercept
    for (let j = 0; j < d; j++) {
      for (let tau = 0; tau < maxLag; tau++) {
        X.set(t, 1 + j * maxLag + tau, data[t + maxLag - 1 - tau]![j]!);
      }
    }
  }

  // Fit OLS for each target variable
  const residuals: number[][] = [];
  for (let j = 0; j < d; j++) {
    const y: number[] = [];
    for (let t = 0; t < effT; t++) y.push(data[t + maxLag]![j]!);

    // OLS via normal equations: beta = (X^T X)^{-1} X^T y
    const k = X.columns;
    const XtX = new Float64Array(k * k);
    const Xty = new Float64Array(k);
    for (let t = 0; t < effT; t++) {
      for (let i = 0; i < k; i++) {
        Xty[i] += X.get(t, i) * y[t]!;
        for (let l = 0; l < k; l++) {
          XtX[i * k + l] += X.get(t, i) * X.get(t, l);
        }
      }
    }

    const beta = solveLinear(XtX, Xty, k);
    // Store lagged coefficients
    for (let i = 0; i < d; i++) {
      for (let tau = 0; tau < maxLag; tau++) {
        const coeff = beta[1 + i * maxLag + tau] ?? 0;
        if (Math.abs(coeff) > threshold) {
          laggedMatrices[tau]![j * d + i] = coeff;
        }
      }
    }

    // Compute residuals
    const res: number[] = [];
    for (let t = 0; t < effT; t++) {
      let pred = beta[0] ?? 0;
      for (let c = 1; c < k; c++) pred += (beta[c] ?? 0) * X.get(t, c);
      res.push(y[t]! - pred);
    }
    residuals.push(res);
  }

  // ── Phase 2: DirectLiNGAM on residuals for instantaneous structure ──
  const resData = new Matrix(effT, d);
  for (let t = 0; t < effT; t++)
    for (let j = 0; j < d; j++)
      resData.set(t, j, residuals[j]![t]!);

  const lingamResult = directLiNGAM(resData, nodeNames);
  const B0 = new Float64Array(d * d);
  const weights = lingamResult.weights;
  for (const [child, parentMap] of weights) {
    const childIdx = nodeNames.indexOf(child);
    for (const [parent, w] of parentMap) {
      const parentIdx = nodeNames.indexOf(parent);
      B0[childIdx * d + parentIdx] = w;
    }
  }

  if (domainKnowledge) lingamResult.graph.applyDomainKnowledge(domainKnowledge);

  return {
    instantaneousGraph: lingamResult.graph,
    B0,
    laggedMatrices,
    order: lingamResult.order,
  };
}

// ── Linear Solver ──────────────────────────────────────────────────

function solveLinear(XtX: Float64Array, Xty: Float64Array, n: number): Float64Array {
  const aug = new Float64Array(n * (n + 1));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug[i * (n + 1) + j] = XtX[i * n + j]!;
    aug[i * (n + 1) + n] = Xty[i]!;
  }
  const cols = n + 1;
  for (let k = 0; k < n; k++) {
    let pivot = k;
    for (let i = k + 1; i < n; i++)
      if (Math.abs(aug[i * cols + k]!) > Math.abs(aug[pivot * cols + k]!)) pivot = i;
    if (Math.abs(aug[pivot * cols + k]!) < 1e-14) continue;
    if (pivot !== k)
      for (let j = 0; j < cols; j++) {
        const tmp = aug[k * cols + j]!; aug[k * cols + j] = aug[pivot * cols + j]!; aug[pivot * cols + j] = tmp;
      }
    const pv = aug[k * cols + k]!;
    for (let j = k; j < cols; j++) aug[k * cols + j] /= pv;
    for (let i = 0; i < n; i++) {
      if (i === k) continue;
      const f = aug[i * cols + k]!;
      for (let j = k; j < cols; j++) aug[i * cols + j] -= f * aug[k * cols + j]!;
    }
  }
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = aug[i * cols + n]!;
  return x;
}
