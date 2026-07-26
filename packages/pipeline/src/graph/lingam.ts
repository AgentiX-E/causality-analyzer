/**
 * DirectLiNGAM — Linear Non-Gaussian Acyclic Model for causal discovery.
 *
 * This implementation is a faithful port of the official Python version,
 * addressing fundamental flaws:
 *   - Replaced HSIC/Kendall's tau with correct pwling dependence measure.
 *   - Replaced naive fixed-threshold OLS with BIC-based backward elimination
 *     (approximating official Adaptive Lasso pruning).
 *   - Fixed diffMutualInfo to use std-division (not full standardization)
 *     for residual entropy, matching official `ri_j / np.std(ri_j)`.
 *
 * Reference: Shimizu et al. (2011). "DirectLiNGAM." JMLR 12:1225-1248.
 * Official Source: https://github.com/cdt15/lingam
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';

// ── pwling helpers ──────────────────────────────────────────────────

function entropy(u: Float64Array): number {
  const k1 = 79.047, k2 = 7.4129, gamma = 0.37457;
  const n = u.length;
  let logCoshSum = 0, uExpSum = 0;
  for (let i = 0; i < n; i++) {
    logCoshSum += Math.log(Math.cosh(u[i]));
    uExpSum += u[i] * Math.exp(-(u[i] ** 2) / 2);
  }
  const term1 = (1 + Math.log(2 * Math.PI)) / 2;
  const term2 = k1 * (logCoshSum / n - gamma) ** 2;
  const term3 = k2 * (uExpSum / n) ** 2;
  return term1 - term2 - term3;
}

/**
 * Matches official: `entropy(ri_j / np.std(ri_j))`
 * NOT full standardization (no centering).
 */
function scaleByStd(x: Float64Array): Float64Array {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let varianceValue = 0;
  for (let i = 0; i < n; i++) varianceValue += (x[i] - mean) ** 2;
  const std = Math.sqrt(varianceValue / n);
  const result = new Float64Array(n);
  const invStd = std > 1e-10 ? 1 / std : 1;
  for (let i = 0; i < n; i++) result[i] = x[i] * invStd;
  return result;
}

function diffMutualInfo(
  xi_std: Float64Array, xj_std: Float64Array,
  ri_j: Float64Array, rj_i: Float64Array,
): number {
  const scaled_ri_j = scaleByStd(ri_j);
  const scaled_rj_i = scaleByStd(rj_i);
  return (entropy(xj_std) + entropy(scaled_ri_j))
       - (entropy(xi_std) + entropy(scaled_rj_i));
}

function standardize(x: Float64Array): Float64Array {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let varianceValue = 0;
  for (let i = 0; i < n; i++) varianceValue += (x[i] - mean) ** 2;
  const std = Math.sqrt(varianceValue / n);
  const invStd = std > 1e-10 ? 1 / std : 1;
  const result = new Float64Array(n);
  for (let i = 0; i < n; i++) result[i] = (x[i] - mean) * invStd;
  return result;
}

function residual(xi: Float64Array, xj: Float64Array): Float64Array {
  const n = xi.length;
  const cov = covariance(xi, xj);
  const varXj = variance(xj);
  const b = cov / varXj;
  const result = new Float64Array(n);
  for (let i = 0; i < n; i++) result[i] = xi[i] - b * xj[i];
  return result;
}

function covariance(x: Float64Array, y: Float64Array): number {
  const n = x.length;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  let cov = 0;
  for (let i = 0; i < n; i++) cov += (x[i] - mx) * (y[i] - my);
  return cov / n;
}

function variance(x: Float64Array): number {
  const n = x.length;
  let m = 0;
  for (let i = 0; i < n; i++) m += x[i];
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - m) ** 2;
  return v / n;
}

// ── BIC-based backward elimination (approximates Adaptive Lasso) ────

/**
 * Selects predictors via BIC-based backward elimination.
 * Approximates official `predict_adaptive_lasso` which uses
 * `LassoLarsIC(criterion="bic")`.
 *
 * Unlike the naive `abs(b) > 1e-4` threshold, this aggressively
 * prunes edges that don't improve the BIC score.
 */
function prunedOLS(
  X: Float64Array[],
  predictors: number[],
  target: number,
): Map<number, number> {
  const n = X[0].length;
  if (predictors.length === 0) return new Map();

  // Start with all predictors
  let active = [...predictors];

  // Fit OLS and compute BIC
  const computeBIC = (activePreds: number[]): [number, Map<number, number>] => {
    const k = activePreds.length + 1; // predictors + intercept
    if (k >= n) return [Infinity, new Map()];

    // OLS: solve Y = X_preds @ beta + intercept
    const Y = X[target];
    const p = activePreds.length;

    // Build X matrix with intercept (column of 1s)
    const Xmat: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [1]; // intercept
      for (const pred of activePreds) row.push(X[pred][i]);
      Xmat.push(row);
    }

    // X^T X
    const cols = p + 1;
    const XtX = new Float64Array(cols * cols);
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < cols; j++) {
        let sum = 0;
        for (let r = 0; r < n; r++) sum += Xmat[r][i] * Xmat[r][j];
        XtX[i * cols + j] = sum;
      }
    }

    // X^T Y
    const XtY = new Float64Array(cols);
    for (let i = 0; i < cols; i++) {
      let sum = 0;
      for (let r = 0; r < n; r++) sum += Xmat[r][i] * Y[r];
      XtY[i] = sum;
    }

    // Solve via Gaussian elimination
    const beta = solveLinearSystem(XtX, XtY, cols);
    if (!beta) return [Infinity, new Map()];

    // Compute RSS
    let rss = 0;
    for (let r = 0; r < n; r++) {
      let pred = beta[0]; // intercept
      for (let j = 0; j < p; j++) pred += beta[j + 1] * Xmat[r][j + 1];
      rss += (Y[r] - pred) ** 2;
    }

    const bic = n * Math.log(rss / n) + k * Math.log(n);

    // Extract coefficients (excluding intercept)
    const coefs = new Map<number, number>();
    for (let j = 0; j < p; j++) {
      coefs.set(activePreds[j], beta[j + 1]);
    }
    return [bic, coefs];
  };

  let [bestBIC, bestCoefs] = computeBIC(active);
  if (!isFinite(bestBIC)) return new Map();

  // Backward elimination
  let improved = true;
  while (improved && active.length > 0) {
    improved = false;
    for (let i = active.length - 1; i >= 0; i--) {
      const candidate = active.filter((_, j) => j !== i);
      const [candidateBIC, candidateCoefs] = computeBIC(candidate);
      if (candidateBIC < bestBIC) {
        bestBIC = candidateBIC;
        bestCoefs = candidateCoefs;
        active = candidate;
        improved = true;
        break;
      }
    }
  }

  return bestCoefs;
}

function solveLinearSystem(A: Float64Array, b: Float64Array, n: number): Float64Array | null {
  const aug = new Float64Array(n * (n + 1));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug[i * (n + 1) + j] = A[i * n + j];
    aug[i * (n + 1) + n] = b[i];
  }
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(aug[row * (n + 1) + col]) > Math.abs(aug[pivot * (n + 1) + col])) pivot = row;
    if (pivot !== col)
      for (let j = 0; j <= n; j++) {
        const tmp = aug[col * (n + 1) + j]; aug[col * (n + 1) + j] = aug[pivot * (n + 1) + j]; aug[pivot * (n + 1) + j] = tmp;
      }
    const pv = aug[col * (n + 1) + col];
    if (Math.abs(pv) < 1e-14) return null;
    for (let j = 0; j <= n; j++) aug[col * (n + 1) + j] /= pv;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const f = aug[row * (n + 1) + col];
      for (let j = 0; j <= n; j++) aug[row * (n + 1) + j] -= f * aug[col * (n + 1) + j];
    }
  }
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = aug[i * (n + 1) + n];
  return x;
}

// ── Main algorithm ──────────────────────────────────────────────────

export function directLiNGAM(
  data: Matrix,
  nodeNames: string[],
): {
  graph: CausalGraph;
  weights: Map<string, Map<string, number>>;
  order: string[];
} {
  const d = nodeNames.length;
  const N = data.rows;

  if (N < 2 || d < 2) {
    const g = new CausalGraph(nodeNames);
    return { graph: g, weights: new Map(), order: [...nodeNames] };
  }

  const X_cols = nodeNames.map((_, i) => data.getColumn(i));
  const X_original = X_cols.map(col => Float64Array.from(col));
  const X = X_original.map(col => new Float64Array(col));

  let U = Array.from({ length: d }, (_, i) => i);
  const K: number[] = [];
  const X_ = X.map(col => new Float64Array(col));

  // Causal order search via pwling
  for (let step = 0; step < d; step++) {
    const M_list: number[] = [];
    for (const i of U) {
      let M = 0;
      for (const j of U) {
        if (i !== j) {
          const xi_std = standardize(X_[i]);
          const xj_std = standardize(X_[j]);
          const ri_j = residual(xi_std, xj_std);
          const rj_i = residual(xj_std, xi_std);
          M += Math.min(0, diffMutualInfo(xi_std, xj_std, ri_j, rj_i)) ** 2;
        }
      }
      M_list.push(-1.0 * M);
    }
    const m = U[M_list.indexOf(Math.max(...M_list))];
    for (const i of U) {
      if (i !== m) {
        X_[i] = residual(X_[i], X_[m]);
      }
    }
    K.push(m);
    U = U.filter(u => u !== m);
  }

  const order = K.map(i => nodeNames[i]);

  // Estimate adjacency matrix via BIC-pruned OLS (approximates Adaptive Lasso)
  const weights = new Map<string, Map<string, number>>();
  for (let i = 1; i < d; i++) {
    const target = K[i];
    const predictors = K.slice(0, i);
    if (predictors.length === 0) continue;

    const child = order[i];
    const coefs = prunedOLS(X_original, predictors, target);
    if (coefs.size > 0) {
      const childMap = new Map<string, number>();
      for (const [predIdx, coef] of coefs.entries()) {
        const parent = nodeNames[predIdx];
        childMap.set(parent, coef);
      }
      weights.set(child, childMap);
    }
  }

  const g = new CausalGraph(nodeNames);
  for (const [child, parents] of weights.entries()) {
    for (const [parent] of parents.entries()) {
      g.addEdge(parent, child);
    }
  }

  return { graph: g, weights, order };
}
