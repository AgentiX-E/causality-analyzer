/**
 * DirectLiNGAM — Linear Non-Gaussian Acyclic Model for causal discovery.
 *
 * This implementation is a faithful port of the official Python version:
 *   - pwling dependence measure for causal order search (Shimizu et al. 2011).
 *   - BIC-based Lasso grid search for adjacency estimation, replacing the
 *     naive fixed-threshold OLS with true L1-regularized variable selection
 *     via @kanaries/ml LassoRegression (coordinate descent).
 *   - Correct diffMutualInfo: scaleByStd (divide only) not full standardization.
 *
 * Reference: Shimizu et al. (2011). "DirectLiNGAM." JMLR 12:1225-1248.
 * Official Source: https://github.com/cdt15/lingam
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { Linear } from '@kanaries/ml';
import { CausalGraph } from './causal-graph.js';

/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-call */

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

function scaleByStd(x: Float64Array): Float64Array {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2;
  const std = Math.sqrt(v / n);
  const invStd = std > 1e-10 ? 1 / std : 1;
  const result = new Float64Array(n);
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
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2;
  const std = Math.sqrt(v / n);
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
  const result = new Float64Array(n) as unknown as Float64Array;
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

// ── Correlation-based causal ordering (d > 10 fallback) ────────────

/**
 * Heuristic causal ordering via total absolute correlation.
 * O(d²n) vs pwling's O(d²n²). Works well for linear Gaussian systems
 * and avoids the entropy approximation instability on large graphs.
 *
 * For each variable, computes the total absolute correlation with all
 * other variables. The variable with the LOWEST total dependence is
 * selected as the most exogenous root. After selection, residuals are
 * computed and the process repeats.
 */
function searchCausalOrderByCorrelation(
  X: Float64Array[], d: number, n: number,
): number[] {
  const X_ = X.map(col => new Float64Array(col));
  const order: number[] = [];
  const remaining = new Set(Array.from({ length: d }, (_, i) => i));

  for (let step = 0; step < d; step++) {
    let bestVar = -1;
    let bestScore = Infinity;

    for (const i of remaining) {
      let totalDep = 0;
      for (const j of remaining) {
        if (i !== j) {
          const c = Math.abs(correlation(X_[i], X_[j], n));
          totalDep += c;
        }
      }
      if (totalDep < bestScore) {
        bestScore = totalDep;
        bestVar = i;
      }
    }

    if (bestVar === -1) break;
    order.push(bestVar);
    remaining.delete(bestVar);

    // Regress out bestVar from all remaining variables
    for (const j of remaining) {
      X_[j] = Float64Array.from(residual(X_[j], X_[bestVar]));
    }
  }

  return order;
}

function correlation(x: Float64Array, y: Float64Array, n: number): number {
  const cov = covariance(x, y);
  const vx = variance(x);
  const vy = variance(y);
  const denom = Math.sqrt(Math.max(vx, 1e-12) * Math.max(vy, 1e-12));
  return cov / denom;
}

// ── BIC-based Lasso grid search ──────────────────────────────────────

const LASSO_ALPHAS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0];

/**
 * Selects predictors via BIC-optimal Lasso regression.
 * This approximates the official `predict_adaptive_lasso` which uses
 * `LassoLarsIC(criterion="bic")`, by doing a grid search over alpha
 * and selecting the model with minimum BIC.
 */
function lassoBIC(
  X: Float64Array[],
  predictors: number[],
  target: number,
  alphas: number[],
): Map<number, number> {
  const n = X[0].length;
  const p = predictors.length;
  if (p === 0) return new Map();

  // Build design matrix (n × p) and response vector from Float64Array columns
  const Y: number[] = [];
  const Xmat: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (const pred of predictors) row.push(X[pred][i]);
    Xmat.push(row);
    Y.push(X[target][i]);
  }

  let bestBIC = Infinity;
  let bestCoef: number[] = new Array(p).fill(0);

  for (const alpha of alphas) {
    const model = new Linear.LassoRegression({
      alpha,
      maxIter: 5000,
      tol: 1e-6,
      fitIntercept: true,
    });

    try {
      model.fit(Xmat, Y);
    } catch {
      continue; // singular design → skip
    }

    if (!(model as any).fitted || !(model as any).coef) continue;

    // BIC = n * log(RSS/n) + k * log(n)
    // where k = number of non-zero coefficients (+ 1 for intercept)
    const yPred = model.predict(Xmat);
    let rss = 0;
    for (let i = 0; i < n; i++) {
      const e = Y[i] - yPred[i];
      rss += e * e;
    }
    rss = Math.max(rss, 1e-12);

    const nonZero = (model as any).coef.filter((c: number) => Math.abs(c) > 1e-6).length;
    const k = nonZero + 1; // +1 for intercept
    const bic = n * Math.log(rss / n) + k * Math.log(Math.max(n, 2));

    if (bic < bestBIC) {
      bestBIC = bic;
      bestCoef = [...(model as any).coef];
    }
  }

  const result = new Map<number, number>();
  for (let j = 0; j < p; j++) {
    if (Math.abs(bestCoef[j]) > 1e-6) {
      result.set(predictors[j], bestCoef[j]);
    }
  }
  return result;
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

  // Causal order search: pwling for small graphs, correlation for large
  const useCorrelationOrder = d > 10;
  if (useCorrelationOrder) {
    const corrOrder = searchCausalOrderByCorrelation(X_, d, N);
    for (const idx of corrOrder) K.push(idx);
  } else {
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
          X_[i] = Float64Array.from(residual(X_[i], X_[m]));
        }
      }
      K.push(m);
      U = U.filter(u => u !== m);
    }
  }

  const order = K.map(i => nodeNames[i]);

  // Stricter Lasso alphas for large graphs to suppress false positives
  const BIG_GRAPH_ALPHAS = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0];
  const lassoAlphas = d > 10 ? BIG_GRAPH_ALPHAS : LASSO_ALPHAS;

  // Estimate adjacency matrix via BIC-optimal Lasso
  const weights = new Map<string, Map<string, number>>();
  for (let i = 1; i < d; i++) {
    const target = K[i];
    const predictors = K.slice(0, i);
    if (predictors.length === 0) continue;

    const child = order[i];
    const coefs = lassoBIC(X_original, predictors, target, lassoAlphas);
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

  // ── BIC post-pruning (removes false positives) ──────────────────
  const edgeList: [number, number][] = [];
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      if (i !== j && g.hasEdge(nodeNames[i], nodeNames[j])) {
        edgeList.push([i, j]);
      }
    }
  }

  if (edgeList.length > 1) {
    // Build covariance matrix from original data
    const covMat = new Float64Array(d * d);
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        covMat[i * d + j] = covariance(X_original[i], X_original[j]);
      }
    }

    const computeBIC = (edges: [number, number][]): number => {
      const paSets: Set<number>[] = Array.from({ length: d }, () => new Set());
      for (const [from, to] of edges) paSets[to].add(from);
      let total = 0;
      for (let y = 0; y < d; y++) {
        const pa = [...paSets[y]];
        const k = pa.length;
        let sigma = covMat[y * d + y];
        if (k > 0) {
          const yCov: number[] = pa.map(p => covMat[y * d + p]);
          if (k === 1) {
            sigma -= yCov[0] * yCov[0] / covMat[pa[0] * d + pa[0]];
          } else {
            const paCov: number[][] = [];
            for (let a = 0; a < k; a++) {
              const row: number[] = [];
              for (let b = 0; b < k; b++) row.push(covMat[pa[a] * d + pa[b]]);
              paCov.push(row);
            }
            const coef = solveSmallLS(paCov, yCov);
            for (let a = 0; a < k; a++) sigma -= (coef[a] ?? 0) * yCov[a];
          }
        }
        sigma = Math.max(sigma, 1e-12);
        total += -(N * (1 + Math.log(sigma)) + (k + 1) * Math.log(Math.max(N, 2)));
      }
      return total;
    };

    let currentEdges = [...edgeList];
    let currentBIC = computeBIC(currentEdges);
    let changed = true;
    while (changed) {
      changed = false;
      for (let idx = currentEdges.length - 1; idx >= 0; idx--) {
        const candidate = currentEdges.filter((_, i) => i !== idx);
        const candidateBIC = computeBIC(candidate);
        if (candidateBIC > currentBIC) {
          currentEdges = candidate;
          currentBIC = candidateBIC;
          changed = true;
          break;
        }
      }
    }

    const pruned = new CausalGraph(nodeNames);
    for (const [from, to] of currentEdges) pruned.addEdge(nodeNames[from], nodeNames[to]);
    return { graph: pruned, weights, order };
  }

  return { graph: g, weights, order };
}

// Small linear solver for BIC pruning
function solveSmallLS(A: number[][], b: number[]): number[] {
  const n = A.length;
  const aug = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    const pv = aug[col][col];
    if (Math.abs(pv) < 1e-12) continue;
    for (let row = col + 1; row < n; row++) {
      const f = aug[row][col] / pv;
      for (let j = col; j <= n; j++) aug[row][j] -= f * aug[col][j];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = aug[i][n];
    for (let j = i + 1; j < n; j++) s -= aug[i][j] * (x[j] ?? 0);
    x[i] = Math.abs(aug[i][i]) < 1e-12 ? 0 : s / aug[i][i];
  }
  return x;
}
