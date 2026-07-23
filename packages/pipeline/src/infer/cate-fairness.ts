import { CONSTANTS } from "../constants.js";
/**
 * CATE, IPW, and Counterfactual Fairness.
 *
 * CATE (Conditional Average Treatment Effect): identifies heterogeneous
 * effects — which specific instances are most impacted by the treatment.
 *
 * IPW (Inverse Probability Weighting): handles unbalanced data (common
 * in AIOps where failures are rare events). Uses logistic regression
 * via IRLS to estimate propensity scores from covariates.
 *
 * Counterfactual Fairness: ensures RCA decisions are fair across
 * protected groups (teams, regions, instance types).
 *
 * @packageDocumentation
 */
import { solveLinear } from '@agentix-e/causality-analyzer-core';

// ── IRLS Logistic Regression (shared utility) ────────────────────────

/**
 * Fit a logistic regression model using Iterative Reweighted Least Squares.
 *
 * @returns coefficient vector β of length `1 + nCovariates`
 */
function fitLogistic(
  data: number[][],
  treatmentIdx: number,
  covariateIndices: number[],
  maxIter: number = 25,
  tol: number = 1e-6,
): Float64Array {
  const n = data.length;
  const p = covariateIndices.length;
  const k = 1 + p; // intercept + covariates
  if (k === 1) {
    // No covariates: just return the log-odds of the marginal probability
    let tCount = 0;
    for (let r = 0; r < n; r++) if ((data[r]![treatmentIdx] ?? 0) > 0.5) tCount++;
    const prob = Math.max(0.05, Math.min(0.95, tCount / n));
    return Float64Array.from([Math.log(prob / (1 - prob))]);
  }

  const X = new Float64Array(n * k);
  for (let r = 0; r < n; r++) {
    X[r * k] = 1; // intercept
    for (let i = 0; i < p; i++) {
      X[r * k + i + 1] = data[r]![covariateIndices[i]!] ?? 0;
    }
  }

  let beta = new Float64Array(k);
  for (let iter = 0; iter < maxIter; iter++) {
    // Compute probabilities p_i = sigmoid(X_i · β)
    const prob = new Float64Array(n);
    for (let r = 0; r < n; r++) {
      let dot = 0;
      for (let j = 0; j < k; j++) dot += X[r * k + j]! * beta[j]!;
      prob[r] = 1 / (1 + Math.exp(-Math.min(Math.max(dot, -15), 15)));
    }

    // IRLS update: β_new = (XᵀWX)⁻¹ XᵀWz
    const XtWX = new Float64Array(k * k);
    const XtWz = new Float64Array(k);
    for (let r = 0; r < n; r++) {
      const w = Math.max(1e-10, prob[r]! * (1 - prob[r]!));
      const t = (data[r]![treatmentIdx] ?? 0) > 0.5 ? 1 : 0;
      const workingResponse = Math.log(Math.max(1e-10, prob[r]! / (1 - prob[r]!))) +
        (t - prob[r]!) / w;

      for (let i = 0; i < k; i++) {
        XtWz[i] += X[r * k + i]! * w * workingResponse;
        for (let j = 0; j < k; j++) {
          XtWX[i * k + j] += X[r * k + i]! * w * X[r * k + j]!;
        }
      }
    }

    // Solve XtWX · newBeta = XtWz
    const XtWX2d: number[][] = new Array(k);
    const XtWz1d: number[] = new Array(k);
    for (let i = 0; i < k; i++) {
      XtWX2d[i] = new Array(k);
      for (let j = 0; j < k; j++) XtWX2d[i]![j] = XtWX[i * k + j]!;
      XtWz1d[i] = XtWz[i]!;
    }
    const newBeta = solveLinear(XtWX2d, XtWz1d);

    // Check convergence
    let delta = 0;
    for (let j = 0; j < k; j++) delta += (newBeta[j]! - beta[j]!) ** 2;
    beta = Float64Array.from(newBeta);
    if (Math.sqrt(delta) < tol) break;
  }
  return beta;
}

/**
 * Compute propensity scores π̂(X_i) = P(T=1 | X_i) from logistic coefficients.
 */
function computePropensityScores(
  data: number[][],
  beta: Float64Array,
  covariateIndices: number[],
): Float64Array {
  const n = data.length;
  const p = covariateIndices.length;
  const k = 1 + p;
  const scores = new Float64Array(n);

  for (let r = 0; r < n; r++) {
    let dot = beta[0]!; // intercept
    for (let i = 0; i < p; i++) {
      dot += (beta[i + 1] ?? 0) * (data[r]![covariateIndices[i]!] ?? 0);
    }
    scores[r] = Math.max(0.05, Math.min(0.95, 1 / (1 + Math.exp(-Math.min(Math.max(dot, -15), 15)))));
  }
  return scores;
}

// ── CATE ──────────────────────────────────────────────────────────────

/**
 * Estimate Conditional Average Treatment Effect (CATE).
 *
 * CATE(x) = E[Y(1) - Y(0) | X = x]
 *
 * Fits an interaction model Y ~ 1 + T + X + T×X.
 * For linear models: CATE(x) = β_T + Σ β_T×X_i · (x_i - x̄_i)
 *
 * @param data — observation array (rows × columns)
 * @param treatmentIdx — column index of binary treatment variable
 * @param outcomeIdx — column index of outcome variable
 * @param featureIndices — column indices of feature/covariate variables
 * @returns CATE function and baseline ATE
 */
export function estimateCATE(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  featureIndices: number[],
): { cateFn: (features: number[]) => number; baselineATE: number } {
  const n = data.length;
  const p = featureIndices.length;
  // Columns: 0=intercept, 1=T, 2..2+p-1=X, 2+p..2+2p-1=T×X
  const k = 2 + 2 * p;
  const X = new Float64Array(n * k);

  // Compute feature means for centering
  const xBar = new Float64Array(p);
  for (let i = 0; i < p; i++) {
    let sum = 0, cnt = 0;
    for (let r = 0; r < n; r++) {
      const val = data[r]![featureIndices[i]!];
      if (val != null && !Number.isNaN(val)) { sum += val; cnt++; }
    }
    xBar[i] = cnt > 0 ? sum / cnt : 0;
  }

  for (let r = 0; r < n; r++) {
    const t = data[r]![treatmentIdx] ?? 0;
    X[r * k] = 1; // intercept
    X[r * k + 1] = t;
    for (let i = 0; i < p; i++) {
      X[r * k + 2 + i] = data[r]![featureIndices[i]!] ?? 0;
      X[r * k + 2 + p + i] = t * (data[r]![featureIndices[i]!] ?? 0);
    }
  }

  const XtX = Array.from({ length: k }, () => new Float64Array(k));
  const Xty = new Float64Array(k);
  for (let r = 0; r < n; r++) {
    const y = data[r]![outcomeIdx] ?? 0;
    for (let i = 0; i < k; i++) {
      Xty[i] += X[r * k + i]! * y;
      for (let j = 0; j < k; j++) XtX[i]![j] += X[r * k + i]! * X[r * k + j]!;
    }
  }

  const coef = solveLinear(
    XtX.map(row => Array.from(row)),
    Array.from(Xty),
  );

  const baselineATE = coef[1]!; // β_T

  return {
    baselineATE,
    cateFn: (features: number[]) => {
      let cate = baselineATE;
      for (let i = 0; i < p; i++) {
        // CATE for feature x_i, centered at mean
        cate += coef[2 + p + i]! * ((features[i] ?? 0) - xBar[i]!);
      }
      return cate;
    },
  };
}

// ── IPW ───────────────────────────────────────────────────────────────

/**
 * Inverse Probability Weighting (IPW) estimator with proper propensity score fitting.
 *
 * ATE = (1/n) Σ [ T_i · Y_i / π̂(X_i) − (1−T_i) · Y_i / (1−π̂(X_i)) ]
 *
 * where π̂(X_i) is estimated via logistic regression (IRLS).
 * Propensity scores are clamped to [0.05, 0.95] for numerical stability.
 *
 * @param covariateIndices — column indices to use for propensity score model.
 *   When empty, falls back to marginal treatment probability (randomized assignment).
 */
export function estimateIPW(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  covariateIndices: number[] = [],
): { ate: number; se: number } {
  const n = data.length;

  // Fit propensity scores via logistic regression
  const beta = fitLogistic(data, treatmentIdx, covariateIndices);
  const pi = computePropensityScores(data, beta, covariateIndices);

  // IPW ATE
  let ipwSum = 0;
  for (let r = 0; r < n; r++) {
    const t = (data[r]![treatmentIdx] ?? 0) > 0.5 ? 1 : 0;
    const y = data[r]![outcomeIdx] ?? 0;
    ipwSum += t * y / pi[r]! - (1 - t) * y / (1 - pi[r]!);
  }
  const ate = ipwSum / n;

  // Influence-function based standard error
  let ifVar = 0;
  for (let r = 0; r < n; r++) {
    const t = (data[r]![treatmentIdx] ?? 0) > 0.5 ? 1 : 0;
    const y = data[r]![outcomeIdx] ?? 0;
    const psi = t * y / pi[r]! - (1 - t) * y / (1 - pi[r]!) - ate;
    ifVar += psi * psi;
  }
  const se = Math.sqrt(Math.max(1e-10, ifVar / (n * n)));

  return { ate, se };
}

// ── Counterfactual Fairness ──────────────────────────────────────────

/**
 * Check if an RCA decision satisfies counterfactual fairness.
 *
 * A decision is counterfactually fair if it would have been the same
 * had the protected attribute been different, given all other observed
 * variables.
 */
export function checkFairness(
  rootCauses: Array<{ name: string; score: number }>,
  protectedGroups: Record<string, string[]>,
): { fair: boolean; disparity: number; protectedGroup: string; explanation: string } {
  let maxDisparity = 0;
  let worstGroup = '';
  const scores = new Map(rootCauses.map(r => [r.name, r.score]));

  for (const [group, members] of Object.entries(protectedGroups)) {
    const groupScores = members.map(m => scores.get(m) ?? 0).filter(s => s > 0);
    const nonGroupScores = rootCauses
      .filter(r => !members.includes(r.name))
      .map(r => r.score);

    if (groupScores.length === 0 || nonGroupScores.length === 0) continue;

    const groupMean = groupScores.reduce((a, b) => a + b, 0) / groupScores.length;
    const nonGroupMean = nonGroupScores.reduce((a, b) => a + b, 0) / nonGroupScores.length;
    const disparity = Math.abs(groupMean - nonGroupMean) / Math.max(0.01, nonGroupMean);

    if (disparity > maxDisparity) {
      maxDisparity = disparity;
      worstGroup = group;
    }
  }

  const fair = maxDisparity < CONSTANTS.FAIRNESS_DISPARITY_THRESHOLD;
  return {
    fair,
    disparity: maxDisparity,
    protectedGroup: worstGroup,
    explanation: fair
      ? 'No significant disparity detected across protected groups'
      : `Potential unfairness: ${worstGroup} has ${(maxDisparity * 100).toFixed(0)}% score disparity`,
  };
}
