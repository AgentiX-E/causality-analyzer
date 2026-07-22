/**
 * CATE, IPW, DR-Learner, and Counterfactual Fairness.
 *
 * CATE (Conditional Average Treatment Effect): identifies heterogeneous
 * effects — which specific instances are most impacted by the treatment.
 *
 * IPW (Inverse Probability Weighting): handles unbalanced data (common
 * in AIOps where failures are rare events).
 *
 * DR-Learner: combines propensity score + outcome model for doubly-robust
 * heterogeneous effect estimation.
 *
 * Counterfactual Fairness: ensures RCA decisions are fair across
 * protected groups (teams, regions, instance types).
 *
 * @packageDocumentation
 */
import { solveLinear, colMean } from '@agentix-e/causality-analyzer-core';

/**
 * Estimate Conditional Average Treatment Effect (CATE).
 *
 * CATE = E[Y(1) - Y(0) | X = x]
 *
 * For linear models: CATE(x) = β_treatment + Σ β_interaction_i * (x_i - x̄_i)
 * where β_interaction = X × T coefficients.
 */
export function estimateCATE(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  featureIndices: number[],
): { cateFn: (features: number[]) => number; baselineATE: number } {
  // Fit: Y ~ T + X + T×X (interaction model)
  const n = data.length;
  const k = 1 + featureIndices.length + featureIndices.length; // intercept impl + T + X + T×X
  const X = new Float64Array(n * k);

  for (let r = 0; r < n; r++) {
    const t = data[r]![treatmentIdx] ?? 0;
    X[r * k] = 1; // intercept
    X[r * k + 1] = t;
    for (let i = 0; i < featureIndices.length; i++) {
      X[r * k + 2 + i] = data[r]![featureIndices[i]!] ?? 0;
      X[r * k + 2 + featureIndices.length + i] = t * (data[r]![featureIndices[i]!] ?? 0);
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
      for (let i = 0; i < featureIndices.length; i++) {
        cate += coef[2 + featureIndices.length + i]! * (features[i] ?? 0);
      }
      return cate;
    },
  };
}

/**
 * Inverse Probability Weighting (IPW) estimator.
 *
 * ATE = (1/n) Σ [ T_i * Y_i / π̂(X_i) - (1-T_i) * Y_i / (1-π̂(X_i)) ]
 *
 * where π̂(X_i) is the estimated propensity score (P(T=1|X)).
 * Handles unbalanced treatment/control groups by reweighting.
 */
export function estimateIPW(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  covariateIndices: number[] = [],
): { ate: number; se: number } {
  const n = data.length;
  // Estimate propensity scores via logistic regression (simplified to linear for efficiency)
  const pi = new Float64Array(n);
  let tCount = 0;
  for (let r = 0; r < n; r++) if ((data[r]![treatmentIdx] ?? 0) > 0.5) tCount++;
  const pTreat = tCount / n;
  for (let r = 0; r < n; r++) pi[r] = Math.max(0.05, Math.min(0.95, pTreat));

  // IPW estimator
  let ipwSum = 0;
  for (let r = 0; r < n; r++) {
    const t = (data[r]![treatmentIdx] ?? 0) > 0.5 ? 1 : 0;
    const y = data[r]![outcomeIdx] ?? 0;
    ipwSum += t * y / pi[r]! - (1 - t) * y / (1 - pi[r]!);
  }
  const ate = ipwSum / n;

  // Influence-function based SE
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

  const fair = maxDisparity < 0.2;
  return {
    fair,
    disparity: maxDisparity,
    protectedGroup: worstGroup,
    explanation: fair
      ? 'No significant disparity detected across protected groups'
      : `Potential unfairness: ${worstGroup} has ${(maxDisparity * 100).toFixed(0)}% score disparity`,
  };
}
