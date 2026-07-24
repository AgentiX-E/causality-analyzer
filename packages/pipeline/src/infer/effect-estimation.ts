/**
 * Causal Effect Estimation — Backdoor, Frontdoor, IV, Propensity Score, Doubly Robust.
 *
 * Implements the estimation layer of the five-step causal analysis framework
 * (Identify → Estimate), complementing the existing identification and refutation
 * methods in causal-inference.ts.
 *
 * Methods:
 *   - adjustBackdoor: ATE via backdoor adjustment with proper criterion
 *   - estimateFrontdoor: Frontdoor adjustment formula
 *   - estimateIV: 2SLS instrumental variables estimator
 *   - estimatePropensityScore: Logistic propensity score + matching
 *   - estimateDoublyRobust: DR estimator (PS + outcome model)
 *
 * @packageDocumentation
 */
import { CausalGraph } from '../graph/causal-graph.js';
import { solveLinear, colMean, createRNG } from '@agentix-e/causality-analyzer-core';
import type { IdentifiedEstimand } from '@agentix-e/causality-analyzer-core';
import { findBackdoorAdjustmentSet } from './backdoor.js';

// ── Backdoor Adjustment ───────────────────────────────────────────────

/**
 * Backdoor adjustment set using Pearl's backdoor criterion.
 *
 * A set Z satisfies the backdoor criterion relative to (X, Y) if:
 * 1. No node in Z is a descendant of X
 * 2. Z d-separates every path from X to Y that contains an arrow into X
 *
 * Uses the unified findBackdoorAdjustmentSet from backdoor.ts
 * which properly verifies d-separation-based path blocking.
 */
export function findBackdoorSet(graph: CausalGraph, treatment: string, outcome: string): string[] {
  return findBackdoorAdjustmentSet(graph, treatment, outcome);
}

/**
 * Compute ATE using backdoor adjustment.
 *
 * ATE = E[ Y|do(X=x₁) ] - E[ Y|do(X=x₀) ]
 * For linear models: β_treatment from OLS(Y ~ X + Z)
 */
export function adjustBackdoor(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
  data: number[][],
  nodeIndex: Map<string, number>,
): { ate: number; se: number; adjustors: string[] } {
  const adjustors = findBackdoorSet(graph, treatment, outcome);
  const tIdx = nodeIndex.get(treatment)!;
  const oIdx = nodeIndex.get(outcome)!;
  const zIdx = adjustors.map(z => nodeIndex.get(z)!);
  const n = data.length;

  if (zIdx.length === 0) {
    // No confounders: ATE = simple mean difference
    let tMean = 0, cMean = 0, tN = 0, cN = 0;
    for (let r = 0; r < n; r++) {
      if (data[r]![tIdx]! > 0.5) { tMean += data[r]![oIdx]!; tN++; }
      else { cMean += data[r]![oIdx]!; cN++; }
    }
    const ate = (tN > 0 ? tMean / tN : 0) - (cN > 0 ? cMean / cN : 0);
    // Pooled SE
    const varT = tN > 0 ? pooledVar(data, oIdx, tIdx, 1, zIdx) / tN : 0;
    const varC = cN > 0 ? pooledVar(data, oIdx, tIdx, 0, zIdx) / cN : 0;
    return { ate, se: Math.sqrt(varT + varC), adjustors };
  }

  // OLS: Y ~ X + Z
  const allPred = [tIdx, ...zIdx];
  const k = allPred.length;
  const XtX = Array.from({ length: k }, () => new Float64Array(k));
  const Xty = new Float64Array(k);
  let ySum = 0;

  for (let r = 0; r < n; r++) {
    const y = data[r]![oIdx] ?? 0;
    ySum += y;
    for (let i = 0; i < k; i++) {
      const xi = data[r]![allPred[i]!] ?? 0;
      Xty[i] += xi * y;
      for (let j = 0; j < k; j++) {
        XtX[i]![j] += xi * (data[r]![allPred[j]!] ?? 0);
      }
    }
  }

  const coef = solveLinear(
    XtX.map(row => Array.from(row)),
    Array.from(Xty),
  );
  const ate = coef[0]!; // treatment coefficient
  const yMean = ySum / n;
  const intercept = yMean - coef.reduce((s, c, i) => s + c * (allPred[i]! >= 0 ? colMean(data, allPred[i]!) : 0), 0);

  // SE via residual variance
  let ss = 0;
  for (let r = 0; r < n; r++) {
    let pred = intercept;
    for (let i = 0; i < k; i++) pred += coef[i]! * (data[r]![allPred[i]!] ?? 0);
    ss += ((data[r]![oIdx] ?? 0) - pred) ** 2;
  }
  const residualVar = ss / Math.max(1, n - k);
  // SE of treatment coef via (X^T X)^-1[0,0] * σ²
  const se = Math.sqrt(residualVar / n); // simplified: Homoskedastic SE

  return { ate, se: Math.max(1e-10, se), adjustors };
}

// ── Frontdoor Adjustment ──────────────────────────────────────────────

/**
 * Frontdoor adjustment ATE.
 *
 * For treatment X → mediator M → outcome Y (with no backdoor X↔Y):
 * P(Y|do(X)) = Σ_m P(m|x) Σ_x' P(Y|x',m) P(x')
 *
 * Linear approximation: β_M * β_X (product of coefficients)
 */
export function estimateFrontdoor(
  graph: CausalGraph,
  treatment: string,
  outcome: string,
  data: number[][],
  nodeIndex: Map<string, number>,
  mediators: string[],
): { ate: number; se: number } {
  const tIdx = nodeIndex.get(treatment)!;
  const oIdx = nodeIndex.get(outcome)!;
  const mIdx = mediators.map(m => nodeIndex.get(m)!);
  const n = data.length;

  // Stage 1: E[X] (treatment mean) — for P(x')
  // Stage 2: M ~ X → β_XM
  // Stage 3: Y ~ M + X → β_MY

  // Simplified: product of regression coefficients
  // Frontdoor ATE = Σ_m β(X→M_m) * β(M_m→Y)
  let ate = 0;
  let varSum = 0;

    for (const mi of mIdx) {
      // X → M coefficient
      let xSum = 0, mSum = 0, xxSum = 0, xmSum = 0;
      for (let r = 0; r < n; r++) {
        const xi = data[r]![tIdx] ?? 0;
        const m = data[r]![mi] ?? 0;
        xSum += xi; mSum += m;
        xxSum += xi * xi; xmSum += xi * m;
      }
      const betaXM = (n * xmSum - xSum * mSum) / Math.max(1e-10, n * xxSum - xSum * xSum);

      // M → Y coefficient (controlling for X)
      let mySum = 0, mmSum = 0, ySum2 = 0, myYm = 0;
      for (let r = 0; r < n; r++) {
        const m = data[r]![mi] ?? 0;
        const y = data[r]![oIdx] ?? 0;
        mySum += m; mmSum += m * m; ySum2 += y; myYm += m * y;
      }
      const betaMY = (n * myYm - mySum * ySum2) /
        Math.max(1e-10, n * mmSum - mySum * mySum);

    ate += betaXM * betaMY;
    varSum += (betaXM * betaMY) ** 2 / n; // delta method approx
  }

  return { ate, se: Math.sqrt(Math.max(1e-10, varSum)) };
}

// ── Instrumental Variables (2SLS) ─────────────────────────────────────

/**
 * Two-Stage Least Squares instrumental variable estimator.
 *
 * Stage 1: X = γ₀ + γ₁ * IV + ε
 * Stage 2: Y = β₀ + β₁ * X̂ + ε
 *
 * ATE = β₁
 */
export function estimateIV(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  ivIdx: number,
  covariateIndices: number[] = [],
): { ate: number; se: number } {
  const n = data.length;
  // Stage 1: regress treatment on IV + covariates
  const pred1 = [ivIdx, ...covariateIndices];
  const k1 = pred1.length;
  const XtX1 = Array.from({ length: k1 }, () => new Float64Array(k1));
  const Xty1 = new Float64Array(k1);

  for (let r = 0; r < n; r++) {
    for (let i = 0; i < k1; i++) {
      const xi = data[r]![pred1[i]!] ?? 0;
      Xty1[i] += xi * (data[r]![treatmentIdx] ?? 0);
      for (let j = 0; j < k1; j++) {
        XtX1[i]![j] += xi * (data[r]![pred1[j]!] ?? 0);
      }
    }
  }

  const gamma = solveLinear(
    XtX1.map(row => Array.from(row)),
    Array.from(Xty1),
  );

  // Fitted treatment values
  const Xhat = new Float64Array(n);
  for (let r = 0; r < n; r++) {
    let pred = 0;
    for (let i = 0; i < k1; i++) pred += gamma[i]! * (data[r]![pred1[i]!] ?? 0);
    Xhat[r] = pred;
  }

  // Stage 2: regress outcome on fitted treatment + covariates
  const allPred2 = [treatmentIdx, ...covariateIndices];
  const k2 = allPred2.length;
  const XtX2 = Array.from({ length: k2 }, () => new Float64Array(k2));
  const Xty2 = new Float64Array(k2);

  for (let r = 0; r < n; r++) {
    const y = data[r]![outcomeIdx] ?? 0;
    const xhat = Xhat[r]!;
    Xty2[0] += xhat * y;
    XtX2[0]![0] += xhat * xhat;
    for (let i = 0; i < covariateIndices.length; i++) {
      const ci = data[r]![covariateIndices[i]!] ?? 0;
      Xty2[i + 1] += ci * y;
      XtX2[0]![i + 1] += xhat * ci;
      XtX2[i + 1]![0] += ci * xhat;
      for (let j = 0; j < covariateIndices.length; j++) {
        XtX2[i + 1]![j + 1] += ci * (data[r]![covariateIndices[j]!] ?? 0);
      }
    }
  }

  const beta = solveLinear(
    XtX2.map(row => Array.from(row)),
    Array.from(Xty2),
  );

  const ate = beta[0]!;
  // SE via residual variance / (n * var(Xhat))
  let ss = 0, xhatSum = 0, xhatSq = 0;
  for (let r = 0; r < n; r++) {
    let pred = 0;
    pred += beta[0]! * Xhat[r]!;
    for (let i = 0; i < covariateIndices.length; i++) pred += beta[i + 1]! * (data[r]![covariateIndices[i]!] ?? 0);
    ss += ((data[r]![outcomeIdx] ?? 0) - pred) ** 2;
    xhatSum += Xhat[r]!;
    xhatSq += Xhat[r]! ** 2;
  }
  const varXhat = xhatSq / n - (xhatSum / n) ** 2;
  const se = Math.sqrt(Math.max(1e-10, ss / (n - k2)) / Math.max(1e-10, n * varXhat));

  return { ate, se };
}

// ── Propensity Score Matching ─────────────────────────────────────────

/**
 * Estimate propensity scores via logistic regression approximation (linear).
 * For binary treatment: P(T=1|Z) ≈ 1/(1 + exp(-Zβ))
 */
export function estimatePropensityScore(
  data: number[][],
  treatmentIdx: number,
  covariateIndices: number[] = [],
): Float64Array {
  const n = data.length;
  const k = covariateIndices.length + 1; // +1 for intercept
  const scores = new Float64Array(n);

  if (covariateIndices.length === 0) {
    // No covariates: propensity = overall treatment probability
    let tCount = 0;
    for (let r = 0; r < n; r++) if ((data[r]![treatmentIdx] ?? 0) > 0.5) tCount++;
    const p = tCount / n;
    scores.fill(p);
    return scores;
  }

  // Logistic regression via iterative reweighted least squares (IRLS)
  const maxIter = 25;
  const tol = 1e-6;
  let beta = new Float64Array(k);
  const X = new Float64Array(n * k);

  for (let r = 0; r < n; r++) {
    X[r * k] = 1; // intercept
    for (let i = 0; i < covariateIndices.length; i++) {
      X[r * k + i + 1] = data[r]![covariateIndices[i]!] ?? 0;
    }
  }

  for (let iter = 0; iter < maxIter; iter++) {
    // Compute p_i = sigmoid(X_i * β)
    const p = new Float64Array(n);
    for (let r = 0; r < n; r++) {
      let dot = 0;
      for (let j = 0; j < k; j++) dot += X[r * k + j]! * beta[j]!;
      p[r] = 1 / (1 + Math.exp(-Math.min(Math.max(dot, -15), 15))); // clamp for stability
      const t = data[r]![treatmentIdx]! > 0.5 ? 1 : 0;
    }

    // IRLS update
    const XtWX = new Float64Array(k * k);
    const XtWz = new Float64Array(k);
    for (let r = 0; r < n; r++) {
      const w = p[r]! * (1 - p[r]!);
      const t = data[r]![treatmentIdx]! > 0.5 ? 1 : 0;
      const z = Math.log(Math.max(1e-10, p[r]! / (1 - p[r]!))) + (t - p[r]!) / Math.max(1e-10, w);
      for (let i = 0; i < k; i++) {
        XtWz[i] += X[r * k + i]! * w * z;
        for (let j = 0; j < k; j++) {
          XtWX[i * k + j] += X[r * k + i]! * w * X[r * k + j]!;
        }
      }
    }

    // Solve XtWX * newBeta = XtWz
    const XtWX2d = new Array(k);
    const XtWz1d = new Array(k);
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

  // Compute final propensity scores
  for (let r = 0; r < n; r++) {
    let dot = 0;
    for (let j = 0; j < k; j++) dot += X[r * k + j]! * beta[j]!;
    scores[r] = 1 / (1 + Math.exp(-Math.min(Math.max(dot, -15), 15)));
  }

  return scores;
}

/**
 * Propensity Score Matching ATE.
 *
 * Matches each treated unit to its nearest control neighbor by propensity score.
 */
export function estimatePSMatching(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  covariateIndices: number[] = [],
  seed?: number,
): { ate: number; se: number } {
  const rng = createRNG(seed ?? null);
  const n = data.length;
  const scores = estimatePropensityScore(data, treatmentIdx, covariateIndices);

  // Separate treated and control
  const treated: Array<{ idx: number; score: number; y: number }> = [];
  const control: Array<{ idx: number; score: number; y: number }> = [];
  for (let r = 0; r < n; r++) {
    const entry = { idx: r, score: scores[r]!, y: data[r]![outcomeIdx] ?? 0 };
    if ((data[r]![treatmentIdx] ?? 0) > 0.5) treated.push(entry);
    else control.push(entry);
  }

  // Nearest-neighbor matching with replacement
  let attSum = 0;
  let matchedCount = 0;
  for (const t of treated) {
    let bestDist = Infinity;
    let bestY = 0;
    for (const c of control) {
      const dist = Math.abs(t.score - c.score);
      if (dist < bestDist) { bestDist = dist; bestY = c.y; }
    }
    attSum += t.y - bestY;
    matchedCount++;
  }

  const ate = matchedCount > 0 ? attSum / matchedCount : 0;

  // Bootstrap SE (simple)
  let seEst = 0;
  const nBoot = 30;
  for (let b = 0; b < nBoot; b++) {
    let sum = 0, cnt = 0;
    for (const t of treated) {
      const ri = Math.floor(rng() * control.length);
      sum += t.y - control[ri]!.y;
      cnt++;
    }
    seEst += ((sum / cnt) - ate) ** 2;
  }
  const se = Math.sqrt(Math.max(1e-10, seEst / nBoot));

  return { ate, se };
}

// ── Doubly Robust Estimation ──────────────────────────────────────────

/**
 * Doubly Robust estimator combining propensity score and outcome model.
 *
 * DR = (1/n) Σ [μ₁(Z_i) - μ₀(Z_i) + T_i*(Y_i - μ₁(Z_i))/π(Z_i) - (1-T_i)*(Y_i - μ₀(Z_i))/(1-π(Z_i))]
 *
 * where μ_t = E[Y|T=t, Z], π = P(T=1|Z)
 */
export function estimateDoublyRobust(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  covariateIndices: number[] = [],
): { ate: number; se: number } {
  const n = data.length;
  const scores = estimatePropensityScore(data, treatmentIdx, covariateIndices);

  // Split data by treatment
  const treatedData: number[][] = [];
  const controlData: number[][] = [];
  for (let r = 0; r < n; r++) {
    if ((data[r]![treatmentIdx] ?? 0) > 0.5) treatedData.push(data[r]!);
    else controlData.push(data[r]!);
  }

  // Pre-fit outcome models ONCE (not per-observation)
  const beta1 = treatedData.length >= 2
    ? fitOLS(treatedData, outcomeIdx, covariateIndices)
    : null;
  const beta0 = controlData.length >= 2
    ? fitOLS(controlData, outcomeIdx, covariateIndices)
    : null;

  // DR computation: apply pre-fitted models
  let drSum = 0;
  const drValues = new Float64Array(n);
  for (let r = 0; r < n; r++) {
    const t = (data[r]![treatmentIdx] ?? 0) > 0.5 ? 1 : 0;
    const y = data[r]![outcomeIdx] ?? 0;
    const pi = Math.min(Math.max(scores[r]!, 0.05), 0.95);
    const m1 = beta1 ? predictFromBeta(covariateIndices, beta1, data[r]!) : 0;
    const m0 = beta0 ? predictFromBeta(covariateIndices, beta0, data[r]!) : 0;
    const dr = m1 - m0 + t * (y - m1) / pi - (1 - t) * (y - m0) / (1 - pi);
    drSum += dr;
    drValues[r] = dr;
  }
  const ate = drSum / n;

  // Influence-function based SE (reuses pre-fitted betas)
  let ifVar = 0;
  for (let r = 0; r < n; r++) {
    const t = (data[r]![treatmentIdx] ?? 0) > 0.5 ? 1 : 0;
    const y = data[r]![outcomeIdx] ?? 0;
    const pi = Math.min(Math.max(scores[r]!, 0.05), 0.95);
    const m1 = beta1 ? predictFromBeta(covariateIndices, beta1, data[r]!) : 0;
    const m0 = beta0 ? predictFromBeta(covariateIndices, beta0, data[r]!) : 0;
    const dr = m1 - m0 + t * (y - m1) / pi - (1 - t) * (y - m0) / (1 - pi);
    ifVar += (dr - ate) ** 2;
  }
  const se = Math.sqrt(Math.max(1e-10, ifVar / (n * n)));

  return { ate, se };
}

// ── Helpers ──────────────────────────────────────────────────────────

function hasDirectedPath(graph: CausalGraph, from: string, to: string): boolean {
  return collectDescendants(graph, from).has(to) || from === to;
}

function collectDescendants(graph: CausalGraph, node: string): Set<string> {
  const result = new Set<string>();
  const stack = [node];
  while (stack.length > 0) {
    const u = stack.pop()!;
    for (const v of graph.children(u)) {
      if (!result.has(v)) { result.add(v); stack.push(v); }
    }
  }
  return result;
}

function pooledVar(
  data: number[][], outcomeIdx: number, treatIdx: number,
  treatVal: number, covIdx: number[],
): number {
  let ss = 0, n = 0;
  for (const row of data) {
    if ((row[treatIdx] ?? 0) > 0.5 !== (treatVal > 0.5)) continue;
    // Simple variance (no cov adjustment for pooled var)
    n++;
  }
  if (n < 2) return 0;
  const rows = data.filter(r => (r[treatIdx] ?? 0) > 0.5 === (treatVal > 0.5));
  const mean = rows.reduce((s, r) => s + (r[outcomeIdx] ?? 0), 0) / rows.length;
  for (const r of rows) ss += ((r[outcomeIdx] ?? 0) - mean) ** 2;
  return ss / (n - 1);
}

/**
 * Fit OLS regression: y ~ X (X = covariate columns from data).
 * Returns coefficients β [β₀, β₁, ..., β_{k-1}] where β₀ corresponds to covariateIndices[0].
 */
function fitOLS(
  data: number[][], outcomeIdx: number,
  covariateIndices: number[],
): number[] {
  const n = data.length;
  const k = covariateIndices.length;
  if (n < k + 1) return new Array(k).fill(0);

  const XtX = Array.from({ length: k }, () => new Float64Array(k));
  const Xty = new Float64Array(k);

  for (let r = 0; r < n; r++) {
    const y = data[r]![outcomeIdx] ?? 0;
    for (let i = 0; i < k; i++) {
      const xi = data[r]![covariateIndices[i]!] ?? 0;
      Xty[i] += xi * y;
      for (let j = 0; j < k; j++) {
        XtX[i]![j] += xi * (data[r]![covariateIndices[j]!] ?? 0);
      }
    }
  }

  return solveLinear(
    XtX.map(row => Array.from(row)),
    Array.from(Xty),
  );
}

/**
 * Predict outcome from pre-fitted OLS coefficients.
 * ŷ = Σ βᵢ·xᵢ  where xᵢ = row[covariateIndices[i]]
 */
function predictFromBeta(
  covariateIndices: number[], beta: number[], row: number[],
): number {
  let pred = 0;
  for (let i = 0; i < covariateIndices.length; i++) {
    pred += (beta[i] ?? 0) * (row[covariateIndices[i]!] ?? 0);
  }
  return pred;
}

// Kept for backward compatibility — prefer fitOLS + predictFromBeta for bulk predictions
function linearPredict(
  data: number[][], outcomeIdx: number,
  covariateIndices: number[], z: number[],
): number {
  const coef = fitOLS(data, outcomeIdx, covariateIndices);
  return predictFromBeta(covariateIndices, coef, z);
}
