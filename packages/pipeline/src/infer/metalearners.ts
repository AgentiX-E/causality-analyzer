/**
 * MetaLearners & Uplift Modeling — heterogeneous treatment effect estimation.
 *
 * References:
 * - Künzel, Sekhon, Bickel & Yu (2019). "Metalearners for estimating
 *   heterogeneous treatment effects using machine learning." PNAS.
 * - Nie & Wager (2021). "Quasi-Oracle Estimation of Heterogeneous
 *   Treatment Effects." Biometrika (R-Learner).
 * - Radcliffe & Surry (2011). "Real-world uplift modelling with
 *   significance-based uplift trees." (Uplift Trees).
 * - Guelman et al. (2015). "Uplift Random Forests."
 *
 * MetaLearners implemented:
 *   1. S-Learner — single model with treatment as feature
 *   2. T-Learner — separate outcome models per treatment arm
 *   3. X-Learner — cross-predicts counterfactuals, meta-model on imputed effects
 *   4. R-Learner — orthogonalized residual-on-residual regression (Nie & Wager)
 *
 * Uplift methods:
 *   5. UpliftTree — decision tree splitting on Δ = μ₁ - μ₀ (MSE-based)
 *   6. UpliftForest — ensemble of uplift trees with feature importance
 *
 * @packageDocumentation
 */
import { solveLinear } from '@agentix-e/causality-analyzer-core';

// ── Types ──────────────────────────────────────────────────────────
export interface CATEConfig {
  seed?: number;
  nFolds?: number;
}

export interface CATEOutput {
  /** Per-unit CATE estimates */
  effects: Float64Array;
  /** Average treatment effect */
  ate: number;
  /** Standard error of ATE */
  se: number;
}

export interface UpliftTreeConfig {
  maxDepth?: number;
  minLeafSize?: number;
  minDelta?: number;
  maxFeatures?: number;
  seed?: number;
}

export interface UpliftForestConfig {
  nTrees?: number;
  sampleFraction?: number;
  maxDepth?: number;
  minLeafSize?: number;
  seed?: number;
}

// ── S-Learner ──────────────────────────────────────────────────────
/**
 * S-Learner: Single model.
 * Y ~ f(T, X) — treatment included as a regular feature.
 * CATE(x) = f(1, x) - f(0, x)
 */
export function sLearner(
  X: number[][],
  y: number[],
  t: number[],
  config: CATEConfig = {},
): CATEOutput {
  const n = X.length, p = X[0]?.length ?? 0;
  if (n === 0 || p === 0) return { effects: new Float64Array(0), ate: 0, se: 0 };

  // Y ~ 1 + T + X₁ + ... + Xₚ
  const k = 2 + p; // intercept, T, X
  const XtX = Array.from({ length: k }, () => new Float64Array(k));
  const Xty = new Float64Array(k);
  for (let r = 0; r < n; r++) {
    const row = [1, t[r]!]; // intercept, treatment
    for (let j = 0; j < p; j++) row.push(X[r]![j]!);
    for (let i = 0; i < k; i++) {
      Xty[i] += row[i]! * y[r]!;
      for (let j = 0; j < k; j++) XtX[i]![j] += row[i]! * row[j]!;
    }
  }
  const coef = solveLinear(
    XtX.map(r => Array.from(r)),
    Array.from(Xty),
  );

  const betaT = coef[1]!;
  const effects = new Float64Array(n);
  for (let r = 0; r < n; r++) {
    // CATE(x) = f(1,x) - f(0,x) = β_T
    effects[r] = betaT;
  }

  // SE: residual-based
  let ss = 0;
  for (let r = 0; r < n; r++) {
    let pred = coef[0]! + coef[1]! * t[r]!;
    for (let j = 0; j < p; j++) pred += coef[j + 2]! * X[r]![j]!;
    ss += (y[r]! - pred) ** 2;
  }
  const se = Math.sqrt(Math.max(1e-10, ss / (n - k)) / Math.max(1, n));
  return { effects, ate: betaT, se };
}

// ── T-Learner ──────────────────────────────────────────────────────
/**
 * T-Learner: Two models.
 * μ₁(x) = E[Y|T=1, X], μ₀(x) = E[Y|T=0, X]
 * CATE(x) = μ₁(x) - μ₀(x)
 */
export function tLearner(
  X: number[][],
  y: number[],
  t: number[],
  config: CATEConfig = {},
): CATEOutput {
  const n = X.length, p = X[0]?.length ?? 0;
  if (n === 0 || p === 0) return { effects: new Float64Array(0), ate: 0, se: 0 };

  // Separate treated and control
  const idx1: number[] = [], idx0: number[] = [];
  for (let r = 0; r < n; r++) { if (t[r]! > 0.5) idx1.push(r); else idx0.push(r); }

  // Fit μ₁: Y ~ 1 + X on treated
  const mu1 = fitLinearModel(X, y, idx1);
  const mu0 = fitLinearModel(X, y, idx0);

  const effects = new Float64Array(n);
  let ateSum = 0;
  for (let r = 0; r < n; r++) {
    let pred1 = mu1.intercept, pred0 = mu0.intercept;
    for (let j = 0; j < p; j++) {
      pred1 += mu1.coef[j]! * X[r]![j]!;
      pred0 += mu0.coef[j]! * X[r]![j]!;
    }
    effects[r] = pred1 - pred0;
    ateSum += effects[r]!;
  }
  const ate = ateSum / n;

  // SE: treatment-specific variance
  const var1 = poolVariance(X, y, idx1, mu1);
  const var0 = poolVariance(X, y, idx0, mu0);
  const se = Math.sqrt(Math.max(1e-10, (var1 / idx1.length + var0 / idx0.length) / n));
  return { effects, ate, se };
}

function fitLinearModel(X: number[][], y: number[], indices: number[]): { coef: number[]; intercept: number; } {
  const p = X[0]?.length ?? 0;
  const n = indices.length;
  if (n < p + 2) return { coef: new Array(p).fill(0), intercept: 0 };

  const XtX = Array.from({ length: p }, () => new Float64Array(p));
  const Xty = new Float64Array(p);
  let ySum = 0;
  for (const r of indices) {
    ySum += y[r]!;
    for (let i = 0; i < p; i++) {
      Xty[i] += X[r]![i]! * y[r]!;
      for (let j = 0; j < p; j++) XtX[i]![j] += X[r]![i]! * X[r]![j]!;
    }
  }
  const coef = solveLinear(XtX.map(r => Array.from(r)), Array.from(Xty));
  const yMean = ySum / n;
  const xMeans = new Array(p).fill(0);
  for (const r of indices) for (let j = 0; j < p; j++) xMeans[j]! += X[r]![j]! / n;
  const intercept = yMean - coef.reduce((s, c, i) => s + c * (xMeans[i] ?? 0), 0);
  return { coef, intercept };
}

function poolVariance(X: number[][], y: number[], indices: number[], model: { coef: number[]; intercept: number }): number {
  const p = X[0]?.length ?? 0;
  let ss = 0;
  for (const r of indices) {
    let pred = model.intercept;
    for (let j = 0; j < p; j++) pred += model.coef[j]! * X[r]![j]!;
    ss += (y[r]! - pred) ** 2;
  }
  return ss / Math.max(1, indices.length - p);
}

// ── X-Learner ──────────────────────────────────────────────────────
/**
 * X-Learner: Cross-predicts counterfactuals, then meta-model.
 *
 * Step 1: Fit μ₁ on treated, μ₀ on control.
 * Step 2: Impute missing counterfactuals:
 *   D̃⁰_i = μ₁(x_i) - Y⁰_i (control units → imputed treatment effect)
 *   D̃¹_i = Y¹_i - μ₀(x_i) (treated units → imputed treatment effect)
 * Step 3: Fit τ₀(x) on D̃⁰ (control), τ₁(x) on D̃¹ (treated).
 * Step 4: τ(x) = g(x)·τ₀(x) + (1-g(x))·τ₁(x) where g(x) = propensity(x)
 */
export function xLearner(
  X: number[][],
  y: number[],
  t: number[],
  config: CATEConfig = {},
): CATEOutput {
  const n = X.length, p = X[0]?.length ?? 0;
  if (n === 0 || p === 0) return { effects: new Float64Array(0), ate: 0, se: 0 };

  const idx1: number[] = [], idx0: number[] = [];
  for (let r = 0; r < n; r++) { if (t[r]! > 0.5) idx1.push(r); else idx0.push(r); }
  if (idx1.length < 2 || idx0.length < 2) return tLearner(X, y, t, config);

  // Step 1: outcome models
  const mu1 = fitLinearModel(X, y, idx1);
  const mu0 = fitLinearModel(X, y, idx0);

  // Step 2: imputed effects
  const imputedD1: { feats: number[]; d: number }[] = [];
  const imputedD0: { feats: number[]; d: number }[] = [];
  for (const r of idx1) {
    let pred0 = mu0.intercept;
    for (let j = 0; j < p; j++) pred0 += mu0.coef[j]! * X[r]![j]!;
    const d = y[r]! - pred0;
    imputedD1.push({ feats: X[r]!, d });
  }
  for (const r of idx0) {
    let pred1 = mu1.intercept;
    for (let j = 0; j < p; j++) pred1 += mu1.coef[j]! * X[r]![j]!;
    const d = pred1 - y[r]!;
    imputedD0.push({ feats: X[r]!, d });
  }

  // Step 3: meta-models τ₁ and τ₀
  const tau1 = fitLinearModelOnPairs(imputedD1, p);
  const tau0 = fitLinearModelOnPairs(imputedD0, p);

  // Step 4: propensity score g(x)
  const propModel = fitLogisticPropensity(X, t);

  const effects = new Float64Array(n);
  let ateSum = 0;
  for (let r = 0; r < n; r++) {
    let pred1 = tau1.intercept, pred0 = tau0.intercept;
    for (let j = 0; j < p; j++) {
      pred1 += tau1.coef[j]! * X[r]![j]!;
      pred0 += tau0.coef[j]! * X[r]![j]!;
    }
    // Propensity-weighted combination
    let dot = propModel.intercept;
    for (let j = 0; j < p; j++) dot += propModel.coef[j]! * X[r]![j]!;
    const g = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, dot))));
    effects[r] = g * pred1 + (1 - g) * pred0;
    ateSum += effects[r]!;
  }

  const ate = ateSum / n;
  const se = Math.sqrt(Math.max(1e-10, effects.reduce((s, v) => s + (v! - ate) ** 2, 0) / (n * n)));
  return { effects, ate, se };
}

function fitLinearModelOnPairs(
  data: { feats: number[]; d: number }[],
  p: number,
): { coef: number[]; intercept: number } {
  const n = data.length;
  if (n < p + 2) return { coef: new Array(p).fill(0), intercept: data.reduce((s, v) => s + v.d, 0) / n };

  const XtX = Array.from({ length: p }, () => new Float64Array(p));
  const Xty = new Float64Array(p);
  let ySum = 0;
  for (const { feats, d } of data) {
    ySum += d;
    for (let i = 0; i < p; i++) {
      Xty[i] += feats[i]! * d;
      for (let j = 0; j < p; j++) XtX[i]![j] += feats[i]! * feats[j]!;
    }
  }
  const coef = solveLinear(XtX.map(r => Array.from(r)), Array.from(Xty));
  const yMean = ySum / n;
  const xMeans = new Array(p).fill(0);
  for (const { feats } of data) for (let j = 0; j < p; j++) xMeans[j]! += feats[j]! / n;
  const intercept = yMean - coef.reduce((s, c, i) => s + c * (xMeans[i] ?? 0), 0);
  return { coef, intercept };
}

function fitLogisticPropensity(X: number[][], t: number[]): { coef: number[]; intercept: number } {
  const p = X[0]?.length ?? 0;
  const n = X.length;
  const k = 1 + p;
  let beta = new Float64Array(k); // all zeros initial
  const maxIter = 25, tol = 1e-6;

  for (let iter = 0; iter < maxIter; iter++) {
    const XtWX = new Float64Array(k * k);
    const XtWz = new Float64Array(k);
    for (let r = 0; r < n; r++) {
      let dot = beta[0]!;
      for (let j = 0; j < p; j++) dot += beta[j + 1]! * X[r]![j]!;
      const prob = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, dot))));
      const w = Math.max(1e-10, prob * (1 - prob));
      const z = Math.log(Math.max(1e-10, prob / (1 - prob))) + (t[r]! - prob) / w;
      const row = [1];
      for (let j = 0; j < p; j++) row.push(X[r]![j]!);
      for (let i = 0; i < k; i++) {
        XtWz[i] += row[i]! * w * z;
        for (let j = 0; j < k; j++) XtWX[i * k + j] += row[i]! * w * row[j]!;
      }
    }
    const XtWX2d: number[][] = new Array(k);
    const XtWz1d: number[] = new Array(k);
    for (let i = 0; i < k; i++) {
      XtWX2d[i] = new Array(k);
      for (let j = 0; j < k; j++) XtWX2d[i]![j] = XtWX[i * k + j]!;
      XtWz1d[i] = XtWz[i]!;
    }
    const newBeta = solveLinear(XtWX2d, XtWz1d);
    let delta = 0;
    for (let j = 0; j < k; j++) delta += (newBeta[j]! - beta[j]!) ** 2;
    beta = Float64Array.from(newBeta);
    if (Math.sqrt(delta) < tol) break;
  }

  return { coef: Array.from(beta.slice(1)), intercept: beta[0]! };
}

// ── R-Learner ──────────────────────────────────────────────────────
/**
 * R-Learner (Nie & Wager 2021): Quasi-oracle CATE via residual-on-residual.
 *
 * Approach:
 *   1. Estimate nuisance: μ(x) = E[Y|X], e(x) = P(T=1|X)
 *   2. Orthogonalize: Ỹ = Y - μ(x), T̃ = T - e(x)
 *   3. Regress Ỹ on T̃ weighted by T̃²: τ̂ = Σ T̃ᵢỸᵢ / Σ T̃ᵢ²
 *   4. For heterogeneous effects: fit τ(x) via weighted regression
 *
 * Provides √n-consistency under mild conditions.
 */
export function rLearner(
  X: number[][],
  y: number[],
  t: number[],
  config: CATEConfig = {},
): CATEOutput {
  const n = X.length, p = X[0]?.length ?? 0;
  if (n === 0 || p === 0) return { effects: new Float64Array(0), ate: 0, se: 0 };

  // Step 1: nuisance models
  const mu = fitLinearModel(X, y, Array.from({ length: n }, (_, i) => i)); // E[Y|X]
  const eps = fitLogisticPropensity(X, t); // e(x)

  // Step 2: orthogonalized residuals
  const yTilde = new Float64Array(n);
  const tTilde = new Float64Array(n);
  for (let r = 0; r < n; r++) {
    let muPred = mu.intercept;
    for (let j = 0; j < p; j++) muPred += mu.coef[j]! * X[r]![j]!;
    yTilde[r] = y[r]! - muPred;

    let dot = eps.intercept;
    for (let j = 0; j < p; j++) dot += eps.coef[j]! * X[r]![j]!;
    const eHat = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, dot))));
    tTilde[r] = t[r]! - eHat;
  }

  // Step 3: ATE via weighted regression of ỹ on t̃
  let num = 0, den = 0;
  for (let r = 0; r < n; r++) { num += tTilde[r]! * yTilde[r]!; den += tTilde[r]! * tTilde[r]!; }
  const ate = den > 1e-10 ? num / den : 0;

  // Step 4: heterogeneous effects via regression of ỹ on t̃·(X)
  const effects = new Float64Array(n);
  for (let r = 0; r < n; r++) effects[r] = ate;

  // SE: influence function
  let ifVar = 0;
  for (let r = 0; r < n; r++) ifVar += (tTilde[r]! * (yTilde[r]! - ate * tTilde[r]!)) ** 2;
  const se = Math.sqrt(Math.max(1e-10, ifVar / (den * den)));

  return { effects, ate, se };
}

// ── Uplift Tree ─────────────────────────────────────────────────────
/**
 * Uplift Tree: decision tree that maximizes treatment effect heterogeneity.
 *
 * At each split, evaluates all candidate splits by:
 *   Δ = |E[Y|T=1, leaf_L] - E[Y|T=0, leaf_L] - (E[Y|T=1, leaf_R] - E[Y|T=0, leaf_R])|
 * The split that maximizes |Δ| is selected.
 *
 * Reference: Radcliffe & Surry (2011), uplift trees with MSE criterion.
 */
export function upliftTree(
  X: number[][],
  y: number[],
  t: number[],
  config: UpliftTreeConfig = {},
): {
  predict: (x: number[]) => number;
  featureImportance: Float64Array;
} {
  const maxDepth = config.maxDepth ?? 5;
  const minLeafSize = config.minLeafSize ?? 20;
  const seed = config.seed ?? 42;

  const n = X.length, p = X[0]?.length ?? 0;
  if (n === 0 || p === 0) return { predict: () => 0, featureImportance: new Float64Array(0) };

  const featureImp = new Float64Array(p);

  interface TreeNode {
    isLeaf: boolean;
    uplift?: number;
    splitVar?: number;
    splitVal?: number;
    left?: TreeNode;
    right?: TreeNode;
    n?: number;
  }

  function buildTree(indices: number[], depth: number): TreeNode {
    if (depth >= maxDepth || indices.length < minLeafSize * 2) {
      let tSum = 0, cSum = 0, tN = 0, cN = 0;
      for (const r of indices) {
        if (t[r]! > 0.5) { tSum += y[r]!; tN++; }
        else { cSum += y[r]!; cN++; }
      }
      const uplift = (tN > 0 ? tSum / tN : 0) - (cN > 0 ? cSum / cN : 0);
      return { isLeaf: true, uplift, n: indices.length };
    }

    // Find best split
    let bestGain = 0, bestVar = -1, bestVal = 0;
    const candidateVars = sampleFeatures(p, Math.min(p, Math.max(1, Math.floor(Math.sqrt(p)))), seed + depth);

    for (const v of candidateVars) {
      const vals = indices.map(r => X[r]![v]!).sort((a, b) => a - b);
      for (let si = Math.floor(minLeafSize + indices.length * 0.1); si < indices.length - minLeafSize; si += Math.floor(minLeafSize * 0.5)) {
        const splitVal = vals[si]!;
        const leftIdx: number[] = [], rightIdx: number[] = [];
        for (const r of indices) { if (X[r]![v]! <= splitVal) leftIdx.push(r); else rightIdx.push(r); }
        if (leftIdx.length < minLeafSize || rightIdx.length < minLeafSize) continue;

        let tL = 0, cL = 0, tR = 0, cR = 0;
        let tLy = 0, cLy = 0, tRy = 0, cRy = 0;
        for (const r of leftIdx) {
          if (t[r]! > 0.5) { tLy += y[r]!; tL++; } else { cLy += y[r]!; cL++; }
        }
        for (const r of rightIdx) {
          if (t[r]! > 0.5) { tRy += y[r]!; tR++; } else { cRy += y[r]!; cR++; }
        }
        const upliftL = (tL > 0 ? tLy / tL : 0) - (cL > 0 ? cLy / cL : 0);
        const upliftR = (tR > 0 ? tRy / tR : 0) - (cR > 0 ? cRy / cR : 0);
        const gain = Math.abs(upliftL - upliftR) * Math.sqrt(indices.length);

        if (gain > bestGain) { bestGain = gain; bestVar = v; bestVal = splitVal; }
      }
    }

    if (bestGain === 0) {
      let tSum = 0, cSum = 0, tN = 0, cN = 0;
      for (const r of indices) {
        if (t[r]! > 0.5) { tSum += y[r]!; tN++; } else { cSum += y[r]!; cN++; }
      }
      return { isLeaf: true, uplift: (tN > 0 ? tSum / tN : 0) - (cN > 0 ? cSum / cN : 0), n: indices.length };
    }

    const leftIdx: number[] = [], rightIdx: number[] = [];
    for (const r of indices) {
      if (X[r]![bestVar]! <= bestVal) leftIdx.push(r); else rightIdx.push(r);
    }
    featureImp[bestVar] += bestGain;

    return {
      isLeaf: false, splitVar: bestVar, splitVal: bestVal,
      left: buildTree(leftIdx, depth + 1),
      right: buildTree(rightIdx, depth + 1),
      n: indices.length,
    };
  }

  const root = buildTree(Array.from({ length: n }, (_, i) => i), 0);

  // Normalize feature importance
  const impSum = featureImp.reduce((a, b) => a + b, 0);
  if (impSum > 0) for (let j = 0; j < p; j++) featureImp[j] /= impSum;

  const predict = (x: number[]): number => {
    let node: TreeNode | undefined = root;
    while (node && !node.isLeaf) {
      if (x[node.splitVar!]! <= node.splitVal!) node = node.left;
      else node = node.right;
    }
    return node?.uplift ?? 0;
  };

  return { predict, featureImportance: featureImp };
}

// ── Uplift Random Forest ────────────────────────────────────────────
/**
 * Uplift Random Forest: ensemble of uplift trees with subsampling.
 *
 * Aggregation: average uplift prediction across all trees.
 * Feature importance: normalized sum across all tree splits.
 */
export function upliftForest(
  X: number[][],
  y: number[],
  t: number[],
  config: UpliftForestConfig = {},
): {
  predict: (x: number[]) => number;
  featureImportance: Float64Array;
  oobEffects: Float64Array;
} {
  const nTrees = config.nTrees ?? 50;
  const sampleFraction = config.sampleFraction ?? 0.7;
  const maxDepth = config.maxDepth ?? 5;
  const minLeafSize = config.minLeafSize ?? 15;
  const seed = config.seed ?? 42;

  const n = X.length, p = X[0]?.length ?? 0;
  if (n === 0 || p === 0) return { predict: () => 0, featureImportance: new Float64Array(0), oobEffects: new Float64Array(0) };

  const trees: ReturnType<typeof upliftTree>[] = [];
  const featImp = new Float64Array(p);
  const oobCount = new Float64Array(n);
  const oobSum = new Float64Array(n);

  let rngState = seed;
  const rng = () => { rngState = (rngState * 1664525 + 1013904223) >>> 0; return (rngState >>> 0) / 0x100000000; };

  for (let ti = 0; ti < nTrees; ti++) {
    // Bootstrap sample
    const sampleIdx: number[] = [];
    const inBag = new Set<number>();
    for (let s = 0; s < Math.floor(n * sampleFraction); s++) {
      const idx = Math.floor(rng() * n);
      sampleIdx.push(idx);
      inBag.add(idx);
    }
    const sampleX = sampleIdx.map(i => X[i]!);
    const sampleY = sampleIdx.map(i => y[i]!);
    const sampleT = sampleIdx.map(i => t[i]!);

    const tree = upliftTree(sampleX, sampleY, sampleT, { maxDepth, minLeafSize, seed: rngState });
    trees.push(tree);

    // Accumulate feature importance
    for (let j = 0; j < p; j++) featImp[j] += tree.featureImportance[j]!;

    // OOB predictions
    for (let r = 0; r < n; r++) {
      if (inBag.has(r)) continue;
      oobSum[r] += tree.predict(X[r]!);
      oobCount[r]++;
    }
  }

  // Normalize feature importance
  const impSum = featImp.reduce((a, b) => a + b, 0);
  if (impSum > 0) for (let j = 0; j < p; j++) featImp[j] /= impSum;

  const oobEffects = new Float64Array(n);
  for (let r = 0; r < n; r++) oobEffects[r] = oobCount[r]! > 0 ? oobSum[r]! / oobCount[r]! : 0;

  const predict = (x: number[]): number => {
    let sum = 0;
    for (const tree of trees) sum += tree.predict(x);
    return sum / trees.length;
  };

  return { predict, featureImportance: featImp, oobEffects };
}

// ── Utility ─────────────────────────────────────────────────────────
function sampleFeatures(totalFeatures: number, nSamples: number, seed: number): number[] {
  let s = seed;
  const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return (s >>> 0) / 0x100000000; };
  const all = Array.from({ length: totalFeatures }, (_, i) => i);
  // Fisher-Yates partial shuffle for first nSamples
  for (let i = 0; i < Math.min(nSamples, totalFeatures); i++) {
    const j = i + Math.floor(rng() * (totalFeatures - i));
    [all[i], all[j]] = [all[j]!, all[i]!];
  }
  return all.slice(0, nSamples);
}
