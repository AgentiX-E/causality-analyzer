/**
 * Double Machine Learning (DML) — Cross-fitting estimator for ATE/CATE.
 *
 * Based on Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018):
 * "Double/Debiased Machine Learning for Treatment and Structural Parameters"
 *
 * DML uses K-fold cross-fitting to estimate treatment effects while
 * controlling for high-dimensional confounders:
 *
 *   ATE = (1/n) Σ [g(1, Z_i) - g(0, Z_i) + (T_i - m(Z_i)) * (Y_i - g(T_i, Z_i)) / (m(Z_i)*(1-m(Z_i)))]
 *
 * where:
 *   m(Z) = E[T|Z]  — propensity score model
 *   g(T, Z) = E[Y|T, Z] — outcome model
 *
 * Cross-fitting eliminates overfitting bias by using out-of-fold predictions.
 *
 * @packageDocumentation
 */
import type { CausalGraph } from '../graph/causal-graph.js';

// ── Learner Interface ────────────────────────────────────────────────

/** A simple regression learner that can be fit/predicted */
export interface MLRegressor {
  fit(X: number[][], y: number[]): void;
  predict(X: number[][]): number[];
  /** Number of parameters (for BIC, optional) */
  nParams?: number;
}

/** A simple classifier that can be fit/predicted (for propensity score) */
export interface MLClassifier {
  fit(X: number[][], y: number[]): void;
  predictProba(X: number[][]): number[];
}

// ── Built-in Learners ─────────────────────────────────────────────────

/** Linear OLS regressor with ridge regularization */
export class RidgeRegressor implements MLRegressor {
  private coef: number[] = [];
  private intercept = 0;
  nParams = 0;

  constructor(private lambda: number = 1e-3) {}

  fit(X: number[][], y: number[]): void {
    const n = X.length;
    if (n === 0) return;
    const p = X[0]!.length;
    this.nParams = p + 1;

    // Ridge: (X^T X + λI) β = X^T y
    const XtX = Array.from({ length: p }, () => new Float64Array(p));
    const Xty = new Float64Array(p);
    const xMeans = new Float64Array(p);
    let ySum = 0;

    for (let r = 0; r < n; r++) {
      ySum += y[r]!;
      for (let i = 0; i < p; i++) {
        const xi = X[r]![i]!;
        xMeans[i] += xi;
        Xty[i] += xi * (y[r]!);
        for (let j = 0; j < p; j++) {
          XtX[i]![j] += xi * X[r]![j]!;
        }
      }
    }

    // Center and add ridge
    for (let i = 0; i < p; i++) xMeans[i] /= n;
    const yMean = ySum / n;

    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        XtX[i]![j] -= n * xMeans[i]! * xMeans[j]!;
        if (i === j) XtX[i]![j] += this.lambda;
      }
      Xty[i] -= n * xMeans[i]! * yMean;
    }

    this.coef = choleskySolveSymmetric(XtX, Array.from(Xty), p);
    this.intercept = yMean;
    for (let i = 0; i < p; i++) {
      this.intercept -= this.coef[i]! * xMeans[i]!;
    }
  }

  predict(X: number[][]): number[] {
    return X.map(row => {
      let pred = this.intercept;
      for (let i = 0; i < this.coef.length; i++) {
        pred += this.coef[i]! * (row[i] ?? 0);
      }
      return pred;
    });
  }
}

/** Logistic regression classifier for propensity score estimation */
export class LogisticClassifier implements MLClassifier {
  private coef: number[] = [];
  private intercept = 0;
  private fitted = false;

  fit(X: number[][], y: number[]): void {
    const n = X.length;
    if (n === 0) return;
    const p = X[0]!.length;
    const k = p + 1; // +1 for intercept

    const Xmat = new Float64Array(n * k);
    for (let r = 0; r < n; r++) {
      Xmat[r * k] = 1; // intercept
      for (let j = 0; j < p; j++) {
        Xmat[r * k + j + 1] = X[r]![j]!;
      }
    }

    let beta = new Float64Array(k);
    for (let iter = 0; iter < 25; iter++) {
      const pVec = new Float64Array(n);
      for (let r = 0; r < n; r++) {
        let dot = 0;
        for (let j = 0; j < k; j++) dot += Xmat[r * k + j]! * beta[j]!;
        pVec[r] = 1 / (1 + Math.exp(-Math.min(Math.max(dot, -15), 15)));
      }

      const XtWX = new Float64Array(k * k);
      const XtWz = new Float64Array(k);
      for (let r = 0; r < n; r++) {
        const w = pVec[r]! * (1 - pVec[r]!);
        const t = y[r] ?? 0;
        const z = Math.log(Math.max(1e-10, pVec[r]! / (1 - pVec[r]!))) + (t - pVec[r]!) / Math.max(1e-10, w);
        for (let i = 0; i < k; i++) {
          XtWz[i] += Xmat[r * k + i]! * w * z;
          for (let j = 0; j < k; j++) {
            XtWX[i * k + j] += Xmat[r * k + i]! * w * Xmat[r * k + j]!;
          }
        }
      }

      const XtWX2d = new Array(k);
      for (let i = 0; i < k; i++) {
        XtWX2d[i] = new Array(k);
        for (let j = 0; j < k; j++) XtWX2d[i]![j] = XtWX[i * k + j]!;
      }
      const newBeta = choleskySolveSymmetric(XtWX2d, Array.from(XtWz), k);

      let delta = 0;
      for (let j = 0; j < k; j++) delta += (newBeta[j]! - beta[j]!) ** 2;
      beta = Float64Array.from(newBeta);
      if (Math.sqrt(delta) < 1e-6) break;
    }

    this.intercept = beta[0]!;
    this.coef = Array.from(beta.slice(1));
    this.fitted = true;
  }

  predictProba(X: number[][]): number[] {
    if (!this.fitted) return X.map(() => 0.5);
    return X.map(row => {
      let dot = this.intercept;
      for (let i = 0; i < this.coef.length; i++) {
        dot += this.coef[i]! * (row[i] ?? 0);
      }
      return 1 / (1 + Math.exp(-Math.min(Math.max(dot, -15), 15)));
    });
  }
}

// ── DML Estimator ─────────────────────────────────────────────────────

export interface DMLConfig {
  /** Number of cross-fitting folds */
  nFolds?: number;
  /** Outcome model (g) */
  outcomeModel?: MLRegressor;
  /** Propensity score model (m) */
  propensityModel?: MLClassifier;
  /** Random seed */
  seed?: number;
}

export interface DMLEstimate {
  /** Average Treatment Effect */
  ate: number;
  /** Standard error */
  se: number;
  /** 95% confidence interval */
  ciLow: number;
  ciHigh: number;
  /** Per-fold ATE estimates */
  foldEstimates: number[];
  /** Number of observations */
  n: number;
}

/**
 * Double Machine Learning ATE estimator.
 *
 * Uses K-fold cross-fitting:
 * 1. Split data into K folds
 * 2. For each fold k:
 *    a. Train outcome model g on data excluding fold k
 *    b. Train propensity model m on data excluding fold k
 *    c. Compute residual for fold k:
 *       Ψ_i = g(1, Z_i) - g(0, Z_i) + T_i * (Y_i - g(1, Z_i)) / m(Z_i)
 *                                      - (1-T_i) * (Y_i - g(0, Z_i)) / (1-m(Z_i))
 * 3. ATE = mean(Ψ_i), SE = sd(Ψ_i) / sqrt(n)
 *
 * @param data — observation matrix (rows × cols)
 * @param treatmentIdx — column index of treatment (0/1 binary)
 * @param outcomeIdx — column index of outcome
 * @param covariateIndices — column indices of covariates Z
 * @param config — DML configuration
 */
export function estimateDML(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  covariateIndices: number[] = [],
  config: DMLConfig = {},
): DMLEstimate {
  const n = data.length;
  if (n < 2) {
    return { ate: 0, se: 0, ciLow: 0, ciHigh: 0, foldEstimates: [], n: 0 };
  }

  const nFolds = config.nFolds ?? 5;
  const seed = config.seed ?? 42;
  const outcomeLearner = config.outcomeModel ?? new RidgeRegressor(1e-3);
  const propensityLearner = config.propensityModel ?? new LogisticClassifier();

  // Shuffle indices for random fold assignment
  const indices = shuffleIndices(n, seed);
  const foldSize = Math.ceil(n / nFolds);

  const psiValues: number[] = [];
  const foldEstimates: number[] = [];

  for (let fold = 0; fold < nFolds; fold++) {
    const start = fold * foldSize;
    const end = Math.min(start + foldSize, n);

    // Split into train (out-of-fold) and test (in-fold)
    const trainIdx: number[] = [];
    const testIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      if (i >= start && i < end) testIdx.push(indices[i]!);
      else trainIdx.push(indices[i]!);
    }

    if (testIdx.length === 0 || trainIdx.length === 0) continue;

    // Train propensity model: P(T=1 | Z) on train data
    const trainZ = trainIdx.map(i => covariateIndices.map(ci => data[i]![ci] ?? 0));
    const trainT = trainIdx.map(i => (data[i]![treatmentIdx] ?? 0) > 0.5 ? 1 : 0);
    propensityLearner.fit(trainZ, trainT);

    // Train outcome model on T=1 and T=0 subsets
    const train1 = trainIdx.filter(i => (data[i]![treatmentIdx] ?? 0) > 0.5);
    const train0 = trainIdx.filter(i => (data[i]![treatmentIdx] ?? 0) <= 0.5);

    const outcomeModel1 = new RidgeRegressor(1e-3);
    const outcomeModel0 = new RidgeRegressor(1e-3);

    if (train1.length >= covariateIndices.length + 2) {
      outcomeModel1.fit(
        train1.map(i => covariateIndices.map(ci => data[i]![ci] ?? 0)),
        train1.map(i => data[i]![outcomeIdx] ?? 0),
      );
    }
    if (train0.length >= covariateIndices.length + 2) {
      outcomeModel0.fit(
        train0.map(i => covariateIndices.map(ci => data[i]![ci] ?? 0)),
        train0.map(i => data[i]![outcomeIdx] ?? 0),
      );
    }

    // Compute DML scores on test fold
    let foldSum = 0;
    for (const ri of testIdx) {
      const zRow = covariateIndices.map(ci => data[ri]![ci] ?? 0);
      const t = (data[ri]![treatmentIdx] ?? 0) > 0.5 ? 1 : 0;
      const y = data[ri]![outcomeIdx] ?? 0;

      const propScore = propensityLearner.predictProba([zRow])[0] ?? 0.5;
      const pClamped = Math.min(Math.max(propScore, 0.05), 0.95);

      const mu1 = outcomeModel1.predict([zRow])[0] ?? 0;
      const mu0 = outcomeModel0.predict([zRow])[0] ?? 0;

      // DML score
      const psi = (mu1 - mu0) +
        t * (y - mu1) / pClamped -
        (1 - t) * (y - mu0) / (1 - pClamped);

      psiValues.push(psi);
      foldSum += psi;
    }
    foldEstimates.push(foldSum / testIdx.length);
  }

  // Final ATE
  const ate = psiValues.reduce((a, b) => a + b, 0) / psiValues.length;

  // Standard error
  const variance = psiValues.reduce((s, v) => s + (v - ate) ** 2, 0) / psiValues.length;
  const se = Math.sqrt(Math.max(1e-10, variance / psiValues.length));

  return {
    ate,
    se,
    ciLow: ate - 1.96 * se,
    ciHigh: ate + 1.96 * se,
    foldEstimates,
    n: psiValues.length,
  };
}

// ── Meta-Learners ─────────────────────────────────────────────────────

/**
 * S-Learner (Single Model).
 * Uses a single model with treatment as a feature:
 *   μ(T, Z) = E[Y | T, Z]
 *   ATE = (1/n) Σ [μ(1, Z_i) - μ(0, Z_i)]
 */
export function sLearnerATE(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  covariateIndices: number[] = [],
  model?: MLRegressor,
): { ate: number; se: number } {
  const n = data.length;
  const learner = model ?? new RidgeRegressor(1e-3);

  // Build features: [T, Z_1, ..., Z_k]
  const X = data.map(row => [
    (row[treatmentIdx] ?? 0) > 0.5 ? 1 : 0,
    ...covariateIndices.map(ci => row[ci] ?? 0),
  ]);
  const y = data.map(row => row[outcomeIdx] ?? 0);

  learner.fit(X, y);

  // Predict with T=1 and T=0
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const zRow = covariateIndices.map(ci => data[i]![ci] ?? 0);
    const mu1 = learner.predict([[1, ...zRow]])[0] ?? 0;
    const mu0 = learner.predict([[0, ...zRow]])[0] ?? 0;
    sum += mu1 - mu0;
  }
  const ate = sum / n;

  // Bootstrap SE
  const rng = createSeeder(42);
  const nBoot = 50;
  let seSq = 0;
  for (let b = 0; b < nBoot; b++) {
    let bs = 0;
    for (let i = 0; i < n; i++) {
      const ri = Math.floor(rng() * n);
      const zR = covariateIndices.map(ci => data[ri]![ci] ?? 0);
      const m1 = learner.predict([[1, ...zR]])[0] ?? 0;
      const m0 = learner.predict([[0, ...zR]])[0] ?? 0;
      bs += m1 - m0;
    }
    seSq += (bs / n - ate) ** 2;
  }
  const se = Math.sqrt(Math.max(1e-10, seSq / nBoot));

  return { ate, se };
}

/**
 * T-Learner (Two Models).
 * Separate models for treated and control:
 *   μ₁(Z) = E[Y | T=1, Z], μ₀(Z) = E[Y | T=0, Z]
 *   ATE = (1/n) Σ [μ₁(Z_i) - μ₀(Z_i)]
 */
export function tLearnerATE(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  covariateIndices: number[] = [],
  model?: MLRegressor,
): { ate: number; se: number } {
  const n = data.length;

  const treated = data.filter(r => (r[treatmentIdx] ?? 0) > 0.5);
  const control = data.filter(r => (r[treatmentIdx] ?? 0) <= 0.5);

  const learner1 = model ?? new RidgeRegressor(1e-3);
  const learner0 = model ?? new RidgeRegressor(1e-3);

  if (treated.length >= covariateIndices.length + 2) {
    learner1.fit(
      treated.map(r => covariateIndices.map(ci => r[ci] ?? 0)),
      treated.map(r => r[outcomeIdx] ?? 0),
    );
  }
  if (control.length >= covariateIndices.length + 2) {
    learner0.fit(
      control.map(r => covariateIndices.map(ci => r[ci] ?? 0)),
      control.map(r => r[outcomeIdx] ?? 0),
    );
  }

  let sum = 0;
  for (let i = 0; i < n; i++) {
    const zRow = covariateIndices.map(ci => data[i]![ci] ?? 0);
    const mu1 = learner1.predict([zRow])[0] ?? 0;
    const mu0 = learner0.predict([zRow])[0] ?? 0;
    sum += mu1 - mu0;
  }
  const ate = sum / n;

  // Bootstrap SE
  const rng = createSeeder(123);
  const nBoot = 50;
  let seSq = 0;
  for (let b = 0; b < nBoot; b++) {
    let bs = 0;
    for (let i = 0; i < n; i++) {
      const ri = Math.floor(rng() * n);
      const zR = covariateIndices.map(ci => data[ri]![ci] ?? 0);
      const m1 = learner1.predict([zR])[0] ?? 0;
      const m0 = learner0.predict([zR])[0] ?? 0;
      bs += m1 - m0;
    }
    seSq += (bs / n - ate) ** 2;
  }
  const se = Math.sqrt(Math.max(1e-10, seSq / nBoot));

  return { ate, se };
}

/**
 * X-Learner.
 * 1. Estimate μ₁ and μ₀ using T-Learner
 * 2. Impute counterfactuals: τ₁ᵢ = Yᵢ(1) - μ₀(Xᵢ) for treated, τ₀ⱼ = μ₁(Xⱼ) - Yⱼ(0) for control
 * 3. Fit τ₁ ~ X and τ₀ ~ X
 * 4. ATE = (1/n) Σ [g(Xᵢ) * τ₁(Xᵢ) + (1-g(Xᵢ)) * τ₀(Xᵢ)]
 */
export function xLearnerATE(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  covariateIndices: number[] = [],
  propensityModel?: MLClassifier,
): { ate: number; se: number } {
  const n = data.length;

  const treated = data.filter(r => (r[treatmentIdx] ?? 0) > 0.5);
  const control = data.filter(r => (r[treatmentIdx] ?? 0) <= 0.5);

  const mu1Model = new RidgeRegressor(1e-3);
  const mu0Model = new RidgeRegressor(1e-3);

  if (treated.length >= covariateIndices.length + 2) {
    mu1Model.fit(
      treated.map(r => covariateIndices.map(ci => r[ci] ?? 0)),
      treated.map(r => r[outcomeIdx] ?? 0),
    );
  }
  if (control.length >= covariateIndices.length + 2) {
    mu0Model.fit(
      control.map(r => covariateIndices.map(ci => r[ci] ?? 0)),
      control.map(r => r[outcomeIdx] ?? 0),
    );
  }

  // Impute counterfactuals
  const tau1X: number[][] = [];
  const tau1Y: number[] = [];
  const tau0X: number[][] = [];
  const tau0Y: number[] = [];

  for (const r of treated) {
    const zRow = covariateIndices.map(ci => r[ci] ?? 0);
    tau1X.push(zRow);
    tau1Y.push((r[outcomeIdx] ?? 0) - (mu0Model.predict([zRow])[0] ?? 0));
  }
  for (const r of control) {
    const zRow = covariateIndices.map(ci => r[ci] ?? 0);
    tau0X.push(zRow);
    tau0Y.push((mu1Model.predict([zRow])[0] ?? 0) - (r[outcomeIdx] ?? 0));
  }

  const tau1Learner = new RidgeRegressor(1e-3);
  const tau0Learner = new RidgeRegressor(1e-3);
  if (tau1X.length >= 2) tau1Learner.fit(tau1X, tau1Y);
  if (tau0X.length >= 2) tau0Learner.fit(tau0X, tau0Y);

  // Propensity score for weighting
  const propensity = propensityModel ?? new LogisticClassifier();
  propensity.fit(
    data.map(r => covariateIndices.map(ci => r[ci] ?? 0)),
    data.map(r => (r[treatmentIdx] ?? 0) > 0.5 ? 1 : 0),
  );

  let sum = 0;
  for (let i = 0; i < n; i++) {
    const zRow = covariateIndices.map(ci => data[i]![ci] ?? 0);
    const g = propensity.predictProba([zRow])[0] ?? 0.5;
    const tau1 = tau1Learner.predict([zRow])[0] ?? 0;
    const tau0 = tau0Learner.predict([zRow])[0] ?? 0;
    sum += g * tau1 + (1 - g) * tau0;
  }
  const ate = sum / n;

  // Bootstrap SE
  const rng = createSeeder(99);
  let seSq = 0;
  for (let b = 0; b < 30; b++) {
    let bs = 0;
    for (let i = 0; i < n; i++) {
      const ri = Math.floor(rng() * n);
      const zR = covariateIndices.map(ci => data[ri]![ci] ?? 0);
      const gB = propensity.predictProba([zR])[0] ?? 0.5;
      const t1B = tau1Learner.predict([zR])[0] ?? 0;
      const t0B = tau0Learner.predict([zR])[0] ?? 0;
      bs += gB * t1B + (1 - gB) * t0B;
    }
    seSq += (bs / n - ate) ** 2;
  }
  const se = Math.sqrt(Math.max(1e-10, seSq / 30));

  return { ate, se };
}

// ── Helpers ──────────────────────────────────────────────────────────

function shuffleIndices(n: number, seed: number): number[] {
  const rng = createSeeder(seed);
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j]!, indices[i]!];
  }
  return indices;
}

function createSeeder(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function choleskySolveSymmetric(A: Float64Array[] | number[][], b: number[], n: number): number[] {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i]![j] ?? 0;
      for (let p = 0; p < j; p++) sum -= L[i * n + p]! * L[j * n + p]!;
      if (i === j) {
        L[i * n + i] = Math.sqrt(Math.max(1e-12, sum));
      } else {
        L[i * n + j] = sum / L[j * n + j]!;
      }
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = b[i] ?? 0;
    for (let j = 0; j < i; j++) sum -= L[i * n + j]! * y[j]!;
    y[i] = sum / L[i * n + i]!;
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = y[i]!;
    for (let j = i + 1; j < n; j++) sum -= L[j * n + i]! * x[j]!;
    x[i] = sum / L[i * n + i]!;
  }
  return Array.from(x);
}
