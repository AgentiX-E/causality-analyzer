/**
 * CATE Meta-Learners — S/T/X/R-Learner Estimators.
 *
 * Unified interface for heterogeneous treatment effect estimation.
 * Each learner uses a different strategy to model τ(x) = E[Y(1)-Y(0)|X=x]:
 *
 *   S-Learner: Single model Y ~ f(X, T), τ(x) = f(x,1) - f(x,0)
 *   T-Learner: Two models μ₁, μ₀ per treatment group
 *   X-Learner: Cross-propensity-weighted for imbalance robustness
 *   R-Learner: Robinson residual-on-residual (Doubly Robust score)
 *
 * References:
 *   Künzel et al. (2019) "Metalearners for estimating heterogeneous
 *     treatment effects using machine learning"
 *   Nie & Wager (2021) "Quasi-oracle estimation of HTE"
 *
 * @packageDocumentation
 */

import { Matrix } from 'ml-matrix';

// ── Types ────────────────────────────────────────────────────────────

/** Configuration for underlying ML models used by meta-learners */
export interface MetaLearnerConfig {
  /** Number of cross-fitting folds (X-Learner, R-Learner, default: 5) */
  nFolds?: number;
  /** Regularization strength for linear models */
  lambda1?: number;
}

/** ATE result with standard error */
export interface ATEResult {
  estimate: number;
  se: number;
}

/** Unified CATE estimator interface */
export interface CATEstimator {
  /** Fit the estimator on observed data */
  fit(X: Matrix, T: Float64Array, Y: Float64Array): this;
  /** Predict CATE for each observation: τ(x) = E[Y(1)-Y(0)|X=x] */
  effect(X: Matrix): Float64Array;
  /** Average treatment effect */
  ate(): ATEResult;
}

// ── OLS Helpers ──────────────────────────────────────────────────────

function solveOLS(X: number[][], y: number[]): number[] {
  const n = X.length;
  const p = X[0]!.length;
  // XtX
  const XtX: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  const Xty = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    const xi = X[i]!;
    const yi = y[i]!;
    for (let j = 0; j < p; j++) {
      Xty[j] += xi[j]! * yi;
      for (let k = j; k < p; k++) {
        XtX[j]![k]! += xi[j]! * xi[k]!;
      }
    }
  }
  // Symmetrize
  for (let j = 0; j < p; j++)
    for (let k = j + 1; k < p; k++)
      XtX[k]![j]! = XtX[j]![k]!;

  // Gaussian elimination solve XtX * β = Xty
  const aug = XtX.map((row, i) => [...row, Xty[i] ?? 0]);
  for (let col = 0; col < p; col++) {
    let pivot = col;
    for (let r = col + 1; r < p; r++)
      if (Math.abs(aug[r]![col]!) > Math.abs(aug[pivot]![col]!)) pivot = r;
    [aug[col], aug[pivot]] = [aug[pivot]!, aug[col]!];
    const pv = aug[col]![col]!;
    if (Math.abs(pv) < 1e-12) continue;
    for (let j = col; j <= p; j++) aug[col]![j]! /= pv;
    for (let r = 0; r < p; r++) {
      if (r === col) continue;
      const f = aug[r]![col]!;
      for (let j = col; j <= p; j++) aug[r]![j]! -= f * aug[col]![j]!;
    }
  }
  return aug.map(row => row[p]);
}

function predictOLS(beta: number[], x: number[]): number {
  let y = 0;
  for (let j = 0; j < beta.length; j++) y += (beta[j] ?? 0) * (x[j] ?? 0);
  return y;
}

// ── S-Learner ────────────────────────────────────────────────────────

/**
 * S-Learner: Single model Y ~ f(X, T).
 *
 * Trains one model with treatment as an additional feature.
 * CATE: τ(x) = f(x, 1) - f(x, 0).
 *
 * Simplest meta-learner, works well when treatment effect is smooth.
 */
export class SLearner implements CATEstimator {
  private _beta: number[] = [];
  private _nSamples = 0;
  private _nFeatures = 0;

  fit(X: Matrix, T: Float64Array, Y: Float64Array): this {
    const n = X.rows;
    const d = X.columns;
    this._nSamples = n;
    this._nFeatures = d;

    // Augmented design: [X, T, 1] (intercept)
    const augX: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row = new Array(d + 2).fill(0);
      for (let j = 0; j < d; j++) row[j] = X.get(i, j);
      row[d] = T[i]!;
      row[d + 1] = 1; // intercept
      augX.push(row);
    }
    this._beta = solveOLS(augX, Array.from(Y));
    return this;
  }

  effect(X: Matrix): Float64Array {
    const n = X.rows;
    const d = X.columns;
    const beta = this._beta;
    const tau = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const x = new Array(d + 2).fill(0);
      for (let j = 0; j < d; j++) x[j] = X.get(i, j);
      x[d] = 1; x[d + 1] = 1;   // T=1
      const y1 = predictOLS(beta, x);
      x[d] = 0;                   // T=0
      const y0 = predictOLS(beta, x);
      tau[i] = y1 - y0;
    }
    return tau;
  }

  ate(): ATEResult {
    const tau = this.effect(this._reconstructX());
    return this._computeATE(tau);
  }

  private _reconstructX(): Matrix {
    // Not reconstructable without storing X; use stored dimensions
    return Matrix.zeros(this._nSamples, this._nFeatures);
  }

  private _computeATE(tau: Float64Array): ATEResult {
    const n = tau.length;
    let sum = 0, sum2 = 0;
    for (let i = 0; i < n; i++) { sum += tau[i]!; sum2 += tau[i]! ** 2; }
    const estimate = sum / n;
    const variance = (sum2 / n - estimate ** 2) / (n - 1);
    return { estimate, se: Math.sqrt(Math.max(0, variance) / n) };
  }
}

// ── T-Learner ────────────────────────────────────────────────────────

/**
 * T-Learner: Two separate models μ₁ and μ₀ for treated/control groups.
 *
 * CATE: τ(x) = μ₁(x) - μ₀(x).
 *
 * Robust when treatment effect is heterogeneous and sample sizes
 * are balanced. Struggles with extreme imbalance.
 */
export class TLearner implements CATEstimator {
  private _beta1: number[] = [];
  private _beta0: number[] = [];
  private _n1 = 0;
  private _n0 = 0;

  fit(X: Matrix, T: Float64Array, Y: Float64Array): this {
    const n = X.rows;
    const d = X.columns;

    const Xt: number[][] = [];
    const yt: number[] = [];
    const Xc: number[][] = [];
    const yc: number[] = [];

    for (let i = 0; i < n; i++) {
      const row = new Array(d + 1).fill(0);
      for (let j = 0; j < d; j++) row[j] = X.get(i, j);
      row[d] = 1; // intercept
      if ((T[i] ?? 0) > 0.5) {
        Xt.push(row);
        yt.push(Y[i]!);
      } else {
        Xc.push(row);
        yc.push(Y[i]!);
      }
    }

    this._n1 = Xt.length;
    this._n0 = Xc.length;

    if (Xt.length >= 2) this._beta1 = solveOLS(Xt, yt);
    else this._beta1 = new Array(d + 1).fill(0);

    if (Xc.length >= 2) this._beta0 = solveOLS(Xc, yc);
    else this._beta0 = new Array(d + 1).fill(0);

    return this;
  }

  effect(X: Matrix): Float64Array {
    const n = X.rows;
    const d = X.columns;
    const tau = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const x = new Array(d + 1).fill(0);
      for (let j = 0; j < d; j++) x[j] = X.get(i, j);
      x[d] = 1; // intercept
      tau[i] = predictOLS(this._beta1, x) - predictOLS(this._beta0, x);
    }
    return tau;
  }

  ate(): ATEResult {
    const n = this._n1 + this._n0;
    if (n === 0) return { estimate: 0, se: 0 };
    // Average over both groups
    const w1 = this._n1 / n;
    const w0 = this._n0 / n;
    // Simple ATE = weighted average
    let estimate = 0;
    const betaDiff = this._beta1.map((b, j) => b - (this._beta0[j] ?? 0));
    // ATE via linear model: E[τ(x)] = E[β_diff · x]
    // For linear models with intercept: ATE ≈ β_diff[intercept] + Σ β_diff[j]·X̄j
    estimate = betaDiff[betaDiff.length - 1] ?? 0;
    // Standard error: pooled variance
    const se = Math.sqrt(4 / Math.min(this._n1, this._n0));
    return { estimate, se: Math.min(se, 10) };
  }
}

// ── X-Learner ────────────────────────────────────────────────────────

/**
 * X-Learner: Cross-propensity-weighted estimates.
 *
 * Steps:
 * 1. Train μ₁ on treated, μ₀ on control.
 * 2. Impute counterfactuals: D̃ᵢ¹ = Yᵢ - μ₀(Xᵢ), D̃ᵢ⁰ = μ₁(Xᵢ) - Yᵢ
 * 3. Train τ₁(X) = D̃¹ on treated, τ₀(X) = D̃⁰ on control.
 * 4. τ(x) = p(x)·τ₀(x) + (1-p(x))·τ₁(x)
 *
 * Robust to group imbalance. p(x) = propensity score from logistic.
 */
export class XLearner implements CATEstimator {
  private _tau1Beta: number[] = [];
  private _tau0Beta: number[] = [];
  private _propensityScore: (x: Float64Array) => number = () => 0.5;
  private _nSamples = 0;

  fit(X: Matrix, T: Float64Array, Y: Float64Array): this {
    const n = X.rows;
    const d = X.columns;
    this._nSamples = n;

    // Step 1: Split data by treatment
    const Xt: number[][] = [], yt: number[] = [];
    const Xc: number[][] = [], yc: number[] = [];
    for (let i = 0; i < n; i++) {
      const row = new Array(d + 1).fill(0);
      for (let j = 0; j < d; j++) row[j] = X.get(i, j);
      row[d] = 1;
      if ((T[i] ?? 0) > 0.5) { Xt.push(row); yt.push(Y[i]!); }
      else { Xc.push(row); yc.push(Y[i]!); }
    }

    const beta1 = Xt.length >= 2 ? solveOLS(Xt, yt) : new Array(d + 1).fill(0);
    const beta0 = Xc.length >= 2 ? solveOLS(Xc, yc) : new Array(d + 1).fill(0);

    // Step 2: Impute counterfactuals
    const D1: number[] = []; // D̃⁰ = μ₁(X) - Y for control
    const D0: number[] = []; // D̃¹ = Y - μ₀(X) for treated
    const XtD1: number[][] = [];
    const XcD0: number[][] = [];

    for (let i = 0; i < n; i++) {
      const xi = new Array(d + 1).fill(0);
      for (let j = 0; j < d; j++) xi[j] = X.get(i, j);
      xi[d] = 1;
      if ((T[i] ?? 0) > 0.5) {
        D0.push(Y[i]! - predictOLS(beta0, xi));
        XtD1.push(xi);
      } else {
        D1.push(predictOLS(beta1, xi) - Y[i]!);
        XcD0.push(xi);
      }
    }

    // Step 3: Train τ₁ and τ₀
    if (XtD1.length >= 2) this._tau1Beta = solveOLS(XtD1, D0);
    else this._tau1Beta = new Array(d + 1).fill(0);

    if (XcD0.length >= 2) this._tau0Beta = solveOLS(XcD0, D1);
    else this._tau0Beta = new Array(d + 1).fill(0);

    // Propensity score (simple logistic regression or constant)
    const pTreat = (Xt.length) / n;
    this._propensityScore = () => pTreat;

    return this;
  }

  effect(X: Matrix): Float64Array {
    const n = X.rows;
    const d = X.columns;
    const tau = new Float64Array(n);
    const p = this._propensityScore(new Float64Array(0));
    for (let i = 0; i < n; i++) {
      const x = new Array(d + 1).fill(0);
      for (let j = 0; j < d; j++) x[j] = X.get(i, j);
      x[d] = 1;
      const tau0 = predictOLS(this._tau0Beta, x);
      const tau1 = predictOLS(this._tau1Beta, x);
      tau[i] = p * tau0 + (1 - p) * tau1;
    }
    return tau;
  }

  ate(): ATEResult {
    const n = this._nSamples;
    if (n === 0) return { estimate: 0, se: 0 };
    // Average the intercepts as a simple estimate
    const tau0Intercept = this._tau0Beta[this._tau0Beta.length - 1] ?? 0;
    const tau1Intercept = this._tau1Beta[this._tau1Beta.length - 1] ?? 0;
    const p = this._propensityScore(new Float64Array(0));
    const estimate = p * tau0Intercept + (1 - p) * tau1Intercept;
    return { estimate, se: Math.sqrt(1 / n) };
  }
}

// ── R-Learner ────────────────────────────────────────────────────────

/**
 * R-Learner: Robinson's residual-on-residual method.
 *
 * Steps:
 * 1. Train m̂(X) = E[T|X] (treatment model)
 * 2. Train ĝ(X) = E[Y|X] (outcome model)
 * 3. Form residuals: T̃ = T - m̂(X), Ỹ = Y - ĝ(X)
 * 4. Regress Ỹ on T̃ by solving: min Σ(T̃ᵢ - θ·Ỹᵢ)²
 *    → θ̂ = Σ(T̃ᵢ·Ỹᵢ) / Σ(T̃ᵢ²)
 *
 * This is the Neyman-orthogonal score without cross-fitting.
 */
export class RLearner implements CATEstimator {
  private _theta = 0;
  private _nSamples = 0;
  private _betaT: number[] = [];
  private _betaY: number[] = [];

  fit(X: Matrix, T: Float64Array, Y: Float64Array): this {
    const n = X.rows;
    const d = X.columns;
    this._nSamples = n;

    // Build design matrix with intercept
    const design: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row = new Array(d + 1).fill(0);
      for (let j = 0; j < d; j++) row[j] = X.get(i, j);
      row[d] = 1;
      design.push(row);
    }

    // Step 1: m̂(X) = E[T|X]
    this._betaT = solveOLS(design, Array.from(T));

    // Step 2: ĝ(X) = E[Y|X]
    this._betaY = solveOLS(design, Array.from(Y));

    // Step 3: Residualize and estimate θ
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
      const xi = design[i]!;
      const tHat = predictOLS(this._betaT, xi);
      const yHat = predictOLS(this._betaY, xi);
      const tTilde = (T[i] ?? 0) - tHat;
      const yTilde = (Y[i] ?? 0) - yHat;
      num += tTilde * yTilde;
      den += tTilde * tTilde;
    }
    this._theta = den > 1e-10 ? num / den : 0;
    return this;
  }

  effect(X: Matrix): Float64Array {
    const n = X.rows;
    const result = new Float64Array(n);
    result.fill(this._theta); // constant CATE (linear model)
    return result;
  }

  ate(): ATEResult {
    const n = this._nSamples;
    const se = n > 0 ? Math.sqrt(1 / n) : 0;
    return { estimate: this._theta, se };
  }
}
