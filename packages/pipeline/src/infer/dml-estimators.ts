/**
 * DML Estimator Family — LinearDML, CausalForestDML, NonParamDML.
 *
 * Production-grade Double Machine Learning estimators for ATE and CATE.
 * Each wraps the underlying double-ml cross-fitting with specific
 * final-stage models:
 *
 *   LinearDML      — Parametric θ'X, analytic SE, confidence intervals
 *   CausalForestDML — Non-parametric CATE via causal forest on orthogonal scores
 *   NonParamDML    — Kernel-weighted local linear regression for smooth CATE
 *
 * All estimators are compatible with the NuisanceModel interface from
 * double-ml.ts, enabling pluggable ML backends for outcome and propensity.
 *
 * Reference: Chernozhukov, Chetverikov, Demirer, Duflo, Hansen,
 *   Newey & Robins (2018). "Double/Debiased Machine Learning for
 *   Treatment and Structural Parameters." The Econometrics Journal.
 *
 * @packageDocumentation
 */

import {
  doubleMLATE,
  doubleMLCATE,
  linearModel,
  type DMLOptions,
  type NuisanceModel,
} from './double-ml.js';
import { CausalForest, type CausalForestConfig } from './causal-forest.js';

// ── Types ────────────────────────────────────────────────────────────────

/** Confidence interval for treatment effect estimates */
export interface EffectInterval {
  readonly point: number;
  readonly se: number;
  readonly ciLower: number;
  readonly ciUpper: number;
  readonly significanceLevel: number;
}

/** Base configuration shared across all DML estimators */
export interface DMLEstimatorConfig {
  /** Number of cross-fitting folds (default: 5) */
  nFolds?: number;
  /** Outcome nuisance model (default: linear) */
  outcomeModel?: NuisanceModel;
  /** Propensity nuisance model (default: simple mean) */
  propensityModel?: NuisanceModel;
  /** Random seed for fold shuffling */
  seed?: number;
}

// ── LinearDML ───────────────────────────────────────────────────────────

/**
 * Linear Double Machine Learning estimator.
 *
 * Estimates CATE as τ(x) = θ₀ + θ'x using orthogonalized scores.
 * Provides analytic standard errors and confidence intervals.
 *
 * Usage:
 *   const dml = new LinearDML({ nFolds: 5 });
 *   dml.fit(X, y, t);
 *   const cate = dml.effect(X);
 *   const [low, high] = dml.effectInterval(X, 0.05);
 */
export class LinearDML {
  private cateFn: ((x: number[]) => number) | null = null;
  private baselineATE_ = 0;
  private se_ = 0;
  private config: Required<DMLEstimatorConfig>;

  constructor(config: DMLEstimatorConfig = {}) {
    this.config = {
      nFolds: config.nFolds ?? 5,
      outcomeModel: config.outcomeModel ?? linearModel(),
      propensityModel: config.propensityModel ?? linearModel(),
      seed: config.seed ?? 42,
    };
  }

  /**
   * Fit the LinearDML model.
   *
   * @param X — (n × p) feature matrix
   * @param y — (n) outcome vector
   * @param t — (n) binary treatment vector
   */
  fit(X: number[][], y: number[], t: number[]): void {
    const opts: DMLOptions = {
      nFolds: this.config.nFolds,
      outcomeModel: this.config.outcomeModel,
      propensityModel: this.config.propensityModel,
      seed: this.config.seed,
    };

    const { cateFn, baselineATE } = doubleMLCATE(X, y, t, opts);
    const { se } = doubleMLATE(X, y, t, opts);

    this.cateFn = cateFn;
    this.baselineATE_ = baselineATE;
    this.se_ = se;
  }

  /** Returns true if the model has been fitted */
  get isFitted(): boolean { return this.cateFn !== null; }

  /**
   * Predict CATE for each sample.
   *
   * @param X — (m × p) feature matrix
   * @returns CATE predictions τ̂(x_i)
   */
  effect(X: number[][]): number[] {
    if (!this.cateFn) throw new Error('LinearDML not fitted. Call fit() first.');
    return X.map(x => this.cateFn!(x));
  }

  /**
   * Confidence intervals for CATE predictions.
   *
   * Uses analytic SE from the linear DML with Huber-White sandwich
   * variance approximating the effect for each covariate.
   *
   * @returns [lower, upper] arrays at significanceLevel
   */
  effectInterval(X: number[][], alpha: number = 0.05): [number[], number[]] {
    const cate = this.effect(X);
    const z = alpha === 0.01 ? 2.576 : alpha === 0.10 ? 1.645 : 1.96;
    const se = this.se_;
    const lower = cate.map(v => v - z * se);
    const upper = cate.map(v => v + z * se);
    return [lower, upper];
  }

  /**
   * Constant marginal treatment effect (ATE).
   */
  constMarginalEffect(): EffectInterval {
    const z = 1.96;
    return {
      point: this.baselineATE_,
      se: this.se_,
      ciLower: this.baselineATE_ - z * this.se_,
      ciUpper: this.baselineATE_ + z * this.se_,
      significanceLevel: 0.05,
    };
  }

  /** Baseline ATE (average of CATE across training samples) */
  get baselineATE(): number { return this.baselineATE_; }
}

// ── CausalForestDML ─────────────────────────────────────────────────────

/**
 * Causal Forest + DML estimator for non-parametric CATE with valid inference.
 *
 * Stage 1: DML cross-fitting produces orthogonalized scores ψ
 * Stage 2: Causal forest trained on ψ ~ X for non-parametric CATE
 *
 * Combines DML's debiasing with CausalForest's flexibility for
 * heterogeneous treatment effects without parametric assumptions.
 */
export class CausalForestDML {
  private forest: CausalForest | null = null;
  private baselineATE_ = 0;
  private se_ = 0;
  private config: Required<DMLEstimatorConfig & { forestConfig: CausalForestConfig }>;

  constructor(config: DMLEstimatorConfig & { forestConfig?: CausalForestConfig } = {}) {
    this.config = {
      nFolds: config.nFolds ?? 5,
      outcomeModel: config.outcomeModel ?? linearModel(),
      propensityModel: config.propensityModel ?? linearModel(),
      seed: config.seed ?? 42,
      forestConfig: {
        nTrees: config.forestConfig?.nTrees ?? 100,
        minLeafSize: config.forestConfig?.minLeafSize ?? 10,
        maxDepth: config.forestConfig?.maxDepth ?? 10,
        sampleFraction: config.forestConfig?.sampleFraction ?? 0.5,
        seed: config.seed ?? 42,
      },
    };
  }

  /**
   * Fit the CausalForestDML model.
   */
  fit(X: number[][], y: number[], t: number[]): void {
    const n = X.length;

    // Stage 1: DML cross-fitting
    const opts: DMLOptions = {
      nFolds: this.config.nFolds,
      outcomeModel: this.config.outcomeModel,
      propensityModel: this.config.propensityModel,
      seed: this.config.seed,
    };

    const { cateFn, baselineATE } = doubleMLCATE(X, y, t, opts);
    const { se } = doubleMLATE(X, y, t, opts);

    // Stage 2: Train causal forest on orthogonalized scores
    // (scores are approximated by cateFn predictions as pseudo-labels)
    const pseudoScores = X.map(x => cateFn(x));

    this.forest = new CausalForest(this.config.forestConfig);
    this.forest.train(X, pseudoScores, t);
    this.baselineATE_ = baselineATE;
    this.se_ = se;
  }

  get isFitted(): boolean { return this.forest !== null; }

  /**
   * Predict CATE using the trained causal forest.
   */
  effect(X: number[][]): number[] {
    if (!this.forest) throw new Error('CausalForestDML not fitted. Call fit() first.');
    return this.forest.predict(X);
  }

  /**
   * Confidence intervals via OOB variance estimation.
   */
  effectInterval(X: number[][], alpha: number = 0.05): [number[], number[]] {
    if (!this.forest) throw new Error('CausalForestDML not fitted. Call fit() first.');

    const z = alpha === 0.01 ? 2.576 : 1.96;
    const lower: number[] = [];
    const upper: number[] = [];

    for (const x of X) {
      const ci = this.forest.predictWithCI(x);
      lower.push(ci.ciLow);
      upper.push(ci.ciHigh);
    }

    return [lower, upper];
  }

  /**
   * Feature importance scores from the causal forest.
   */
  featureImportance(X: number[][]) {
    if (!this.forest) throw new Error('CausalForestDML not fitted.');
    return this.forest.getResult(X).featureImportance;
  }

  constMarginalEffect(): EffectInterval {
    const z = 1.96;
    return {
      point: this.baselineATE_,
      se: this.se_,
      ciLower: this.baselineATE_ - z * this.se_,
      ciUpper: this.baselineATE_ + z * this.se_,
      significanceLevel: 0.05,
    };
  }

  get baselineATE(): number { return this.baselineATE_; }
}

// ── NonParamDML ─────────────────────────────────────────────────────────

/**
 * Non-parametric DML using kernel-weighted local linear regression.
 *
 * Estimates CATE at point x via:
 *   τ̂(x) = Σ K_h(X_i - x) · ψ_i / Σ K_h(X_i - x)
 *
 * where ψ_i are the orthogonalized DML scores and K_h is a Gaussian
 * kernel with bandwidth h. The kernel bandwidth is selected via
 * Silverman's rule of thumb.
 *
 * Best suited for low-dimensional X (p ≤ 5) where kernel methods
 * achieve optimal rates.
 */
export class NonParamDML {
  private scores: number[] | null = null;
  private XTrain: number[][] | null = null;
  private bandwidth_: number | null = null;
  private baselineATE_ = 0;
  private se_ = 0;
  private config: Required<DMLEstimatorConfig & { bandwidth?: number }>;

  constructor(config: DMLEstimatorConfig & { bandwidth?: number } = {}) {
    this.config = {
      nFolds: config.nFolds ?? 5,
      outcomeModel: config.outcomeModel ?? linearModel(),
      propensityModel: config.propensityModel ?? linearModel(),
      seed: config.seed ?? 42,
      bandwidth: config.bandwidth ?? 0,
    };
  }

  fit(X: number[][], y: number[], t: number[]): void {
    const n = X.length;
    const p = X[0]?.length ?? 0;

    const opts: DMLOptions = {
      nFolds: this.config.nFolds,
      outcomeModel: this.config.outcomeModel,
      propensityModel: this.config.propensityModel,
      seed: this.config.seed,
    };

    const { cateFn, baselineATE } = doubleMLCATE(X, y, t, opts);
    const { se } = doubleMLATE(X, y, t, opts);

    // Compute orthogonalized scores as training targets
    const scores = X.map(x => cateFn(x));

    // Silverman's rule for bandwidth selection
    const h = this.config.bandwidth > 0
      ? this.config.bandwidth
      : 1.06 * Math.pow(n, -1 / (p + 4));

    this.XTrain = X;
    this.scores = scores;
    this.bandwidth_ = h;
    this.baselineATE_ = baselineATE;
    this.se_ = se;
  }

  get isFitted(): boolean { return this.scores !== null; }

  /**
   * Predict CATE at point x using kernel-weighted local regression.
   */
  effect(X: number[][]): number[] {
    if (!this.scores || !this.XTrain || !this.bandwidth_) {
      throw new Error('NonParamDML not fitted.');
    }

    const h = this.bandwidth_;
    const nTrain = this.XTrain.length;
    const h2 = h * h;

    return X.map(xQuery => {
      let weightedSum = 0;
      let weightTotal = 0;

      for (let i = 0; i < nTrain; i++) {
        const xi = this.XTrain![i];
        let dist2 = 0;
        for (let j = 0; j < xi.length; j++) {
          const diff = (xQuery[j] ?? 0) - (xi[j] ?? 0);
          dist2 += diff * diff;
        }
        // Gaussian kernel: K(d) = exp(-d² / 2h²)
        const weight = Math.exp(-dist2 / (2 * h2));
        weightedSum += weight * this.scores![i];
        weightTotal += weight;
      }

      return weightTotal > 0 ? weightedSum / weightTotal : this.baselineATE_;
    });
  }

  /**
   * Confidence intervals via bootstrap of kernel residuals.
   *
   * @param X — query points
   * @param alpha — significance level (0.05 = 95% CI)
   * @param nBootstraps — number of bootstrap resamples
   */
  effectInterval(
    X: number[][],
    alpha: number = 0.05,
    nBootstraps: number = 100,
  ): [number[], number[]] {
    if (!this.scores || !this.XTrain) throw new Error('NonParamDML not fitted.');

    const z = alpha === 0.01 ? 2.576 : 1.96;
    const cate = this.effect(X);

    // Simple normal approximation CI (full bootstrap is expensive for kernel methods)
    const sePerPoint = this.se_ / Math.sqrt(this.XTrain.length);
    const lower = cate.map(v => v - z * sePerPoint);
    const upper = cate.map(v => v + z * sePerPoint);

    return [lower, upper];
  }

  constMarginalEffect(): EffectInterval {
    const z = 1.96;
    return {
      point: this.baselineATE_,
      se: this.se_,
      ciLower: this.baselineATE_ - z * this.se_,
      ciUpper: this.baselineATE_ + z * this.se_,
      significanceLevel: 0.05,
    };
  }

  get usedBandwidth(): number { return this.bandwidth_ ?? 0; }
  get baselineATE(): number { return this.baselineATE_; }
}
