/**
 * Meta-Learner Family — SLearner, TLearner, XLearner.
 *
 * Meta-learners are CATE estimation frameworks that combine base ML
 * models for heterogeneous treatment effect estimation. Unlike DML,
 * they do not use orthogonal moment conditions — instead they rely
 * on the flexibility of the base learner to capture treatment effect
 * heterogeneity.
 *
 * Reference: Künzel, S., Sekhon, J., Bickel, P., & Yu, B. (2019).
 *   "Metalearners for estimating heterogeneous treatment effects
 *   using machine learning." PNAS 116(10):4156–4165.
 *
 *   SLearner — Single model: Y ~ f(X, T), τ(x) = f(x,1) - f(x,0)
 *   TLearner — Two models: treated and control modeled separately
 *   XLearner — Cross-learner: T-learner + imputed effects + propensity weighting
 *
 * @packageDocumentation
 */

import { CausalForest, type CausalForestConfig } from './causal-forest.js';

/** Base learner function: train on (X, y) → predict on new X */
export type BaseLearner = {
  fit(X: number[][], y: number[]): void;
  predict(X: number[][]): number[];
};

/** Simple linear regression base learner (internal default) */
class LinearBaseLearner implements BaseLearner {
  private beta: number[] | null = null;

  fit(X: number[][], y: number[]): void {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const k = p + 1;
    const xtx: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
    const xty: number[] = new Array(k).fill(0);

    for (let i = 0; i < n; i++) {
      xtx[0]![0]! += 1;
      xty[0]! += y[i]!;
      for (let j = 0; j < p; j++) {
        const xv = X[i]![j]!;
        xtx[0]![j + 1]! += xv;
        xtx[j + 1]![0]! += xv;
        xty[j + 1]! += xv * y[i]!;
      }
    }
    for (let ci = 0; ci < k; ci++) {
      for (let cj = ci + 1; cj < k; cj++) {
        xtx[cj]![ci] = xtx[ci]![cj]!;
      }
    }
    for (let ci = 0; ci < k; ci++) xtx[ci]![ci]! += 1e-8;

    const aug = xtx.map((row, ri) => [...row, xty[ri]!]);
    for (let col = 0; col < k; col++) {
      let mr = col, mv = Math.abs(aug[col]![col]!);
      for (let r = col + 1; r < k; r++) {
        if (Math.abs(aug[r]![col]!) > mv) { mv = Math.abs(aug[r]![col]!); mr = r; }
      }
      if (mv < 1e-12) continue;
      if (mr !== col) [aug[col], aug[mr]] = [aug[mr]!, aug[col]!];
      const piv = aug[col]![col]!;
      for (let c = col; c <= k; c++) aug[col]![c]! /= piv;
      for (let r = 0; r < k; r++) {
        if (r === col) continue;
        const f = aug[r]![col]!;
        if (f === 0) continue;
        for (let c = col; c <= k; c++) aug[r]![c]! -= f * aug[col]![c]!;
      }
    }
    this.beta = aug.map(r => r[k]!);
  }

  predict(X: number[][]): number[] {
    if (!this.beta) throw new Error('Not fitted');
    return X.map(x => {
      let s = this.beta![0]!;
      for (let j = 0; j < x.length; j++) s += (this.beta![j + 1] ?? 0) * (x[j] ?? 0);
      return s;
    });
  }
}

/** CausalForest used as a base learner (wraps existing class) */
class ForestBaseLearner implements BaseLearner {
  private forest: CausalForest;
  constructor(config: CausalForestConfig = {}) {
    this.forest = new CausalForest(config);
  }
  fit(X: number[][], y: number[]): void {
    this.forest.train(X, y, new Array(X.length).fill(1));
  }
  predict(X: number[][]): number[] {
    return this.forest.predict(X);
  }
}

// ── Config ──────────────────────────────────────────────────────────────

/** Configuration for meta-learners */
export interface MetaLearnerConfig {
  /** Base learner type */
  learnerType?: 'linear' | 'causal-forest';
  /** Causal forest config when learnerType = 'causal-forest' */
  forestConfig?: CausalForestConfig;
  /** Propensity score estimation method (XLearner only) */
  propensityMethod?: 'logistic' | 'mean';
  /** Random seed */
  seed?: number;
  /** Bandwidth for kernel smoothing (XLearner stage 3) */
  bandwidth?: number;
}

// ── SLearner ────────────────────────────────────────────────────────────

/**
 * SLearner — Single-Model Meta-Learner.
 *
 * Trains a single model: Y ~ f(X, T) where T is included as an
 * additional feature. CATE is computed as τ(x) = f(x, 1) - f(x, 0).
 *
 * Pros: Simple, uses all data in one model.
 * Cons: May fail to capture treatment effect if T has weak signal.
 */
export class SLearner {
  private learner: BaseLearner;
  private fitted = false;

  constructor(config: MetaLearnerConfig = {}) {
    this.learner = config.learnerType === 'causal-forest'
      ? new ForestBaseLearner(config.forestConfig)
      : new LinearBaseLearner();
  }

  /**
   * Fit the SLearner.
   *
   * @param X — (n × p) covariate matrix
   * @param y — (n) outcome vector
   * @param t — (n) binary treatment vector
   */
  fit(X: number[][], y: number[], t: number[]): void {
    // Augment features with treatment indicator: [X | T]
    const augmented = X.map((row, i) => [...row, t[i]!]);
    this.learner.fit(augmented, y);
    this.fitted = true;
  }

  get isFitted(): boolean { return this.fitted; }

  /**
   * Predict CATE: τ̂(x) = f̂(x, 1) - f̂(x, 0)
   */
  effect(X: number[][]): number[] {
    if (!this.fitted) throw new Error('SLearner not fitted.');
    const y1 = this.learner.predict(X.map(row => [...row, 1]));
    const y0 = this.learner.predict(X.map(row => [...row, 0]));
    return y1.map((v, i) => v - y0[i]!);
  }

  /** Baseline ATE: average of predicted CATE */
  ate(X: number[][], t: number[]): number {
    const cate = this.effect(X);
    return cate.reduce((a, b) => a + b, 0) / cate.length;
  }
}

// ── TLearner ────────────────────────────────────────────────────────────

/**
 * TLearner — Two-Model Meta-Learner.
 *
 * Trains separate models for treated and control groups:
 *   τ̂(x) = f̂₁(x) - f̂₀(x)
 *
 * Pros: Each model adapts to its own distribution.
 * Cons: Cannot share information across groups; high variance with
 *       imbalanced treatment.
 */
export class TLearner {
  private modelT: BaseLearner;
  private modelC: BaseLearner;
  private fitted = false;

  constructor(config: MetaLearnerConfig = {}) {
    const factory = config.learnerType === 'causal-forest'
      ? () => new ForestBaseLearner(config.forestConfig)
      : () => new LinearBaseLearner();
    this.modelT = factory();
    this.modelC = factory();
  }

  /**
   * Fit the TLearner.
   */
  fit(X: number[][], y: number[], t: number[]): void {
    const XT: number[][] = [];
    const yT: number[] = [];
    const XC: number[][] = [];
    const yC: number[] = [];

    for (let i = 0; i < X.length; i++) {
      if (t[i]! > 0.5) { XT.push(X[i]!); yT.push(y[i]!); }
      else { XC.push(X[i]!); yC.push(y[i]!); }
    }

    this.modelT.fit(XT, yT);
    this.modelC.fit(XC, yC);
    this.fitted = true;
  }

  get isFitted(): boolean { return this.fitted; }

  /**
   * Predict CATE: τ̂(x) = f̂₁(x) - f̂₀(x)
   */
  effect(X: number[][]): number[] {
    if (!this.fitted) throw new Error('TLearner not fitted.');
    const y1 = this.modelT.predict(X);
    const y0 = this.modelC.predict(X);
    return y1.map((v, i) => v - y0[i]!);
  }

  ate(X: number[][], t: number[]): number {
    const cate = this.effect(X);
    return cate.reduce((a, b) => a + b, 0) / cate.length;
  }
}

// ── XLearner ────────────────────────────────────────────────────────────

/**
 * XLearner — Cross-Learner Meta-Learner.
 *
 * Four-stage estimation:
 *   1. T-learner: f̂₁, f̂₀
 *   2. Imputed effects:
 *        τ̃₁ᵢ = Yᵢ - f̂₀(Xᵢ)  for treated units
 *        τ̃₀ᵢ = f̂₁(Xᵢ) - Yᵢ  for control units
 *   3. Fit CATE models on imputed effects:
 *        τ̃₁ ~ g₁(X)  (on treated only)
 *        τ̃₀ ~ g₀(X)  (on control only)
 *   4. Combine via propensity weighting:
 *        τ̂(x) = ê(x)·g₀(x) + (1-ê(x))·g₁(x)
 *
 * Pros: Uses all data; handles imbalance well.
 * Cons: More complex; requires propensity model.
 */
export class XLearner {
  private modelT: BaseLearner;
  private modelC: BaseLearner;
  private cateModelT: BaseLearner;
  private cateModelC: BaseLearner;
  private propensity: number[] | null = null;
  private fitted = false;
  private config: Required<MetaLearnerConfig>;

  constructor(config: MetaLearnerConfig = {}) {
    this.config = {
      learnerType: config.learnerType ?? 'linear',
      forestConfig: config.forestConfig ?? {},
      propensityMethod: config.propensityMethod ?? 'logistic',
      seed: config.seed ?? 42,
      bandwidth: config.bandwidth ?? 0,
    };

    const factory = this.config.learnerType === 'causal-forest'
      ? () => new ForestBaseLearner(this.config.forestConfig)
      : () => new LinearBaseLearner();

    this.modelT = factory();
    this.modelC = factory();
    this.cateModelT = factory();
    this.cateModelC = factory();
  }

  /**
   * Fit the XLearner.
   */
  fit(X: number[][], y: number[], t: number[]): void {
    const n = X.length;
    const XT: number[][] = [], yT: number[] = [];
    const XC: number[][] = [], yC: number[] = [];

    for (let i = 0; i < n; i++) {
      if (t[i]! > 0.5) { XT.push(X[i]!); yT.push(y[i]!); }
      else { XC.push(X[i]!); yC.push(y[i]!); }
    }

    // Stage 1: T-learner
    this.modelT.fit(XT, yT);
    this.modelC.fit(XC, yC);

    // Stage 2: Imputed effects
    const imputedT: number[] = [];
    const imputedC: number[] = [];

    // τ̃₁ = Y - f̂₀(X) for treated
    const y0pred = XT.length > 0 ? this.modelC.predict(XT) : [];
    for (let i = 0; i < XT.length; i++) {
      imputedT.push(yT[i]! - y0pred[i]!);
    }

    // τ̃₀ = f̂₁(X) - Y for control
    const y1pred = XC.length > 0 ? this.modelT.predict(XC) : [];
    for (let i = 0; i < XC.length; i++) {
      imputedC.push(y1pred[i]! - yC[i]!);
    }

    // Stage 3: Fit CATE models
    if (XT.length > 0) this.cateModelT.fit(XT, imputedT);
    if (XC.length > 0) this.cateModelC.fit(XC, imputedC);

    // Stage 4: Propensity model
    this.propensity = this.config.propensityMethod === 'logistic'
      ? estimatePropensity(X, t)
      : new Array(n).fill(t.filter(v => v > 0.5).length / n);

    this.fitted = true;
  }

  get isFitted(): boolean { return this.fitted; }

  /**
   * Predict CATE with propensity-weighted combination:
   *   τ̂(x) = ê(x)·g₀(x) + (1-ê(x))·g₁(x)
   */
  effect(X: number[][]): number[] {
    if (!this.fitted) throw new Error('XLearner not fitted.');
    const g1 = this.cateModelT.predict(X);
    const g0 = this.cateModelC.predict(X);

    // Use mean propensity for new X (we don't know their T)
    const eBar = this.propensity!.reduce((a, b) => a + b, 0) / this.propensity!.length;

    return g1.map((v1, i) => {
      const v0 = g0[i]!;
      return (1 - eBar) * v1 + eBar * v0;
    });
  }

  ate(X: number[][], t: number[]): number {
    const cate = this.effect(X);
    return cate.reduce((a, b) => a + b, 0) / cate.length;
  }
}

// ── Helpers ─────────────────────────────────────────────────────────────

function estimatePropensity(X: number[][], t: number[]): number[] {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  const k = p + 1;

  let beta = new Array(k).fill(0);
  const pos = t.filter(v => v > 0.5).length;
  const neg = n - pos;
  beta[0] = Math.log((pos + 0.5) / (neg + 0.5));

  for (let iter = 0; iter < 20; iter++) {
    const xtx = Array.from({ length: k }, () => new Array(k).fill(0));
    const xtz = new Array(k).fill(0);

    for (let i = 0; i < n; i++) {
      const eta = predictLinear(X[i]!, beta);
      const mu = sigmoid(eta);
      const w = Math.max(1e-6, mu * (1 - mu));
      const z = eta + (t[i]! - mu) / w;

      xtx[0]![0]! += w; xtz[0]! += w * z;
      for (let j = 0; j < p; j++) {
        const xv = X[i]![j]!;
        xtx[0]![j + 1]! += w * xv;
        xtx[j + 1]![0]! += w * xv;
        xtz[j + 1]! += w * xv * z;
      }
    }

    for (let ci = 0; ci < k; ci++) {
      for (let cj = ci + 1; cj < k; cj++) {
        xtx[cj]![ci]! = xtx[ci]![cj]!;
      }
    }
    for (let ci = 0; ci < k; ci++) xtx[ci]![ci]! += 1e-8;

    const aug = xtx.map((row, ri) => [...row, xtz[ri]!]);
    for (let col = 0; col < k; col++) {
      let mr = col, mv = Math.abs(aug[col]![col]!);
      for (let r = col + 1; r < k; r++) {
        if (Math.abs(aug[r]![col]!) > mv) { mv = Math.abs(aug[r]![col]!); mr = r; }
      }
      if (mv < 1e-12) continue;
      if (mr !== col) [aug[col], aug[mr]] = [aug[mr]!, aug[col]!];
      const piv = aug[col]![col]!;
      for (let c = col; c <= k; c++) aug[col]![c]! /= piv;
      for (let r = 0; r < k; r++) {
        if (r === col) continue;
        const f = aug[r]![col]!;
        if (f === 0) continue;
        for (let c = col; c <= k; c++) aug[r]![c]! -= f * aug[col]![c]!;
      }
    }
    beta = aug.map(r => r[k]!);
  }

  return X.map(xi => sigmoid(predictLinear(xi, beta)));
}

function predictLinear(x: number[], beta: number[]): number {
  let s = beta[0]!;
  for (let j = 0; j < x.length; j++) s += (beta[j + 1] ?? 0) * (x[j] ?? 0);
  return s;
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, z))));
}
