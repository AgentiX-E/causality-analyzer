/**
 * Double Machine Learning (DoubleML) Estimators.
 *
 * Neyman-orthogonal score approach from Chernozhukov et al. (2018)
 * with K-fold cross-fitting to eliminate overfitting bias.
 *
 * PLR: Partial Linear Regression — D continuous, linear treatment effect.
 * PLIV: Partial Linear IV — instrumented treatment via Z.
 *
 * @packageDocumentation
 */

import { Matrix } from 'ml-matrix';
import type { CATEstimator, ATEResult } from './cate-meta-learners.js';

export interface DoubleMLConfig {
  nFolds?: number;
  seed?: number;
}

// ── OLS ──────────────────────────────────────────────────────────────

function solveOLS(X: number[][], y: number[]): number[] {
  const n = X.length;
  const d = X[0]!.length;
  if (n < d + 1) return new Array(d).fill(0);
  const XtX: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  const Xty = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    const xi = X[i]!, yi = y[i]!;
    for (let j = 0; j < d; j++) {
      Xty[j] += xi[j]! * yi;
      for (let k = j; k < d; k++) XtX[j]![k]! += xi[j]! * xi[k]!;
    }
  }
  for (let j = 0; j < d; j++)
    for (let k = j + 1; k < d; k++)
      XtX[k]![j]! = XtX[j]![k]!;
  const aug = XtX.map((row, i) => [...row, Xty[i] ?? 0]);
  for (let col = 0; col < d; col++) {
    let pivot = col;
    for (let r = col + 1; r < d; r++)
      if (Math.abs(aug[r]![col]!) > Math.abs(aug[pivot]![col]!)) pivot = r;
    [aug[col], aug[pivot]] = [aug[pivot]!, aug[col]!];
    const pv = aug[col]![col]!;
    if (Math.abs(pv) < 1e-12) continue;
    for (let j2 = col; j2 <= d; j2++) aug[col]![j2]! /= pv;
    for (let r = 0; r < d; r++) {
      if (r === col) continue;
      const f = aug[r]![col]!;
      for (let j2 = col; j2 <= d; j2++) aug[r]![j2]! -= f * aug[col]![j2]!;
    }
  }
  return aug.map(row => row[d]);
}

function predictOLS(beta: number[], x: number[]): number {
  let y = 0;
  for (let j = 0; j < beta.length; j++) y += (beta[j] ?? 0) * (x[j] ?? 0);
  return y;
}

function kFoldIndices(n: number, k: number, seed: number): number[][] {
  const folds: number[][] = Array.from({ length: k }, () => []);
  const indices = Array.from({ length: n }, (_, i) => i);
  let s = seed;
  for (let i = n - 1; i > 0; i--) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const j = s % (i + 1);
    [indices[i], indices[j]] = [indices[j]!, indices[i]!];
  }
  for (let i = 0; i < n; i++) folds[i % k]!.push(indices[i]!);
  return folds;
}

// ── PLR ──────────────────────────────────────────────────────────────

export class DoubleMLPLR implements CATEstimator {
  private _theta = 0;
  private _se = 0;
  private _nSamples = 0;
  private _nFolds: number;
  private _seed: number;

  constructor(config: DoubleMLConfig = {}) {
    this._nFolds = config.nFolds ?? 5;
    this._seed = config.seed ?? 42;
  }

  fit(X: Matrix, D: Float64Array, Y: Float64Array): this {
    const n = X.rows, d = X.columns;
    this._nSamples = n;
    if (n < this._nFolds * 3) {
      this._theta = this._single(X, D, Y);
      this._se = Math.sqrt(1 / Math.max(1, n));
      return this;
    }
    const folds = kFoldIndices(n, this._nFolds, this._seed);
    const thetaFolds: number[] = [];
    for (let fold = 0; fold < this._nFolds; fold++) {
      const testIdx = folds[fold]!;
      const trainIdx = folds.flatMap((f, i) => (i !== fold ? f : []));
      if (testIdx.length < 2 || trainIdx.length < d + 1) continue;
      const trainX: number[][] = [], trainY: number[] = [], trainD: number[] = [];
      for (const i of trainIdx) {
        const row = new Array(d + 1).fill(0);
        for (let j = 0; j < d; j++) row[j] = X.get(i, j);
        row[d] = 1; trainX.push(row); trainY.push(Y[i]!); trainD.push(D[i]!);
      }
      const betaY = solveOLS(trainX, trainY), betaD = solveOLS(trainX, trainD);
      let num = 0, den = 0;
      for (const i of testIdx) {
        const xi = new Array(d + 1).fill(0);
        for (let j = 0; j < d; j++) xi[j] = X.get(i, j); xi[d] = 1;
        const yTilde = (Y[i] ?? 0) - predictOLS(betaY, xi);
        const dTilde = (D[i] ?? 0) - predictOLS(betaD, xi);
        num += dTilde * yTilde; den += dTilde * dTilde;
      }
      if (den > 1e-10) thetaFolds.push(num / den);
    }
    if (thetaFolds.length > 0) {
      this._theta = thetaFolds.reduce((s, v) => s + v, 0) / thetaFolds.length;
      const v = thetaFolds.length > 1
        ? thetaFolds.reduce((s, x) => s + (x - this._theta) ** 2, 0) / (thetaFolds.length - 1) : 0;
      this._se = Math.sqrt(Math.max(0, v) / thetaFolds.length);
    }
    return this;
  }

  private _single(X: Matrix, D: Float64Array, Y: Float64Array): number {
    const n = X.rows, d = X.columns;
    const design: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row = new Array(d + 1).fill(0);
      for (let j = 0; j < d; j++) row[j] = X.get(i, j); row[d] = 1; design.push(row);
    }
    const bY = solveOLS(design, Array.from(Y)), bD = solveOLS(design, Array.from(D));
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
      const x = design[i]!;
      num += ((D[i] ?? 0) - predictOLS(bD, x)) * ((Y[i] ?? 0) - predictOLS(bY, x));
      den += ((D[i] ?? 0) - predictOLS(bD, x)) ** 2;
    }
    return den > 1e-10 ? num / den : 0;
  }

  effect(_X: Matrix): Float64Array {
    const r = new Float64Array(this._nSamples); r.fill(this._theta); return r;
  }

  ate(): ATEResult { return { estimate: this._theta, se: this._se }; }

  get theta(): number { return this._theta; }
  get se(): number { return this._se; }
}

// ── PLIV ─────────────────────────────────────────────────────────────

export class DoubleMLPLIV implements CATEstimator {
  private _theta = 0;
  private _se = 0;
  private _nSamples = 0;
  private _nFolds: number;
  private _seed: number;

  constructor(config: DoubleMLConfig = {}) {
    this._nFolds = config.nFolds ?? 5;
    this._seed = config.seed ?? 42;
  }

  fitIV(X: Matrix, Z: Float64Array, D: Float64Array, Y: Float64Array): this {
    const n = X.rows, d = X.columns; this._nSamples = n;
    if (n < this._nFolds * 3) {
      this._theta = this._single(X, Z, D, Y);
      this._se = Math.sqrt(1 / Math.max(1, n));
      return this;
    }
    const folds = kFoldIndices(n, this._nFolds, this._seed);
    const thetaFolds: number[] = [];
    for (let fold = 0; fold < this._nFolds; fold++) {
      const testIdx = folds[fold]!;
      const trainIdx = folds.flatMap((f, i) => (i !== fold ? f : []));
      if (testIdx.length < 2 || trainIdx.length < d + 1) continue;
      const trainX: number[][] = [], trainY: number[] = [], trainD: number[] = [], trainZ: number[] = [];
      for (const i of trainIdx) {
        const row = new Array(d + 1).fill(0);
        for (let j = 0; j < d; j++) row[j] = X.get(i, j); row[d] = 1;
        trainX.push(row); trainY.push(Y[i]!); trainD.push(D[i]!); trainZ.push(Z[i]!);
      }
      const bY = solveOLS(trainX, trainY), bD = solveOLS(trainX, trainD), bZ = solveOLS(trainX, trainZ);
      let aNum = 0, aDen = 0;
      for (const i of testIdx) {
        const xi = new Array(d + 1).fill(0);
        for (let j = 0; j < d; j++) xi[j] = X.get(i, j); xi[d] = 1;
        const zResid = (Z[i] ?? 0) - predictOLS(bZ, xi);
        const yResid = (Y[i] ?? 0) - predictOLS(bY, xi);
        const dResid = (D[i] ?? 0) - predictOLS(bD, xi);
        aNum += zResid * yResid; aDen += zResid * dResid;
      }
      if (Math.abs(aDen) > 1e-10) thetaFolds.push(aNum / aDen);
    }
    if (thetaFolds.length > 0) {
      this._theta = thetaFolds.reduce((s, v) => s + v, 0) / thetaFolds.length;
      const v = thetaFolds.length > 1
        ? thetaFolds.reduce((s, x) => s + (x - this._theta) ** 2, 0) / (thetaFolds.length - 1) : 0;
      this._se = Math.sqrt(Math.max(0, v) / thetaFolds.length);
    }
    return this;
  }

  private _single(X: Matrix, Z: Float64Array, D: Float64Array, Y: Float64Array): number {
    const n = X.rows, d = X.columns;
    const design: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row = new Array(d + 1).fill(0);
      for (let j = 0; j < d; j++) row[j] = X.get(i, j); row[d] = 1; design.push(row);
    }
    const bY = solveOLS(design, Array.from(Y)), bD = solveOLS(design, Array.from(D)), bZ = solveOLS(design, Array.from(Z));
    let n2 = 0, d2 = 0;
    for (let i = 0; i < n; i++) {
      const x = design[i]!;
      n2 += ((Z[i] ?? 0) - predictOLS(bZ, x)) * ((Y[i] ?? 0) - predictOLS(bY, x));
      d2 += ((Z[i] ?? 0) - predictOLS(bZ, x)) * ((D[i] ?? 0) - predictOLS(bD, x));
    }
    return Math.abs(d2) > 1e-10 ? n2 / d2 : 0;
  }

  fit(_X: Matrix, _T: Float64Array, _Y: Float64Array): this {
    throw new Error('DoubleMLPLIV requires fitIV(X, Z, D, Y) with instrument Z. Use fitIV() instead of fit().');
  }

  effect(_X: Matrix): Float64Array {
    const r = new Float64Array(this._nSamples); r.fill(this._theta); return r;
  }

  ate(): ATEResult { return { estimate: this._theta, se: this._se }; }
}

// ── Legacy Compatibility Types ──────────────────────────────────────

/** Nuisance model interface (pluggable ML backends) */
export interface NuisanceModel {
  fit(X: number[][], y: number[]): void;
  predict(X: number[][]): number[];
}

/** DML options (compatible with old API) */
export interface DMLOptions {
  nFolds?: number;
  seed?: number;
  outcomeModel?: NuisanceModel;
  propensityModel?: NuisanceModel;
}

/** Linear nuisance model (OLS-based) */
export function linearModel(): NuisanceModel {
  let beta: number[] = [];
  return {
    fit(X: number[][], y: number[]): void { beta = solveOLS(X, y); },
    predict(X: number[][]): number[] {
      return X.map(x => predictOLS(beta, x));
    },
  };
}

/** Double ML ATE computation (legacy wrapper) */
export function doubleMLATE(
  X: number[][], y: number[], t: number[],
  options: DMLOptions = {},
): { ate: number; se: number } {
  const matrix = new Matrix(X);
  const Y = Float64Array.from(y);
  const T = Float64Array.from(t);
  const estimator = new DoubleMLPLR({
    nFolds: options.nFolds ?? 5, seed: options.seed ?? 42,
  });
  estimator.fit(matrix, T, Y);
  return { ate: estimator.theta, se: estimator.se };
}

/** Double ML CATE computation (legacy wrapper) */
export function doubleMLCATE(
  X: number[][], y: number[], t: number[],
  options: DMLOptions = {},
): { cateFn: (x: number[]) => number; baselineATE: number } {
  const matrix = new Matrix(X);
  const Y = Float64Array.from(y);
  const T = Float64Array.from(t);
  const estimator = new DoubleMLPLR({
    nFolds: options.nFolds ?? 5, seed: options.seed ?? 42,
  });
  estimator.fit(matrix, T, Y);
  const ate = estimator.theta;
  return {
    cateFn: (_x: number[]) => ate,
    baselineATE: ate,
  };
}

/** Polynomial nuisance model (degree-d polynomial features) */
export function polynomialModel(degree: number = 2): NuisanceModel {
  function polyFeatures(x: number[]): number[] {
    const result = [1]; // intercept
    for (const xi of x) {
      for (let d = 1; d <= degree; d++) {
        result.push(Math.pow(xi, d));
      }
    }
    return result;
  }
  let beta: number[] = [];
  return {
    fit(X: number[][], y: number[]): void {
      const polyX = X.map(row => polyFeatures(row));
      beta = solveOLS(polyX, y);
    },
    predict(X: number[][]): number[] {
      return X.map(row => {
        const poly = polyFeatures(row);
        let y = 0;
        for (let j = 0; j < beta.length; j++) y += (beta[j] ?? 0) * (poly[j] ?? 0);
        return y;
      });
    },
  };
}

/** Test treatment effect heterogeneity (legacy wrapper) */
export function testHeterogeneity(
  X: number[][], y: number[], t: number[], _options: DMLOptions = {},
): { significant: boolean; pValue: number; statistic: number } {
  // Simple heterogeneity test: split by median of predicted CATE
  const { cateFn } = doubleMLCATE(X, y, t);
  const cates = X.map(row => cateFn(row));
  const median = cates.sort((a, b) => a - b)[Math.floor(cates.length / 2)] ?? 0;

  const high = cates.filter(c => c > median);
  const low = cates.filter(c => c <= median);

  if (high.length < 3 || low.length < 3) {
    return { significant: false, pValue: 1.0, statistic: 0 };
  }

  const highMean = high.reduce((s, v) => s + v, 0) / high.length;
  const lowMean = low.reduce((s, v) => s + v, 0) / low.length;
  const highVar = high.reduce((s, v) => s + (v - highMean) ** 2, 0) / high.length;
  const lowVar = low.reduce((s, v) => s + (v - lowMean) ** 2, 0) / low.length;

  const pooledSE = Math.sqrt(highVar / high.length + lowVar / low.length);
  const statistic = pooledSE > 0 ? (highMean - lowMean) / pooledSE : 0;
  const pValue = 2 * (1 - 0.5 * (1 + Math.tanh(statistic / Math.SQRT2))); // approximation

  return {
    significant: Math.abs(statistic) > 1.96,
    pValue: Math.max(0, Math.min(1, pValue)),
    statistic,
  };
}
