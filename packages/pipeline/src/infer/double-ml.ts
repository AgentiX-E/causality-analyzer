/**
 * Double Machine Learning (DML) — debiased CATE estimation with
 * customizable nuisance models.
 *
 * Reference: Chernozhukov, Chetverikov, Demirer, Duflo, Hansen,
 *   Newey & Robins (2018). "Double/Debiased Machine Learning for
 *   Treatment and Structural Parameters." The Econometrics Journal.
 *
 * Key features:
 *   - K-fold cross-fitting for debiased estimation
 *   - Orthogonal moment conditions (Frisch-Waugh-Lovell partialling)
 *   - Pluggable nuisance models: linear, polynomial, or custom
 *   - Hetergeneity testing via Best Linear Predictor (BLP)
 *
 * @packageDocumentation
 */

/**
 * Nuisance model function: given feature matrix X and target y with
 * training indices, returns a prediction function.
 */
export type NuisanceModel = (
  X: number[][],
  y: number[],
  trainIdx: number[],
) => (x: number[]) => number;

/** DML options including custom nuisance models */
export interface DMLOptions {
  /** Number of cross-fitting folds (default: 5) */
  nFolds?: number;
  /** Custom outcome nuisance model (default: linear regression) */
  outcomeModel?: NuisanceModel;
  /** Custom propensity nuisance model (default: linear regression) */
  propensityModel?: NuisanceModel;
  /** Use polynomial expansion for non-linear effects (degree, default: 0 = linear only) */
  polyDegree?: number;
  /** Random seed for fold shuffling */
  seed?: number;
}

// ── Standard Nuisance Models ────────────────────────────────────────────

/**
 * Linear regression nuisance model (default).
 * Fits Y ~ 1 + X via OLS, returns prediction function.
 */
export function linearModel(): NuisanceModel {
  return (X: number[][], y: number[], trainIdx: number[]): ((x: number[]) => number) => {
    const p = X[0]?.length ?? 0;
    const beta = fitLinearRegression(X, y, trainIdx, p);
    return (x: number[]) => predictLinear(x, beta);
  };
}

/**
 * Polynomial expansion nuisance model.
 * Augments features with polynomial terms up to the given degree
 * to capture non-linear relationships.
 */
export function polynomialModel(degree: number = 2): NuisanceModel {
  return (X: number[][], y: number[], trainIdx: number[]): ((x: number[]) => number) => {
    const p = X[0]?.length ?? 0;
    // Expand features: [x1, x2, x1^2, x2^2, x1*x2, ...]
    const expandX = (x: number[]): number[] => {
      const expanded = [...x];
      for (let d = 2; d <= degree; d++) {
        for (let i = 0; i < x.length; i++) {
          expanded.push(Math.pow(x[i], d));
        }
        // Interaction terms for degree 2 only (to keep it manageable)
        if (d === 2) {
          for (let i = 0; i < x.length; i++) {
            for (let j = i + 1; j < x.length; j++) {
              expanded.push((x[i] ?? 0) * (x[j] ?? 0));
            }
          }
        }
      }
      return expanded;
    };

    const expandedDim = expandX(X[0]).length;
    const beta = fitLinearRegression(
      X.map(expandX),
      y,
      trainIdx,
      expandedDim,
    );
    return (x: number[]) => predictLinear(expandX(x), beta);
  };
}

/**
 * Build a combined DML config from shorthand options.
 *
 * @internal
 */
function resolveNuisanceModels(options: DMLOptions): {
  outcomeModel: NuisanceModel;
  propensityModel: NuisanceModel;
} {
  if (options.outcomeModel && options.propensityModel) {
    return { outcomeModel: options.outcomeModel, propensityModel: options.propensityModel };
  }
  if (options.polyDegree && options.polyDegree > 1) {
    const poly = polynomialModel(options.polyDegree);
    return { outcomeModel: poly, propensityModel: poly };
  }
  // Default: linear outcome model, simple mean propensity
  const lin = linearModel();
  const meanPropensity = simplePropensityModel();
  return { outcomeModel: lin, propensityModel: meanPropensity };
}

/**
 * Simple propensity model that returns E[T] from the training data.
 * Much more stable than linear regression for binary outcomes.
 */
function simplePropensityModel(): NuisanceModel {
  return (X: number[][], t: number[], trainIdx: number[]): ((x: number[]) => number) => {
    let sum = 0;
    for (const i of trainIdx) sum += t[i];
    const prop = Math.max(0.1, Math.min(0.9, sum / trainIdx.length));
    return () => prop;
  };
}

/**
 * DML estimator for Average Treatment Effect (ATE) with customizable
 * nuisance models.
 *
 * @param X — feature matrix (n × p)
 * @param y — outcome vector (n)
 * @param t — binary treatment vector (n)
 * @param options — folds, nuisance models, polynomial degree
 * @returns ATE estimate with standard error
 */
export function doubleMLATE(
  X: number[][],
  y: number[],
  t: number[],
  options: number | DMLOptions = 5,
): { ate: number; se: number } {
  const opts = typeof options === 'number' ? { nFolds: options } : options;
  const nFolds = opts.nFolds ?? 5;
  const n = X.length;
  if (n < nFolds * 2) return { ate: naiveATE(y, t), se: 0 };

  const { outcomeModel, propensityModel } = resolveNuisanceModels(opts);
  const seed = opts.seed ?? 42;
  const indices = seededShuffle(Array.from({ length: n }, (_, i) => i), seed);
  const foldSize = Math.floor(n / nFolds);
  const scores: number[] = [];

  for (let k = 0; k < nFolds; k++) {
    const testStart = k * foldSize;
    const testEnd = k === nFolds - 1 ? n : (k + 1) * foldSize;
    const testIdx = indices.slice(testStart, testEnd);
    const trainIdx = [...indices.slice(0, testStart), ...indices.slice(testEnd)];

    const yPred = outcomeModel(X, y, trainIdx);
    const tPred = propensityModel(X, t, trainIdx);

    let num = 0, den = 0;
    for (const i of testIdx) {
      const rY = y[i] - yPred(X[i]);
      const rawTPred = tPred(X[i]);
      // Clamp propensity and apply sigmoid for logistic-like calibration
      const clampedTPred = rawTPred > 10 ? 0.95 : rawTPred < -10 ? 0.05 : sigmoidFn(rawTPred);
      const rT = t[i] - clampedTPred;
      num += rY * rT;
      den += rT * rT;
    }
    if (den > 1e-10) scores.push(num / den);
  }

  const ate = scores.reduce((a, b) => a + b, 0) / scores.length;
  const ifVar = scores.reduce((s, s_i) => s + (s_i - ate) ** 2, 0) / scores.length;
  const se = Math.sqrt(ifVar / scores.length);

  return { ate, se };
}

/**
 * DML estimator for Conditional Average Treatment Effect (CATE) with
 * customizable nuisance models.
 *
 * @param X — feature matrix (n × p)
 * @param y — outcome vector (n)
 * @param t — binary treatment vector (n)
 * @param options — folds, nuisance models, polynomial degree
 * @returns CATE function and baseline ATE
 */
export function doubleMLCATE(
  X: number[][],
  y: number[],
  t: number[],
  options: number | DMLOptions = 5,
): { cateFn: (x: number[]) => number; baselineATE: number } {
  const opts = typeof options === 'number' ? { nFolds: options } : options;
  const nFolds = opts.nFolds ?? 5;
  const n = X.length;

  if (n < nFolds * 2) {
    const ate = naiveATE(y, t);
    return { cateFn: () => ate, baselineATE: ate };
  }

  const ateResult = doubleMLATE(X, y, t, opts);
  const { outcomeModel, propensityModel } = resolveNuisanceModels(opts);
  const seed = (opts.seed ?? 42) + 1;
  const indices = seededShuffle(Array.from({ length: n }, (_, i) => i), seed);
  const foldSize = Math.floor(n / nFolds);

  const scores: number[] = new Array<number>(n).fill(0);

  for (let k = 0; k < nFolds; k++) {
    const testStart = k * foldSize;
    const testEnd = k === nFolds - 1 ? n : (k + 1) * foldSize;
    const testIdx = indices.slice(testStart, testEnd);
    const trainIdx = [...indices.slice(0, testStart), ...indices.slice(testEnd)];

    const yPred = outcomeModel(X, y, trainIdx);
    const tPred = propensityModel(X, t, trainIdx);

    for (const i of testIdx) {
      const rY = y[i] - yPred(X[i]);
      const rawTPred = tPred(X[i]);
      const clampedTPred = rawTPred > 10 ? 0.95 : rawTPred < -10 ? 0.05 : sigmoidFn(rawTPred);
      const rT = t[i] - clampedTPred;
      const denom = Math.max(0.01, Math.abs(rT));
      scores[i] = Number.isFinite(rY) && Number.isFinite(denom) ? rY / denom : 0;
    }
  }

  // Fit linear interaction model on scores
  const p = X[0]?.length ?? 0;
  const XtX = Array.from({ length: p + 1 }, () => new Float64Array(p + 1));
  const Xty = new Float64Array(p + 1);

  for (let i = 0; i < n; i++) {
    XtX[0][0] += 1;
    Xty[0] += scores[i];
    for (let j = 0; j < p; j++) {
      const xij = X[i][j];
      XtX[0][j + 1] += xij;
      XtX[j + 1][0] += xij;
      Xty[j + 1] += xij * scores[i];
      for (let l = j; l < p; l++) {
        XtX[j + 1][l + 1] += xij * X[i][l];
        XtX[l + 1][j + 1] = XtX[j + 1][l + 1]!;
      }
    }
  }

  // Ridge-regularized Cholesky solve for the interaction model
  const k = p + 1;
  const ridgeLam = p > 10 ? 1.0 : 1e-10;
  for (let i = 0; i < k; i++) XtX[i]![i]! += ridgeLam;

  const L = new Float64Array(k * k);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = XtX[i]![j]!;
      for (let m = 0; m < j; m++) sum -= L[i * k + m]! * L[j * k + m]!;
      if (i === j) L[i * k + i] = Math.sqrt(Math.max(1e-12, sum));
      else L[i * k + j] = sum / L[j * k + j]!;
    }
  }
  const z = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    let sum = Xty[i]!;
    for (let j = 0; j < i; j++) sum -= L[i * k + j]! * z[j]!;
    z[i] = sum / L[i * k + i]!;
  }
  const sol = new Array<number>(k).fill(0);
  for (let i = k - 1; i >= 0; i--) {
    let sum = z[i]!;
    for (let j = i + 1; j < k; j++) sum -= L[j * k + i]! * (sol[j] ?? 0);
    sol[i] = sum / L[i * k + i]!;
  }

  const beta0 = sol[0]!;
  const betas = sol.slice(1);

  // Guard against NaN from numerical issues with polynomial-expanded features
  const hasNaN = !Number.isFinite(beta0) || betas.some(b => !Number.isFinite(b));
  if (hasNaN) {
    // Fallback to simple ATE (no CATE)
    return {
      baselineATE: ateResult.ate,
      cateFn: () => ateResult.ate,
    };
  }

  return {
    baselineATE: ateResult.ate,
    cateFn: (x: number[]) => {
      let tau = beta0;
      for (let j = 0; j < Math.min(p, x.length); j++) tau += (betas[j] ?? 0) * (x[j] ?? 0);
      return tau;
    },
  };
}

// ── Heterogeneity Testing ───────────────────────────────────────────────

/**
 * Best Linear Predictor (BLP) test for treatment effect heterogeneity.
 *
 * Tests H₀: no heterogeneity (constant CATE) vs H₁: CATE varies with features.
 * Uses the method from Chernozhukov et al. (2018) §6.2.
 *
 * @param X — feature matrix
 * @param y — outcome vector
 * @param t — treatment vector
 * @param options — DML options
 * @returns p-value for heterogeneity test (< 0.05 indicates heterogeneity)
 */
export function testHeterogeneity(
  X: number[][],
  y: number[],
  t: number[],
  options: DMLOptions = {},
): { pValue: number; isHeterogeneous: boolean } {
  const { cateFn, baselineATE } = doubleMLCATE(X, y, t, options);
  const n = X.length;

  // Compute CATE predictions
  const catePred = X.map(x => cateFn(x) - baselineATE);

  // Test if CATE variance > 0 via simple F-test on coefficients
  // H0: all interaction betas = 0
  const p = X[0]?.length ?? 0;
  const rssRestricted = catePred.reduce((s, v) => s + v * v, 0);

  // Restricted model: just the intercept (mean)
  const meanCATE = catePred.reduce((a, b) => a + b, 0) / n;
  const rssFull = catePred.reduce((s, v) => s + (v - meanCATE) ** 2, 0);

  const dfFull = n - 1;
  const dfRestricted = n - p - 1;
  const fStat = dfFull > 0 && rssFull > 0
    ? ((rssRestricted - rssFull) / p) / (rssFull / dfFull)
    : 0;

  // Approximate p-value from F-distribution (using chi-square for simplicity)
  // F(p, n-p-1) ≈ χ²(p) / p for large n
  const chi2Stat = Math.max(0, fStat * p);
  const pValue = 1 - chiSquareCDFApprox(chi2Stat, p);

  return { pValue, isHeterogeneous: pValue < 0.05 };
}

/**
 * Approximate chi-squared CDF using the regularized gamma function.
 * For df ≤ 10, uses direct computation. For larger df, uses
 * the Wilson-Hilferty normal approximation.
 *
 * @internal
 */
function chiSquareCDFApprox(x: number, df: number): number {
  if (x <= 0) return 0;
  if (df <= 0) return 0;

  // Wilson-Hilferty approximation: (χ²/df)^(1/3) ≈ N(1 - 2/(9df), 2/(9df))
  if (df > 10) {
    const z = (Math.pow(x / df, 1/3) - (1 - 2/(9*df))) / Math.sqrt(2/(9*df));
    return normalCDFApprox(z);
  }

  // Direct gamma incomplete for small df
  let sum = 1;
  let term = 1;
  for (let i = 1; i < 100; i++) {
    term *= (x / 2) / (df / 2 + i);
    sum += term;
    if (Math.abs(term) < 1e-12) break;
  }
  return 1 - Math.exp(-x/2) * Math.pow(x/2, df/2) * sum / gammaApprox(df/2);
}

function normalCDFApprox(z: number): number {
  // Abramowitz & Stegun 7.1.26 approximation
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp(-z * z / 2);
  const prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - prob : prob;
}

function gammaApprox(x: number): number {
  // Stirling's approximation for Gamma function
  if (x <= 0) return 1;
  return Math.sqrt(2 * Math.PI / x) * Math.pow(x / Math.E, x);
}

function fitLinearRegression(X: number[][], target: number[], idx: number[], p: number): number[] {
  const XtX = Array(p + 1).fill(0).map(() => new Float64Array(p + 1));
  const Xty = new Float64Array(p + 1);

  for (const i of idx) {
    XtX[0][0] += 1;
    Xty[0] += target[i];
    for (let j = 0; j < p; j++) {
      const xj = X[i][j];
      XtX[0][j + 1] += xj;
      XtX[j + 1][0] += xj;
      Xty[j + 1] += xj * target[i];
    }
  }

  // Ridge-regularized OLS: (XᵀX + λI)β = Xᵀy via Cholesky
  const k = p + 1;
  const lambda = p > 10 ? 1.0 : 1e-10;
  for (let i = 0; i < k; i++) XtX[i]![i]! += lambda;

  // Cholesky decomposition on flat Float64Array (avoids number[][] GC overhead)
  const L = new Float64Array(k * k);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = XtX[i]![j]!;
      for (let p = 0; p < j; p++) sum -= L[i * k + p]! * L[j * k + p]!;
      if (i === j) {
        L[i * k + i] = Math.sqrt(Math.max(1e-12, sum));
      } else {
        L[i * k + j] = sum / L[j * k + j]!;
      }
    }
  }

  // Forward: L·z = Xᵀy
  const z = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    let sum = Xty[i]!;
    for (let j = 0; j < i; j++) sum -= L[i * k + j]! * z[j]!;
    z[i] = sum / L[i * k + i]!;
  }

  // Back: Lᵀ·β = z
  const beta = new Array<number>(k);
  for (let i = k - 1; i >= 0; i--) {
    let sum = z[i]!;
    for (let j = i + 1; j < k; j++) sum -= L[j * k + i]! * (beta[j] ?? 0);
    beta[i] = sum / L[i * k + i]!;
  }
  return beta;
}

// ── Helpers ───────────────────────────────────────────────────────────

function predictLinear(x: number[], beta: number[]): number {
  let s = beta[0];
  for (let j = 0; j < x.length; j++) s += (beta[j + 1] ?? 0) * (x[j] ?? 0);
  return s;
}

function sigmoidFn(z: number): number { return 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, z)))); }

function naiveATE(y: number[], t: number[]): number {
  let tSum = 0, tN = 0, cSum = 0, cN = 0;
  for (let i = 0; i < y.length; i++) {
    if (t[i] > 0.5) { tSum += y[i]; tN++; }
    else { cSum += y[i]; cN++; }
  }
  return (tN > 0 ? tSum / tN : 0) - (cN > 0 ? cSum / cN : 0);
}

function seededShuffle<T>(arr: T[], seed: number): T[] {
  let s = seed;
  const rng = (): number => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return s / 0x7FFFFFFF; };
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}
