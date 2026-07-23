/**
 * Double Machine Learning (DML) — debiased CATE estimation.
 *
 * Reference: Chernozhukov, Chetverikov, Demirer, Duflo, Hansen,
 *   Newey & Robins (2018). "Double/Debiased Machine Learning for
 *   Treatment and Structural Parameters." The Econometrics Journal.
 *
 * DML uses orthogonal moment conditions and cross-fitting to remove
 * the regularization bias that plagues naive "plug-in" ML approaches.
 * It provides √n-consistent, asymptotically normal estimates even when
 * high-dimensional ML models are used for nuisance functions.
 *
 * The key innovation: split data into K folds. On each fold:
 *   1. Estimate E[Y|X] and E[T|X] using ML on the OTHER K-1 folds
 *   2. Compute orthogonalized scores on the held-out fold
 *   3. Average across folds for the final estimate
 *
 * @packageDocumentation
 */

/**
 * DML estimator for Average Treatment Effect (ATE).
 *
 * @param X — feature matrix (n × p)
 * @param y — outcome vector (n)
 * @param t — binary treatment vector (n)
 * @param nFolds — number of cross-fitting folds (default 5)
 * @returns ATE estimate with standard error
 */
export function doubleMLATE(
  X: number[][],
  y: number[],
  t: number[],
  nFolds: number = 5,
): { ate: number; se: number } {
  const n = X.length;
  if (n < nFolds * 2) return { ate: naiveATE(y, t), se: 0 };

  // Create K-fold splits
  const indices = shuffle(Array.from({ length: n }, (_, i) => i));
  const foldSize = Math.floor(n / nFolds);

  const scores: number[] = [];

  for (let k = 0; k < nFolds; k++) {
    const testStart = k * foldSize;
    const testEnd = k === nFolds - 1 ? n : (k + 1) * foldSize;
    const testIdx = indices.slice(testStart, testEnd);
    const trainIdx = [...indices.slice(0, testStart), ...indices.slice(testEnd)];

    // Step 1: Regress Y on X on training folds → residual R_Y = Y - Ê[Y|X]
    const yModel = estimateOutcomeModel(X, y, trainIdx);
    // Step 2: Regress T on X on training folds → residual R_T = T - Ê[T|X]
    const tModel = estimatePropensityModel(X, t, trainIdx);

    // Step 3: Compute FWL estimate on test fold: ATE = Σ(R_Y·R_T) / Σ(R_T²)
    let num = 0, den = 0;
    for (const i of testIdx) {
      const rY = y[i]! - predictLinear(X[i]!, yModel);
      const rT = t[i]! - predictLinear(X[i]!, tModel);
      num += rY * rT;
      den += rT * rT;
    }
    if (den > 1e-10) scores.push(num / den);
  }

  // Average score across all folds
  const ate = scores.reduce((a, b) => a + b, 0) / scores.length;

  // Standard error via influence function
  const ifVar = scores.reduce((s, s_i) => s + (s_i - ate) ** 2, 0) / scores.length;
  const se = Math.sqrt(ifVar / scores.length);

  return { ate, se };
}

/**
 * DML estimator for Conditional Average Treatment Effect (CATE).
 *
 * Estimates τ(x) = E[Y(1) - Y(0) | X = x] using cross-fitting.
 *
 * @returns CATE function that maps features to treatment effect
 */
export function doubleMLCATE(
  X: number[][],
  y: number[],
  t: number[],
  nFolds: number = 5,
): { cateFn: (x: number[]) => number; baselineATE: number } {
  const n = X.length;
  if (n < nFolds * 2) {
    const ate = naiveATE(y, t);
    return { cateFn: () => ate, baselineATE: ate };
  }

  const ateResult = doubleMLATE(X, y, t, nFolds);

  // For CATE, fit a linear interaction model on the orthogonalized scores
  // This provides a simple parametric approximation to τ(x)
  const scores: number[] = [];
  const foldSize = Math.floor(n / nFolds);
  const indices = shuffle(Array.from({ length: n }, (_, i) => i));

  for (let k = 0; k < nFolds; k++) {
    const testStart = k * foldSize;
    const testEnd = k === nFolds - 1 ? n : (k + 1) * foldSize;
    const testIdx = indices.slice(testStart, testEnd);
    const trainIdx = [...indices.slice(0, testStart), ...indices.slice(testEnd)];

    const yHat = estimateOutcomeModel(X, y, trainIdx);
    const tHat = estimatePropensityModel(X, t, trainIdx);

    for (const i of testIdx) {
      const rY = y[i]! - predictLinear(X[i]!, yHat);
      const rT = t[i]! - sigmoid(predictLinear(X[i]!, tHat));
      scores[i] = rY / Math.max(0.1, Math.abs(rT) < 0.01 ? 0.1 : rT);
    }
  }

  // Fit linear model: τ̂(x) = β₀ + Σ β_j · x_j using orthogonal scores
  const p = X[0]?.length ?? 0;
  const XtX = Array.from({ length: p + 1 }, () => new Float64Array(p + 1));
  const Xty = new Float64Array(p + 1);

  for (let i = 0; i < n; i++) {
    XtX[0]![0] += 1;
    Xty[0] += scores[i]!;
    for (let j = 0; j < p; j++) {
      const xij = X[i]![j]!;
      XtX[0]![j + 1] += xij;
      XtX[j + 1]![0] += xij;
      Xty[j + 1] += xij * scores[i]!;
      for (let l = j; l < p; l++) {
        XtX[j + 1]![l + 1] += xij * X[i]![l]!;
        XtX[l + 1]![j + 1] = XtX[j + 1]![l + 1];
      }
    }
  }

  // Solve with Gaussian elimination
  const aug = XtX.map((row, ri) => [...Array.from(row), Xty[ri]!]);
  const k = p + 1;
  for (let col = 0; col < k; col++) {
    let pivot = col;
    for (let row = col + 1; row < k; row++) {
      if (Math.abs(aug[row]![col]!) > Math.abs(aug[pivot]![col]!)) pivot = row;
    }
    [aug[col], aug[pivot]] = [aug[pivot]!, aug[col]!];
    if (Math.abs(aug[col]![col]!) < 1e-12) continue;
    for (let j = col; j <= k; j++) aug[col]![j]! /= aug[col]![col]!;
    for (let row = 0; row < k; row++) {
      if (row === col) continue;
      const f = aug[row]![col]!;
      for (let j = col; j <= k; j++) aug[row]![j]! -= f * aug[col]![j]!;
    }
  }

  const beta0 = aug[0]![k]!;
  const betas = aug.slice(1).map(r => r[k]!);

  return {
    baselineATE: ateResult.ate,
    cateFn: (x: number[]) => {
      let tau = beta0;
      for (let j = 0; j < Math.min(p, x.length); j++) tau += (betas[j] ?? 0) * (x[j] ?? 0);
      return tau;
    },
  };
}

// ── Nuisance function estimation ──────────────────────────────────────

function estimateOutcomeModel(X: number[][], y: number[], trainIdx: number[]): number[] {
  const p = X[0]?.length ?? 0;
  return fitLinearRegression(X, y, trainIdx, p);
}

function estimatePropensityModel(X: number[][], t: number[], trainIdx: number[]): number[] {
  const p = X[0]?.length ?? 0;
  return fitLinearRegression(X, t, trainIdx, p);
}

function fitLinearRegression(X: number[][], target: number[], idx: number[], p: number): number[] {
  const XtX = Array(p + 1).fill(0).map(() => new Float64Array(p + 1));
  const Xty = new Float64Array(p + 1);

  for (const i of idx) {
    XtX[0]![0] += 1;
    Xty[0] += target[i]!;
    for (let j = 0; j < p; j++) {
      const xj = X[i]![j]!;
      XtX[0]![j + 1] += xj;
      XtX[j + 1]![0] += xj;
      Xty[j + 1] += xj * target[i]!;
    }
  }

  // Simple OLS: solve XtX * beta = Xty
  const k = p + 1;
  const A = Array.from({ length: k }, () => Array(k + 1).fill(0));
  for (let i = 0; i < k; i++) { for (let j = 0; j < k; j++) A[i]![j] = XtX[i]![j]!; A[i]![k] = Xty[i]!; }

  for (let col = 0; col < k; col++) {
    let pivot = col;
    for (let row = col + 1; row < k; row++) if (Math.abs(A[row]![col]!) > Math.abs(A[pivot]![col]!)) pivot = row;
    [A[col], A[pivot]] = [A[pivot]!, A[col]!];
    if (Math.abs(A[col]![col]!) < 1e-12) continue;
    for (let j = col; j <= k; j++) A[col]![j]! /= A[col]![col]!;
    for (let row = 0; row < k; row++) {
      if (row === col) continue;
      const f = A[row]![col]!;
      for (let j = col; j <= k; j++) A[row]![j]! -= f * A[col]![j]!;
    }
  }
  return A.map(r => r[k]!);
}

// ── Helpers ───────────────────────────────────────────────────────────

function predictLinear(x: number[], beta: number[]): number {
  let s = beta[0]!;
  for (let j = 0; j < x.length; j++) s += (beta[j + 1] ?? 0) * (x[j] ?? 0);
  return s;
}

function sigmoid(z: number): number { return 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, z)))); }

function naiveATE(y: number[], t: number[]): number {
  let tSum = 0, tN = 0, cSum = 0, cN = 0;
  for (let i = 0; i < y.length; i++) {
    if (t[i]! > 0.5) { tSum += y[i]!; tN++; }
    else { cSum += y[i]!; cN++; }
  }
  return (tN > 0 ? tSum / tN : 0) - (cN > 0 ? cSum / cN : 0);
}

function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j]!, arr[i]!];
  }
  return arr;
}
