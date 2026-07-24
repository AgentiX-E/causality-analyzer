/**
 * Numerical Optimization — L-BFGS and Adam optimizers.
 *
 * L-BFGS: Limited-memory BFGS quasi-Newton method for smooth unconstrained
 * optimization (Nocedal & Wright, 2006, §7.2). Suitable for NOTEARS
 * DAG-constrained optimization and other smooth objective functions.
 *
 * Adam: Adaptive moment estimation (Kingma & Ba, 2015) for stochastic
 * gradient-based optimization with per-parameter step sizes.
 *
 * Both operate on flat Float64Array parameters for maximum flexibility.
 *
 * @packageDocumentation
 */

// ── L-BFGS ────────────────────────────────────────────────────────────

export interface LBFGSConfig {
  /** Convergence tolerance on gradient norm */
  gtol: number;
  /** Maximum iterations */
  maxIter: number;
  /** History size for quasi-Newton approximation */
  m: number;
  /** Initial step size (line search) — 0 = auto */
  stepLength: number;
}

const DEFAULT_LBFGS: LBFGSConfig = {
  gtol: 1e-6,
  maxIter: 200,
  m: 10,
  stepLength: 0,
};

export interface LBFGSResult {
  x: Float64Array;
  value: number;
  gradNorm: number;
  iterations: number;
  converged: boolean;
}

/**
 * L-BFGS minimizer.
 *
 * @param f — objective function (x) => [value, gradient]
 * @param x0 — initial point
 * @param config — optimizer settings
 */
export function lbfgs(
  f: (x: Float64Array) => [number, Float64Array],
  x0: Float64Array,
  config: Partial<LBFGSConfig> = {},
): LBFGSResult {
  const cfg = { ...DEFAULT_LBFGS, ...config };
  const n = x0.length;
  const x = new Float64Array(x0);
  const [, g] = f(x);
  let gNorm = norm2(g);

  if (gNorm < cfg.gtol) {
    const [v] = f(x);
    return { x, value: v, gradNorm: gNorm, iterations: 0, converged: true };
  }

  // Storage for s_k, y_k pairs (limited memory)
  const sList: Float64Array[] = [];
  const yList: Float64Array[] = [];
  const rhoList: number[] = [];

  let grad: Float64Array = new Float64Array(g);
  const dir = new Float64Array(n);
  const q = new Float64Array(n);
  const alphaArr = new Float64Array(cfg.m);

  for (let k = 0; k < n; k++) dir[k] = -grad[k]!;

  for (let iter = 0; iter < cfg.maxIter; iter++) {
    // Line search (Armijo backtracking)
    const step = cfg.stepLength > 0 ? cfg.stepLength : 1.0;
    const [alpha, xNew, fNew, gradNew] = lineSearch(f, x, dir, grad, step);
    if (alpha === 0) break;

    const gradNewNorm = norm2(gradNew);
    if (gradNewNorm < cfg.gtol) {
      return { x: xNew, value: fNew, gradNorm: gradNewNorm, iterations: iter + 1, converged: true };
    }

    // Compute s_k = x_{k+1} - x_k, y_k = g_{k+1} - g_k
    const s = new Float64Array(n);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      s[i] = xNew[i]! - x[i]!;
      y[i] = gradNew[i]! - grad[i]!;
    }

    const ys = dot(y, s);
    if (ys <= 1e-15) {
      // Reset direction if curvature is near-zero
      grad = gradNew as unknown as Float64Array;
      for (let i = 0; i < n; i++) { dir[i] = -grad[i]!; x[i] = xNew[i]!; }
      continue;
    }

    const rho = 1 / ys;

    // Store pair
    if (sList.length >= cfg.m) { sList.shift(); yList.shift(); rhoList.shift(); }
    sList.push(s);
    yList.push(y);
    rhoList.push(rho);

    // Two-loop recursion
    for (let i = 0; i < n; i++) q[i] = gradNew[i]!;
    const len = sList.length;

    for (let j = len - 1; j >= 0; j--) {
      alphaArr[j] = rhoList[j]! * dot(sList[j]!, q);
      for (let i = 0; i < n; i++) q[i] = q[i]! - alphaArr[j]! * yList[j]![i]!;
    }

    // Scale initial Hessian
    const gamma = ys / dot(y, y);
    for (let i = 0; i < n; i++) dir[i] = gamma * q[i]!;

    for (let j = 0; j < len; j++) {
      const beta = rhoList[j]! * dot(yList[j]!, dir);
      for (let i = 0; i < n; i++) dir[i] = dir[i]! + sList[j]![i]! * (alphaArr[j]! - beta);
    }

    // Negate for descent
    for (let i = 0; i < n; i++) { dir[i] = -dir[i]!; x[i] = xNew[i]!; }
    grad = gradNew;
    gNorm = gradNewNorm;
  }

  const [v] = f(x);
  return { x, value: v, gradNorm: gNorm, iterations: cfg.maxIter, converged: false };
}

// ── Adam ──────────────────────────────────────────────────────────────

export interface AdamConfig {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  maxIter: number;
  gtol: number;
}

const DEFAULT_ADAM: AdamConfig = {
  lr: 1e-3,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
  maxIter: 1000,
  gtol: 1e-5,
};

/**
 * Adam optimizer for stochastic objectives.
 */
export function adam(
  f: (x: Float64Array) => [number, Float64Array],
  x0: Float64Array,
  config: Partial<AdamConfig> = {},
): LBFGSResult {
  const cfg = { ...DEFAULT_ADAM, ...config };
  const n = x0.length;
  const x = new Float64Array(x0);
  const m = new Float64Array(n);
  const v = new Float64Array(n);
  let bestVal = Infinity;
  let bestX = new Float64Array(x0);

  for (let t = 0; t < cfg.maxIter; t++) {
    const [, g] = f(x);
    const gNorm = norm2(g);
    if (gNorm < cfg.gtol) {
      const [val] = f(x);
      return { x, value: val, gradNorm: gNorm, iterations: t + 1, converged: true };
    }

    for (let i = 0; i < n; i++) {
      const gi = g[i]!;
      m[i] = cfg.beta1 * m[i]! + (1 - cfg.beta1) * gi;
      v[i] = cfg.beta2 * v[i]! + (1 - cfg.beta2) * gi * gi;

      const mHat = m[i]! / (1 - Math.pow(cfg.beta1, t + 1));
      const vHat = v[i]! / (1 - Math.pow(cfg.beta2, t + 1));

      x[i] = x[i]! - cfg.lr * mHat / (Math.sqrt(vHat) + cfg.eps);
    }

    const [val] = f(x);
    if (val < bestVal) { bestVal = val; bestX = new Float64Array(x); }
  }

  const [val] = f(bestX);
  const [, gFinal] = f(bestX);
  return { x: bestX, value: val, gradNorm: norm2(gFinal), iterations: cfg.maxIter, converged: false };
}

// ── Line Search (Armijo backtracking) ─────────────────────────────────

function lineSearch(
  f: (x: Float64Array) => [number, Float64Array],
  x: Float64Array,
  dir: Float64Array,
  _g: Float64Array,
  step0: number,
): [number, Float64Array, number, Float64Array] {
  const n = x.length;
  const c = 1e-4;
  const tau = 0.5;
  let step = step0;
  const [f0] = f(x);

  // Compute directional derivative g^T * dir = -||dir||² (since dir = -g initially)
  const dgInit = -dot(dir, dir); // initial descent = -g^T * g after first iteration

  for (let j = 0; j < 30; j++) {
    const xNew = new Float64Array(n);
    for (let i = 0; i < n; i++) xNew[i] = x[i]! + step * dir[i]!;

    const [fNew, gNew] = f(xNew);
    if (isNaN(fNew)) { step *= tau; continue; }

    // Armijo condition: f(x + step*d) ≤ f(x) + c*step*∇f^T*d
    // We approximate with the initial descent
    if (fNew <= f0 + c * step * dgInit) {
      return [step, xNew, fNew, gNew];
    }
    step *= tau;
  }
  return [0, x, f0, _g]; // failure
}

// ── Vector Utilities ──────────────────────────────────────────────────

function norm2(v: Float64Array): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i]! * v[i]!;
  return Math.sqrt(s);
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) s += a[i]! * b[i]!;
  return s;
}
