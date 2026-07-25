/**
 * DAGMA — Directed Acyclic Graphs via M-matrices for Acyclicity.
 *
 * A continuous DAG optimization alternative to NOTEARS (Zheng et al. 2018)
 * that uses the log-determinant constraint instead of trace-exponential.
 *
 * Key advantages over NOTEARS:
 *   - No matrix exponential (which was numerically unstable)
 *   - Log-det constraint h(W) = -log det(I - W⊙W) has cleaner gradient
 *   - Faster convergence in practice
 *   - More robust to scaling
 *
 * Reference: Bello et al. (NeurIPS 2022).
 *            "DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization."
 *
 * @packageDocumentation
 */
import { CausalGraph } from './causal-graph.js';
import { lbfgs } from '@agentix-e/causality-analyzer-core';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';

export interface DAGMAConfig {
  lambda1: number;
  rho: number;
  rhoFactor: number;
  maxOuterIter: number;
  tol: number;
  wThreshold: number;
  seed?: number;
}

const DEFAULTS: DAGMAConfig = {
  lambda1: 0.1,
  rho: 1.0,
  rhoFactor: 10,
  maxOuterIter: 15,
  tol: 1e-8,
  wThreshold: 0.3,
};

export function dagmaAlgorithm(
  XArr: number[][],
  nodeNames: string[],
  config: Partial<DAGMAConfig> = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph; W: Float64Array; h: number } {
  const cfg = { ...DEFAULTS, ...config };
  const n = XArr.length;
  const d = nodeNames.length;

  // Z-score normalize
  const X = new Float64Array(n * d);
  for (let j = 0; j < d; j++) {
    let sum = 0, sq = 0;
    for (let i = 0; i < n; i++) {
      const v = XArr[i]![j]!;
      sum += v; sq += v * v;
    }
    const mean = sum / n;
    const std = Math.sqrt(Math.max(1e-10, sq / n - mean * mean));
    for (let i = 0; i < n; i++) X[i * d + j] = (XArr[i]![j]! - mean) / std;
  }

  // Precompute covariance X^T X / n
  const cov = new Float64Array(d * d);
  for (let j = 0; j < d; j++)
    for (let k = j; k < d; k++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += X[i * d + j]! * X[i * d + k]!;
      cov[j * d + k] = cov[k * d + j] = s / n;
    }

  // Augmented Lagrangian
  let W = new Float64Array(d * d);
  let alpha = 0;
  let rho = cfg.rho;

  for (let outer = 0; outer < cfg.maxOuterIter; outer++) {
    const sub = lbfgs(
      w => dagmaLoss(w, d, cov, alpha, rho, cfg.lambda1),
      W, { maxIter: 300, gtol: cfg.tol, m: 15 },
    );
    W = new Float64Array(sub.x);

    const h = logDetH(W, d);
    if (h <= 1e-6) break;
    alpha += rho * h;
    rho *= cfg.rhoFactor;
  }

  // Threshold to DAG
  const g = new CausalGraph([...nodeNames]);
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      if (i !== j && Math.abs(W[i * d + j]!) > cfg.wThreshold)
        g.addEdge(nodeNames[i]!, nodeNames[j]!);

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);
  return { graph: g, W, h: logDetH(W, d) };
}

// ── DAGMA Loss ─────────────────────────────────────────────────────

function dagmaLoss(
  w: Float64Array, d: number, cov: Float64Array,
  alpha: number, rho: number, lambda1: number,
): [number, Float64Array] {
  // f(W) = 0.5 * ||X - XW||² / n
  let f = 0;
  const gf = new Float64Array(d * d);

  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      let wCov = 0;
      for (let l = 0; l < d; l++) wCov += w[i * d + l]! * cov[l * d + j]!;
      gf[i * d + j] = -cov[i * d + j]! + wCov;
      f -= w[i * d + j]! * cov[i * d + j]!;
      f += 0.5 * w[i * d + j]! * wCov;
    }
  }
  for (let i = 0; i < d; i++) f += 0.5 * cov[i * d + i]!;

  // h(W) = -log det(I - W⊙W) and gradient ∂h = 2·(I - W⊙W)^{-1}ᵀ ⊙ W
  const [h, dh] = logDetHAndGrad(w, d);

  // Augmented Lagrangian
  const loss = f + lambda1 * l1Norm(w, d * d) + alpha * h + 0.5 * rho * h * h;
  const grad = new Float64Array(d * d);
  const coeff = alpha + rho * h;

  for (let i = 0; i < d * d; i++) {
    grad[i] = gf[i]! + lambda1 * (w[i]! > 0 ? 1 : w[i]! < 0 ? -1 : 0) + coeff * dh[i]!;
  }

  return [loss, grad];
}

// ── Log-Det Constraint ─────────────────────────────────────────────

/**
 * h(W) = -log det(I - W⊙W)
 *
 * The constraint is h(W) = 0 iff W represents a DAG.
 * Uses Cholesky decomposition of I - W⊙W for log-det computation.
 * Gradient: ∂h/∂W = 2·(I - W⊙W)^{-T} ⊙ W
 */
function logDetHAndGrad(W: Float64Array, d: number): [number, Float64Array] {
  // Build M = I - W⊙W (element-wise square)
  const M = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    M[i * d + i] = 1;
    for (let j = 0; j < d; j++) {
      const w2 = W[i * d + j]! * W[i * d + j]!;
      M[i * d + j] -= w2;
    }
  }

  // Cholesky: M = L·Lᵀ, log det(M) = 2·Σ log(Lᵢᵢ)
  const L = cholesky(M, d);
  if (!L) {
    // M is not positive definite → h = Infinity (constraint violated)
    return [1e10, new Float64Array(d * d)];
  }

  let logDet = 0;
  for (let i = 0; i < d; i++) {
    const lii = L[i * d + i]!;
    if (lii <= 1e-10) return [1e10, new Float64Array(d * d)];
    logDet += Math.log(lii);
  }
  const h = -2 * logDet;

  // Gradient: (I - W⊙W)^{-1} via Cholesky back-solve
  // We need M^{-1}. Compute (L·Lᵀ)^{-1} = L^{-T}·L^{-1}
  const invM = invertFromCholesky(L, d);
  if (!invM) return [h, new Float64Array(d * d)];

  // dh = 2 * invMᵀ ⊙ W
  const grad = new Float64Array(d * d);
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      grad[i * d + j] = 2 * invM[j * d + i]! * W[i * d + j]!;

  return [h, grad];
}

function logDetH(W: Float64Array, d: number): number {
  return logDetHAndGrad(W, d)[0];
}

// ── Cholesky Decomposition ─────────────────────────────────────────

function cholesky(A: Float64Array, n: number): Float64Array | null {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i * n + j]!;
      for (let k = 0; k < j; k++) sum -= L[i * n + k]! * L[j * n + k]!;
      if (i === j) {
        if (sum <= 0) return null;
        L[i * n + i] = Math.sqrt(sum);
      } else {
        L[i * n + j] = sum / L[j * n + j]!;
      }
    }
  }
  return L;
}

function invertFromCholesky(L: Float64Array, n: number): Float64Array | null {
  // Invert lower triangular L via forward substitution, then L^{-T}·L^{-1}
  const invL = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    invL[i * n + i] = 1 / L[i * n + i]!;
    for (let j = 0; j < i; j++) {
      let sum = 0;
      for (let k = j; k < i; k++) sum += L[i * n + k]! * invL[k * n + j]!;
      invL[i * n + j] = -sum / L[i * n + i]!;
    }
  }

  // M^{-1} = L^{-T} · L^{-1}
  const invM = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = Math.max(i, j); k < n; k++) sum += invL[k * n + i]! * invL[k * n + j]!;
      invM[i * n + j] = sum;
    }

  return invM;
}

// ── Utilities ──────────────────────────────────────────────────────

function l1Norm(v: Float64Array, len: number): number {
  let s = 0;
  for (let i = 0; i < len; i++) s += Math.abs(v[i]!);
  return s;
}
