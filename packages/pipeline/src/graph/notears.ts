/**
 * NOTEARS — Non-combinatorial Optimization via Trace Exponential
 * Augmented Lagrangian for Structure learning (Zheng et al., NeurIPS 2018).
 *
 * Reformulates the NP-hard DAG learning problem as a continuous constrained
 * optimization over the weighted adjacency matrix:
 *
 *   min_W  1/(2n)‖X - XW‖²_F + λ‖W‖₁
 *   s.t.   h(W) = tr(e^(W⊙W)) - d = 0
 *
 * where ⊙ is element-wise product (Hadamard square).
 * Solved via augmented Lagrangian + L-BFGS.
 *
 * @packageDocumentation
 */
import { CausalGraph } from './causal-graph.js';
import { lbfgs } from '@agentix-e/causality-analyzer-core';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';

// ── Config ────────────────────────────────────────────────────────────

export interface NOTEARSConfig {
  lambda1: number;
  rho: number;
  rhoFactor: number;
  maxOuterIter: number;
  tol: number;
  wThreshold: number;
  seed?: number;
}

const DEFAULTS: NOTEARSConfig = {
  lambda1: 0.1,
  rho: 1.0,
  rhoFactor: 10,
  maxOuterIter: 20,
  tol: 1e-8,
  wThreshold: 0.3,
};

// ── Public API ────────────────────────────────────────────────────────

export function notearsAlgorithm(
  XArr: number[][],
  nodeNames: string[],
  config: Partial<NOTEARSConfig> = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph; W: Float64Array; h: number; iterations: number } {
  const cfg = { ...DEFAULTS, ...config };
  const n = XArr.length;
  const d = nodeNames.length;

  // Flatten + z-score normalize
  const X = new Float64Array(n * d);
  for (let j = 0; j < d; j++) {
    let sum = 0,
      sq = 0;
    for (let i = 0; i < n; i++) {
      const v = XArr[i]![j]!;
      sum += v;
      sq += v * v;
    }
    const mean = sum / n,
      std = Math.sqrt(Math.max(1e-10, sq / n - mean * mean));
    for (let i = 0; i < n; i++) X[i * d + j] = (XArr[i]![j]! - mean) / std;
  }

  // Precompute X^T X / n (scaled covariance, d×d)
  const cov = new Float64Array(d * d);
  for (let j = 0; j < d; j++)
    for (let k = j; k < d; k++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += X[i * d + j]! * X[i * d + k]!;
      cov[j * d + k] = cov[k * d + j] = s / n;
    }

  // Augmented Lagrangian outer loop
  // α is a SCALAR Lagrange multiplier for the single constraint h(W)=0.
  // Zheng et al. (2018) §3.2: L_ρ(W,α) = f(W) + λ‖W‖₁ + α·h(W) + (ρ/2)·h(W)²
  let W = new Float64Array(d * d);
  let alpha = 0;
  let rho = cfg.rho,
    totalIter = 0;

  for (let outer = 0; outer < cfg.maxOuterIter; outer++) {
    const sub = lbfgs(
      (w: Float64Array) => noteLoss(w, d, cov, alpha, rho, cfg.lambda1),
      W,
      { maxIter: 500, gtol: cfg.tol, m: 20 },
    );
    W = new Float64Array(sub.x);
    totalIter += sub.iterations;

    const h = dagH(W, d);
    if (h <= 1e-6) break;

    // α ← α + ρ·h  (standard augmented Lagrangian dual update — scalar)
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

  return { graph: g, W, h: dagH(W, d), iterations: totalIter };
}

// ── Loss Function ─────────────────────────────────────────────────────

/**
 * Augmented Lagrangian: L_ρ(W, α) = f(W) + λ‖W‖₁ + α·h(W) + (ρ/2)·h(W)²
 *
 * α is a SCALAR (single constraint h(W)=0).
 * Gradient: ∇L = ∇f + λ·sign(W) + (α + ρ·h)·∇h
 */
function noteLoss(
  w: Float64Array,
  d: number,
  cov: Float64Array,
  alpha: number,
  rho: number,
  lambda1: number,
): [number, Float64Array] {
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
  // Add constant trace term (doesn't affect gradient)
  for (let i = 0; i < d; i++) f += 0.5 * cov[i * d + i]!;

  // h(W) = tr(e^(W⊙W)) - d  and  dh = 2·[(exp(W⊙W))ᵀ ⊙ W]
  const [h, dh] = hAndGrad(w, d);

  // Augmented Lagrangian: L = f + λ‖W‖₁ + α·h + (ρ/2)·h²
  const loss = f + lambda1 * l1Norm(w, d * d) + alpha * h + 0.5 * rho * h * h;

  // Gradient: ∇L = ∇f + λ·sign(W) + (α + ρ·h)·∇h
  const grad = new Float64Array(d * d);
  const multiplierCoeff = alpha + rho * h;
  for (let i = 0; i < d * d; i++) {
    grad[i] =
      gf[i]! +
      lambda1 * (w[i]! > 0 ? 1 : w[i]! < 0 ? -1 : 0) +
      multiplierCoeff * dh[i]!;
  }

  return [loss, grad];
}

// ── DAG Constraint ────────────────────────────────────────────────────

/**
 * h(W) = tr(e^(W⊙W)) - d
 * Gradient: dh/dW = 2·[(e^(W⊙W))ᵀ ⊙ W]
 *
 * Uses "scaling and squaring" for matrix exponential:
 * 1. Scale: B = (W⊙W) / 2^j, where j ensures ‖B‖ ≤ 4
 * 2. Compute exp(B) via Taylor series (15 terms — converges for ‖B‖ ≤ 4)
 * 3. Unscale: exp(W⊙W) = exp(B)^(2^j) via j repeated matrix squarings
 *
 * The scaling factor is always a power of 2, enabling exact unscaling.
 */
function hAndGrad(W: Float64Array, d: number): [number, Float64Array] {
  // Compute S = W⊙W
  const S = new Float64Array(d * d);
  let normSq = 0;
  for (let i = 0; i < d * d; i++) {
    const v = W[i]! * W[i]!;
    S[i] = v;
    normSq += v;
  }

  // Scaling: find j such that ‖S‖ / 2^j ≤ 4
  // j = max(0, ceil(log2(‖S‖)) - 2)
  const normS = Math.sqrt(normSq);
  const j = normS > 0 ? Math.max(0, Math.ceil(Math.log2(Math.max(normS, 0.125))) - 2) : 0;
  const scale = 1 << j;

  // Scaled matrix: B = S / 2^j
  const invScale = 1 / scale;
  const B = new Float64Array(d * d);
  for (let i = 0; i < d * d; i++) B[i] = S[i]! * invScale;

  // exp(B) via Taylor series
  let expCurr = matExp(B, d);

  // Unscale: exp(S) = exp(B)^(2^j) via j matrix squarings
  for (let s = 0; s < j; s++) {
    const next = new Float64Array(d * d);
    for (let i = 0; i < d; i++)
      for (let k = 0; k < d; k++) {
        let v = 0;
        for (let l = 0; l < d; l++) v += expCurr[i * d + l]! * expCurr[l * d + k]!;
        next[i * d + k] = v;
      }
    expCurr = next;
  }

  // h = tr(expS) - d
  let tr = 0;
  for (let i = 0; i < d; i++) tr += expCurr[i * d + i]!;
  const h = tr - d;

  // Gradient: dh = 2 * expSᵀ ⊙ W  (element-wise)
  const grad = new Float64Array(d * d);
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      grad[i * d + j] = 2 * expCurr[j * d + i]! * W[i * d + j]!;

  return [h, grad];
}

function dagH(W: Float64Array, d: number): number {
  return hAndGrad(W, d)[0];
}

// ── Matrix Exponential (Taylor series) ────────────────────────────────

/**
 * Compute matrix exponential via Taylor series.
 * Converges well when ‖A‖ < 4 (after scaling).
 */
function matExp(A: Float64Array, d: number): Float64Array {
  // I (identity)
  const result = new Float64Array(d * d);
  for (let i = 0; i < d; i++) result[i * d + i] = 1;

  let Ak = new Float64Array(d * d);
  for (let i = 0; i < d; i++) Ak[i * d + i] = 1;

  let fact = 1;
  const buf1 = new Float64Array(d * d);
  const buf2 = new Float64Array(d * d);

  for (let k = 1; k < 15; k++) {
    // buf1 = Ak * A
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++) {
        let v = 0;
        for (let l = 0; l < d; l++) v += Ak[i * d + l]! * A[l * d + j]!;
        buf1[i * d + j] = v;
      }
    fact *= k;

    // result += buf1 / k!
    for (let i = 0; i < d * d; i++) result[i] += buf1[i]! / fact;

    // Swap: Ak = buf1, but we need buf1 for next iteration
    // Copy buf1 -> buf2, then point Ak to buf2
    for (let i = 0; i < d * d; i++) buf2[i] = buf1[i]!;
    Ak = buf2;
  }

  return result;
}

// ── Utilities ─────────────────────────────────────────────────────────

function l1Norm(v: Float64Array, len: number): number {
  let s = 0;
  for (let i = 0; i < len; i++) s += Math.abs(v[i]!);
  return s;
}
