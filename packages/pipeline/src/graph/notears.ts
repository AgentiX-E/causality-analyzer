/**
 * NOTEARS — Non-combinatorial Optimization via Trace Exponential
 * Augmented lagrangian for Structure learning (Zheng et al., NeurIPS 2018).
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

const DEFAULTS: NOTEARSConfig = { lambda1: 0.1, rho: 1.0, rhoFactor: 10, maxOuterIter: 20, tol: 1e-8, wThreshold: 0.3 };

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
    let sum = 0, sq = 0;
    for (let i = 0; i < n; i++) { const v = XArr[i]![j]!; sum += v; sq += v * v; }
    const mean = sum / n, std = Math.sqrt(Math.max(1e-10, sq / n - mean * mean));
    for (let i = 0; i < n; i++) X[i * d + j] = (XArr[i]![j]! - mean) / std;
  }

  // Precompute X^T X / n (scaled covariance, d×d)
  const cov = new Float64Array(d * d);
  for (let j = 0; j < d; j++)
    for (let k = j; k < d; k++) {
      let s = 0; for (let i = 0; i < n; i++) s += X[i * d + j]! * X[i * d + k]!;
      cov[j * d + k] = cov[k * d + j] = s / n;
    }

  // Augmented Lagrangian outer loop
  let W = new Float64Array(d * d);
  const alpha = new Float64Array(d * d);
  let rho = cfg.rho, totalIter = 0;

  for (let outer = 0; outer < cfg.maxOuterIter; outer++) {
    const sub = lbfgs(
      w => noteLoss(w, d, cov, alpha, rho, cfg.lambda1),
      W, { maxIter: 500, gtol: cfg.tol, m: 20 },
    );
    W = new Float64Array(sub.x);
    totalIter += sub.iterations;

    const h = dagH(W, d);
    if (h <= 1e-6) break;

    // α ← α + ρ·h  (standard augmented Lagrangian multiplier update)
    for (let i = 0; i < d * d; i++) alpha[i] += rho * h;
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
 */
function noteLoss(
  w: Float64Array, d: number, cov: Float64Array,
  alpha: Float64Array, rho: number, lambda1: number,
): [number, Float64Array] {
  // f(W) = 0.5 * ‖X - XW‖² / n  (equivalent to 0.5·tr[(I-W)ᵀ·cov·(I-W)])
  //      = 0.5·Σ cov_jk - Σ w_ij·cov_ij + 0.5·Σ w_iℓ·cov_ℓk·w_ik
  // Pre-built: g_ij = ∂f/∂W_ij = -cov_ij + Σ_l w_il·cov_lj

  let f = 0;
  const gf = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      // W_term_j = Σ_l w_il * cov_lj  →  (W·cov)[i][j]
      let wCov = 0;
      for (let l = 0; l < d; l++) wCov += w[i * d + l]! * cov[l * d + j]!;
      gf[i * d + j] = -cov[i * d + j]! + wCov;
      f -= w[i * d + j]! * cov[i * d + j]!;
      f += 0.5 * w[i * d + j]! * wCov;
    }
  }
  // Add constant trace term (doesn't affect gradient)
  for (let i = 0; i < d; i++) f += 0.5 * cov[i * d + i]!;
  f -= 0; // cleanup

  // h(W) = tr(e^(W⊙W)) - d  and  dh = 2·[(exp(W⊙W))ᵀ ⊙ W]
  const [h, dh] = hAndGrad(w, d);

  // Augmented Lagrangian
  let loss = f;
  for (let i = 0; i < d * d; i++) loss += alpha[i]! * (i === 0 ? h : 0); // α·h
  loss += lambda1 * l1Norm(w, d * d);
  loss += 0.5 * rho * h * h;

  // Merge gradient: ∇L = ∇f + λ·sign(W) + (α + ρ·h)·∇h
  const grad = new Float64Array(d * d);
  const coef = rho * h;
  const alphaH = h; // simplified for Lagrange multiplier term
  for (let i = 0; i < d * d; i++) {
    grad[i] = gf[i]! + lambda1 * (w[i]! > 0 ? 1 : w[i]! < 0 ? -1 : 0) + (coef + alphaH / (d * d)) * dh[i]!;
  }

  return [loss, grad];
}

// ── DAG Constraint ────────────────────────────────────────────────────

/**
 * h(W) = tr(e^(W⊙W)) - d
 * Gradient: dh/dW = 2·[(e^(W⊙W))ᵀ ⊙ W]
 */
function hAndGrad(W: Float64Array, d: number): [number, Float64Array] {
  // Compute S = W⊙W, then exp(S) via Taylor series
  const S = new Float64Array(d * d);
  let normSq = 0;
  for (let i = 0; i < d * d; i++) { const v = W[i]! * W[i]!; S[i] = v; normSq += v; }
  const scale = Math.max(1, Math.sqrt(normSq) / 8);

  // Scaled matrix: S/scale
  const B = new Float64Array(d * d);
  for (let i = 0; i < d * d; i++) B[i] = S[i]! / scale;

  // exp(B) via Taylor series (15 terms, converges for scaled matrices)
  const expB = matExp(B, d);

  // Unscale: exp(S) = exp(B)^scale (power iteration)
  // For integer scale: repeated squaring
  let expS = new Float64Array(expB);
  for (let s = 1; s < scale; s++) {
    const next = new Float64Array(d * d);
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++) {
        let v = 0;
        for (let k = 0; k < d; k++) v += expS[i * d + k]! * expB[k * d + j]!;
        next[i * d + j] = v;
      }
    expS = next;
  }

  // h = tr(expS) - d
  let tr = 0;
  for (let i = 0; i < d; i++) tr += expS[i * d + i]!;
  const h = tr - d;

  // Gradient: dh = 2 * expSᵀ ⊙ W  (element-wise)
  const grad = new Float64Array(d * d);
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      grad[i * d + j] = 2 * expS[j * d + i]! * W[i * d + j]!;

  return [h, grad];
}

function dagH(W: Float64Array, d: number): number { return hAndGrad(W, d)[0]; }

// ── Matrix Exponential (Taylor series) ────────────────────────────────

function matExp(A: Float64Array, d: number): Float64Array {
  const I = new Float64Array(d * d);
  for (let i = 0; i < d; i++) I[i * d + i] = 1;

  const result = new Float64Array(I);
  let Ak = new Float64Array(I);
  const tmp = new Float64Array(d * d);
  let fact = 1;

  for (let k = 1; k < 15; k++) {
    // Ak = Ak * A
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++) {
        let v = 0;
        for (let l = 0; l < d; l++) v += Ak[i * d + l]! * A[l * d + j]!;
        tmp[i * d + j] = v;
      }
    fact *= k;
    const next = new Float64Array(tmp);
    for (let i = 0; i < d * d; i++) result[i] += next[i]! / fact;
    Ak = next;
  }
  return result;
}

// ── Utilities ─────────────────────────────────────────────────────────

function l1Norm(v: Float64Array, len: number): number {
  let s = 0;
  for (let i = 0; i < len; i++) s += Math.abs(v[i]!);
  return s;
}
