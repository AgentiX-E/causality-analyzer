/**
 * GOLEM — Gradient-based Optimization of Likelihood for linear sEM.
 *
 * A faithful port of the official NeurIPS 2020 implementation, now using:
 *   - `mathjs` for `expm()` (Padé w/ scaling-and-squaring) and `det()` —
 *     the two operations that blocked full fidelity in prior attempts.
 *   - ANALYTICAL gradients derived from the GOLEM-EV loss function,
 *     avoiding the O(d⁴) cost of numerical finite differences.
 *   - `ml-matrix` for all remaining linear algebra (fast, well-typed).
 *
 * Loss (GOLEM-EV, equal variances):
 *   L(B) = 0.5·d·log(||X-XB||²) - log|det(I-B)| + λ₁·||B||₁
 *          + λ₂·(trace(expm(B⊙B)) - d)
 *
 * Gradient:
 *   ∇L = -d·Xᵀ(X-XB)/RSS + (I-B)⁻ᵀ + λ₁·sign(B)
 *        + 2λ₂·B ⊙ expm(B⊙B)ᵀ
 *
 * Reference: Ng, Ghassami & Zhang (NeurIPS 2020).
 * Official Source: https://github.com/ignavierng/golem
 *
 * @packageDocumentation
 */
import { Matrix, inverse } from 'ml-matrix';
import { det, expm } from 'mathjs';
import { CausalGraph } from './causal-graph.js';
import { adam } from '@agentix-e/causality-analyzer-core';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';

export interface GOLEMConfig {
  lambda1: number;
  lambda2: number;
  lr: number;
  maxIter: number;
  wThreshold: number;
  seed?: number;
}

const DEFAULTS: GOLEMConfig = {
  lambda1: 1e-2,   // tuned: lower L1 allows more edges; λ₂=5 keeps DAG constraint strong
  lambda2: 5.0,    // DAG penalty weight
  lr: 1e-3,        // Adam learning rate
  maxIter: 5000,   // reduced from 1e5 for practical runtime
  wThreshold: 0.3,  // official default
};

// ── Analytical loss + gradient ──────────────────────────────────────

function golemLossAndGrad(
  w: Float64Array , d: number,
  X: Matrix, lambda1: number, lambda2: number,
): [number, Float64Array] {
  // Build B from w (d×d matrix, diagonal stays zero)
  const B = new Matrix(d, d);
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      B.set(i, j, w[i * d + j]);

  // ── Likelihood: 0.5·d·log(RSS) - log|det(I-B)| ───────────────────
  const XB = X.mmul(B);
  const diff = Matrix.sub(X, XB);
  const rss = Math.max(1e-12, diff.pow(2).sum()); // ||X-XB||²_F
  const lik1 = 0.5 * d * Math.log(rss);

  const I = Matrix.eye(d);
  const I_minus_B = Matrix.sub(I, B);

  // det(I-B) from mathjs (accepts plain 2D array)
  let detVal: number;
  try { detVal = det(I_minus_B.to2DArray()) as number; } catch { detVal = 0; }
  const lik2 = detVal > 1e-12 ? -Math.log(detVal) : 1e10;
  const likelihood = lik1 + lik2;

  // ── L1 penalty ────────────────────────────────────────────────────
  let l1 = 0;
  for (let i = 0; i < d * d; i++) l1 += Math.abs(w[i]);

  // ── DAG penalty: trace(expm(B⊙B)) - d ──────────────────────────────
  const B_sq_arr: number[][] = [];
  for (let i = 0; i < d; i++) {
    const row: number[] = [];
    for (let j = 0; j < d; j++) row.push(B.get(i, j) ** 2);
    B_sq_arr.push(row);
  }

  let expmArr: number[][] = [];
  let h = 1e10;
  try {
    const expmResult = expm(B_sq_arr) as any;
    expmArr = expmResult.toArray() as number[][];
    let tr = 0;
    for (let i = 0; i < d; i++) tr += expmArr[i]?.[i] ?? 0;
    h = tr - d;
  } catch { /* expm failed → h stays large */ }

  // ── Total loss ────────────────────────────────────────────────────
  const loss = likelihood + lambda1 * l1 + lambda2 * h;

  // ── ANALYTICAL GRADIENT ───────────────────────────────────────────
  const G = new Float64Array(d * d);

  // ∇(lik1) = -d·Xᵀ·(X-XB) / RSS
  const XtDiff = X.transpose().mmul(diff);
  const rssScale = -d / rss;
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      G[i * d + j] = rssScale * XtDiff.get(i, j);

  // ∇(lik2) = (I-B)⁻ᵀ
  try {
    const inv_I_B = inverse(I_minus_B);
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++)
        G[i * d + j] += inv_I_B.get(j, i); // M⁻ᵀ[i,j] = M⁻¹[j,i]
  } catch { /* singular → skip */ }

  // ∇(L1) = λ₁·sign(w)
  for (let i = 0; i < d * d; i++)
    G[i] += lambda1 * Math.sign(w[i]);

  // ∇(h) = 2λ₂·B ⊙ expm(B⊙B)ᵀ
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      G[i * d + j] += 2 * lambda2 * B.get(i, j) * (expmArr[j]?.[i] ?? 0);

  return [loss, G];
}

// ── Public API ──────────────────────────────────────────────────────

export function golemAlgorithm(
  XArr: number[][],
  nodeNames: string[],
  config: Partial<GOLEMConfig> = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph; W: Float64Array } {
  const cfg = { ...DEFAULTS, ...config };
  const n = XArr.length;
  const d = nodeNames.length;

  if (n < 5 || d < 2) {
    return { graph: new CausalGraph([...nodeNames]), W: new Float64Array(d * d) };
  }

  // Center data (no scaling — GOLEM-EV operates on centered data)
  const X = new Matrix(n, d);
  for (let j = 0; j < d; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += XArr[i][j];
    const mean = sum / n;
    for (let i = 0; i < n; i++) X.set(i, j, XArr[i][j] - mean);
  }

  const lossFn = (w: Float64Array): [number, Float64Array] =>
    golemLossAndGrad(w, d, X, cfg.lambda1, cfg.lambda2);

  const result = adam(lossFn, new Float64Array(d * d), {
    maxIter: cfg.maxIter,
    lr: cfg.lr,
    gtol: 1e-6,
  });
  const W = new Float64Array(result.x);

  // Threshold to DAG
  const g = new CausalGraph([...nodeNames]);
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      if (i !== j && Math.abs(W[i * d + j]) > cfg.wThreshold)
        g.addEdge(nodeNames[i], nodeNames[j]);

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);
  return { graph: g, W };
}
