/**
 * GOLEM — Gradient-based Optimization of Likelihood for linear sEM.
 *
 * A continuous DAG optimization alternative to NOTEARS (Ng et al., NeurIPS 2020).
 * Uses the log-determinant of (I-W) as a natural acyclicity measure.
 *
 * Reference: Ng, Ghassami & Zhang (NeurIPS 2020).
 *
 * @packageDocumentation
 */
import { CausalGraph } from './causal-graph.js';
import { adam, lbfgs } from '@agentix-e/causality-analyzer-core';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';

export interface GOLEMConfig {
  lambda1: number;
  lr: number;
  maxIter: number;
  tol: number;
  wThreshold: number;
  optimizer: 'adam' | 'lbfgs';
  seed?: number;
}

const DEFAULTS: GOLEMConfig = {
  lambda1: 0.01,
  lr: 0.01,
  maxIter: 5000,
  tol: 1e-6,
  wThreshold: 0.3,
  optimizer: 'adam',
};

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

  // Z-score normalize
  const X = new Float64Array(n * d);
  for (let j = 0; j < d; j++) {
    let sum = 0, sq = 0;
    for (let i = 0; i < n; i++) { const v = XArr[i][j]; sum += v; sq += v * v; }
    const mean = sum / n;
    const std = Math.sqrt(Math.max(1e-10, sq / n - mean * mean));
    for (let i = 0; i < n; i++) X[i * d + j] = (XArr[i][j] - mean) / std;
  }

  // Precompute covariance S = X^T X / n
  const S = new Float64Array(d * d);
  for (let j = 0; j < d; j++)
    for (let k = j; k < d; k++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += X[i * d + j] * X[i * d + k];
      S[j * d + k] = S[k * d + j] = s / n;
    }

  let W = new Float64Array(d * d);

  const lossFn = (w: Float64Array): [number, Float64Array] => golemLoss(w, d, S, cfg.lambda1);

  if (cfg.optimizer === 'lbfgs') {
    const result = lbfgs(lossFn, W, { maxIter: cfg.maxIter, gtol: cfg.tol, m: 15 });
    W = new Float64Array(result.x);
  } else {
    const result = adam(lossFn, W, { maxIter: cfg.maxIter, lr: cfg.lr, gtol: cfg.tol });
    W = new Float64Array(result.x);
  }

  // Threshold to DAG
  const g = new CausalGraph([...nodeNames]);
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      if (i !== j && Math.abs(W[i * d + j]) > cfg.wThreshold)
        g.addEdge(nodeNames[i], nodeNames[j]);

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);
  return { graph: g, W };
}

// ── GOLEM Loss: L(W) = (d/2)log(RSS/n) - log|det(I-W)| + λ₁‖W‖₁ ──

function golemLoss(
  w: Float64Array, d: number, S: Float64Array, lambda1: number,
): [number, Float64Array] {
  // Build M = I - W
  const M = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    M[i * d + i] = 1;
    for (let j = 0; j < d; j++) M[i * d + j] -= w[i * d + j];
  }

  // log|det(M)| via Gaussian elimination with partial pivoting
  // det(M) = product of diagonal entries after elimination
  const detM = determinant(M, d);
  if (detM <= 1e-10) return [1e10, new Float64Array(d * d)];

  // RSS = tr(M^T S M) — matching GOLEM Eq.7 for equal variance
  let rss = 0;
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      let sMj = 0;
      for (let k = 0; k < d; k++) sMj += S[i * d + k] * M[k * d + j];
      rss += M[i * d + j] * sMj;
    }
  }

  const rssN = Math.max(1e-10, rss / d); // per-variable average RSS
  const loss = (d / 2) * Math.log(rssN) - Math.log(detM) + lambda1 * l1Norm(w, d * d);

  // Gradient: ∇L = ∇RSS + ∇det + ∇L1
  const grad = new Float64Array(d * d);
  const rssCoeff = d / (2 * rssN * d); // d/(2·RSS·d) = 1/(2·RSS_avg)

  // ∇RSS: -2 S M (derivative of tr(M^T S M) w.r.t. W[i,j] is -2 (S M)[j,i])
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      let sm = 0;
      for (let k = 0; k < d; k++) sm += S[j * d + k] * M[k * d + i];
      grad[i * d + j] = rssCoeff * (-2 * sm);
    }
  }

  // ∇(-log|det(M)|) = (M^{-1})^T — each row of M^{-T} = column of M^{-1}
  const invM = invertWithElimination(M, d);
  if (invM) {
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++)
        grad[i * d + j] = (grad[i * d + j] ?? 0) + invM[j * d + i]; // M^{-T}[i,j] = M^{-1}[j,i]
  }

  // L1 subgradient
  for (let i = 0; i < d * d; i++)
    grad[i] = (grad[i] ?? 0) + lambda1 * (w[i] > 0 ? 1 : w[i] < 0 ? -1 : 0);

  return [loss, grad];
}

// ── Determinant via Gaussian Elimination ───────────────────────────

function determinant(A: Float64Array, n: number): number {
  const B = new Float64Array(A);
  let det = 1;
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(B[row * n + col]) > Math.abs(B[pivot * n + col])) pivot = row;
    if (pivot !== col) {
      for (let j = col; j < n; j++) {
        const tmp = B[col * n + j]; B[col * n + j] = B[pivot * n + j]!; B[pivot * n + j] = tmp;
      }
      det = -det;
    }
    const pv = B[col * n + col];
    if (Math.abs(pv) < 1e-14) return 0;
    det *= pv;
    for (let row = col + 1; row < n; row++) {
      const f = B[row * n + col] / pv;
      for (let j = col; j < n; j++) B[row * n + j] -= f * B[col * n + j];
    }
  }
  return det;
}

// ── Matrix Inverse via Gaussian Elimination ────────────────────────

function invertWithElimination(A: Float64Array, n: number): Float64Array | null {
  const aug = new Float64Array(n * n * 2);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug[i * (2 * n) + j] = A[i * n + j]!;
    aug[i * (2 * n) + n + i] = 1;
  }
  const cols = 2 * n;

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(aug[row * cols + col]) > Math.abs(aug[pivot * cols + col])) pivot = row;
    if (pivot !== col)
      for (let j = 0; j < cols; j++) {
        const tmp = aug[col * cols + j]; aug[col * cols + j] = aug[pivot * cols + j]!; aug[pivot * cols + j] = tmp;
      }
    const pv = aug[col * cols + col];
    if (Math.abs(pv) < 1e-14) return null;
    for (let j = 0; j < cols; j++) aug[col * cols + j] /= pv;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const f = aug[row * cols + col];
      for (let j = 0; j < cols; j++) aug[row * cols + j] -= f * aug[col * cols + j];
    }
  }

  const inv = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      inv[i * n + j] = aug[i * cols + n + j]!;
  return inv;
}

function l1Norm(v: Float64Array, len: number): number {
  let s = 0;
  for (let i = 0; i < len; i++) s += Math.abs(v[i]);
  return s;
}
