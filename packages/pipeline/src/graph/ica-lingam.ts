/**
 * ICA-LiNGAM — ICA-based Linear Non-Gaussian Acyclic Model.
 *
 * Uses Independent Component Analysis (FastICA) to estimate the
 * causal ordering and edge coefficients.  Complementary to
 * DirectLiNGAM — ICA-LiNGAM uses the full mixing matrix from ICA
 * rather than pairwise independence tests.
 *
 * Algorithm (Shimizu et al., 2006):
 *  1. Center and whiten the data
 *  2. Apply FastICA with non-Gaussian contrast (tanh/logcosh)
 *  3. Estimate mixing matrix A = whitening^{-1} · W_ica
 *  4. Compute B = I - A^{-1} (the causal coefficient matrix)
 *  5. Find permutation minimizing upper-triangular |B| (causal order)
 *  6. Permute + threshold B → DAG
 *
 * References:
 *  - Shimizu et al. (2006). "A linear non-Gaussian acyclic model
 *    for causal discovery." JMLR 7:2003-2030.
 *  - Hyvärinen & Oja (2000). "Independent Component Analysis:
 *    Algorithms and Applications." Neural Networks 13(4-5):411-430.
 *
 * @packageDocumentation
 */
import { CausalGraph } from './causal-graph.js';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';

export interface ICALiNGAMConfig {
  /** Maximum ICA iterations */
  maxIter?: number;
  /** Convergence tolerance */
  tol?: number;
  /** Edge weight threshold */
  threshold?: number;
}

export function icaLiNGAM(
  XArr: number[][],
  nodeNames: string[],
  config: ICALiNGAMConfig = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph; B: Float64Array; order: string[] } {
  const maxIter = config.maxIter ?? 1000;
  const tol = config.tol ?? 1e-6;
  const threshold = config.threshold ?? 0.1;

  const n = XArr.length;
  const d = nodeNames.length;

  if (n < d || d < 2) {
    return { graph: new CausalGraph([...nodeNames]), B: new Float64Array(d * d), order: [...nodeNames] };
  }

  // 1. Center
  const X = new Float64Array(n * d);
  const means = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += XArr[i]![j]!;
    means[j] = sum / n;
  }
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++)
      X[i * d + j] = XArr[i]![j]! - means[j]!;

  // 2. Whiten: X_white = V · X where V = D^{-1/2} · E^T
  // Covariance = X^T X / n, eigenvalue decomposition
  const cov = new Float64Array(d * d);
  for (let j = 0; j < d; j++)
    for (let k = j; k < d; k++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += X[i * d + j]! * X[i * d + k]!;
      cov[j * d + k] = cov[k * d + j] = s / n;
    }

  const { values: eigenvals, vectors: eigenvecs } = eigh(cov, d);
  const V = new Float64Array(d * d); // whitening matrix
  for (let j = 0; j < d; j++) {
    const invSqrt = eigenvals[j]! > 1e-10 ? 1 / Math.sqrt(eigenvals[j]!) : 0;
    for (let i = 0; i < d; i++)
      V[i * d + j] = eigenvecs[i * d + j]! * invSqrt;
  }

  // Apply whitening
  const Xwhite = new Float64Array(n * d);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let k = 0; k < d; k++) s += X[i * d + k]! * V[k * d + j]!;
      Xwhite[i * d + j] = s;
    }

  // 3. FastICA — one component at a time with deflation
  const W = new Float64Array(d * d); // unmixing matrix
  for (let comp = 0; comp < d; comp++) {
    let w = randomUnitVector(d, comp + 1);

    for (let iter = 0; iter < maxIter; iter++) {
      // w^+ = E{x·g(w^T x)} - E{g'(w^T x)}·w
      const wPlus = new Float64Array(d);
      let gDerivSum = 0;

      for (let i = 0; i < n; i++) {
        let wx = 0;
        for (let j = 0; j < d; j++) wx += w[j]! * Xwhite[i * d + j]!;
        const g = Math.tanh(wx);
        const gDeriv = 1 - g * g;
        gDerivSum += gDeriv;
        for (let j = 0; j < d; j++) wPlus[j] = (wPlus[j] ?? 0) + Xwhite[i * d + j]! * g;
      }
      for (let j = 0; j < d; j++) wPlus[j] = (wPlus[j] ?? 0) / n;
      const gDerivMean = gDerivSum / n;
      for (let j = 0; j < d; j++) wPlus[j] = (wPlus[j] ?? 0) - gDerivMean * (w[j] ?? 0);

      // Deflation: orthogonalize against previous components
      for (let c = 0; c < comp; c++) {
        let dot = 0;
        for (let j = 0; j < d; j++) dot += wPlus[j]! * W[c * d + j]!;
        for (let j = 0; j < d; j++) wPlus[j] = (wPlus[j] ?? 0) - dot * W[c * d + j]!;
      }

      // Normalize
      const norm = Math.sqrt(wPlus.reduce((s, v) => s + v * v, 0));
      if (norm < 1e-10) break;
      for (let j = 0; j < d; j++) wPlus[j] = (wPlus[j] ?? 0) / norm;

      // Check convergence
      let dist = 0;
      for (let j = 0; j < d; j++) {
        const diff = Math.abs(Math.abs(wPlus[j]!) - Math.abs(w[j] ?? 0));
        dist = Math.max(dist, diff);
      }

      w = wPlus;
      if (dist < tol) break;
    }

    for (let j = 0; j < d; j++) W[comp * d + j] = w[j] ?? 0;
  }

  // 4. Estimate mixing matrix: A = V^{-1} · W^T = eigenvecs · sqrt(eigenvals) · W^T
  // Since we want B = I - A^{-1}, compute A first
  const A = new Float64Array(d * d); // A = V^{-1} · W^T
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let k = 0; k < d; k++) {
        const vInv = eigenvecs[i * d + k]! * Math.sqrt(Math.max(1e-10, eigenvals[k]!));
        s += vInv * W[j * d + k]!;
      }
      A[i * d + j] = s;
    }
  }

  // 5. B = I - A^{-1}
  const Ainv = invertMatrix(A, d);
  if (!Ainv) {
    return { graph: new CausalGraph([...nodeNames]), B: new Float64Array(d * d), order: [...nodeNames] };
  }

  let B = new Float64Array(d * d);
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      B[i * d + j] = (i === j ? 1 : 0) - Ainv[i * d + j]!;

  // 6. Find permutation minimizing upper-triangular sum → causal order
  const perm = findCausalOrder(B, d);

  // 7. Apply permutation and threshold
  const BPermuted = new Float64Array(d * d);
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      BPermuted[i * d + j] = B[perm[i]! * d + perm[j]!]!;

  // Zero elements below diagonal (ensure acyclicity) and threshold
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      if (i >= j || Math.abs(BPermuted[i * d + j]!) < threshold)
        BPermuted[i * d + j] = 0;
    }
  }

  // 8. Build graph
  const g = new CausalGraph([...nodeNames]);
  const order = perm.map(i => nodeNames[i]!);
  for (let i = 0; i < d; i++)
    for (let j = i + 1; j < d; j++)
      if (BPermuted[i * d + j] !== 0)
        g.addEdge(order[i]!, order[j]!);

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);
  return { graph: g, B: B, order };
}

// ── Eigenvalue Decomposition (power iteration) ────────────────────

function eigh(cov: Float64Array, d: number): { values: Float64Array; vectors: Float64Array } {
  // Simple Jacobi eigenvalue decomposition for symmetric matrix
  const values = new Float64Array(d);
  const vectors = new Float64Array(d * d);
  // Initialize to identity
  for (let i = 0; i < d; i++) vectors[i * d + i] = 1;

  const A = new Float64Array(cov);
  for (let sweep = 0; sweep < 50; sweep++) {
    let maxOff = 0;
    for (let p = 0; p < d; p++)
      for (let q = p + 1; q < d; q++)
        maxOff = Math.max(maxOff, Math.abs(A[p * d + q]!));
    if (maxOff < 1e-12) break;

    for (let p = 0; p < d; p++) {
      for (let q = p + 1; q < d; q++) {
        const app = A[p * d + p]!;
        const aqq = A[q * d + q]!;
        const apq = A[p * d + q]!;
        if (Math.abs(apq) < 1e-12) continue;

        const theta = 0.5 * Math.atan2(2 * apq, app - aqq);
        const c = Math.cos(theta), s = Math.sin(theta);

        // Rotate A
        A[p * d + p] = c * c * app + 2 * c * s * apq + s * s * aqq;
        A[q * d + q] = s * s * app - 2 * c * s * apq + c * c * aqq;
        A[p * d + q] = A[q * d + p] = 0;

        for (let r = 0; r < d; r++) {
          if (r === p || r === q) continue;
          const arp = A[r * d + p]!;
          const arq = A[r * d + q]!;
          A[r * d + p] = A[p * d + r] = c * arp + s * arq;
          A[r * d + q] = A[q * d + r] = -s * arp + c * arq;
        }

        // Rotate eigenvectors
        for (let r = 0; r < d; r++) {
          const vrp = vectors[r * d + p]!;
          const vrq = vectors[r * d + q]!;
          vectors[r * d + p] = c * vrp + s * vrq;
          vectors[r * d + q] = -s * vrp + c * vrq;
        }
      }
    }
  }

  for (let i = 0; i < d; i++) values[i] = A[i * d + i]!;
  return { values, vectors };
}

// ── Matrix Inversion ──────────────────────────────────────────────

function invertMatrix(A: Float64Array, n: number): Float64Array | null {
  const aug = new Float64Array(n * 2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug[i * 2 * n + j] = A[i * n + j]!;
    aug[i * 2 * n + n + i] = 1;
  }
  const cols = 2 * n;

  for (let k = 0; k < n; k++) {
    let pivot = k;
    for (let i = k + 1; i < n; i++)
      if (Math.abs(aug[i * cols + k]!) > Math.abs(aug[pivot * cols + k]!)) pivot = i;
    if (Math.abs(aug[pivot * cols + k]!) < 1e-14) return null;
    if (pivot !== k)
      for (let j = 0; j < cols; j++) {
        const tmp = aug[k * cols + j]!; aug[k * cols + j] = aug[pivot * cols + j]!; aug[pivot * cols + j] = tmp;
      }
    const pv = aug[k * cols + k]!;
    for (let j = 0; j < cols; j++) aug[k * cols + j] /= pv;
    for (let i = 0; i < n; i++) {
      if (i === k) continue;
      const f = aug[i * cols + k]!;
      for (let j = 0; j < cols; j++) aug[i * cols + j] -= f * aug[k * cols + j]!;
    }
  }

  const inv = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      inv[i * n + j] = aug[i * cols + n + j]!;
  return inv;
}

// ── Causal Order ──────────────────────────────────────────────────

function findCausalOrder(B: Float64Array, d: number): number[] {
  // Greedy: repeatedly find row with minimum absolute sum (most exogenous)
  const perm: number[] = [];
  const remaining = new Set(Array.from({ length: d }, (_, i) => i));

  while (remaining.size > 0) {
    let bestIdx = -1;
    let bestSum = Infinity;
    for (const i of remaining) {
      let sum = 0;
      for (let j = 0; j < d; j++)
        if (i !== j) sum += Math.abs(B[i * d + j] ?? 0);
      if (sum < bestSum) { bestSum = sum; bestIdx = i; }
    }
    perm.push(bestIdx!);
    remaining.delete(bestIdx!);
  }
  return perm;
}

// ── Random Unit Vector ────────────────────────────────────────────

function randomUnitVector(d: number, seed: number): Float64Array {
  const v = new Float64Array(d);
  let s = seed * 1664525 + 1013904223;
  for (let i = 0; i < d; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    v[i] = (s / 0x100000000) - 0.5;
  }
  const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
  if (norm > 0) for (let i = 0; i < d; i++) v[i] /= norm;
  return v;
}
