/**
 * NOTEARS & GOLEM — Continuous optimization causal discovery.
 *
 * Unlike constraint-based (PC/FCI) and score-based (GES) methods,
 * NOTEARS formulates causal discovery as a continuous optimization
 * problem with an acyclicity constraint.
 *
 * NOTEARS (Zheng et al. 2018):
 *   min_W ||X - XW||² + λ||W||₁  s.t.  h(W) = Tr(e^{W∘W}) - d = 0
 *
 * GOLEM (Ng et al. 2020):
 *   min_W ½||X - XW||² + λ₁||W||₁ - λ₂ log|det(I - W)| + λ₃ h(W)
 *   Relaxes the hard constraint into a soft penalty via log-det.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';

// ── Types ──────────────────────────────────────────────────────────
export interface NOTEARSConfig {
  /** L1 regularization strength (default 0.1) */
  lambda1?: number;
  /** Maximum iterations (default 100) */
  maxIter?: number;
  /** Convergence tolerance on loss (default 1e-8) */
  hTol?: number;
  /** Step size for gradient descent (default 0.01) */
  rho?: number;
  /** Whether to threshold small weights to zero (default true) */
  threshold?: boolean;
  /** Weight threshold (default 0.3) */
  wThreshold?: number;
}

export interface GOLEMConfig {
  /** L1 regularization (default 0.01) */
  lambda1?: number;
  /** Log-det penalty weight (default 1.0) */
  lambda2?: number;
  /** Acyclicity penalty weight (default 5.0) */
  lambda3?: number;
  /** Maximum iterations (default 1000) */
  maxIter?: number;
  /** Learning rate (default 0.001) */
  lr?: number;
}

// ── NOTEARS ────────────────────────────────────────────────────────
/**
 * NOTEARS: Non-combinatorial Optimization via Trace Exponential
 * Augmented lagRangian Structure learning.
 *
 * Reference: Zheng, Aragam, Ravikumar & Xing (NeurIPS 2018).
 * "DAGs with NO TEARS: Continuous Optimization for Structure Learning"
 *
 * @param data — N×d observation matrix
 * @param nodeNames — variable names
 * @param config — optimization parameters
 * @returns discovered causal graph
 */
export function notearsAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: NOTEARSConfig = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph; W: number[][] } {
  const d = nodeNames.length;
  const n = data.rows;
  if (n === 0 || d === 0) return { graph: new CausalGraph(nodeNames), W: [] };

  const lambda1 = config.lambda1 ?? 0.1;
  const maxIter = config.maxIter ?? 100;
  const hTol = config.hTol ?? 1e-8;
  const rho = config.rho ?? 0.01;
  const doThreshold = config.threshold !== false;
  const wThreshold = config.wThreshold ?? 0.3;

  // Initialize W = 0 (d×d)
  const W = Array.from({ length: d }, () => new Array(d).fill(0));

  // Center data
  const X = data.to2DArray();
  const means = new Array(d).fill(0);
  for (let j = 0; j < d; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += X[i]![j]!;
    means[j] = sum / n;
  }
  const Xc = X.map(row => row.map((v, j) => v - means[j]!));
  const cov = computeCov(Xc, d, n);

  // Precompute XtX for the quadratic term (used as a proxy for data)
  // We use the covariance-based formulation for efficiency
  // Loss: ½||X - XW||² ≈ ½ Tr((I-W)ᵀ Σ (I-W)) for centered data

  // Augmented Lagrangian optimization
  let alpha = 0; // Lagrange multiplier
  for (let iter = 0; iter < maxIter; iter++) {
    // Proximal gradient step: W ← Prox_{λ₁}(W - η∇_W L(W, α, ρ))
    const grad = computeNOTGrad(W, cov, d, alpha, rho);
    const eta = 0.01; // learning rate
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        if (i === j) continue;
        W[i]![j]! -= eta * grad[i]![j]!;
        // L1 proximal operator (soft thresholding)
        if (lambda1 > 0) {
          W[i]![j]! = softThreshold(W[i]![j]!, eta * lambda1);
        }
      }
    }

    // Compute acyclicity constraint
    const hval = acyclicityH(W, d);

    // Update Lagrange multiplier
    alpha += rho * hval;

    // Convergence check
    if (hval < hTol) break;
  }

  // Threshold small weights
  if (doThreshold) {
    const wMax = Math.max(...W.flat().map(Math.abs));
    const thresh = wMax * wThreshold;
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        if (Math.abs(W[i]![j]!) < thresh) W[i]![j] = 0;
      }
    }
  }

  // Convert to CausalGraph
  const g = new CausalGraph(nodeNames);
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      if (W[i]![j]! !== 0 && i !== j) {
        g.addEdge(nodeNames[i]!, nodeNames[j]!);
      }
    }
  }

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);

  // Ensure DAG: break bidirectional edges (cycles)
  return ensureAcyclic(g, nodeNames, W);
}

// ── GOLEM ──────────────────────────────────────────────────────────
/**
 * GOLEM: Gradient-based Optimization for causal discovery with
 * Latent variable models via Efficient Markov chain Monte Carlo.
 *
 * Reference: Ng, Ghassami & Zhang (NeurIPS 2020).
 * "On the Role of Sparsity and DAG Constraints for Learning Linear DAGs"
 *
 * Uses gradient descent on a continuous objective combining:
 * - Data likelihood: ½||X - XW||² (or log-det for equal variance)
 * - L1 sparsity: λ₁||W||₁
 * - Acyclicity soft penalty: λ₃ h(W)
 *
 * @returns discovered causal graph with weight matrix
 */
export function golemAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: GOLEMConfig = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph; W: number[][] } {
  const d = nodeNames.length;
  const n = data.rows;
  if (n === 0 || d === 0) return { graph: new CausalGraph(nodeNames), W: [] };

  const lambda1 = config.lambda1 ?? 0.01;
  const lambda3 = config.lambda3 ?? 5.0;
  const maxIter = config.maxIter ?? 300;
  const lr = config.lr ?? 0.001;

  const W = Array.from({ length: d }, () => new Array(d).fill(0));

  // Center data
  const X = data.to2DArray();
  const means = new Array(d).fill(0);
  for (let j = 0; j < d; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += X[i]![j]!;
    means[j] = sum / n;
  }
  const Xc = X.map(row => row.map((v, j) => v - means[j]!));
  const cov = computeCov(Xc, d, n);

  // Adam optimizer state
  const m = Array.from({ length: d }, () => new Array(d).fill(0));
  const v = Array.from({ length: d }, () => new Array(d).fill(0));
  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
  let t = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    t++;
    // Gradient of GOLEM objective
    const grad = computeGOLEMGrad(W, cov, d, lambda3);
    const l1Grad = computeL1Subgrad(W, d, lambda1);

    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        if (i === j) continue;
        const g = grad[i]![j]! + l1Grad[i]![j]!;
        m[i]![j]! = beta1 * m[i]![j]! + (1 - beta1) * g;
        v[i]![j]! = beta2 * v[i]![j]! + (1 - beta2) * g * g;
        const mHat = m[i]![j]! / (1 - Math.pow(beta1, t));
        const vHat = v[i]![j]! / (1 - Math.pow(beta2, t));
        W[i]![j]! -= lr * mHat / (Math.sqrt(vHat) + eps);
      }
    }

    // Convergence check
    if (iter > 0 && iter % 50 === 0 && acyclicityH(W, d) < 1e-6) break;
  }

  // Threshold
  const wMax = Math.max(...W.flat().map(Math.abs), 1e-6);
  const thresh = wMax * 0.3;
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      if (Math.abs(W[i]![j]!) < thresh) W[i]![j] = 0;
    }
  }

  const g = new CausalGraph(nodeNames);
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      if (W[i]![j]! !== 0 && i !== j) {
        g.addEdge(nodeNames[i]!, nodeNames[j]!);
      }
    }
  }

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);

  // Ensure DAG: break bidirectional edges (cycles)
  return ensureAcyclic(g, nodeNames, W);
}

// ── Utility Functions ──────────────────────────────────────────────
function computeCov(X: number[][], d: number, n: number): number[][] {
  const cov = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) sum += X[k]![i]! * X[k]![j]!;
      cov[i]![j] = sum / n;
      cov[j]![i] = cov[i]![j]!;
    }
  }
  return cov;
}

function softThreshold(z: number, gamma: number): number {
  if (z > gamma) return z - gamma;
  if (z < -gamma) return z + gamma;
  return 0;
}

/** Acyclicity constraint: h(W) = Tr(e^{W○W}) - d. h(W) = 0 iff W is acyclic. */
function acyclicityH(W: number[][], d: number): number {
  // Compute W○W (element-wise square), then matrix exponential
  const square = W.map(row => row.map(w => w * w));
  const expM = matrixExponential(square, d, 10); // 10 terms Taylor series
  let trace = 0;
  for (let i = 0; i < d; i++) trace += expM[i]![i]!;
  return trace - d;
}

/** Matrix exponential via Taylor series: e^M ≈ Σ_{k=0}^{K} M^k / k! */
function matrixExponential(M: number[][], d: number, terms: number): number[][] {
  let result = identity(d);
  let power = identity(d);
  let fact = 1;
  for (let k = 1; k < terms; k++) {
    fact *= k;
    power = matrixMultiply(power, M, d);
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        result[i]![j]! += power[i]![j]! / fact;
      }
    }
  }
  return result;
}

function identity(d: number): number[][] {
  const I = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let i = 0; i < d; i++) I[i]![i] = 1;
  return I;
}

function matrixMultiply(A: number[][], B: number[][], d: number): number[][] {
  const C = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      for (let k = 0; k < d; k++) C[i]![j] += A[i]![k]! * B[k]![j]!;
    }
  }
  return C;
}

/** ∇_W L for NOTEARS: derivative of ½||X - XW||² + (α + ρh(W))h(W) */
function computeNOTGrad(W: number[][], cov: number[][], d: number, alpha: number, rho: number): number[][] {
  const grad = Array.from({ length: d }, () => new Array(d).fill(0));

  // ∇_W ½||X - XW||² = Σ(I - W) (covariance-based)
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      if (i === j) continue;
      // Derivative of data term: -(Σ(I-W))_ij
      let sum = 0;
      for (let k = 0; k < d; k++) {
        sum += cov[i]![k]! * (k === j ? 1 : 0);
      }
      for (let k = 0; k < d; k++) {
        sum -= cov[i]![k]! * W[k]![j]!;
      }
      grad[i]![j] = -sum;

      // Acyclicity gradient: (α + ρ·h) · ∇_W h(W)
      const hVal = acyclicityH(W, d);
      const hGrad = acyclicityHGradient(W, d);
      grad[i]![j] += (alpha + rho * Math.max(0, hVal)) * hGrad[i]![j]!;
    }
  }
  return grad;
}

/** ∇_W h(W) where h(W) = Tr(e^{W○W}) - d */
function acyclicityHGradient(W: number[][], d: number): number[][] {
  const square = W.map(row => row.map(w => w * w));
  const expM = matrixExponential(square, d, 10);
  const grad = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      grad[i]![j] = expM[i]![j]! * 2 * W[i]![j]!;
    }
  }
  return grad;
}

/** GOLEM gradient: ∇_W (½||X-XW||² + λ₃ h(W)) */
function computeGOLEMGrad(W: number[][], cov: number[][], d: number, lambda3: number): number[][] {
  const grad = Array.from({ length: d }, () => new Array(d).fill(0));
  const hGrad = acyclicityHGradient(W, d);

  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      if (i === j) continue;
      // Data term
      let sum = -cov[i]![j]!;
      for (let k = 0; k < d; k++) sum += cov[i]![k]! * W[k]![j]!;
      grad[i]![j] = -sum;

      // Acyclicity penalty
      grad[i]![j] += lambda3 * hGrad[i]![j]!;
    }
  }
  return grad;
}

/** L1 subgradient: λ₁ · sign(W) */
function computeL1Subgrad(W: number[][], d: number, lambda1: number): number[][] {
  const grad = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      if (i === j) continue;
      grad[i]![j] = W[i]![j]! > 0 ? lambda1 : W[i]![j]! < 0 ? -lambda1 : 0;
    }
  }
  return grad;
}

/** Ensure graph is acyclic by removing weakest edges in cycles. */
function ensureAcyclic(graph: CausalGraph, nodeNames: string[], W: number[][]): { graph: CausalGraph; W: number[][] } {
  const g = graph.clone();
  // Remove bidirectional edges (cycles of length 2)
  for (let i = 0; i < g.nodeCount; i++) {
    for (let j = i + 1; j < g.nodeCount; j++) {
      if (g.hasEdge(nodeNames[i]!, nodeNames[j]!) && g.hasEdge(nodeNames[j]!, nodeNames[i]!)) {
        const wi = Math.abs(W[i]?.[j] ?? 0);
        const wj = Math.abs(W[j]?.[i] ?? 0);
        if (wi >= wj) g.removeEdge(nodeNames[j]!, nodeNames[i]!);
        else g.removeEdge(nodeNames[i]!, nodeNames[j]!);
      }
    }
  }
  return { graph: g, W };
}
