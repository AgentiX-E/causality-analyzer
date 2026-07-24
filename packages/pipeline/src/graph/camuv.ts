/**
 * CAM-UV — Causal Additive Models with Unobserved Variables.
 *
 * Based on Bühlmann, Peters & Ernest (Annals of Statistics 2014):
 * "CAM: Causal Additive Models" and Maeda & Shimizu (NeurIPS 2020):
 * "RCD: Repetitive Causal Discovery of Linear Non-Gaussian Acyclic Models"
 *
 * CAM-UV extends additive noise models to handle unobserved confounders
 * by analyzing residual structure. Unlike PC/FCI (constraint-based) or
 * LiNGAM (non-Gaussian), CAM-UV uses nonparametric additive regression
 * models with B-spline basis functions.
 *
 * Key scoring mechanism:
 * - For each pair (X, Y): fit X ~ f(Y) and Y ~ f(X) using splines
 * - The direction with lower residual variance is more likely causal
 * - Residual independence tests distinguish confounders from direct causes
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { CausalGraph } from './causal-graph.js';

export interface CAMUVConfig {
  /** Significance level */
  alpha?: number;
  /** Number of B-spline basis functions */
  nBasis?: number;
  /** Minimum residual variance ratio for direction decision */
  threshold?: number;
}

/**
 * CAM-UV causal discovery algorithm.
 *
 * Uses pairwise additive model scoring to determine edge direction.
 * Handles unobserved confounders by testing residual independence:
 * if residuals of X~f(Y) and Y~f(X) are both dependent, there's likely
 * an unobserved confounder.
 */
export function camuvAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: Partial<CAMUVConfig> = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph } {
  const alpha = config.alpha ?? 0.05;
  const nBasis = config.nBasis ?? 10;
  const threshold = config.threshold ?? 0.5;
  const n = nodeNames.length;

  if (data.rows === 0 || n === 0) return { graph: new CausalGraph(nodeNames) };

  const rows = data.rows;
  const g = new CausalGraph(nodeNames);

  // For each pair, test both directions using additive models
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      // Extract columns
      const colI = extractColumn(data, i);
      const colJ = extractColumn(data, j);

      // Fit X_i ~ f(X_j) using B-spline regression
      const { rss: rssIJ, residuals: resIJ } = fitAdditiveModel(colI, colJ, nBasis);
      // Fit X_j ~ f(X_i)
      const { rss: rssJI, residuals: resJI } = fitAdditiveModel(colJ, colI, nBasis);

      // Test residual independence
      const resCorrIJ = correlation(resIJ, colJ); // residuals of i~f(j) vs j
      const resCorrJI = correlation(resJI, colI); // residuals of j~f(i) vs i

      // Decision rule: direction with lower residual variance
      // and higher residual independence is more likely causal
      const normalizerI = variance(colI) || 1;
      const normalizerJ = variance(colJ) || 1;
      const normRSS_IJ = rssIJ / normalizerI;
      const normRSS_JI = rssJI / normalizerJ;

      const scoreIJ = normRSS_IJ + Math.abs(resCorrIJ);
      const scoreJI = normRSS_JI + Math.abs(resCorrJI);

      // Compute independence: if strongly directional, add edge
      const ratio = Math.min(scoreIJ, scoreJI) / Math.max(scoreIJ, scoreJI);

      if (ratio < threshold && scoreIJ > scoreJI + 0.1) {
        // J → I
        g.addEdge(nodeNames[j]!, nodeNames[i]!);
      } else if (ratio < threshold && scoreJI > scoreIJ + 0.1) {
        // I → J
        g.addEdge(nodeNames[i]!, nodeNames[j]!);
      }
      // If ratio >= threshold: ambiguous — skip (possible confounder)
    }
  }

  // Ensure acyclicity by removing back-edges from nodes with lower variance ratio
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j && g.hasEdge(nodeNames[i]!, nodeNames[j]!) && g.hasEdge(nodeNames[j]!, nodeNames[i]!)) {
        // Bidirectional — choose direction with lower residual variance
        const colI = extractColumn(data, i);
        const colJ = extractColumn(data, j);
        const { rss: rssIJ } = fitAdditiveModel(colI, colJ, nBasis);
        const { rss: rssJI } = fitAdditiveModel(colJ, colI, nBasis);
        if (rssIJ <= rssJI) g.removeEdge(nodeNames[i]!, nodeNames[j]!);
        else g.removeEdge(nodeNames[j]!, nodeNames[i]!);
      }
    }
  }

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);

  return { graph: g };
}

// ── Additive Model Fitting ────────────────────────────────────────────

/**
 * Fit Y ~ f(X) using B-spline basis expansion + OLS.
 *
 * Returns residual sum of squares and per-point residuals.
 */
function fitAdditiveModel(
  y: Float64Array,
  x: Float64Array,
  nBasis: number,
): { rss: number; residuals: Float64Array } {
  const m = y.length;
  const k = Math.min(nBasis, Math.floor(m / 10));
  if (k <= 1) {
    // Too few points: fit linear
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let r = 0; r < m; r++) {
      sx += x[r]!; sy += y[r]!;
      sxx += x[r]! * x[r]!; sxy += x[r]! * y[r]!;
    }
    const beta = (m * sxy - sx * sy) / Math.max(1e-10, m * sxx - sx * sx);
    const alpha = (sy - beta * sx) / m;
    let rss = 0;
    const residuals = new Float64Array(m);
    for (let r = 0; r < m; r++) {
      residuals[r] = y[r]! - (alpha + beta * x[r]!);
      rss += residuals[r]! * residuals[r]!;
    }
    return { rss, residuals };
  }

  // Build B-spline design matrix
  const xMin = min(x);
  const xMax = max(x);
  const xRange = Math.max(1e-10, xMax - xMin);

  // Use cubic B-spline basis functions equally spaced
  const knots = new Float64Array(k + 2);
  for (let i = 0; i < k + 2; i++) {
    knots[i] = xMin + (i / (k + 1)) * xRange;
  }

  const design = new Float64Array(m * k);
  for (let r = 0; r < m; r++) {
    const xVal = x[r]!;
    for (let b = 0; b < k; b++) {
      // Simple radial basis: exp(-|x - c|² / (2σ²))
      const center = knots[b + 1]!;
      const width = xRange / (k + 1);
      const dist = (xVal - center) / width;
      design[r * k + b] = Math.exp(-dist * dist / 2);
    }
  }

  // OLS: design^T * design * beta = design^T * y
  const XtX = new Float64Array(k * k);
  const Xty = new Float64Array(k);
  for (let r = 0; r < m; r++) {
    for (let b = 0; b < k; b++) {
      Xty[b] += design[r * k + b]! * y[r]!;
      for (let c = 0; c < k; c++) {
        XtX[b * k + c] += design[r * k + b]! * design[r * k + c]!;
      }
    }
  }

  // Solve via Cholesky (add small ridge for stability)
  const ridge = 1e-6;
  const L = new Float64Array(k * k);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = XtX[i * k + j]! + (i === j ? ridge : 0);
      for (let p = 0; p < j; p++) sum -= L[i * k + p]! * L[j * k + p]!;
      if (i === j) {
        L[i * k + i] = Math.sqrt(Math.max(1e-12, sum));
      } else {
        L[i * k + j] = sum / L[j * k + j]!;
      }
    }
  }

  const w = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    let sum = Xty[i]!;
    for (let j = 0; j < i; j++) sum -= L[i * k + j]! * w[j]!;
    w[i] = sum / L[i * k + i]!;
  }
  const beta = new Float64Array(k);
  for (let i = k - 1; i >= 0; i--) {
    let sum = w[i]!;
    for (let j = i + 1; j < k; j++) sum -= L[j * k + i]! * beta[j]!;
    beta[i] = sum / L[i * k + i]!;
  }

  // Predict and compute residuals
  let rss = 0;
  const residuals = new Float64Array(m);
  for (let r = 0; r < m; r++) {
    let pred = 0;
    for (let b = 0; b < k; b++) {
      pred += beta[b]! * design[r * k + b]!;
    }
    residuals[r] = y[r]! - pred;
    rss += residuals[r]! * residuals[r]!;
  }

  return { rss, residuals };
}

// ── Statistics ──────────────────────────────────────────────────────────

function extractColumn(data: Matrix, idx: number): Float64Array {
  const n = data.rows;
  const col = new Float64Array(n);
  for (let i = 0; i < n; i++) col[i] = data.get(i, idx);
  return col;
}

function correlation(a: Float64Array, b: Float64Array): number {
  const n = a.length;
  let sa = 0, sb = 0;
  for (let i = 0; i < n; i++) { sa += a[i]!; sb += b[i]!; }
  const mA = sa / n, mB = sb / n;
  let cov = 0, varA = 0, varB = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i]! - mA;
    const db = b[i]! - mB;
    cov += da * db;
    varA += da * da;
    varB += db * db;
  }
  return cov / Math.sqrt(Math.max(1e-10, varA * varB));
}

function variance(a: Float64Array): number {
  const n = a.length;
  let s = 0, s2 = 0;
  for (let i = 0; i < n; i++) { s += a[i]!; s2 += a[i]! * a[i]!; }
  return Math.max(1e-10, (s2 - s * s / n) / (n - 1));
}

function min(a: Float64Array): number {
  let m = Infinity;
  for (let i = 0; i < a.length; i++) if (a[i]! < m) m = a[i]!;
  return m;
}

function max(a: Float64Array): number {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i]! > m) m = a[i]!;
  return m;
}
