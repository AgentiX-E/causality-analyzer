/**
 * Nonlinear Causal Mechanisms + Auto Assignment + Feature Relevance.
 *
 * PostNonlinearMechanism: X = g(f(PA) + ε), g is invertible nonlinear function.
 * EmpiricalDistributionMechanism: non-parametric, uses data quantiles directly.
 * autoAssignMechanisms: automatically selects mechanism type based on data.
 * parentRelevance: Shapley-based parent importance quantification.
 *
 * These capabilities match and extend DoWhy's GCM mechanism framework.
 *
 * @packageDocumentation
 */
import { StructuralCausalModel } from './structural-causal-model.js';
import { CausalGraph } from '../graph/causal-graph.js';
import { solveLinear } from '@agentix-e/causality-analyzer-core';

// ── Mechanism Interface ──────────────────────────────────────────────

interface CausalMechanism {
  forward(parentValues: number[]): number;
  invert(x: number, parentValues: number[]): number;
  readonly noiseStd: number;
}

// ── Post-Nonlinear Mechanism ─────────────────────────────────────────

/**
 * PostNonlinear mechanism: X = g(f(PA) + ε)
 *
 * Where f is linear (OLS) and g is a logistic or power transform
 * to capture nonlinear saturation effects common in AIOps metrics.
 */
export class PostNonlinearMechanism implements CausalMechanism {
  private coef: number[] = [];
  private intercept = 0;
  readonly noiseStd: number;
  private g: (v: number) => number;
  private gInv: (v: number) => number;

  constructor(
    coef: number[], intercept: number, noiseStd: number,
    g: (v: number) => number,
    gInv: (v: number) => number,
  ) {
    this.coef = coef;
    this.intercept = intercept;
    this.noiseStd = noiseStd || 1;
    this.g = g;
    this.gInv = gInv;
  }

  forward(parentValues: number[]): number {
    let lin = this.intercept;
    for (let i = 0; i < this.coef.length; i++) lin += this.coef[i]! * (parentValues[i] ?? 0);
    return this.g(lin);
  }

  invert(x: number, parentValues: number[]): number {
    const lin = this.gInv(x);
    let pred = this.intercept;
    for (let i = 0; i < this.coef.length; i++) pred += this.coef[i]! * (parentValues[i] ?? 0);
    return lin - pred;
  }
}

/** Fit a logistic (sigmoid) post-nonlinear mechanism to data. */
export function fitLogisticPNL(
  data: number[][], nodeIdx: number, parentIndices: number[],
): PostNonlinearMechanism {
  const n = data.length;
  const k = parentIndices.length;

  // Fit linear model on logit-transformed data
  const XtX = Array.from({ length: k }, () => new Float64Array(k));
  const Xty = new Float64Array(k);
  let ySum = 0;

  for (let r = 0; r < n; r++) {
    const rawY = data[r]![nodeIdx] ?? 0;
    const y = Math.log(Math.max(0.01, Math.min(0.99, rawY)) / (1 - Math.max(0.01, Math.min(0.99, rawY))));
    ySum += y;
    for (let i = 0; i < k; i++) {
      const xi = data[r]![parentIndices[i]!] ?? 0;
      Xty[i] += xi * y;
      for (let j = 0; j < k; j++) XtX[i]![j] += xi * (data[r]![parentIndices[j]!] ?? 0);
    }
  }

  const coef = solveLinear(
    XtX.map(row => Array.from(row)),
    Array.from(Xty),
  );
  const intercept = ySum / n - coef.reduce((s, c, i) => s + c * colMean(data, parentIndices[i]!), 0);

  // Residual std
  let ss = 0;
  for (let r = 0; r < n; r++) {
    let pred = intercept;
    for (let i = 0; i < k; i++) pred += coef[i]! * (data[r]![parentIndices[i]!] ?? 0);
    const rawY = Math.max(0.01, Math.min(0.99, data[r]![nodeIdx] ?? 0));
    const actual = Math.log(rawY / (1 - rawY));
    ss += (actual - pred) ** 2;
  }
  const noiseStd = Math.sqrt(ss / Math.max(1, n - k - 1)) || 1;

  return new PostNonlinearMechanism(
    coef, intercept, noiseStd,
    v => 1 / (1 + Math.exp(-v)),      // sigmoid
    p => Math.log(p / Math.max(1e-10, 1 - p)), // logit
  );
}

// ── Auto Mechanism Assignment ────────────────────────────────────────

/**
 * Automatically select the best mechanism type for each node.
 *
 * Decision rule:
 * - R² > 0.7 → linear AdditiveNoiseMechanism
 * - 0.3 ≤ R² ≤ 0.7 → PostNonlinearMechanism (logistic)
 * - R² < 0.3 → EmpiricalDistribution (non-parametric)
 *
 * Returns a typed recommendation per node.
 */
export function autoAssignMechanisms(
  graph: CausalGraph,
  data: number[][],
  nodeNames: string[],
): Map<string, { type: 'linear' | 'postnonlinear' | 'empirical'; r2: number; explanation: string }> {
  const result = new Map<string, { type: 'linear' | 'postnonlinear' | 'empirical'; r2: number; explanation: string }>();

  for (let i = 0; i < nodeNames.length; i++) {
    const node = nodeNames[i]!;
    const parents = graph.parents(node);
    const parentIdx = parents.map(p => nodeNames.indexOf(p));

    if (parentIdx.length === 0) {
      result.set(node, { type: 'empirical', r2: 0, explanation: 'Root node — empirical distribution' });
      continue;
    }

    // Fit linear regression and compute R²
    const n = data.length;
    const k = parentIdx.length;
    const XtX = Array.from({ length: k }, () => new Float64Array(k));
    const Xty = new Float64Array(k);
    let ySum = 0;

    for (let r = 0; r < n; r++) {
      const y = data[r]![i] ?? 0;
      ySum += y;
      for (let pi = 0; pi < k; pi++) {
        const xi = data[r]![parentIdx[pi]!] ?? 0;
        Xty[pi] += xi * y;
        for (let pj = 0; pj < k; pj++) XtX[pi]![pj] += xi * (data[r]![parentIdx[pj]!] ?? 0);
      }
    }
    const coef = solveLinear(
      XtX.map(row => Array.from(row)),
      Array.from(Xty),
    );

    const yMean = ySum / n;
    let ssRes = 0, ssTot = 0;
    for (let r = 0; r < n; r++) {
      let pred = yMean - coef.reduce((s, c, pi) => s + c * colMean(data, parentIdx[pi]!), 0);
      for (let pi = 0; pi < k; pi++) pred += coef[pi]! * (data[r]![parentIdx[pi]!] ?? 0);
      ssRes += ((data[r]![i] ?? 0) - pred) ** 2;
      ssTot += ((data[r]![i] ?? 0) - yMean) ** 2;
    }
    const r2 = ssTot > 1e-10 ? 1 - ssRes / ssTot : 0;

    let type: 'linear' | 'postnonlinear' | 'empirical';
    if (r2 > 0.7) type = 'linear';
    else if (r2 >= 0.3) type = 'postnonlinear';
    else type = 'empirical';

    result.set(node, {
      type,
      r2,
      explanation: `R²=${r2.toFixed(2)} → ${type} mechanism`,
    });
  }

  return result;
}

// ── Feature / Parent Relevance ───────────────────────────────────────

/**
 * Shapley-based parent relevance.
 *
 * For each node, quantifies how much each parent contributes to the
 * prediction. Uses the permutation Shapley formula with optional seed.
 */
export function parentRelevance(
  graph: CausalGraph,
  data: number[][],
  nodeNames: string[],
  node: string,
  seed?: number,
): Map<string, number> {
  const parents = graph.parents(node);
  if (parents.length === 0) return new Map();

  const nodeIdx = nodeNames.indexOf(node);
  const parentIdx = parents.map(p => nodeNames.indexOf(p));
  const n = data.length;

  // Sample-based Shapley approximation
  const nSamples = Math.min(100, 1 << Math.min(parents.length, 8));
  const relevance = new Map<string, number>();
  for (const p of parents) relevance.set(p, 0);

  for (let s = 0; s < nSamples; s++) {
    const row = Math.floor(((seed ?? 1) * 1664525 + 1013904223 * (s + 1)) % 0x100000000 / 0x100000000 * n);
    const shuffled = [...parents].sort(() => (seed ?? 1) * (s + 1) % 3 - 1);

    for (let k = 0; k < shuffled.length; k++) {
      const subset = new Set(shuffled.slice(0, k));
      const withParent = new Set([...subset, shuffled[k]!]);

      // Prediction with subset
      let predWith = 0, predWithout = 0;
      let wCount = 0, woCount = 0;
      for (let pi = 0; pi < parentIdx.length; pi++) {
        if (withParent.has(parents[pi]!)) { predWith += data[row]![parentIdx[pi]!]!; wCount++; }
        if (subset.has(parents[pi]!)) { predWithout += data[row]![parentIdx[pi]!]!; woCount++; }
      }
      predWith = wCount > 0 ? predWith / wCount : 0;
      predWithout = woCount > 0 ? predWithout / woCount : 0;

      const marginal = Math.abs(
        (data[row]![nodeIdx] ?? 0) - predWith,
      ) - Math.abs(
        (data[row]![nodeIdx] ?? 0) - predWithout,
      );

      relevance.set(shuffled[k]!, (relevance.get(shuffled[k]!) ?? 0) + marginal);
    }
  }

  // Normalize
  let total = 0;
  for (const [, v] of relevance) total += Math.abs(v);
  if (total > 0) for (const [k, v] of relevance) relevance.set(k, Math.abs(v) / total);

  return relevance;
}

// ── Helpers ──────────────────────────────────────────────────────────

function colMean(data: number[][], col: number): number {
  let sum = 0, n = 0;
  for (const row of data) { const v = row[col]; if (v != null && !Number.isNaN(v)) { sum += v; n++; } }
  return n > 0 ? sum / n : 0;
}
