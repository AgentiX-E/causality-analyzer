/**
 * CATE Unification — common interface for heterogeneous treatment effect
 * estimation across all three CA implementations.
 *
 * @packageDocumentation
 */

import { CausalForest, type CausalForestConfig } from './causal-forest.js';
import { doubleMLCATE } from './double-ml.js';
import { estimateCATE } from './cate-fairness.js';

// ── Types ────────────────────────────────────────────────────────────────

export type CATEstimator = 'linear' | 'double-ml' | 'causal-forest';

export interface CATEstimate {
  readonly estimator: CATEstimator;
  readonly cate: ReadonlyArray<number>;
  readonly baselineATE: number;
  readonly cateSD: number;
  readonly baselineSE?: number;
  readonly config: Record<string, unknown>;
}

export interface CATEModelComparison {
  readonly models: Array<{
    readonly name: string;
    readonly cate: ReadonlyArray<number>;
    readonly ate: number;
    readonly sd: number;
  }>;
  readonly correlations: Array<{
    readonly modelA: string;
    readonly modelB: string;
    readonly correlation: number;
  }>;
  readonly concordance: number;
}

// ── Unified CATE ────────────────────────────────────────────────────────

export function unifiedCATE(
  X: number[][],
  y: number[],
  t: number[],
  estimator: CATEstimator,
  config: Record<string, unknown> = {},
): CATEstimate {
  switch (estimator) {
    case 'linear': {
      // Build combined data: [T, Y, X...]
      const data = X.map((row, i) => [t[i], y[i], ...row]);
      const featureIndices = Array.from({ length: X[0]?.length ?? 0 }, (_, j) => j + 2);
      const { cateFn } = estimateCATE(data, 0, 1, featureIndices);
      const cate = X.map(row => cateFn(row));
      const baselineATE = cate.reduce((a, b) => a + b, 0) / cate.length;
      const cateSD = Math.sqrt(cate.reduce((s, v) => s + (v - baselineATE) ** 2, 0) / cate.length);
      return { estimator: 'linear', cate, baselineATE, cateSD, config };
    }

    case 'double-ml': {
      const { cateFn, baselineATE } = doubleMLCATE(X, y, t, { nFolds: (config.nFolds as number) ?? 5 });
      const cate = X.map(row => cateFn(row));
      const sd = Math.sqrt(cate.reduce((s, v) => s + (v - baselineATE) ** 2, 0) / cate.length);
      return { estimator: 'double-ml', cate, baselineATE, cateSD: sd, config };
    }

    case 'causal-forest': {
      const forest = new CausalForest(config);
      forest.train(X, y, t);
      const cate = forest.predict(X);
      const baselineATE = cate.reduce((a, b) => a + b, 0) / cate.length;
      const sd = Math.sqrt(cate.reduce((s, v) => s + (v - baselineATE) ** 2, 0) / cate.length);
      return { estimator: 'causal-forest', cate, baselineATE, cateSD: sd, config };
    }

    default:
      throw new Error(`Unknown CATE estimator: ${estimator as string}`);
  }
}

// ── Model Comparison ────────────────────────────────────────────────────

export function compareCATEModels(
  X: number[][],
  y: number[],
  t: number[],
  config: Record<string, unknown> = {},
): CATEModelComparison {
  const modelNames: CATEstimator[] = ['linear', 'double-ml', 'causal-forest'];
  const results: Array<{ name: string; cate: number[]; ate: number; sd: number }> = [];

  for (const name of modelNames) {
    const r = unifiedCATE(X, y, t, name, { ...config, nFolds: 3, nTrees: 20 });
    results.push({ name, cate: [...r.cate], ate: r.baselineATE, sd: r.cateSD });
  }

  // Pairwise Pearson correlations
  const correlations: Array<{ modelA: string; modelB: string; correlation: number }> = [];
  for (let i = 0; i < results.length; i++) {
    for (let j = i + 1; j < results.length; j++) {
      const a = results[i].cate;
      const b = results[j].cate;
      const n = a.length;
      let sumA = 0, sumB = 0, sumAB = 0, sumA2 = 0, sumB2 = 0;
      for (let k = 0; k < n; k++) {
        sumA += a[k]; sumB += b[k];
        sumAB += a[k] * b[k];
        sumA2 += a[k] * a[k]; sumB2 += b[k] * b[k];
      }
      const denom = Math.sqrt((n * sumA2 - sumA * sumA) * (n * sumB2 - sumB * sumB));
      correlations.push({
        modelA: results[i].name,
        modelB: results[j].name,
        correlation: denom === 0 ? 0 : (n * sumAB - sumA * sumB) / denom,
      });
    }
  }

  // Kendall's W
  const ranks = results.map(r => {
    const sorted = [...r.cate].map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
    const rank = new Array<number>(r.cate.length);
    sorted.forEach(({ i }, idx) => { rank[i] = idx + 1; });
    return rank;
  });

  const n = ranks[0].length;
  const m = ranks.length;
  let sumRSq = 0;
  const meanRank = (n + 1) / 2;
  for (let i = 0; i < n; i++) {
    let total = 0;
    for (let j = 0; j < m; j++) total += ranks[j][i];
    sumRSq += (total - m * meanRank) ** 2;
  }
  const concordance = n > 1 ? (12 * sumRSq) / (m * m * (n * n * n - n)) : 1;

  return { models: results, correlations, concordance };
}
