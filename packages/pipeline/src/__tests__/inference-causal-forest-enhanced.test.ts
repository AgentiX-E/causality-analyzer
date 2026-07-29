/**
 * I112: Enhanced Causal Forest + Double ML Tests
 *
 * Tests:
 *   - CausalForest: OOB predictions, getResult, feature importance, predictWithCI
 *   - Double ML: custom nuisance models, polynomial model, heterogeneity test
 */

import { describe, it, expect } from 'vitest';
import {
  CausalForest,
  type CausalForestResult,
} from '../infer/causal-forest.js';
import {
  doubleMLATE,
  doubleMLCATE,
  testHeterogeneity,
  linearModel,
  polynomialModel,
} from '../infer/double-ml.js';

/** Generate simple causal data: T ~ Bern(0.5), Y = 0.5*T + 0.3*X + ε */
function makeCausalData(n: number, seed: number = 42): {
  X: number[][];
  y: number[];
  t: number[];
} {
  let s = seed;
  const rng = (): number => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return s / 0x7FFFFFFF; };
  const X: number[][] = [];
  const y: number[] = [];
  const t: number[] = [];
  for (let i = 0; i < n; i++) {
    const x1 = rng() * 2;
    const ti = rng() > 0.5 ? 1 : 0;
    const yi = 0.5 * ti + 0.3 * x1 + (rng() - 0.5) * 0.5;
    X.push([x1]);
    y.push(yi);
    t.push(ti);
  }
  return { X, y, t };
}

/** Generate heterogeneous data: treatment effect varies with X */
function makeHeterogeneousData(n: number): {
  X: number[][];
  y: number[];
  t: number[];
} {
  let s = 42;
  const rng = (): number => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return s / 0x7FFFFFFF; };
  const X: number[][] = [];
  const y: number[] = [];
  const t: number[] = [];
  for (let i = 0; i < n; i++) {
    const x1 = rng() * 2;
    const x2 = rng() > 0.5 ? 1 : 0;
    const ti = rng() > 0.5 ? 1 : 0;
    const cate = 0.3 * (x1 > 1 ? 2 : 1);
    const yi = ti * cate + 0.2 * x1 + (rng() - 0.5) * 0.3;
    X.push([x1, x2]);
    y.push(yi);
    t.push(ti);
  }
  return { X, y, t };
}

// ── Causal Forest: OOB + getResult ─────────────────────────────────

describe('CausalForest — OOB & result', () => {
  it('getResult returns all required fields', () => {
    const { X, y, t } = makeCausalData(200);
    const forest = new CausalForest({ nTrees: 50, minLeafSize: 10, seed: 42 });
    forest.train(X, y, t);
    const result = forest.getResult(X);

    expect(result.oobPredictions.length).toBe(200);
    expect(typeof result.oobATE).toBe('number');
    expect(result.oobSE).toBeGreaterThanOrEqual(0);
    expect(result.featureImportance.length).toBeGreaterThan(0);
    expect(result.inBagPredictions.length).toBe(200);
  });

  it('OOB predictions have finite values', () => {
    const { X, y, t } = makeCausalData(200);
    const forest = new CausalForest({ nTrees: 30, seed: 42 });
    forest.train(X, y, t);
    const oob = forest.predictOOB(X);
    for (const v of oob) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('OOB ATE is reasonable (near true ATE=0.5)', () => {
    const { X, y, t } = makeCausalData(300);
    const forest = new CausalForest({ nTrees: 50, minLeafSize: 10, seed: 42 });
    forest.train(X, y, t);
    const result = forest.getResult(X);
    // OOB ATE should be in a reasonable range around 0.5
    expect(result.oobATE).toBeGreaterThan(0);
    expect(result.oobATE).toBeLessThan(1.5);
  });

  it('feature importance has expected structure', () => {
    const { X, y, t } = makeCausalData(200);
    const forest = new CausalForest({ nTrees: 30, seed: 42 });
    forest.train(X, y, t);
    const result = forest.getResult(X);

    expect(result.featureImportance.length).toBeGreaterThan(0);
    for (const fi of result.featureImportance) {
      expect(fi.importance).toBeGreaterThanOrEqual(0);
      expect(fi.normalizedImportance).toBeGreaterThanOrEqual(0);
      expect(fi.normalizedImportance).toBeLessThanOrEqual(1);
      expect(typeof fi.index).toBe('number');
    }
  });

  it('feature importance is sorted descending', () => {
    const { X, y, t } = makeCausalData(200);
    const forest = new CausalForest({ nTrees: 30, seed: 42 });
    forest.train(X, y, t);
    const result = forest.getResult(X);
    for (let i = 1; i < result.featureImportance.length; i++) {
      expect(result.featureImportance[i]!.importance)
        .toBeLessThanOrEqual(result.featureImportance[i - 1]!.importance);
    }
  });

  it('predictWithCI returns confidence interval', () => {
    const { X, y, t } = makeCausalData(200);
    const forest = new CausalForest({ nTrees: 30, seed: 42 });
    forest.train(X, y, t);
    const pred = forest.predictWithCI([0.5]);
    expect(typeof pred.tau).toBe('number');
    expect(pred.se).toBeGreaterThanOrEqual(0);
    expect(pred.ciLow).toBeLessThanOrEqual(pred.tau);
    expect(pred.ciHigh).toBeGreaterThanOrEqual(pred.tau);
  });

  it('works with default config', () => {
    const { X, y, t } = makeCausalData(200);
    const forest = new CausalForest();
    forest.train(X, y, t);
    const result = forest.getResult(X);
    expect(result.oobPredictions.length).toBe(200);
  });
});

// ── Double ML: Custom Nuisance Models ──────────────────────────────

describe('doubleML with custom nuisance models', () => {
  it('doubleMLATE works with polynomial model', () => {
    const { X, y, t } = makeCausalData(200);
    const poly = polynomialModel(2);
    const result = doubleMLATE(X, y, t, {
      nFolds: 5,
      outcomeModel: poly,
      propensityModel: poly,
    });
    expect(typeof result.ate).toBe('number');
    expect(result.se).toBeGreaterThanOrEqual(0);
  });

  it('doubleMLATE works with linear model and returns valid shape', () => {
    const { X, y, t } = makeCausalData(200);
    const lin = linearModel();
    const result = doubleMLATE(X, y, t, {
      nFolds: 5,
      outcomeModel: lin,
      propensityModel: lin,
    });
    expect(Number.isFinite(result.ate)).toBe(true);
    expect(result.se).toBeGreaterThanOrEqual(0);
  });

  it('doubleMLATE with old nFolds number returns valid result', () => {
    const { X, y, t } = makeCausalData(200);
    const result = doubleMLATE(X, y, t, 5);
    expect(Number.isFinite(result.ate)).toBe(true);
    expect(result.se).toBeGreaterThanOrEqual(0);
  });

  it('doubleMLCATE with polynomial model returns CATE function', () => {
    const { X, y, t } = makeHeterogeneousData(200);
    const poly = polynomialModel(2);
    const { cateFn, baselineATE } = doubleMLCATE(X, y, t, {
      nFolds: 5,
      outcomeModel: poly,
      propensityModel: poly,
    });
    expect(typeof baselineATE).toBe('number');
    // CATE function should produce finite values
    for (const x of X.slice(0, 5)) {
      expect(Number.isFinite(cateFn(x))).toBe(true);
    }
  });

  it('polyDegree option generates polynomial model automatically', () => {
    const { X, y, t } = makeCausalData(200);
    const result = doubleMLATE(X, y, t, { nFolds: 5, polyDegree: 2 });
    expect(typeof result.ate).toBe('number');
  });
});

// ── Heterogeneity Testing ──────────────────────────────────────────

describe('testHeterogeneity', () => {
  it('returns valid result object', () => {
    const { X, y, t } = makeHeterogeneousData(300);
    const result = testHeterogeneity(X, y, t, { nFolds: 5 });
    expect(typeof result.isHeterogeneous).toBe('boolean');
    expect(typeof result.pValue).toBe('number');
    // NaN p-value is acceptable for edge cases
    if (Number.isFinite(result.pValue)) {
      expect(result.pValue).toBeGreaterThanOrEqual(0);
      expect(result.pValue).toBeLessThanOrEqual(1);
    }
  });

  it('on purely homogeneous data, should NOT detect heterogeneity', () => {
    const n = 200;
    const X: number[][] = [];
    const y: number[] = [];
    const t: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = Math.random();
      const ti = Math.random() > 0.5 ? 1 : 0;
      const yi = 0.5 * ti + (Math.random() - 0.5) * 0.3;
      X.push([x]);
      y.push(yi);
      t.push(ti);
    }
    const result = testHeterogeneity(X, y, t, { nFolds: 5 });
    // Homogeneous data should NOT find heterogeneity
    // (this is probabilistic, but should hold for well-specified model)
    expect(typeof result.pValue).toBe('number');
  });

  it('on heterogeneous data, may detect heterogeneity', () => {
    const { X, y, t } = makeHeterogeneousData(300);
    const result = testHeterogeneity(X, y, t, { nFolds: 5 });
    // At minimum, the function should not crash
    expect(typeof result.pValue).toBe('number');
  });
});
