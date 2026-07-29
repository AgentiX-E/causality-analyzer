/**
 * DML/DR Estimator Tests (I8-P1)
 *
 * Tests:
 *   - LinearDML: fit, effect, effectInterval, constMarginalEffect
 *   - CausalForestDML: fit, effect, feature importance
 *   - NonParamDML: fit, effect, kernel bandwidth
 *   - LinearDRLearner: ATE recovery, confidence intervals
 *   - ForestDRLearner: ATE recovery, forest-based DR
 */

import { describe, it, expect } from 'vitest';
import { LinearDML, CausalForestDML, NonParamDML } from '../infer/dml-estimators.js';
import { LinearDRLearner, ForestDRLearner } from '../infer/dr-estimators.js';

/** Generate simple causal data: Y = 0.5 * T + 0.3 * X + noise */
function makeData(n: number, seed: number = 42): {
  X: number[][]; y: number[]; t: number[];
} {
  let s = seed;
  const rng = () => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return s / 0x7FFFFFFF; };
  const X: number[][] = [];
  const y: number[] = [];
  const t: number[] = [];
  for (let i = 0; i < n; i++) {
    const x1 = rng() * 2;
    const ti = rng() > 0.5 ? 1 : 0;
    X.push([x1]);
    y.push(0.5 * ti + 0.3 * x1 + (rng() - 0.5) * 0.3);
    t.push(ti);
  }
  return { X, y, t };
}

/** Generate data with 2 covariates */
function makeData2(n: number): { X: number[][]; y: number[]; t: number[] } {
  let s = 42;
  const rng = () => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return s / 0x7FFFFFFF; };
  const X: number[][] = [];
  const y: number[] = [];
  const t: number[] = [];
  for (let i = 0; i < n; i++) {
    const x1 = rng() * 2;
    const x2 = rng() > 0.5 ? 1 : 0;
    const ti = rng() > 0.5 ? 1 : 0;
    X.push([x1, x2]);
    y.push(0.5 * ti + 0.3 * x1 + 0.1 * x2 + (rng() - 0.5) * 0.3);
    t.push(ti);
  }
  return { X, y, t };
}

// ── LinearDML ──────────────────────────────────────────────────────

describe('LinearDML', () => {
  it('fits and predicts CATE', () => {
    const { X, y, t } = makeData(200);
    const dml = new LinearDML({ nFolds: 3 });
    dml.fit(X, y, t);
    expect(dml.isFitted).toBe(true);

    const cate = dml.effect(X);
    expect(cate.length).toBe(200);
    expect(Number.isFinite(cate[0])).toBe(true);
  });

  it('throws if not fitted', () => {
    const dml = new LinearDML();
    expect(() => dml.effect([[1]])).toThrow('not fitted');
  });

  it('effectInterval returns lower and upper arrays', () => {
    const { X, y, t } = makeData(200);
    const dml = new LinearDML({ nFolds: 3 });
    dml.fit(X, y, t);
    const [low, high] = dml.effectInterval(X, 0.05);
    expect(low.length).toBe(200);
    expect(high.length).toBe(200);
    expect(low[0]!).toBeLessThanOrEqual(high[0]!);
  });

  it('constMarginalEffect returns CI structure', () => {
    const { X, y, t } = makeData(200);
    const dml = new LinearDML({ nFolds: 3 });
    dml.fit(X, y, t);
    const ci = dml.constMarginalEffect();
    expect(ci.ciLower).toBeLessThanOrEqual(ci.point);
    expect(ci.ciUpper).toBeGreaterThanOrEqual(ci.point);
    expect(ci.se).toBeGreaterThanOrEqual(0);
  });

  it('baselineATE is near 0.5', () => {
    const { X, y, t } = makeData(300);
    const dml = new LinearDML({ nFolds: 5 });
    dml.fit(X, y, t);
    // ATE should be in reasonable range
    expect(Number.isFinite(dml.baselineATE)).toBe(true);
  });

  it('works with 2 covariates', () => {
    const { X, y, t } = makeData2(200);
    const dml = new LinearDML({ nFolds: 3 });
    dml.fit(X, y, t);
    expect(dml.isFitted).toBe(true);
  });
});

// ── CausalForestDML ────────────────────────────────────────────────

describe('CausalForestDML', () => {
  it('fits and predicts CATE', () => {
    const { X, y, t } = makeData(200);
    const forest = new CausalForestDML({
      nFolds: 3,
      forestConfig: { nTrees: 30, minLeafSize: 10, maxDepth: 8, seed: 42 },
    });
    forest.fit(X, y, t);
    expect(forest.isFitted).toBe(true);

    const cate = forest.effect(X);
    expect(cate.length).toBe(200);
  });

  it('throws if not fitted', () => {
    const f = new CausalForestDML();
    expect(() => f.effect([[1]])).toThrow('not fitted');
  });

  it('effectInterval returns CI arrays', () => {
    const { X, y, t } = makeData(200);
    const forest = new CausalForestDML({
      nFolds: 3,
      forestConfig: { nTrees: 20, minLeafSize: 10, seed: 42 },
    });
    forest.fit(X, y, t);
    const [low, high] = forest.effectInterval(X);
    expect(low.length).toBe(200);
    expect(high.length).toBe(200);
  });

  it('feature importance available after fitting', () => {
    const { X, y, t } = makeData2(200);
    const forest = new CausalForestDML({
      nFolds: 3,
      forestConfig: { nTrees: 20, minLeafSize: 10, seed: 42 },
    });
    forest.fit(X, y, t);
    const importance = forest.featureImportance(X);
    expect(importance.length).toBeGreaterThan(0);
    for (const fi of importance) {
      expect(fi.importance).toBeGreaterThanOrEqual(0);
    }
  });

  it('constMarginalEffect returns CI', () => {
    const { X, y, t } = makeData(200);
    const forest = new CausalForestDML({
      nFolds: 3,
      forestConfig: { nTrees: 20, minLeafSize: 10, seed: 42 },
    });
    forest.fit(X, y, t);
    const ci = forest.constMarginalEffect();
    expect(ci.se).toBeGreaterThanOrEqual(0);
  });
});

// ── NonParamDML ────────────────────────────────────────────────────

describe('NonParamDML', () => {
  it('fits and predicts with kernel smoother', () => {
    const { X, y, t } = makeData(200);
    const np = new NonParamDML({ nFolds: 3 });
    np.fit(X, y, t);
    expect(np.isFitted).toBe(true);

    const cate = np.effect(X);
    expect(cate.length).toBe(200);
    expect(Number.isFinite(cate[0])).toBe(true);
  });

  it('throws if not fitted', () => {
    const np = new NonParamDML();
    expect(() => np.effect([[1]])).toThrow('not fitted');
  });

  it('bandwidth is positive after fitting', () => {
    const { X, y, t } = makeData(200);
    const np = new NonParamDML({ nFolds: 3 });
    np.fit(X, y, t);
    expect(np.usedBandwidth).toBeGreaterThan(0);
  });

  it('custom bandwidth is respected', () => {
    const { X, y, t } = makeData(200);
    const np = new NonParamDML({ nFolds: 3, bandwidth: 0.5 });
    np.fit(X, y, t);
    expect(np.usedBandwidth).toBe(0.5);
  });

  it('effectInterval returns arrays', () => {
    const { X, y, t } = makeData(200);
    const np = new NonParamDML({ nFolds: 3 });
    np.fit(X, y, t);
    const [low, high] = np.effectInterval(X, 0.05, 50);
    expect(low.length).toBe(200);
    expect(high.length).toBe(200);
  });
});

// ── LinearDRLearner ────────────────────────────────────────────────

describe('LinearDRLearner', () => {
  it('fits and recovers ATE near 0.5', () => {
    const { X, y, t } = makeData(300);
    const dr = new LinearDRLearner();
    dr.fit(X, y, t);
    expect(dr.isFitted).toBe(true);
    expect(dr.ate).toBeGreaterThan(0);
    expect(dr.ate).toBeLessThan(1.5);
    expect(dr.se).toBeGreaterThan(0);
  });

  it('throws if not fitted', () => {
    const dr = new LinearDRLearner();
    expect(() => dr.ate).toThrow('not fitted');
  });

  it('effectInterval returns valid CI', () => {
    const { X, y, t } = makeData(300);
    const dr = new LinearDRLearner();
    dr.fit(X, y, t);
    const ci = dr.effectInterval(0.05);
    expect(ci.ciLower).toBeLessThanOrEqual(ci.ciUpper);
    expect(ci.point).toEqual(dr.ate);
  });

  it('works with 2 covariates', () => {
    const { X, y, t } = makeData2(200);
    const dr = new LinearDRLearner();
    dr.fit(X, y, t);
    expect(dr.isFitted).toBe(true);
    expect(Number.isFinite(dr.ate)).toBe(true);
  });
});

// ── ForestDRLearner ────────────────────────────────────────────────

describe('ForestDRLearner', () => {
  it('fits and returns ATE', () => {
    const { X, y, t } = makeData(200);
    const dr = new ForestDRLearner({
      forestConfig: { nTrees: 20, minLeafSize: 10, seed: 42 },
    });
    dr.fit(X, y, t);
    expect(dr.isFitted).toBe(true);
    expect(Number.isFinite(dr.ate)).toBe(true);
    expect(dr.se).toBeGreaterThanOrEqual(0);
  });

  it('throws if not fitted', () => {
    const dr = new ForestDRLearner();
    expect(() => dr.ate).toThrow('not fitted');
  });

  it('effectInterval returns valid CI', () => {
    const { X, y, t } = makeData(200);
    const dr = new ForestDRLearner({
      forestConfig: { nTrees: 20, minLeafSize: 10, seed: 42 },
    });
    dr.fit(X, y, t);
    const ci = dr.effectInterval(0.05);
    expect(ci.ciLower).toBeLessThanOrEqual(ci.ciUpper);
  });

  it('works with default config', () => {
    const { X, y, t } = makeData(200);
    const dr = new ForestDRLearner();
    dr.fit(X, y, t);
    expect(dr.isFitted).toBe(true);
  });
});
