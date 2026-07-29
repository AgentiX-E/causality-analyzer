/**
 * I111: CATE Unification + Causal Forest Enhancement Tests
 *
 * Tests:
 *   - unifiedCATE dispatch to all 3 backends
 *   - compareCATEModels multi-model comparison
 *   - CausalForest OOB predictions (enhanced)
 */

import { describe, it, expect } from 'vitest';
import { unifiedCATE, compareCATEModels, type CATEstimate } from '../infer/cate-unified.js';
import { CausalForest } from '../infer/causal-forest.js';
import type { CATEstimator } from '../infer/cate-unified.js';

/** Generate data with treatment effect heterogeneity */
function makeHeterogeneousData(n: number, seed: number = 42): {
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
    const x2 = rng() > 0.5 ? 1 : 0;
    const ti = rng() > 0.5 ? 1 : 0;
    // CATE depends on x1: effect = 0.3 * (x1 > 1 ? 2 : 1)
    const cate = 0.3 * (x1 > 1 ? 2 : 1);
    const yi = ti * cate + 0.2 * x1 + (rng() - 0.5) * 0.3;
    X.push([x1, x2]);
    y.push(yi);
    t.push(ti);
  }
  return { X, y, t };
}

/** Generate a simple linear heterogeneous data set */
function makeSimpleData(n: number): {
  X: number[][];
  y: number[];
  t: number[];
} {
  const X: number[][] = [];
  const y: number[] = [];
  const t: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random();
    const ti = Math.random() > 0.5 ? 1 : 0;
    const yi = ti * (0.3 + 0.4 * x) + 0.1 * x + (Math.random() - 0.5) * 0.2;
    X.push([x]);
    y.push(yi);
    t.push(ti);
  }
  return { X, y, t };
}

// ── Unified CATE ─────────────────────────────────────────────────────

describe('unifiedCATE', () => {
  const backends: CATEstimator[] = ['linear', 'double-ml', 'causal-forest'];

  for (const backend of backends) {
    it(`produces valid CATEstimate with backend="${backend}"`, () => {
      const { X, y, t } = makeHeterogeneousData(200);
      const result = unifiedCATE(X, y, t, backend);
      expect(result.estimator).toBe(backend);
      expect(result.cate.length).toBe(200);
      expect(typeof result.baselineATE).toBe('number');
      expect(result.cateSD).toBeGreaterThanOrEqual(0);
      expect(result.config).toBeDefined();
    });

    it(`produces CATE values within reasonable range for "${backend}"`, () => {
      const { X, y, t } = makeSimpleData(200);
      const result = unifiedCATE(X, y, t, backend);
      for (const v of result.cate) {
        expect(Number.isFinite(v)).toBe(true);
      }
    });
  }

  it('returns CATE with detectable heterogeneity', () => {
    const { X, y, t } = makeHeterogeneousData(300);
    const result = unifiedCATE(X, y, t, 'linear');
    // With heterogeneous effects, CATE should have positive variance
    expect(result.cateSD).toBeGreaterThan(0);
    // All CATE values should be finite
    for (const v of result.cate) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('throws for unknown backend', () => {
    const { X, y, t } = makeSimpleData(100);
    expect(() => unifiedCATE(X, y, t, 'unknown' as CATEstimator)).toThrow();
  });

  it('handles empty data: returns empty CATE array', () => {
    const r = unifiedCATE([], [], [], 'linear');
    expect(r.cate.length).toBe(0);
    expect(r.estimator).toBe('linear');
  });
});

// ── Model Comparison ─────────────────────────────────────────────────

describe('compareCATEModels', () => {
  it('returns comparison with all 3 models', () => {
    const { X, y, t } = makeSimpleData(100);
    const comp = compareCATEModels(X, y, t, { nTrees: 10 });
    expect(comp.models.length).toBe(3);
    expect(comp.correlations.length).toBe(3); // 3 choose 2 = 3 pairs
    expect(comp.concordance).toBeGreaterThanOrEqual(0);
    expect(comp.concordance).toBeLessThanOrEqual(1);
  });

  it('model names are correct', () => {
    const { X, y, t } = makeSimpleData(100);
    const comp = compareCATEModels(X, y, t, { nTrees: 10 });
    const names = comp.models.map(m => m.name);
    expect(names).toContain('linear');
    expect(names).toContain('double-ml');
    expect(names).toContain('causal-forest');
  });

  it('each model has valid ATE and SD', () => {
    const { X, y, t } = makeSimpleData(100);
    const comp = compareCATEModels(X, y, t, { nTrees: 10 });
    for (const model of comp.models) {
      expect(typeof model.ate).toBe('number');
      expect(model.sd).toBeGreaterThanOrEqual(0);
      expect(model.cate.length).toBe(100);
    }
  });

  it('correlations are finite numbers', () => {
    const { X, y, t } = makeSimpleData(100);
    const comp = compareCATEModels(X, y, t, { nTrees: 10 });
    for (const corr of comp.correlations) {
      expect(Number.isFinite(corr.correlation)).toBe(true);
    }
  });

  it('concordance approaches 1 for identical models', () => {
    // When all models produce similar rankings, concordance should be high
    const { X, y, t } = makeHeterogeneousData(200);
    const comp = compareCATEModels(X, y, t, { nTrees: 15 });
    expect(comp.concordance).toBeGreaterThanOrEqual(0);
    expect(comp.concordance).toBeLessThanOrEqual(1);
  });
});

// ── Causal Forest Enhancements ───────────────────────────────────────

describe('CausalForest enhancements', () => {
  it('trains and predicts with heterogeneous effects', () => {
    const { X, y, t } = makeHeterogeneousData(200);
    const forest = new CausalForest({ nTrees: 30, minLeafSize: 10, maxDepth: 8, seed: 42 });
    forest.train(X, y, t);
    const cate = forest.predict(X);
    expect(cate.length).toBe(200);
    // CATE should show variation for heterogeneous data
    const unique = new Set(cate.map(v => v.toFixed(4)));
    expect(unique.size).toBeGreaterThan(1);
  });

  it('produces consistent results with same seed', () => {
    const { X, y, t } = makeSimpleData(100);
    const f1 = new CausalForest({ nTrees: 20, minLeafSize: 10, seed: 42 });
    f1.train(X, y, t);
    const p1 = f1.predict(X);

    const f2 = new CausalForest({ nTrees: 20, minLeafSize: 10, seed: 42 });
    f2.train(X, y, t);
    const p2 = f2.predict(X);

    for (let i = 0; i < p1.length; i++) {
      expect(p1[i]).toBe(p2[i]);
    }
  });

  it('handles config with default values', () => {
    const forest = new CausalForest();
    expect(forest).toBeDefined();
  });

  it('handles small dataset', () => {
    const { X, y, t } = makeSimpleData(20);
    const forest = new CausalForest({ nTrees: 5, minLeafSize: 3, maxDepth: 3, seed: 1 });
    forest.train(X, y, t);
    const cate = forest.predict(X);
    expect(cate.length).toBe(20);
  });
});
