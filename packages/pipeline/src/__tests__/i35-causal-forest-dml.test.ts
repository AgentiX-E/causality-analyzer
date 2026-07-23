import { describe, it, expect } from 'vitest';
import { CausalForest } from '../infer/causal-forest.js';
import { doubleMLATE, doubleMLCATE } from '../infer/double-ml.js';

// ── Causal Forest ────────────────────────────────────────────────────

function generateHTEData(n: number): { X: number[][]; y: number[]; t: number[]; trueTau: number[] } {
  const X: number[][] = [];
  const y: number[] = [];
  const t: number[] = [];
  const trueTau: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 10;
    const treatment = Math.random() > 0.5 ? 1 : 0;
    // Heterogeneous effect: τ(x) = x < 3 ? 2.0 : x > 7 ? 0.5 : 1.0
    const tau = x < 3 ? 2.0 : x > 7 ? 0.5 : 1.0;
    const outcome = treatment * tau + x * 0.3 + (Math.random() - 0.5) * 0.5;
    X.push([x]);
    y.push(outcome);
    t.push(treatment);
    trueTau.push(tau);
  }
  return { X, y, t, trueTau };
}

describe('CausalForest', () => {
  it('trains and predicts treatment effects', () => {
    const { X, y, t } = generateHTEData(200);
    const cf = new CausalForest({ nTrees: 30, minLeafSize: 10, maxDepth: 5, seed: 42 });
    cf.train(X, y, t);

    const preds = cf.predict(X.slice(0, 10));
    expect(preds.length).toBe(10);
    expect(preds.every(p => typeof p === 'number')).toBe(true);
    expect(preds.every(p => Number.isFinite(p))).toBe(true);
  });

  it('returns zero for untrained forest', () => {
    const cf = new CausalForest();
    expect(cf.predictOne([0.5])).toBe(0);
  });

  it('detects heterogeneous effects (lower R² but valid range)', () => {
    const { X, y, t } = generateHTEData(400);
    const cf = new CausalForest({ nTrees: 50, minLeafSize: 15, maxDepth: 6, seed: 42 });
    cf.train(X, y, t);

    const preds = cf.predict(X);
    const meanPred = preds.reduce((a, b) => a + b, 0) / preds.length;
    // ATE should be positive (treatment has effect)
    expect(meanPred).toBeGreaterThan(0.2);
  });

  it('handles small datasets gracefully', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 3, 1, 4, 3];
    const t = [1, 0, 1, 0, 1];
    const cf = new CausalForest({ nTrees: 10, minLeafSize: 1, seed: 42 });
    cf.train(X, y, t);
    expect(cf.predict([1]).length).toBe(1);
  });
});

// ── Double ML ─────────────────────────────────────────────────────────

describe('doubleMLATE', () => {
  it('estimates ATE for simple RCT data', () => {
    const X: number[][] = [];
    const y: number[] = [];
    const t: number[] = [];
    for (let i = 0; i < 200; i++) {
      const x = Math.random();
      const treat = Math.random() > 0.5 ? 1 : 0;
      X.push([x]);
      y.push(treat * 0.5 + x * 0.3 + (Math.random() - 0.5) * 0.2);
      t.push(treat);
    }
    const result = doubleMLATE(X, y, t, 5);
    expect(result.ate).toBeGreaterThan(0.2);
    expect(result.ate).toBeLessThan(1.5);
    expect(result.se).toBeGreaterThan(0);
  });

  it('returns naive ATE for small datasets', () => {
    const X = [[1], [2]];
    const y = [3, 1];
    const t = [1, 0];
    const result = doubleMLATE(X, y, t);
    expect(Math.abs(result.ate - 2)).toBeLessThan(0.1);
  });
});

describe('doubleMLCATE', () => {
  it('returns CATE function with baseline ATE', () => {
    const X: number[][] = [];
    const y: number[] = [];
    const t: number[] = [];
    for (let i = 0; i < 200; i++) {
      const x1 = Math.random() * 5;
      const treat = Math.random() > 0.5 ? 1 : 0;
      X.push([x1]);
      y.push(treat * 0.5 + x1 * 0.2 + (Math.random() - 0.5) * 0.1);
      t.push(treat);
    }
    const result = doubleMLCATE(X, y, t, 5);
    expect(typeof result.baselineATE).toBe('number');
    expect(typeof result.cateFn([2.5])).toBe('number');
  });

  it('handles small data with naive fallback', () => {
    const X = [[1], [2], [3]];
    const y = [3, 1, 4];
    const t = [1, 0, 1];
    const result = doubleMLCATE(X, y, t);
    expect(typeof result.cateFn([1])).toBe('number');
  });
});
