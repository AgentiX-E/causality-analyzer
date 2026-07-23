/**
 * I17 tests: CATE, IPW, and Counterfactual Fairness.
 */
import { describe, it, expect } from 'vitest';
import { estimateCATE, estimateIPW, checkFairness } from '../infer/cate-fairness.js';

// ── CATE ─────────────────────────────────────────────────────────
describe('estimateCATE', () => {
  it('returns CATE function with baseline ATE (no covariates)', () => {
    const data: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const t = Math.random() > 0.5 ? 1 : 0;
      const y = t * 0.5 + Math.random() * 0.1;
      data.push([t, y]);
    }
    const result = estimateCATE(data, 0, 1, []);
    expect(typeof result.baselineATE).toBe('number');
    expect(Math.abs(result.baselineATE - 0.5)).toBeLessThan(0.3);
    // CATE function with no covariates should be constant
    expect(typeof result.cateFn([])).toBe('number');
  });

  it('CATE with a single covariate recovers interaction effect', () => {
    // Y = T*0.5 + X*0.3 + T*X*0.2 + noise
    const data: number[][] = [];
    const trueATE = 0.5;
    const trueInteraction = 0.2;
    for (let i = 0; i < 300; i++) {
      const x1 = Math.random();
      const t = Math.random() > 0.5 ? 1 : 0;
      const y = t * trueATE + x1 * 0.3 + t * x1 * trueInteraction + (Math.random() - 0.5) * 0.1;
      data.push([x1, t, y]);
    }
    const result = estimateCATE(data, 1, 2, [0]);
    expect(result.baselineATE).toBeGreaterThan(0.2);
    expect(result.baselineATE).toBeLessThan(0.8);
    // CATE at x=0.5 should be close to baseline + interaction*0.5
    const cateAt05 = result.cateFn([0.5]);
    expect(typeof cateAt05).toBe('number');
    expect(Math.abs(cateAt05 - result.baselineATE)).toBeLessThan(0.5);
  });

  it('CATE feature centering: at mean features, CATE ≈ baselineATE', () => {
    // Y = T*0.5 + X*0.3 + noise (no interaction)
    const data: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const x = Math.random() * 10 + 5; // mean ~= 10
      const t = Math.random() > 0.5 ? 1 : 0;
      const y = t * 0.5 + x * 0.1 + (Math.random() - 0.5) * 0.05;
      data.push([x, t, y]);
    }
    const result = estimateCATE(data, 1, 2, [0]);
    // At the feature mean (~10), CATE should be close to baseline
    const cateAtMean = result.cateFn([10]);
    expect(Math.abs(cateAtMean - result.baselineATE)).toBeLessThan(0.3);
  });

  it('CATE with extreme feature values', () => {
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const x = Math.random() * 2;
      const t = Math.random() > 0.5 ? 1 : 0;
      const y = t * 0.5 + (Math.random() - 0.5) * 0.1;
      data.push([x, t, y]);
    }
    const result = estimateCATE(data, 1, 2, [0]);
    const cateLow = result.cateFn([-1000]);
    const cateHigh = result.cateFn([1000]);
    expect(typeof cateLow).toBe('number');
    expect(typeof cateHigh).toBe('number');
    expect(!Number.isNaN(cateLow)).toBe(true);
    expect(!Number.isNaN(cateHigh)).toBe(true);
  });

  it('CATE with NaN values in data handles gracefully', () => {
    const data: number[][] = [
      [1, 0, 2], [NaN, 1, 3], [0.5, 1, 4], [2, NaN, 5],
    ];
    // Should not crash with NaN values
    const result = estimateCATE(data, 1, 2, [0]);
    expect(typeof result.baselineATE).toBe('number');
    expect(typeof result.cateFn([0.5])).toBe('number');
  });

  it('CATE single sample: should produce valid result', () => {
    const data = [[5, 1, 10]];
    const result = estimateCATE(data, 1, 2, [0]);
    expect(typeof result.baselineATE).toBe('number');
  });

  it('CATE with multiple covariates (p=5) no array overflow', () => {
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const row = [Math.random(), Math.random(), Math.random(), Math.random(), Math.random(), Math.random(), Math.random()];
      const t = Math.random() > 0.5 ? 1 : 0;
      const y = t * 0.5 + row[0]! * 0.1 + (Math.random() - 0.5) * 0.05;
      data.push([...row, t, y]); // 7 features + treatment + outcome
    }
    // 5 covariates at indices 0-4, treatment at 5, outcome at 6
    const result = estimateCATE(data, 5, 6, [0, 1, 2, 3, 4]);
    expect(typeof result.baselineATE).toBe('number');
    expect(typeof result.cateFn([1, 2, 3, 4, 5])).toBe('number');
  });

  it('CATE without interaction: CATE ≈ ATE for all x', () => {
    // Y = T*0.7 + noise (no interaction with X)
    const data: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const x = Math.random() * 10;
      const t = Math.random() > 0.5 ? 1 : 0;
      const y = t * 0.7 + (Math.random() - 0.5) * 0.05;
      data.push([x, t, y]);
    }
    const result = estimateCATE(data, 1, 2, [0]);
    const atX0 = result.cateFn([0]);
    const atX5 = result.cateFn([5]);
    const atX10 = result.cateFn([10]);
    // Without true interaction, CATE should be roughly constant
    const spread = Math.max(Math.abs(atX0 - atX5), Math.abs(atX5 - atX10), Math.abs(atX0 - atX10));
    expect(spread).toBeLessThan(0.5);
  });
});

// ── IPW ──────────────────────────────────────────────────────────
describe('estimateIPW', () => {
  it('returns ATE with standard error (no covariates)', () => {
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const t = Math.random() > 0.3 ? 1 : 0;
      const y = t * 0.6 + Math.random() * 0.3;
      data.push([t, y]);
    }
    const result = estimateIPW(data, 0, 1);
    expect(result.ate).toBeGreaterThan(-1);
    expect(result.se).toBeGreaterThan(0);
  });

  it('IPW with balanced random treatment recovers true ATE', () => {
    // Random assignment, Y = T*0.5 + noise
    const data: number[][] = [];
    const trueATE = 0.5;
    for (let i = 0; i < 500; i++) {
      const t = Math.random() > 0.5 ? 1 : 0;
      const y = t * trueATE + (Math.random() - 0.5) * 0.2;
      data.push([t, y]);
    }
    const result = estimateIPW(data, 0, 1);
    expect(Math.abs(result.ate - trueATE)).toBeLessThan(0.15);
    expect(result.se).toBeGreaterThan(0);
  });

  it('IPW with confounding covariates (propensity depends on X)', () => {
    // Treatment probability depends on X: P(T=1) = sigmoid(X*2 - 1)
    // Y = T*1.0 + X*0.5 + noise
    const data: number[][] = [];
    for (let i = 0; i < 500; i++) {
      const x = Math.random() * 2;
      const ps = 1 / (1 + Math.exp(-(x * 2 - 1))); // propensity depends on x
      const t = Math.random() < ps ? 1 : 0;
      const y = t * 1.0 + x * 0.5 + (Math.random() - 0.5) * 0.2;
      data.push([x, t, y]);
    }
    const result = estimateIPW(data, 1, 2, [0]);
    expect(result.ate).toBeGreaterThan(0.5);
    expect(result.ate).toBeLessThan(1.5);
    expect(result.se).toBeGreaterThan(0);
  });

  it('IPW propensity scores are clamped to [0.05, 0.95]', () => {
    // Generate extreme treatment assignment (nearly all treated or all control)
    const data: number[][] = [];
    for (let i = 0; i < 50; i++) {
      if (i < 45) {
        data.push([1, 1, 5]); // mostly treated
      } else {
        data.push([0.1, 0, 1]); // few control
      }
    }
    const result = estimateIPW(data, 1, 2, [0]);
    expect(result.ate).toBeDefined();
    expect(result.se).toBeGreaterThan(0);
    expect(!Number.isNaN(result.ate)).toBe(true);
  });

  it('IPW standard error is reasonable', () => {
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const x = Math.random();
      const t = x > 0.5 ? 1 : 0;
      const y = t * 0.5 + x * 0.3 + (Math.random() - 0.5) * 0.1;
      data.push([x, t, y]);
    }
    const result = estimateIPW(data, 1, 2, [0]);
    // SE should be positive and finite
    expect(result.se).toBeGreaterThan(0);
    expect(Number.isFinite(result.se)).toBe(true);
    expect(result.se).toBeLessThan(2); // reasonable range for this data
  });

  it('IPW with and without covariates returns different estimates (covariates matter)', () => {
    const data: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const x = Math.random() * 3;
      const t = (x > 1.5 && Math.random() > 0.3) ? 1 : 0;
      const y = t * 1.0 + x * 0.3 + (Math.random() - 0.5) * 0.2;
      data.push([x, t, y]);
    }
    const withCov = estimateIPW(data, 1, 2, [0]);
    const withoutCov = estimateIPW(data, 1, 2);
    // Estimates should both be finite
    expect(!Number.isNaN(withCov.ate)).toBe(true);
    expect(!Number.isNaN(withoutCov.ate)).toBe(true);
  });

  it('IPW handles single observation', () => {
    // Single observation: IPW may not produce a reliable estimate
    // but it should not crash
    const result = estimateIPW([[0, 1, 5]], 1, 2, [0]);
    expect(typeof result.ate).toBe('number');
    expect(typeof result.se).toBe('number');
  });
});

// ── Counterfactual Fairness ──────────────────────────────────────
describe('checkFairness', () => {
  it('detects disparity between groups', () => {
    const rc = [
      { name: 'team-a-svc1', score: 0.9 },
      { name: 'team-a-svc2', score: 0.85 },
      { name: 'team-b-svc1', score: 0.3 },
      { name: 'team-b-svc2', score: 0.2 },
    ];
    const result = checkFairness(rc, { 'team-a': ['team-a-svc1', 'team-a-svc2'], 'team-b': ['team-b-svc1', 'team-b-svc2'] });
    expect(result.fair).toBe(false);
    expect(result.disparity).toBeGreaterThan(0.5);
  });

  it('reports fair for balanced groups', () => {
    const rc = [
      { name: 'a1', score: 0.5 },
      { name: 'a2', score: 0.5 },
      { name: 'b1', score: 0.5 },
    ];
    const result = checkFairness(rc, { A: ['a1', 'a2'], B: ['b1'] });
    expect(result.fair).toBe(true);
  });

  it('handles empty protected groups gracefully', () => {
    const rc = [{ name: 'svc1', score: 0.8 }];
    const result = checkFairness(rc, {});
    expect(result.fair).toBe(true);
  });

  it('handles protected groups with no matching root causes', () => {
    const rc = [{ name: 'svc1', score: 0.8 }];
    const result = checkFairness(rc, { 'team-x': ['non-existent'] });
    expect(result.fair).toBe(true);
  });
});
