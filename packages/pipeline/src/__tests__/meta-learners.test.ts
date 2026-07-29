/**
 * Meta-Learner Tests (I8-P2)
 *
 * Tests SLearner, TLearner, XLearner for CATE estimation.
 */

import { describe, it, expect } from 'vitest';
import { SLearner, TLearner, XLearner } from '../infer/meta-learners.js';

function makeData(n: number): { X: number[][]; y: number[]; t: number[] } {
  let s = 42;
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

describe('SLearner', () => {
  it('fits and predicts CATE', () => {
    const { X, y, t } = makeData(200);
    const sl = new SLearner();
    sl.fit(X, y, t);
    expect(sl.isFitted).toBe(true);
    const cate = sl.effect(X);
    expect(cate.length).toBe(200);
    expect(Number.isFinite(cate[0])).toBe(true);
  });

  it('throws if not fitted', () => {
    expect(() => new SLearner().effect([[1]])).toThrow();
  });

  it('ate returns scalar', () => {
    const { X, y, t } = makeData(200);
    const sl = new SLearner();
    sl.fit(X, y, t);
    const ate = sl.ate(X, t);
    expect(Number.isFinite(ate)).toBe(true);
  });

  it('works with 2 covariates', () => {
    const X = []; const y = []; const t = [];
    for (let i = 0; i < 150; i++) {
      const ti = Math.random() > 0.5 ? 1 : 0;
      X.push([Math.random(), Math.random()]);
      y.push(0.5 * ti + Math.random() * 0.3);
      t.push(ti);
    }
    const sl = new SLearner();
    sl.fit(X, y, t);
    expect(sl.effect(X).length).toBe(150);
  });
});

describe('TLearner', () => {
  it('fits and predicts CATE', () => {
    const { X, y, t } = makeData(200);
    const tl = new TLearner();
    tl.fit(X, y, t);
    expect(tl.isFitted).toBe(true);
    const cate = tl.effect(X);
    expect(cate.length).toBe(200);
  });

  it('throws if not fitted', () => {
    expect(() => new TLearner().effect([[1]])).toThrow();
  });

  it('ate returns scalar', () => {
    const { X, y, t } = makeData(200);
    const tl = new TLearner();
    tl.fit(X, y, t);
    expect(Number.isFinite(tl.ate(X, t))).toBe(true);
  });
});

describe('XLearner', () => {
  it('fits and predicts CATE', () => {
    const { X, y, t } = makeData(200);
    const xl = new XLearner();
    xl.fit(X, y, t);
    expect(xl.isFitted).toBe(true);
    const cate = xl.effect(X);
    expect(cate.length).toBe(200);
  });

  it('throws if not fitted', () => {
    expect(() => new XLearner().effect([[1]])).toThrow();
  });

  it('ate returns scalar', () => {
    const { X, y, t } = makeData(200);
    const xl = new XLearner();
    xl.fit(X, y, t);
    expect(Number.isFinite(xl.ate(X, t))).toBe(true);
  });

  it('works with default config', () => {
    const { X, y, t } = makeData(200);
    const xl = new XLearner({ propensityMethod: 'mean' });
    xl.fit(X, y, t);
    expect(xl.isFitted).toBe(true);
  });
});
