import { describe, it, expect } from 'vitest';
import { estimateCATE, estimateIPW, checkFairness } from '../infer/cate-fairness.js';

describe('estimateCATE', () => {
  it('returns CATE function with baseline ATE', () => {
    const data: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const x1 = Math.random();
      const t = Math.random() > 0.5 ? 1 : 0;
      const y = t * 0.5 + x1 * 0.3 + t * x1 * 0.2 + Math.random() * 0.1;
      data.push([x1, t, y]);
    }
    const result = estimateCATE(data, 1, 2, [0]);
    expect(typeof result.baselineATE).toBe('number');
    expect(typeof result.cateFn([0.5])).toBe('number');
  });
});

describe('estimateIPW', () => {
  it('returns ATE with standard error', () => {
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
});

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
});
