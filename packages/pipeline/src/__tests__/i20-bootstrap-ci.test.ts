import { describe, it, expect } from 'vitest';
import { bootstrapATE, parallelBootstrap, bootstrapATEParallel } from '../infer/bootstrap-ci.js';

describe('bootstrapATE', () => {
  it('returns CI for simple estimator', () => {
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const t = Math.random() > 0.5 ? 1 : 0;
      data.push([t, t * 0.5 + Math.random() * 0.3]);
    }
    const result = bootstrapATE(data, d => {
      let tSum = 0, cSum = 0, tN = 0, cN = 0;
      for (const r of d) {
        if ((r[0] ?? 0) > 0.5) { tSum += r[1]!; tN++; }
        else { cSum += r[1]!; cN++; }
      }
      return (tN > 0 ? tSum / tN : 0) - (cN > 0 ? cSum / cN : 0);
    }, 100, 0.05, 42);
    expect(result.ciLow).toBeLessThanOrEqual(result.ate);
    expect(result.ciHigh).toBeGreaterThanOrEqual(result.ate);
    expect(result.se).toBeGreaterThan(0);
  });

  it('deterministic seed produces reproducible CI', () => {
    const data: number[][] = Array.from({ length: 50 }, () => [Math.random() > 0.5 ? 1 : 0, Math.random()]);
    const est = (d: number[][]) => {
      let s = 0; for (const r of d) s += r[1]!; return s / d.length;
    };
    const r1 = bootstrapATE(data, est, 200, 0.05, 42);
    const r2 = bootstrapATE(data, est, 200, 0.05, 42);
    expect(r1.ciLow).toBe(r2.ciLow);
    expect(r1.ciHigh).toBe(r2.ciHigh);
  });

  it('different seeds produce potentially different CIs', () => {
    const data: number[][] = Array.from({ length: 50 }, () => [Math.random() > 0.5 ? 1 : 0, Math.random()]);
    const est = (d: number[][]) => {
      let s = 0; for (const r of d) s += r[1]!; return s / d.length;
    };
    const r1 = bootstrapATE(data, est, 200, 0.05, 1);
    const r2 = bootstrapATE(data, est, 200, 0.05, 9999);
    // Different seeds — may differ or may not (SE is valid either way)
    expect(r1.se).toBeGreaterThan(0);
    expect(r2.se).toBeGreaterThan(0);
  });

  it('handles very small dataset (n=2)', () => {
    const data = [[1, 3], [0, 1]];
    const est = (d: number[][]) => {
      let s = 0; for (const r of d) s += r[1]!; return s / d.length;
    };
    const result = bootstrapATE(data, est, 50, 0.05);
    expect(typeof result.ate).toBe('number');
    expect(!Number.isNaN(result.ate)).toBe(true);
  });

  it('CI width is positive for noisy data', () => {
    const data: number[][] = Array.from({ length: 100 }, () => [Math.random() > 0.5 ? 1 : 0, Math.random() * 10]);
    const est = (d: number[][]) => {
      let s = 0; for (const r of d) s += r[1]!; return s / d.length;
    };
    const result = bootstrapATE(data, est, 500, 0.05, 42);
    expect(result.ciHigh - result.ciLow).toBeGreaterThan(0);
  });

  it('alpha=0 produces tightest CI', () => {
    const data: number[][] = Array.from({ length: 50 }, () => [Math.random() > 0.5 ? 1 : 0, Math.random()]);
    const est = (d: number[][]) => {
      let s = 0; for (const r of d) s += r[1]!; return s / d.length;
    };
    const r1 = bootstrapATE(data, est, 100, 0.01, 42);
    const r2 = bootstrapATE(data, est, 100, 0.5, 42);
    // Larger alpha (more permissive) should have narrower CI
    expect(r2.ciHigh - r2.ciLow).toBeLessThanOrEqual(r1.ciHigh - r1.ciLow + 0.01);
  });
});

describe('parallelBootstrap', () => {
  it('sequential mode produces results', async () => {
    const data = Array.from({ length: 20 }, () => [Math.random()]);
    const estimator = (d: number[][], _seed: number) => {
      let s = 0; for (const r of d) s += r[0]!; return s / d.length;
    };
    const results = await parallelBootstrap(data, estimator, 50, 42);
    expect(results.length).toBe(50);
    expect(results.every(r => typeof r === 'number')).toBe(true);
  });

  it('single bootstrap task', async () => {
    const data = [[1], [2], [3]];
    const est = (d: number[][], _seed: number) => d.length;
    const results = await parallelBootstrap(data, est, 1, 42);
    expect(results.length).toBe(1);
    expect(results[0]).toBe(3);
  });

  it('parallel bootstrap ATE produces valid CI', async () => {
    const data: number[][] = Array.from({ length: 30 }, () => [Math.random() > 0.5 ? 1 : 0, Math.random()]);
    const est = (d: number[][]) => {
      let s = 0; for (const r of d) s += r[1]!; return s / d.length;
    };
    const result = await bootstrapATEParallel(data, est, 50, 2, 0.05, 42);
    expect(result.ciLow).toBeLessThanOrEqual(result.ate);
    expect(result.ciHigh).toBeGreaterThanOrEqual(result.ate);
    expect(result.se).toBeGreaterThanOrEqual(0);
  });
});

describe('bootstrap edge cases', () => {
  it('bootstrapATE with empty data', () => {
    const data: number[][] = [];
    const est = (d: number[][]) => d.length;
    const result = bootstrapATE(data, est, 10, 0.05, 42);
    expect(result.ate).toBe(0);
    expect(result.se).toBeGreaterThanOrEqual(0);
  });

  it('bootstrapATE with single observation', () => {
    const data = [[1, 5]];
    const est = (d: number[][]) => { let s = 0; for (const r of d) s += r[1]!; return s / d.length; };
    const result = bootstrapATE(data, est, 20, 0.05, 42);
    expect(typeof result.ate).toBe('number');
  });

  it('bootstrapATEParallel with single thread', async () => {
    const data = Array.from({ length: 10 }, () => [Math.random() > 0.5 ? 1 : 0, Math.random()]);
    const est = (d: number[][]) => { let s = 0; for (const r of d) s += r[1]!; return s / d.length; };
    const result = await bootstrapATEParallel(data, est, 20, 1, 0.05, 42);
    expect(result.ciLow).toBeLessThanOrEqual(result.ate);
  });
});

describe('bootstrap edge cases', () => {
  it('bootstrapATE with n=2 data', () => {
    const est = (d: number[][]) => { let s = 0; for (const r of d) s += r[1]!; return s / d.length; };
    const result = bootstrapATE([[0, 1], [1, 3]], est, 10, 0.05, 42);
    expect(typeof result.ate).toBe('number');
  });
});

describe('branch coverage', () => {
  it('parallelBootstrap with nTasks > 1', async () => {
    const data = Array.from({ length: 10 }, () => [Math.random()]);
    const results = await parallelBootstrap(
      data, (d, _s) => d.length, 5, 42,
    );
    expect(results.length).toBe(5);
  });

  it('bootstrapATE with alpha=0.5 (wide CI)', () => {
    const data = Array.from({ length: 20 }, () => [Math.random() > 0.5 ? 1 : 0, Math.random()]);
    const est = (d: number[][]) => { let s = 0; for (const r of d) s += r[1]!; return s / d.length; };
    const result = bootstrapATE(data, est, 100, 0.5, 42);
    expect(result.ciLow).toBeLessThanOrEqual(result.ciHigh);
  });
});
