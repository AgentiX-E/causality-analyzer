import { describe, it, expect } from 'vitest';
import { bootstrapATE } from '../infer/bootstrap-ci.js';

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
});
