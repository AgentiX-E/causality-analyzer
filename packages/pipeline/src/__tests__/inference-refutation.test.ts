/**
 * I76: Refutation base + advanced refutation tests.
 */
import { describe, it, expect } from 'vitest';
import type { RefutationResult } from '../../src/infer/causal-inference.js';
import { runRefutationBatch, summarizeRefutation } from '../../src/infer/refutation-base.js';
import { refuteAddUnobservedCommonCause } from '../../src/infer/refutation-advanced.js';

describe('Refutation Base', () => {
  describe('runRefutationBatch', () => {
    it('returns empty array for empty refuters', () => {
      const result = runRefutationBatch([], [[1, 2]], 0, 1);
      expect(result).toEqual([]);
    });

    it('collects results from multiple refuters', () => {
      const mockResult: RefutationResult = {
        method: 'mock_refuter',
        originalEstimate: 1.5,
        newEstimate: 1.45,
        pValue: 0.8,
        isRobust: true,
      };
      const mockRefuter = {
        method: 'mock_refuter',
        refute: () => mockResult,
      };
      const results = runRefutationBatch([mockRefuter, mockRefuter], [[1, 2, 3]], 0, 1);
      expect(results).toHaveLength(2);
      expect(results[0]!.method).toBe('mock_refuter');
      expect(results[0]!.originalEstimate).toBe(1.5);
      expect(results[0]!.isRobust).toBe(true);
    });

    it('passes treatment and outcome indices', () => {
      const captured: number[][] = [];
      const refuter = {
        method: 'capture',
        refute: (_data: number[][], tIdx: number, oIdx: number) => {
          captured.push([tIdx, oIdx]);
          return { method: 'capture', originalEstimate: 0, newEstimate: 0, pValue: 1, isRobust: true };
        },
      };
      runRefutationBatch([refuter], [[1, 2, 3]], 1, 2);
      expect(captured[0]).toEqual([1, 2]);
    });
  });

  describe('summarizeRefutation', () => {
    it('returns no-refutations message for empty results', () => {
      const summary = summarizeRefutation([]);
      expect(summary).toBe('No refutations performed.');
    });

    it('includes robust status in summary', () => {
      const results = [
        { method: 'test_a', originalEstimate: 1.0, newEstimate: 0.95, pValue: 0.3, isRobust: true },
        { method: 'test_b', originalEstimate: 1.0, newEstimate: 0.5, pValue: 0.01, isRobust: false },
      ];
      const summary = summarizeRefutation(results);
      expect(summary).toContain('ROBUST');
      expect(summary).toContain('SENSITIVE');
      expect(summary).toContain('Passed: 1/2');
    });

    it('formats numbers correctly', () => {
      const results = [
        { method: 'precise', originalEstimate: 2.12345, newEstimate: 2.1, pValue: 0.05, isRobust: true },
      ];
      const summary = summarizeRefutation(results);
      expect(summary).toContain('2.1235');
      expect(summary).toContain('2.1000');
      expect(summary).toContain('0.0500');
    });
  });
});

describe('refuteAddUnobservedCommonCause', () => {
  it('returns valid RefutationResult', () => {
    const data = Array.from({ length: 50 }, () => [Math.random(), Math.random() + Math.random()]);
    const estimateFn = (d: number[][]) => {
      const n = d.length;
      let treated = 0, control = 0;
      let tSum = 0, cSum = 0;
      for (let i = 0; i < n; i++) {
        if ((d[i]![0] ?? 0) > 0.5) { treated++; tSum += d[i]![1]!; }
        else { control++; cSum += d[i]![1]!; }
      }
      const ate = (tSum / Math.max(1, treated)) - (cSum / Math.max(1, control));
      return { ate, se: 0.1 };
    };
    const result = refuteAddUnobservedCommonCause(data, 0, 1, estimateFn);
    expect(result.method).toBe('add_unobserved_common_cause');
    expect(typeof result.originalEstimate).toBe('number');
    expect(typeof result.newEstimate).toBe('number');
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it('accepts custom confound correlation', () => {
    const data = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [1.5, 3.0], [2.5, 5.0]];
    const estimateFn = () => ({ ate: 2.0, se: 0.1 });
    const r1 = refuteAddUnobservedCommonCause(data, 0, 1, estimateFn, { confoundCorrelation: 0.1, seed: 42 });
    const r2 = refuteAddUnobservedCommonCause(data, 0, 1, estimateFn, { confoundCorrelation: 0.8, seed: 42 });
    expect(typeof r1.newEstimate).toBe('number');
    expect(typeof r2.newEstimate).toBe('number');
  });

  it('accepts custom number of simulations', () => {
    const data = [[1.0, 2.0], [2.0, 4.0]];
    const estimateFn = () => ({ ate: 2.0, se: 0.1 });
    const result = refuteAddUnobservedCommonCause(data, 0, 1, estimateFn, { numSimulations: 10, seed: 42 });
    expect(result.pValue).toBeGreaterThanOrEqual(0);
  });
});
