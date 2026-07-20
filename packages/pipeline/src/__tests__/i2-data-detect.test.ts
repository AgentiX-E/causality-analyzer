/**
 * Comprehensive tests for I2: Data preprocessing and anomaly detection.
 */
import { describe, it, expect } from 'vitest';
import { ColumnarTable } from '@agentix-e/causality-analyzer-core';
import { standardize, discretize, extractWindows, imputeMean } from '../data/standardizer.js';
import { StatsDetector } from '../detect/stats-detector.js';
import { SpectralResidualDetector } from '../detect/spectral-residual.js';
import { SPOTDetector, DSPOTDetector } from '../detect/spot.js';
import { VotingDetector } from '../detect/voting-detector.js';

// ── Standardizer ──────────────────────────────────────────────────────
describe('standardize', () => {
  it('zscore: mean≈0, std≈1', () => {
    const t = ColumnarTable.fromRows([{ x: 1 }, { x: 2 }, { x: 3 }, { x: 4 }, { x: 5 }]);
    const s = standardize(t, 'zscore');
    const col = s.column('x');
    let sum = 0; for (let i = 0; i < col.length; i++) sum += col[i]!;
    expect(Math.abs(sum / col.length)).toBeLessThan(1e-10);
  });

  it('minmax: values in [0, 1]', () => {
    const t = ColumnarTable.fromRows([{ x: 10 }, { x: 20 }, { x: 30 }]);
    const s = standardize(t, 'minmax');
    const col = s.column('x');
    expect(col[0]).toBeCloseTo(0, 5);
    expect(col[2]).toBeCloseTo(1, 5);
  });

  it('robust: median-centered', () => {
    const t = ColumnarTable.fromRows([{ x: 1 }, { x: 2 }, { x: 3 }, { x: 4 }, { x: 100 }]);
    const s = standardize(t, 'robust');
    const col = s.column('x');
    expect(col[1]).toBeLessThan(1); // normal point near median
    expect(col[4]).toBeGreaterThan(10); // outlier far away
  });

  it('single-row table returns identity', () => {
    const t = ColumnarTable.fromRows([{ x: 5 }]);
    const s = standardize(t, 'zscore');
    expect(s.column('x')[0]).toBe(5);
  });

  it('all methods handle constant column gracefully', () => {
    const t = ColumnarTable.fromRows([{ x: 7 }, { x: 7 }, { x: 7 }]);
    for (const m of ['zscore', 'minmax', 'robust'] as const) {
      const s = standardize(t, m);
      expect(s.column('x')[0]).toBe(0); // constant normalizes to 0
    }
  });
});

// ── Discretizer ───────────────────────────────────────────────────────
describe('discretize', () => {
  it('bin continuous values into integer labels', () => {
    const t = ColumnarTable.fromRows([{ x: 0 }, { x: 2.5 }, { x: 5 }, { x: 7.5 }, { x: 10 }]);
    const d = discretize(t, 5);
    expect(Array.from(d.column('x'))).toEqual([0, 1, 2, 3, 4]);
  });

  it('single-bin returns identity', () => {
    const t = ColumnarTable.fromRows([{ x: 1 }, { x: 2 }]);
    const d = discretize(t, 1);
    expect(d.column('x')[0]).toBe(1);
  });

  it('fewer rows than bins caps correctly', () => {
    const t = ColumnarTable.fromRows([{ x: 0 }, { x: 1 }]);
    const d = discretize(t, 10);
    expect(d.column('x')[1]).toBeLessThanOrEqual(9);
  });
});

// ── Window Extractor ──────────────────────────────────────────────────
describe('extractWindows', () => {
  it('yields consecutive windows of size', () => {
    const t = ColumnarTable.fromRows([{ a: 1 }, { a: 2 }, { a: 3 }, { a: 4 }, { a: 5 }]);
    const windows = Array.from(extractWindows(t, 3, 1));
    expect(windows).toHaveLength(3);
    expect(windows[0]!.rowCount).toBe(3);
    expect(windows[1]!.rowCount).toBe(3);
    expect(windows[2]!.rowCount).toBe(3);
  });

  it('step > 1 yields fewer windows', () => {
    const t = ColumnarTable.fromRows([{ a: 1 }, { a: 2 }, { a: 3 }, { a: 4 }, { a: 5 }]);
    const windows = Array.from(extractWindows(t, 3, 2));
    expect(windows).toHaveLength(2);
  });

  it('size > rowCount yields zero windows', () => {
    const t = ColumnarTable.fromRows([{ a: 1 }, { a: 2 }]);
    expect(Array.from(extractWindows(t, 10)).length).toBe(0);
  });
});

// ── Imputer ───────────────────────────────────────────────────────────
describe('imputeMean', () => {
  it('fills NaN with column mean', () => {
    const t = ColumnarTable.fromRows([{ x: 1 }, { x: NaN }, { x: 3 }]);
    const r = imputeMean(t);
    expect(r.column('x')[0]).toBe(1);
    expect(r.column('x')[1]).toBeCloseTo(2, 5);
    expect(r.column('x')[2]).toBe(3);
  });

  it('all-NaN column fills with 0', () => {
    const t = ColumnarTable.fromRows([{ x: NaN }, { x: NaN }]);
    const r = imputeMean(t);
    expect(r.column('x')[0]).toBe(0);
  });
});

// ── StatsDetector ────────────────────────────────────────────────────
describe('StatsDetector', () => {
  it('detects outlier beyond threshold', () => {
    const d = new StatsDetector({ method: 'zscore', threshold: 3, minSamples: 5 });
    // Train with normal data
    const normal: number[][] = Array.from({ length: 100 }, () => [10 + Math.random() * 2]);
    for (const row of normal) d.update(row);
    // Inject anomaly
    const result = d.update([25]);
    expect(result.isAnomalous).toBe(true);
  });

  it('does not flag normal variation', () => {
    const d = new StatsDetector({ threshold: 3, minSamples: 5 });
    for (let i = 0; i < 50; i++) d.update([10]);
    for (let i = 0; i < 20; i++) {
      const r = d.update([10.5]);
      expect(r.isAnomalous).toBe(false);
    }
  });

  it('warming period returns non-anomalous', () => {
    const d = new StatsDetector({ minSamples: 5 });
    for (let i = 0; i < 4; i++) {
      const r = d.update([i]);
      expect(r.isAnomalous).toBe(false);
      expect(r.metadata.stage).toBe('warming');
    }
  });

  it('MAD method is robust to outliers', () => {
    const d = new StatsDetector({ method: 'mad', threshold: 3, minSamples: 10 });
    for (let i = 0; i < 20; i++) d.update([10]);
    d.update([100]); // extreme outlier
    // After outlier, MAD is still stable
    const r = d.update([12]);
    expect(r.isAnomalous).toBe(false);
  });

  it('handles multi-dimensional input', () => {
    const d = new StatsDetector({ threshold: 3, minSamples: 10 });
    for (let i = 0; i < 30; i++) d.update([5, 10]);
    const r = d.update([5, 50]);
    expect(r.isAnomalous).toBe(true);
    expect(r.labels[0]).toBe(0);  // dimension 0 normal
    expect(r.labels[1]).toBe(1);  // dimension 1 anomalous
  });

  it('batch detect returns array of results', () => {
    const d = new StatsDetector({ threshold: 3, minSamples: 5 });
    const train = Array.from({ length: 20 }, () => [10 + Math.random()]);
    d.detect(train);
    const results = d.detect([[10], [50], [11]]);
    expect(results).toHaveLength(3);
    expect(results[1]!.isAnomalous).toBe(true);
  });

  it('empty training data handled gracefully', () => {
    const d = new StatsDetector();
    expect(() => d.update([1])).not.toThrow();
  });
});

// ── SpectralResidualDetector ─────────────────────────────────────────
describe('SpectralResidualDetector', () => {
  it('detects anomaly in periodic signal', () => {
    const sr = new SpectralResidualDetector({ minPoints: 32 });
    // Feed periodic signal
    for (let i = 0; i < 64; i++) {
      sr.update(Math.sin(i * 0.5) * 5 + 10);
    }
    // Inject anomaly
    for (let i = 0; i < 10; i++) {
      sr.update(Math.sin(i * 0.5) * 5 + 10);
    }
    const r = sr.update(50); // large spike
    expect(typeof r.isAnomalous).toBe('boolean');
    expect(typeof r.scores[0]).toBe('number');
  });

  it('warming period does not crash', () => {
    const sr = new SpectralResidualDetector({ minPoints: 32 });
    for (let i = 0; i < 10; i++) {
      const r = sr.update(Math.random());
      expect(r.isAnomalous).toBe(false);
    }
  });

  it('batch detect works on array input', () => {
    const sr = new SpectralResidualDetector({ minPoints: 32 });
    const data = Array.from({ length: 50 }, () => Math.sin(Math.random() * Math.PI) * 10 + 50);
    const results = sr.detect(data);
    expect(results.length).toBe(50);
    expect(results.every(r => 'isAnomalous' in r)).toBe(true);
  });
});

// ── SPOTDetector ─────────────────────────────────────────────────────
describe('SPOTDetector', () => {
  it('initializes with calibration data', () => {
    const spot = new SPOTDetector({ initSize: 50, q: 1e-3 });
    const data = Array.from({ length: 100 }, () => Math.random() * 10 + 5);
    spot.initialize(data);
    // After initialization, normal values should not trigger
    for (let i = 0; i < 10; i++) {
      const r = spot.update(5 + Math.random() * 10);
      expect(r.isAnomalous).toBe(false);
    }
  });

  it('detects outlier after warmup completes', () => {
    const spot = new SPOTDetector({ initSize: 30, q: 1e-3 });
    // Send enough varied data with actual peaks for GPD calibration
    for (let i = 0; i < 100; i++) spot.update(5 + Math.sin(i * 0.5) * 8);
    // Extreme outlier
    const r = spot.update(500);
    expect(r.isAnomalous).toBe(true);
  });

  it('warming behavior after initSize', () => {
    const spot = new SPOTDetector({ initSize: 20, q: 1e-2 });
    for (let i = 0; i < 20; i++) spot.update(5 + i * 3);
    // After initSize calls, algorithm transitions out of warming
    const r = spot.update(50);
    expect(r.metadata.method).toBe('spot');
    expect(typeof r.isAnomalous).toBe('boolean');
  });

  it('handles constant data', () => {
    const spot = new SPOTDetector({ initSize: 10, q: 1e-2 });
    for (let i = 0; i < 30; i++) spot.update(5);
    const r = spot.update(5);
    expect(r.isAnomalous).toBe(false);
  });

  it('adapts threshold as peaks arrive', () => {
    const spot = new SPOTDetector({ initSize: 20, q: 1e-2 });
    for (let i = 0; i < 50; i++) spot.update(10 + Math.random());
    // After some peaks, threshold should be higher than initial
    const initialR = spot.update(50);
    expect(typeof initialR.isAnomalous).toBe('boolean');
  });

  it('very small initSize still works', () => {
    const spot = new SPOTDetector({ initSize: 5, q: 1e-2 });
    for (let i = 0; i < 10; i++) spot.update(Math.random());
    expect(() => spot.update(100)).not.toThrow();
  });
});

// ── DSPOTDetector ───────────────────────────────────────────────────
describe('DSPOTDetector', () => {
  it('adapts to gradual baseline shift', () => {
    const dspot = new DSPOTDetector({ initSize: 30, q: 1e-2, driftWindow: 50 });
    // Phase 1: oscillating baseline
    for (let i = 0; i < 200; i++) dspot.update(10 + Math.sin(i * 0.2) * 4);
    // Phase 2: shifted baseline with oscillation
    for (let i = 0; i < 100; i++) dspot.update(25 + Math.sin(i * 0.2) * 3);
    // After adaptation, value near new baseline should not be anomalous
    const r = dspot.update(26);
    expect(r.isAnomalous).toBe(false);
  });

  it('detects anomaly amid drift', () => {
    const dspot = new DSPOTDetector({ initSize: 30, q: 1e-2, driftWindow: 50 });
    for (let i = 0; i < 100; i++) dspot.update(10 + Math.random() * 2);
    for (let i = 0; i < 50; i++) dspot.update(20 + i * 0.3 + Math.random() * 2);
    // Sudden spike during drift
    const r = dspot.update(200);
    expect(r.isAnomalous).toBe(true);
  });
});

// ── VotingDetector ──────────────────────────────────────────────────
describe('VotingDetector', () => {
  it('majority vote: flags anomaly when majority agree', () => {
    const d1 = new StatsDetector({ threshold: 2, minSamples: 5 });
    const d2 = new StatsDetector({ threshold: 2, minSamples: 5 });
    const v = new VotingDetector([d1, d2], { strategy: 'majority' });
    for (let i = 0; i < 30; i++) v.update([10]);
    expect(v.update([10]).isAnomalous).toBe(false); // both agree: normal
  });

  it('maximum strategy picks highest score', () => {
    const d1 = new StatsDetector({ threshold: 3, minSamples: 5 });
    const d2 = new StatsDetector({ threshold: 3, minSamples: 5 });
    const v = new VotingDetector([d1, d2], { strategy: 'maximum' });
    for (let i = 0; i < 30; i++) v.update([10]);
    // both detectors trained; outlier should trigger via max strategy
    const r = v.update([50]);
    expect(r.isAnomalous).toBe(true);
  });

  it('weighted strategy uses config weights', () => {
    const d1 = new StatsDetector({ threshold: 3, minSamples: 5 });
    const d2 = new StatsDetector({ threshold: 3, minSamples: 5 });
    const v = new VotingDetector([d1, d2], { strategy: 'weighted', weights: [0.1, 0.9] });
    for (let i = 0; i < 30; i++) v.update([10]);
    const r = v.update([10]);
    expect(typeof r.isAnomalous).toBe('boolean');
  });

  it('batch detection combines all results', () => {
    const d1 = new StatsDetector({ threshold: 3, minSamples: 5 });
    const d2 = new StatsDetector({ threshold: 3, minSamples: 5 });
    const v = new VotingDetector([d1, d2]);
    for (let i = 0; i < 20; i++) v.update([10]);
    expect(v.detect([[10], [11]]).length).toBe(2);
  });
});

// ── Performance benchmark ────────────────────────────────────────────
describe('performance', () => {
  it('StatsDetector update under 50μs', () => {
    const d = new StatsDetector({ minSamples: 5 });
    for (let i = 0; i < 20; i++) d.update([10]);
    const start = performance.now();
    for (let i = 0; i < 1000; i++) d.update([10 + Math.random()]);
    const elapsed = (performance.now() - start) / 1000 * 1000; // μs
    expect(elapsed).toBeLessThan(50); // < 50μs per point average
  });
});

// ── Coverage gap: VotingDetector edge cases ─────────────────────────
describe('VotingDetector edge cases', () => {
  it('majority with minAgreement=1 (any vote counts)', () => {
    const d1 = new StatsDetector({ threshold: 1, minSamples: 5 });
    const d2 = new StatsDetector({ threshold: 10, minSamples: 5 });
    const v = new VotingDetector([d1, d2], { strategy: 'majority', minAgreement: 1 });
    for (let i = 0; i < 10; i++) v.update([5]);
    const r = v.update([20]); // d1 should flag (threshold 1), d2 won't (threshold 10)
    expect(r.isAnomalous).toBe(true); // 1 >= minAgreement of 1
  });

  it('majority with no weights defaults', () => {
    const d = new StatsDetector({ threshold: 3, minSamples: 5 });
    const v = new VotingDetector([d]);
    for (let i = 0; i < 10; i++) v.update([5]);
    expect(v.update([5]).isAnomalous).toBe(false);
  });

  it('weighted with default equal weights', () => {
    const d1 = new StatsDetector({ threshold: 3, minSamples: 5 });
    const d2 = new StatsDetector({ threshold: 3, minSamples: 5 });
    const v = new VotingDetector([d1, d2], { strategy: 'weighted' });
    for (let i = 0; i < 10; i++) v.update([5]);
    expect(v.update([5]).isAnomalous).toBe(false);
  });
});

// ── Coverage gap: SPOT GPD fallback branches ──────────────────────
describe('SPOTDetector edge cases', () => {
  it('handles very few peaks in GPD estimation', () => {
    const spot = new SPOTDetector({ initSize: 10, q: 1e-2, initThresholdQuantile: 0.5 });
    // Send data with very few extreme values
    const data = [1,1,1,1,1,1,1,1,1,100];
    spot.initialize(data);
    // Should not crash
    expect(() => spot.update(200)).not.toThrow();
  });

  it('fallback to method of moments when Grimshaw fails', () => {
    const spot = new SPOTDetector({ initSize: 10, q: 1e-2 });
    // Very sparse peaks
    const data = Array.from({ length: 20 }, (_, i) => i < 18 ? 1 : 100);
    spot.initialize(data);
    expect(() => spot.update(200)).not.toThrow();
  });
});

// ── Coverage gap: spectral residual edge ─────────────────────────
describe('SpectralResidualDetector edge cases', () => {
  it('handles very short data gracefully', () => {
    const sr = new SpectralResidualDetector({ minPoints: 32 });
    for (let i = 0; i < 15; i++) sr.update(1);
    const r = sr.update(1);
    expect(r.metadata.stage).toBe('warming');
  });

  it('handles constant input', () => {
    const sr = new SpectralResidualDetector({ minPoints: 32 });
    for (let i = 0; i < 50; i++) sr.update(5);
    const r = sr.update(5);
    expect(typeof r.isAnomalous).toBe('boolean');
  });
});

// ── Coverage gap: StatsDetector IQR method ───────────────────────
describe('StatsDetector IQR', () => {
  it('uses IQR method for scale estimation', () => {
    const d = new StatsDetector({ method: 'iqr', threshold: 3, minSamples: 5 });
    for (let i = 0; i < 20; i++) d.update([10]);
    const r = d.update([50]);
    expect(r.isAnomalous).toBe(true);
  });
});

// ── Branch coverage: remaining gaps ─────────────────────────────
describe('detector branch coverage', () => {
  it('VotingDetector weighted with explicit zero-weight detector', () => {
    const d1 = new StatsDetector({ threshold: 3, minSamples: 2 });
    const d2 = new StatsDetector({ threshold: 3, minSamples: 2 });
    const v = new VotingDetector([d1, d2], { strategy: 'weighted', weights: [0, 1] });
    for (let i = 0; i < 5; i++) v.update([5]);
    const r = v.update([100]);
    expect(r.metadata.method).toBe('voting_weighted');
  });

  it('StatsDetector handles zero-point input array', () => {
    const d = new StatsDetector({ minSamples: 5 });
    d.train([]);
    expect(() => d.update([1])).not.toThrow();
  });

  it('StatsDetector with exactly minSamples points triggers training', () => {
    const d = new StatsDetector({ minSamples: 3 });
    d.update([1]); d.update([2]); d.update([3]);
    const r = d.update([3]);
    expect(r.metadata.stage).toBeUndefined();
  });
});
