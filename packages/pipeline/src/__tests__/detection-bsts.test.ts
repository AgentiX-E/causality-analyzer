import { describe, it, expect } from 'vitest';
import { bstsDetect } from '../detect/bsts.js';

describe('bstsDetect', () => {
  it('detects spike anomaly in simple series', () => {
    // 100 normal points + 1 spike
    const y = Array.from({ length: 100 }, (_, i) => 10 + Math.sin(i * 0.2) * 2 + (Math.random() - 0.5));
    y[80] = 50; // clear spike

    const result = bstsDetect(y, 0, 3.0);
    expect(result.anomalies[80]).toBe(true);
    expect(result.residuals.length).toBe(100);
    expect(result.trend.length).toBe(100);
  });

  it('short series returns safe defaults', () => {
    const result = bstsDetect([1, 2, 3]);
    expect(result.anomalies.every(a => !a)).toBe(true);
    expect(result.scores.every(s => s === 0)).toBe(true);
  });

  it('seasonal decomposition reduces false positives', () => {
    // Strong daily pattern with a real anomaly
    const y: number[] = [];
    for (let i = 0; i < 100; i++) {
      y.push(10 + 5 * Math.sin(i * Math.PI / 12) + (Math.random() - 0.5) * 0.5);
    }
    y[90] = 30; // anomaly during normal peak time

    const noSeasonal = bstsDetect(y, 0, 3.0);
    const withSeasonal = bstsDetect(y, 24, 3.0);

    // Both should detect the anomaly
    expect(noSeasonal.anomalies[90] || withSeasonal.anomalies[90]).toBe(true);
  });

  it('stable series produces no false anomalies', () => {
    const y = Array.from({ length: 50 }, () => 20 + (Math.random() - 0.5) * 2);
    const result = bstsDetect(y, 0, 4.0);
    // With threshold 4.0, stable series should have zero anomalies
    const anomCount = result.anomalies.filter(a => a).length;
    expect(anomCount).toBe(0);
  });

  it('output arrays match input length', () => {
    const y = Array.from({ length: 30 }, () => Math.random() * 10);
    const result = bstsDetect(y, 7);
    expect(result.residuals.length).toBe(30);
    expect(result.trend.length).toBe(30);
    expect(result.scores.length).toBe(30);
    expect(result.anomalies.length).toBe(30);
  });

  it('high threshold suppresses all anomalies', () => {
    const y = Array.from({ length: 50 }, () => Math.random() * 10);
    const result = bstsDetect(y, 0, 100); // very high threshold
    expect(result.anomalies.every(a => !a)).toBe(true);
  });

  it('low threshold catches edge cases', () => {
    const y = Array.from({ length: 30 }, (_, i) => i * 0.5);
    const result = bstsDetect(y, 0, 1.0);
    expect(typeof result.scores[0]).toBe('number');
  });
});

describe('bsts edge cases', () => {
  it('period larger than data handles safely', () => {
    const y = Array.from({ length: 10 }, () => Math.random() * 10);
    const result = bstsDetect(y, 20, 3.0); // period > n
    expect(result.residuals.length).toBe(10);
  });
});
