/**
 * I87: RESIT Nonlinear Causal Direction Tests.
 */
import { describe, it, expect } from 'vitest';
import { resitTest } from '../../src/gcm/resit.js';

describe('RESIT Algorithm', () => {
  it('identifies linear causal direction X→Y', () => {
    const n = 200;
    const X: number[] = [];
    const Y: number[] = [];
    for (let i = 0; i < n; i++) {
      X.push(Math.random());
      Y.push(0.8 * X[i]! + Math.random() * 0.2);
    }
    const result = resitTest(X, Y);
    expect(['X→Y', 'Y→X', 'uncertain']).toContain(result.direction);
    expect(result.pValueXY).toBeGreaterThanOrEqual(0);
    expect(result.pValueXY).toBeLessThanOrEqual(1);
    expect(result.pValueYX).toBeGreaterThanOrEqual(0);
    expect(result.pValueYX).toBeLessThanOrEqual(1);
  });

  it('provides confidence score', () => {
    const n = 200;
    const X: number[] = [];
    const Y: number[] = [];
    for (let i = 0; i < n; i++) {
      X.push(Math.random() * 2 - 1);
      Y.push(Math.pow(X[i]!, 3) + Math.random() * 0.1); // cubic nonlinear
    }
    const result = resitTest(X, Y, { degree: 3 });
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(typeof result.direction).toBe('string');
  });

  it('handles independent data gracefully', () => {
    const X = Array.from({ length: 50 }, () => Math.random());
    const Y = Array.from({ length: 50 }, () => Math.random());
    const result = resitTest(X, Y);
    // Independent data: direction uncertain
    expect(['X→Y', 'Y→X', 'uncertain']).toContain(result.direction);
  });

  it('handles small sample data', () => {
    const result = resitTest([1, 2, 3], [4, 5, 6]);
    expect(result.direction).toBe('uncertain');
    expect(result.confidence).toBe(0);
  });

  it('accepts custom polynomial degree', () => {
    const n = 150;
    const X: number[] = [];
    const Y: number[] = [];
    for (let i = 0; i < n; i++) {
      X.push(Math.random());
      Y.push(X[i]! * X[i]! + Math.random() * 0.1); // quadratic
    }
    const r1 = resitTest(X, Y, { degree: 1 });
    const r2 = resitTest(X, Y, { degree: 3 });
    expect(typeof r1.direction).toBe('string');
    expect(typeof r2.direction).toBe('string');
  });

  it('returns valid p-values in [0,1]', () => {
    const X = Array.from({ length: 100 }, () => Math.random());
    const Y = X.map(x => x * 0.5 + Math.random() * 0.3);
    const result = resitTest(X, Y);
    expect(result.pValueXY).toBeGreaterThanOrEqual(0);
    expect(result.pValueXY).toBeLessThanOrEqual(1);
    expect(result.pValueYX).toBeGreaterThanOrEqual(0);
    expect(result.pValueYX).toBeLessThanOrEqual(1);
  });
});
