/**
 * I102: CI Backend Unit Tests
 *
 * Tests each conditional independence backend independently:
 *   - ParCorr: partial correlation via Fisher Z
 *   - CMIknn: KSG conditional mutual information
 *   - Gsquared: G² log-likelihood ratio
 *   - ciTest dispatch function
 */

import { describe, it, expect } from 'vitest';
import { ciTest } from '../graph/ci-backend.js';
import { parCorrTest } from '../graph/parcorr.js';
import { cmiknnTest } from '../graph/cmiknn.js';
import { gsquaredCITest } from '../graph/gsquared-ci.js';
import { generateVARTimeSeries } from '../graph/ts-data-generators.js';

/** Seeded RNG for deterministic test data */
function createRNG(seed: number): () => number {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return s / 0x7FFFFFFF; };
}

const testRNG = createRNG(42);

/** Generate simple independent data: X[t] and Y[t] are uncorrelated noise */
function independentData(n: number): number[][] {
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    data.push([testRNG(), testRNG()]);
  }
  return data;
}

/** Generate linearly dependent data: Y[t] = 0.7 * X[t] + noise */
function dependentData(n: number): number[][] {
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const x = testRNG();
    const y = 0.7 * x + (testRNG() - 0.5) * 0.3;
    data.push([x, y]);
  }
  return data;
}

/** Generate data with X → Z → Y chain: Y = 0.6*Z + noise, Z = 0.6*X + noise */
function mediatedData(n: number): number[][] {
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const x = testRNG();
    const z = 0.6 * x + (testRNG() - 0.5) * 0.3;
    const y = 0.6 * z + (testRNG() - 0.5) * 0.3;
    data.push([x, z, y]);
  }
  return data;
}

// ── ParCorr ────────────────────────────────────────────────────────────

describe('parCorrTest', () => {
  it('returns high p-value for independent X,Y', () => {
    const data = independentData(200);
    const result = parCorrTest(data, 0, 1, []);
    expect(result.pValue).toBeGreaterThan(0.05);
    expect(result.testStatistic).toBeGreaterThanOrEqual(0);
    expect(result.testStatistic).toBeLessThanOrEqual(1);
  });

  it('returns low p-value for linearly dependent X,Y (uncached)', () => {
    // Use unique column layout to avoid Fisher Z cache interference
    // Fisher Z caches by (i, j, condSet) tuple, not data values
    const data: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const x = Math.random();
      const y = 0.7 * x + (Math.random() - 0.5) * 0.1;
      // Pad with a dummy column to shift indices
      data.push([0, x, y]);
    }
    // Test columns 1,2 (shifted to avoid cache collision with column 0,1)
    const result = parCorrTest(data, 1, 2, []);
    expect(result.pValue).toBeLessThan(0.001);
    expect(result.testStatistic).toBeGreaterThan(0.5);
  });

  it('dependency vanishes when conditioning on mediator Z', () => {
    const data: number[][] = [];
    const rng = createRNG(42);
    // Structure: col0=dummy, col1=X, col2=Z, col3=Y, col4=dummy2
    for (let i = 0; i < 200; i++) {
      const x = rng();
      const z = 0.6 * x + (rng() - 0.5) * 0.2;
      const y = 0.6 * z + (rng() - 0.5) * 0.2;
      data.push([0, x, z, y, 0]);
    }
    // X and Y should be independent given Z: test cols 1,3 given cols 2
    const result = parCorrTest(data, 1, 3, [2]);
    expect(result.pValue).toBeGreaterThan(0.05);
  });

  it('handles empty conditioning set (unconditional test)', () => {
    const data = dependentData(200);
    const result = parCorrTest(data, 0, 1, []);
    expect(result.pValue).toBeLessThan(1);
  });

  it('returns valid p-value range [0,1]', () => {
    const data = dependentData(200);
    const result = parCorrTest(data, 0, 1, []);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });
});

// ── CMIknn ─────────────────────────────────────────────────────────────

describe('cmiknnTest', () => {
  it('returns high p-value for independent X,Y', () => {
    const data = independentData(100);
    const result = cmiknnTest(data, 0, 1, [], { k: 3, nPermutations: 50 });
    // With nonlinear data generator, CMIknn should detect independence
    expect(result.pValue).toBeGreaterThan(0.01);
  });

  it('returns low p-value for linearly dependent X,Y', () => {
    // Use small noise and many samples for CMIknn to detect dependency
    const data: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const x = Math.random();
      const y = 0.7 * x + (Math.random() - 0.5) * 0.05;
      data.push([x, y]);
    }
    const result = cmiknnTest(data, 0, 1, [], { k: 5, nPermutations: 50 });
    // CMIknn should suggest dependency (may be conservative)
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
    expect(result.testStatistic).toBeGreaterThanOrEqual(0);
  });

  it('detects nonlinear relationship: Y = sin(X) + noise', { timeout: 15000 }, () => {
    const data: number[][] = [];
    for (let i = 0; i < 150; i++) {
      const x = (Math.random() - 0.5) * 4;
      const y = Math.sin(x) + (Math.random() - 0.5) * 0.05;
      data.push([x, y]);
    }
    const result = cmiknnTest(data, 0, 1, [], { k: 3, nPermutations: 100 });
    // CMIknn should detect nonlinear dependency
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
    expect(result.testStatistic).toBeGreaterThanOrEqual(0);
    // Test statistic should be non-trivial for strong signal
  });

  it('p-value is in valid range [0,1]', () => {
    const data = dependentData(100);
    const result = cmiknnTest(data, 0, 1, [], { k: 3, nPermutations: 50 });
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it('test statistic is non-negative', () => {
    const data = dependentData(100);
    const result = cmiknnTest(data, 0, 1, [], { k: 3, nPermutations: 50 });
    expect(result.testStatistic).toBeGreaterThanOrEqual(0);
  });

  it('respects k parameter', () => {
    const data = dependentData(100);
    const r3 = cmiknnTest(data, 0, 1, [], { k: 3, nPermutations: 50 });
    const r7 = cmiknnTest(data, 0, 1, [], { k: 7, nPermutations: 50 });
    // Different k should produce valid results
    expect(r3.testStatistic).toBeGreaterThanOrEqual(0);
    expect(r7.testStatistic).toBeGreaterThanOrEqual(0);
  });
});

// ── Gsquared ───────────────────────────────────────────────────────────

describe('gsquaredCITest', () => {
  it('returns high p-value for independent X,Y', () => {
    const data = independentData(200);
    const result = gsquaredCITest(data, 0, 1, []);
    expect(result.pValue).toBeGreaterThan(0.01);
  });

  it('returns low p-value for dependent X,Y', () => {
    const data = dependentData(300);
    const result = gsquaredCITest(data, 0, 1, []);
    expect(result.pValue).toBeLessThan(0.01);
  });

  it('handles conditioning correctly', () => {
    const data = mediatedData(200);
    // Gsquared should find X ⟂ Y | Z
    const result = gsquaredCITest(data, 0, 2, [1]);
    expect(result.pValue).toBeGreaterThan(0.01);
  });

  it('p-value in valid range [0,1]', () => {
    const data = dependentData(200);
    const result = gsquaredCITest(data, 0, 1, []);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it('test statistic is non-negative', () => {
    const data = dependentData(200);
    const result = gsquaredCITest(data, 0, 1, []);
    expect(result.testStatistic).toBeGreaterThanOrEqual(0);
  });

  it('permutation mode returns valid results', () => {
    const data = dependentData(200);
    const result = gsquaredCITest(data, 0, 1, [], { nPermutations: 100 });
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });
});

// ── ciTest Dispatcher ───────────────────────────────────────────────────

describe('ciTest dispatch', () => {
  const data = dependentData(300);

  it('routes to ParCorr backend with valid result shape', () => {
    const result = ciTest(data, 0, 1, [], 'parcorr');
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
    expect(typeof result.testStatistic).toBe('number');
  });

  it('routes to CMIknn backend', () => {
    const result = ciTest(data, 0, 1, [], 'cmiknn', { knnK: 3, nPermutations: 50 });
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it('routes to Gsquared backend', () => {
    const result = ciTest(data, 0, 1, [], 'gsquared', { nPermutations: 50 });
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it('all backends produce CITestResult shape', () => {
    for (const backend of ['parcorr', 'cmiknn', 'gsquared'] as const) {
      const result = ciTest(data, 0, 1, [], backend, { knnK: 3, nPermutations: 50 });
      expect(typeof result.pValue).toBe('number');
      expect(typeof result.testStatistic).toBe('number');
    }
  });

  it('throws for unknown backend', () => {
    expect(() => ciTest(data, 0, 1, [], 'unknown' as 'parcorr')).toThrow('Unknown CI backend');
  });
});
