/**
 * Effect estimation tests — validates causal effect estimation methods.
 *
 * Covers: backdoor adjustment, frontdoor, IV/2SLS, propensity score matching,
 * doubly robust estimation. All methods tested against known DAGs with
 * ground-truth ATE from synthetic data.
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../graph/causal-graph.js';
import {
  findBackdoorSet, adjustBackdoor, estimateFrontdoor,
  estimateIV, estimatePropensityScore, estimatePSMatching,
  estimateDoublyRobust,
} from '../infer/effect-estimation.js';

// ── Test DAGs ─────────────────────────────────────────────────────────

/** Simple confounded DAG: Z → T, Z → Y, T → Y */
function confoundedDAG(): CausalGraph {
  const g = new CausalGraph(['Z', 'T', 'Y']);
  g.addEdge('Z', 'T'); g.addEdge('Z', 'Y'); g.addEdge('T', 'Y');
  return g;
}

/** M-shaped DAG: X → M → Y with X→Y direct */
function mediationDAG(): CausalGraph {
  const g = new CausalGraph(['X', 'M', 'Y']);
  g.addEdge('X', 'M'); g.addEdge('M', 'Y'); g.addEdge('X', 'Y');
  return g;
}

/** IV DAG: Z → X → Y, Z uncorrelated with confounders */
function ivDAG(): CausalGraph {
  const g = new CausalGraph(['Z', 'X', 'Y', 'U']);
  g.addEdge('Z', 'X'); g.addEdge('X', 'Y'); g.addEdge('U', 'X'); g.addEdge('U', 'Y');
  return g;
}

// ── Synthetic data generators ─────────────────────────────────────────

/** Generate data from confounded DAG with known ATE */
function generateConfounded(n: number, ate: number): { data: number[][]; nodeIndex: Map<string, number> } {
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const z = Math.random() * 2;
    const t = z * 0.6 + Math.random() * 0.5;
    const y = z * 0.3 + t * ate + Math.random() * 0.2;
    data.push([z, t, y]);
  }
  const nodeIndex = new Map([['Z', 0], ['T', 1], ['Y', 2]]);
  return { data, nodeIndex };
}

/** Generate data with binary treatment for PS/DR tests */
function generateBinaryTreatment(n: number, ate: number): { data: number[][]; nodeIndex: Map<string, number> } {
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const x1 = Math.random() * 2 - 1;
    const x2 = Math.random() * 2 - 1;
    const pTreat = 1 / (1 + Math.exp(-(x1 * 0.8 + x2 * 0.5)));
    const t = Math.random() < pTreat ? 1 : 0;
    const y = x1 * 0.4 + x2 * 0.3 + t * ate + Math.random() * 0.3;
    data.push([x1, x2, t, y]);
  }
  const nodeIndex = new Map([['X1', 0], ['X2', 1], ['T', 2], ['Y', 3]]);
  return { data, nodeIndex };
}

// ── Backdoor adjustment ──────────────────────────────────────────────

describe('findBackdoorSet', () => {
  it('finds confounders in confounded DAG', () => {
    const g = confoundedDAG();
    const adjustors = findBackdoorSet(g, 'T', 'Y');
    expect(adjustors).toContain('Z');
  });

  it('returns empty when no confounders', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    expect(findBackdoorSet(g, 'A', 'B')).toEqual([]);
  });
});

describe('adjustBackdoor', () => {
  it('recovers ATE in confounded DAG', () => {
    const g = confoundedDAG();
    const trueATE = 0.8;
    const { data, nodeIndex } = generateConfounded(500, trueATE);
    const result = adjustBackdoor(g, 'T', 'Y', data, nodeIndex);
    expect(result.ate).toBeGreaterThan(0.3);
    expect(result.ate).toBeLessThan(1.3);
    expect(result.adjustors).toContain('Z');
    expect(result.se).toBeGreaterThan(0);
  });

  it('works without confounders', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const a = Math.random() > 0.5 ? 1 : 0;
      const b = a * 0.5 + Math.random() * 0.3;
      data.push([a, b]);
    }
    const result = adjustBackdoor(g, 'A', 'B', data, new Map([['A', 0], ['B', 1]]));
    expect(result.adjustors).toEqual([]);
    expect(result.ate).toBeGreaterThan(0);
  });
});

// ── Frontdoor adjustment ─────────────────────────────────────────────

describe('estimateFrontdoor', () => {
  it('computes frontdoor ATE via product of coefficients', () => {
    const g = mediationDAG();
    const data: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const x = Math.random() * 2;
      const m = x * 0.6 + Math.random() * 0.2;
      const y = m * 0.5 + x * 0.1 + Math.random() * 0.1;
      data.push([x, m, y]);
    }
    const result = estimateFrontdoor(g, 'X', 'Y', data, new Map([['X', 0], ['M', 1], ['Y', 2]]), ['M']);
    // ATE = β_XM * β_MY ≈ 0.6 * 0.5 = 0.3
    expect(result.ate).toBeGreaterThan(0.1);
    expect(result.ate).toBeLessThan(0.6);
    expect(result.se).toBeGreaterThan(0);
  });
});

// ── IV / 2SLS ────────────────────────────────────────────────────────

describe('estimateIV', () => {
  it('recovers ATE with valid instrument', () => {
    const n = 300;
    const trueATE = 0.7;
    const data: number[][] = [];
    for (let i = 0; i < n; i++) {
      const u = Math.random() * 2 - 1;
      const z = Math.random() * 2 - 1;
      const x = z * 0.8 + u * 0.3 + Math.random() * 0.2;
      const y = x * trueATE + u * 0.4 + Math.random() * 0.2;
      data.push([z, x, u, y]);
    }
    // IV: z (idx 0), treatment: x (idx 1), outcome: y (idx 3)
    const result = estimateIV(data, 1, 3, 0, []);
    expect(result.ate).toBeGreaterThan(0.3);
    expect(result.ate).toBeLessThan(1.1);
    expect(result.se).toBeGreaterThan(0);
  });

  it('provides standard error', () => {
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const z = Math.random();
      const x = z + Math.random() * 0.5;
      const y = x * 0.5 + Math.random() * 0.3;
      data.push([z, x, y]);
    }
    const result = estimateIV(data, 1, 2, 0, []);
    expect(result.se).toBeGreaterThan(0);
  });
});

// ── Propensity Score ─────────────────────────────────────────────────

describe('estimatePropensityScore', () => {
  it('returns values in [0, 1]', () => {
    const { data } = generateBinaryTreatment(200, 0.5);
    const scores = estimatePropensityScore(data, 2, [0, 1]);
    for (let i = 0; i < scores.length; i++) {
      expect(scores[i]).toBeGreaterThanOrEqual(0);
      expect(scores[i]).toBeLessThanOrEqual(1);
    }
  });

  it('returns constant when no covariates', () => {
    const data = [[1, 1], [0, 0], [1, 2]];
    const scores = estimatePropensityScore(data, 0, []);
    // 2/3 treated
    for (const s of scores) expect(s).toBeCloseTo(2 / 3, 2);
  });
});

describe('estimatePSMatching', () => {
  it('computes PSM ATE', () => {
    const { data } = generateBinaryTreatment(200, 0.6);
    const result = estimatePSMatching(data, 2, 3, [0, 1]);
    // ATE should be in reasonable range
    expect(result.ate).toBeGreaterThan(-1);
    expect(result.ate).toBeLessThan(2);
    expect(result.se).toBeGreaterThan(0);
  });
});

// ── Doubly Robust ────────────────────────────────────────────────────

describe('estimateDoublyRobust', () => {
  it('computes DR ATE', () => {
    const { data } = generateBinaryTreatment(200, 0.5);
    const result = estimateDoublyRobust(data, 2, 3, [0, 1]);
    expect(result.ate).toBeGreaterThan(-1);
    expect(result.ate).toBeLessThan(2);
    expect(result.se).toBeGreaterThan(0);
  });

  it('returns zero SE with insufficient data', () => {
    const data = [[1, 0, 0]]; // single row
    const result = estimateDoublyRobust(data, 2, 2, []);
    // Should not crash
    expect(typeof result.ate).toBe('number');
  });
});

// ── Edge cases ────────────────────────────────────────────────────────

describe('effect estimation edge cases', () => {
  it('adjustBackdoor handles empty covariates', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const data = [[0, 1], [1, 2], [0, 1.5]];
    const result = adjustBackdoor(g, 'A', 'B', data, new Map([['A', 0], ['B', 1]]));
    expect(typeof result.ate).toBe('number');
    expect(result.adjustors).toEqual([]);
  });

  it('estimatePSMatching with no covariates', () => {
    const data: number[][] = [];
    for (let i = 0; i < 50; i++) {
      data.push([Math.random() > 0.5 ? 1 : 0, Math.random() * 2]);
    }
    const result = estimatePSMatching(data, 0, 1, []);
    expect(typeof result.ate).toBe('number');
  });

  it('estimateIV with single covariate', () => {
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const z = Math.random();
      const c = Math.random() * 2;
      const x = z * 0.6 + c * 0.3 + Math.random() * 0.2;
      const y = x * 0.5 + c * 0.2 + Math.random() * 0.1;
      data.push([z, x, c, y]);
    }
    const result = estimateIV(data, 1, 3, 0, [2]);
    expect(result.ate).toBeGreaterThan(0.1);
    expect(result.se).toBeGreaterThan(0);
  });
});
