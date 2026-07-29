/**
 * Policy Tree + Federated DP Tests (I8-P3)
 */
import { describe, it, expect } from 'vitest';
import { PolicyTree, PolicyForest, type PolicyEvaluation } from '../infer/policy-learning.js';
import {
  laplaceMechanism,
  gaussianMechanism,
  computeATESensitivity,
  federatedDMLWithDP,
  secureAggregate,
  totalPrivacyCost,
} from '../infer/federated-dp.js';

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
    const cate = x1 > 1 ? 0.8 : 0.2;
    y.push(cate * ti + 0.3 * x1 + (rng() - 0.5) * 0.3);
    t.push(ti);
  }
  return { X, y, t };
}

describe('PolicyTree', () => {
  it('fits and predicts', () => {
    const { X, y, t } = makeData(200);
    const cate = X.map(x => (x[0]! > 1 ? 0.8 : 0.2));
    const tree = new PolicyTree({ maxDepth: 3, minLeafSize: 20, seed: 42 });
    tree.fit(X, cate, y, t);
    expect(tree.isFitted).toBe(true);
    const decisions = tree.predict(X);
    expect(decisions.length).toBe(200);
    expect(decisions.every(d => d === 0 || d === 1)).toBe(true);
  });

  it('evaluate returns policy metrics', () => {
    const { X, y, t } = makeData(200);
    const cate = X.map(x => (x[0]! > 1 ? 0.8 : 0.2));
    const tree = new PolicyTree({ maxDepth: 3, minLeafSize: 20 });
    tree.fit(X, cate, y, t);
    const eval_ = tree.evaluate(X, y, t);
    expect(eval_.treatmentFraction).toBeGreaterThanOrEqual(0);
    expect(eval_.treatmentFraction).toBeLessThanOrEqual(1);
    expect(typeof eval_.policyValue).toBe('number');
  });

  it('has leaves after fitting', () => {
    const { X, y, t } = makeData(200);
    const tree = new PolicyTree({ maxDepth: 3, minLeafSize: 20 });
    tree.fit(X, X.map(x => x[0]!), y, t);
    expect(tree.leafCount).toBeGreaterThanOrEqual(1);
  });
});

describe('PolicyForest', () => {
  it('fits and predicts via majority vote', () => {
    const { X, y, t } = makeData(200);
    const cate = X.map(x => (x[0]! > 1 ? 0.8 : 0.2));
    const forest = new PolicyForest({ nTrees: 30, maxDepth: 3, minLeafSize: 20, seed: 42 });
    forest.fit(X, cate, y, t);
    expect(forest.isFitted).toBe(true);
    const decisions = forest.predict(X);
    expect(decisions.length).toBe(200);
  });

  it('evaluate returns metrics', () => {
    const { X, y, t } = makeData(200);
    const cate = X.map(x => (x[0]! > 1 ? 0.8 : 0.2));
    const forest = new PolicyForest({ nTrees: 20, maxDepth: 3 });
    forest.fit(X, cate, y, t);
    const eval_ = forest.evaluate(X, y, t);
    expect(typeof eval_.policyValue).toBe('number');
    expect(typeof eval_.improvement).toBe('number');
  });
});

// ── Differential Privacy ───────────────────────────────────────────

describe('laplaceMechanism', () => {
  it('adds noise to the value', () => {
    const { privatizedValue, noiseAdded } = laplaceMechanism(0.5, 0.1, 1.0, 42);
    expect(Math.abs(privatizedValue - 0.5)).toBeGreaterThan(0);
    expect(typeof noiseAdded).toBe('number');
  });

  it('is reproducible with same seed', () => {
    const r1 = laplaceMechanism(0.5, 0.1, 1.0, 42);
    const r2 = laplaceMechanism(0.5, 0.1, 1.0, 42);
    expect(r1.privatizedValue).toBe(r2.privatizedValue);
  });

  it('larger epsilon = less noise', () => {
    const loose = laplaceMechanism(0.5, 0.1, 10, 42);
    const tight = laplaceMechanism(0.5, 0.1, 0.1, 42);
    expect(Math.abs(loose.noiseAdded)).toBeLessThan(Math.abs(tight.noiseAdded));
  });
});

describe('gaussianMechanism', () => {
  it('adds Gaussian noise', () => {
    const { privatizedValue } = gaussianMechanism(0.5, 0.1, 1.0, 1e-5, 42);
    expect(typeof privatizedValue).toBe('number');
  });

  it('is reproducible', () => {
    const r1 = gaussianMechanism(0.5, 0.1, 1.0, 1e-5, 42);
    const r2 = gaussianMechanism(0.5, 0.1, 1.0, 1e-5, 42);
    expect(r1.privatizedValue).toBe(r2.privatizedValue);
  });
});

describe('computeATESensitivity', () => {
  it('returns larger value for smaller groups', () => {
    const s1 = computeATESensitivity(1, 10, 10);
    const s2 = computeATESensitivity(1, 5, 100);
    expect(s2).toBeGreaterThan(s1);
  });
});

// ── Federated DML with DP ──────────────────────────────────────────

describe('federatedDMLWithDP', () => {
  it('aggregates across nodes', () => {
    const nodes = [
      { nodeId: 'n1', localATE: 0.5, localSE: 0.1, nTreated: 50, nControl: 50, outcomeMax: 5 },
      { nodeId: 'n2', localATE: 0.55, localSE: 0.12, nTreated: 40, nControl: 60, outcomeMax: 5 },
      { nodeId: 'n3', localATE: 0.48, localSE: 0.11, nTreated: 60, nControl: 40, outcomeMax: 5 },
    ];
    const result = federatedDMLWithDP(nodes, {
      epsilon: 1.0, mechanism: 'laplace', sensitivity: 0.1, minNodes: 3, seed: 42,
    });
    expect(result.nodes.length).toBe(3);
    expect(result.aggregation.nodeCount).toBe(3);
    expect(result.aggregation.totalSamples).toBe(300);
    expect(typeof result.aggregation.aggregateValue).toBe('number');
  });

  it('throws for insufficient nodes', () => {
    expect(() => federatedDMLWithDP([], {
      epsilon: 1, mechanism: 'laplace', sensitivity: 0.1, minNodes: 2, seed: 42,
    })).toThrow('Need at least');
  });

  it('works with Gaussian mechanism', () => {
    const nodes = [
      { nodeId: 'n1', localATE: 0.5, localSE: 0.1, nTreated: 50, nControl: 50, outcomeMax: 5 },
      { nodeId: 'n2', localATE: 0.55, localSE: 0.12, nTreated: 50, nControl: 50, outcomeMax: 5 },
    ];
    const result = federatedDMLWithDP(nodes, {
      epsilon: 1.0, mechanism: 'gaussian', sensitivity: 0.1, minNodes: 2, delta: 1e-5, seed: 42,
    });
    expect(result.nodes.length).toBe(2);
    expect(Number.isFinite(result.aggregation.aggregateValue)).toBe(true);
  });
});

describe('secureAggregate', () => {
  it('computes weighted mean', () => {
    const result = secureAggregate([0.5, 0.7, 0.6], [100, 200, 100]);
    expect(result).toBeCloseTo(0.625, 2);
  });

  it('handles empty inputs', () => {
    expect(secureAggregate([], [])).toBe(0);
  });
});

describe('totalPrivacyCost', () => {
  it('sums epsilon values', () => {
    expect(totalPrivacyCost([1.0, 0.5, 0.5])).toBe(2.0);
  });

  it('returns 0 for empty', () => {
    expect(totalPrivacyCost([])).toBe(0);
  });
});
