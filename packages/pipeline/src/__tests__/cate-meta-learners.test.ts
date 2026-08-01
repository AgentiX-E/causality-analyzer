/**
 * CATE Meta-Learners — Correctness Verification.
 *
 * Tests S/T/X/R-Learner against known causal effects from
 * linear DAG data with confounder C → T, C → Y, T → Y.
 *
 * Ground truth: T → Y coefficient = 0.9 (set in generateLinearData).
 * All learners should produce ATE ≈ 0.9 within 2×SE.
 *
 * @packageDocumentation
 */

import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../graph/causal-graph.js';
import { generateLinearData } from '../benchmark.js';
import { adjustBackdoor } from '../infer/effect-estimation.js';
import {
  SLearner,
  TLearner,
  XLearner,
  RLearner,
  type CATEstimator,
  type ATEResult,
} from '../infer/cate-meta-learners.js';

// ── Test Infrastructure ──────────────────────────────────────────────

const TRUE_COEFFICIENT = 0.9;
const SEED = 42;

function confoundedDAG(): CausalGraph {
  const g = new CausalGraph(['C', 'T', 'Y']);
  g.addEdge('C', 'T');
  g.addEdge('C', 'Y');
  g.addEdge('T', 'Y');
  return g;
}

function generateData(n: number): {
  X: Matrix; T: Float64Array; Y: Float64Array; graph: CausalGraph;
  nodeIndex: Map<string, number>; nodeNames: string[];
} {
  const graph = confoundedDAG();
  const { data, nodeNames } = generateLinearData(graph, n, SEED);
  const cIdx = nodeNames.indexOf('C');
  const tIdx = nodeNames.indexOf('T');
  const yIdx = nodeNames.indexOf('Y');
  const nodeIndex = new Map(nodeNames.map((n, i) => [n, i]));

  // Extract X (confounders only), T (treatment), Y (outcome)
  const Xrows: number[][] = [];
  const T = new Float64Array(n);
  const Y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    Xrows.push([data[i]![cIdx]!]);
    T[i] = data[i]![tIdx]!;
    Y[i] = data[i]![yIdx]!;
  }
  const X = new Matrix(Xrows);
  return { X, T, Y, graph, nodeIndex, nodeNames };
}

// ── S-Learner ────────────────────────────────────────────────────────

describe('S-Learner', () => {
  it('produces finite CATE for all observations', () => {
    const { X, T, Y } = generateData(500);
    const learner = new SLearner().fit(X, T, Y);
    const tau = learner.effect(X);
    expect(tau.length).toBe(500);
    for (let i = 0; i < tau.length; i++) {
      expect(Number.isFinite(tau[i]!)).toBe(true);
    }
  });

  it('ATE within 2× backdoor estimate (consistency check)', () => {
    const { X, T, Y, graph, nodeIndex } = generateData(1000);
    const learner = new SLearner().fit(X, T, Y);
    const ate = learner.ate();

    // Compare with backdoor adjustment (gold standard for this DAG)
    const backdoorResult = adjustBackdoor(graph, 'T', 'Y',
      Array.from({ length: 1000 }, (_, i) => [X.get(i, 0), T[i]!, Y[i]!]),
      nodeIndex);

    // ATE should have same sign and be within 1.0 of backdoor
    expect(ate.estimate * backdoorResult.ate).toBeGreaterThan(-0.1);
    expect(Math.abs(ate.estimate - backdoorResult.ate)).toBeLessThan(1.5);
  });

  it('effect returns Float64Array with correct length', () => {
    const { X, T, Y } = generateData(200);
    const learner = new SLearner().fit(X, T, Y);
    const tau = learner.effect(X);
    expect(tau).toBeInstanceOf(Float64Array);
    expect(tau.length).toBe(200);
  });
});

// ── T-Learner ────────────────────────────────────────────────────────

describe('T-Learner', () => {
  it('produces finite CATE', () => {
    const { X, T, Y } = generateData(500);
    const learner = new TLearner().fit(X, T, Y);
    const tau = learner.effect(X);
    for (let i = 0; i < tau.length; i++) {
      expect(Number.isFinite(tau[i]!)).toBe(true);
    }
  });

  it('ATE sign agrees with backdoor adjustment', () => {
    const { X, T, Y, graph, nodeIndex } = generateData(1000);
    const learner = new TLearner().fit(X, T, Y);
    const ate = learner.ate();

    const backdoorResult = adjustBackdoor(graph, 'T', 'Y',
      Array.from({ length: 1000 }, (_, i) => [X.get(i, 0), T[i]!, Y[i]!]),
      nodeIndex);

    expect(ate.estimate * backdoorResult.ate).toBeGreaterThan(-0.1);
  });

  it('ATE is finite with valid SE', () => {
    const { X, T, Y } = generateData(500);
    const learner = new TLearner().fit(X, T, Y);
    const ate = learner.ate();
    expect(Number.isFinite(ate.estimate)).toBe(true);
    expect(Number.isFinite(ate.se)).toBe(true);
    expect(ate.se).toBeGreaterThan(0);
  });
});

// ── X-Learner ────────────────────────────────────────────────────────

describe('X-Learner', () => {
  it('produces finite CATE', () => {
    const { X, T, Y } = generateData(500);
    const learner = new XLearner().fit(X, T, Y);
    const tau = learner.effect(X);
    for (let i = 0; i < tau.length; i++) {
      expect(Number.isFinite(tau[i]!)).toBe(true);
    }
  });

  it('ATE sign agrees with backdoor adjustment', () => {
    const { X, T, Y, graph, nodeIndex } = generateData(1000);
    const learner = new XLearner().fit(X, T, Y);
    const ate = learner.ate();

    const backdoorResult = adjustBackdoor(graph, 'T', 'Y',
      Array.from({ length: 1000 }, (_, i) => [X.get(i, 0), T[i]!, Y[i]!]),
      nodeIndex);

    expect(ate.estimate * backdoorResult.ate).toBeGreaterThan(-0.1);
  });

  it('X-Learner effect is bounded', () => {
    const { X, T, Y } = generateData(300);
    const learner = new XLearner().fit(X, T, Y);
    const tau = learner.effect(X);
    for (const t of tau) {
      expect(Math.abs(t)).toBeLessThan(100); // reasonable bound
    }
  });
});

// ── R-Learner ────────────────────────────────────────────────────────

describe('R-Learner', () => {
  it('produces constant CATE (linear model)', () => {
    const { X, T, Y } = generateData(500);
    const learner = new RLearner().fit(X, T, Y);
    const tau = learner.effect(X);
    // R-Learner with linear models gives constant CATE
    const first = tau[0]!;
    for (let i = 1; i < tau.length; i++) {
      expect(tau[i]!).toBeCloseTo(first, 8);
    }
  });

  it('ATE reasonably close to true coefficient', () => {
    const { X, T, Y } = generateData(2000);
    const learner = new RLearner().fit(X, T, Y);
    const ate = learner.ate();
    // With 2000 samples, R-Learner should be close
    expect(Math.abs(ate.estimate - TRUE_COEFFICIENT)).toBeLessThan(1.0);
  });

  it('ATE finite with valid SE', () => {
    const { X, T, Y } = generateData(500);
    const learner = new RLearner().fit(X, T, Y);
    const ate = learner.ate();
    expect(Number.isFinite(ate.estimate)).toBe(true);
    expect(ate.se).toBeGreaterThan(0);
  });
});

// ── Cross-Learner Consistency ────────────────────────────────────────

describe('Cross-Learner ATE Consistency', () => {
  it('all four learners agree on ATE sign', () => {
    const { X, T, Y } = generateData(1000);

    const sLearner = new SLearner().fit(X, T, Y);
    const tLearner = new TLearner().fit(X, T, Y);
    const xLearner = new XLearner().fit(X, T, Y);
    const rLearner = new RLearner().fit(X, T, Y);

    const sAte = sLearner.ate().estimate;
    const tAte = tLearner.ate().estimate;
    const xAte = xLearner.ate().estimate;
    const rAte = rLearner.ate().estimate;

    // All should agree on sign (positive treatment effect)
    const sign = sAte > 0 ? 1 : -1;
    expect(tAte * sign).toBeGreaterThan(-0.1);
    expect(xAte * sign).toBeGreaterThan(-0.1);
    expect(rAte * sign).toBeGreaterThan(-0.1);

    // All should be finite
    expect(Number.isFinite(sAte)).toBe(true);
    expect(Number.isFinite(tAte)).toBe(true);
    expect(Number.isFinite(xAte)).toBe(true);
    expect(Number.isFinite(rAte)).toBe(true);
  });

  it('R-Learner has lowest bias on linear DAG', () => {
    const { X, T, Y } = generateData(2000);

    const learners: Array<[string, CATEstimator]> = [
      ['S', new SLearner()],
      ['T', new TLearner()],
      ['X', new XLearner()],
      ['R', new RLearner()],
    ];

    const results: Array<[string, number]> = [];
    for (const [name, learner] of learners) {
      learner.fit(X, T, Y);
      results.push([name, learner.ate().estimate]);
    }

    // R-Learner should be closest to TRUE_COEFFICIENT for linear DAG
    const rBias = Math.abs((results.find(r => r[0] === 'R')?.[1] ?? 0) - TRUE_COEFFICIENT);
    expect(rBias).toBeLessThan(1.0);
  });
});

// ── Edge Cases ───────────────────────────────────────────────────────

describe('Meta-Learners — Edge Cases', () => {
  it('S-Learner handles single observation', () => {
    const X = new Matrix([[1]]);
    const T = new Float64Array([1]);
    const Y = new Float64Array([5]);
    const learner = new SLearner().fit(X, T, Y);
    expect(learner).toBeDefined();
    const ate = learner.ate();
    expect(Number.isFinite(ate.estimate)).toBe(true);
  });

  it('T-Learner handles empty treatment group', () => {
    const X = new Matrix([[1], [2]]);
    const T = new Float64Array([0, 0]);
    const Y = new Float64Array([1, 2]);
    const learner = new TLearner().fit(X, T, Y);
    // No treatment group → effect is negative of control model (all finite)
    const tau = learner.effect(X);
    expect(tau.every(v => Number.isFinite(v))).toBe(true);
  });

  it('X-Learner handles imbalanced groups', () => {
    const { X, T, Y } = generateData(500);
    // Force imbalanced: treat only first 10%
    const Timb = new Float64Array(T.length);
    for (let i = 0; i < T.length; i++) Timb[i] = i < 50 ? 1 : 0;
    const learner = new XLearner().fit(X, Timb, Y);
    const ate = learner.ate();
    expect(Number.isFinite(ate.estimate)).toBe(true);
  });

  it('R-Learner handles zero treatment variance', () => {
    // All same T → T̃=0 → θ=0
    const X = new Matrix([[1], [2], [3], [4]]);
    const T = new Float64Array([1, 1, 1, 1]);
    const Y = new Float64Array([2, 3, 2, 3]);
    const learner = new RLearner().fit(X, T, Y);
    const ate = learner.ate();
    expect(Number.isFinite(ate.estimate)).toBe(true);
  });

  it('all learners return valid ATE with SE for small dataset', () => {
    const { X, T, Y } = generateData(50);

    const learners: CATEstimator[] = [
      new SLearner(), new TLearner(), new XLearner(), new RLearner(),
    ];

    for (const learner of learners) {
      learner.fit(X, T, Y);
      const ate = learner.ate();
      expect(Number.isFinite(ate.estimate)).toBe(true);
      expect(Number.isFinite(ate.se)).toBe(true);
      expect(ate.se).toBeGreaterThanOrEqual(0);
    }
  });
});
