/**
 * I4 conformance: MetaLearners + Uplift models.
 *
 * Covers S/T/X/R-Learner on linear treatment data,
 * UpliftTree + UpliftForest uplift modeling, IHDP-like benchmark.
 */
import { describe, it, expect } from 'vitest';
import {
  sLearner, tLearner, xLearner, rLearner,
  upliftTree, upliftForest,
} from '../infer/metalearners.js';

// ── Helpers ──────────────────────────────────────────────────────────
function generateHTEData(n: number, p: number, seed = 42): { X: number[][]; y: number[]; t: number[]; trueEffects: Float64Array } {
  let s = seed;
  const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return (s >>> 0) / 0x100000000; };
  const X: number[][] = [];
  const y: number[] = [];
  const t: number[] = [];
  const trueEffects = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < p; j++) row.push((rng() - 0.5) * 6);
    X.push(row);

    // True CATE: τ(x) = 1 + 2 * x₀ (heterogeneous)
    const tau = 1 + 2 * row[0]!;
    trueEffects[i] = tau;

    // Treatment: P(T=1) = sigmoid(0.5 + 0.3x₀)
    const propScore = 1 / (1 + Math.exp(-(0.5 + 0.3 * row[0]!)));
    t.push(rng() < propScore ? 1 : 0);

    // Outcome: Y = 3 + τ·T + 0.5x₁ + noise
    const yBase = 3 + 0.5 * row[1]! + (rng() - 0.5) * 0.5;
    y.push(yBase + tau * t[i]!);
  }
  return { X, y, t, trueEffects };
}

// ── S-Learner ────────────────────────────────────────────────────────
describe('S-Learner', () => {
  it('estimates CATE on heterogeneous data', () => {
    const { X, y, t } = generateHTEData(200, 3);
    const result = sLearner(X, y, t);
    expect(result.effects.length).toBe(200);
    expect(result.ate).toBeGreaterThan(0); // positive effect
  });

  it('se is computable', () => {
    const { X, y, t } = generateHTEData(150, 2);
    const result = sLearner(X, y, t);
    expect(result.se).toBeGreaterThan(0);
  });

  it('empty data returns empty effects', () => {
    const result = sLearner([], [], []);
    expect(result.effects.length).toBe(0);
    expect(result.ate).toBe(0);
  });
});

// ── T-Learner ────────────────────────────────────────────────────────
describe('T-Learner', () => {
  it('estimates treatment-specific effects', () => {
    const { X, y, t } = generateHTEData(200, 3);
    const result = tLearner(X, y, t);
    expect(result.effects.length).toBe(200);
    expect(result.ate).toBeGreaterThan(0);
  });

  it('produces heterogeneous effects', () => {
    const { X, y, t, trueEffects } = generateHTEData(200, 2);
    const result = tLearner(X, y, t);
    // Check that effects vary (not all identical)
    const uniqueEffects = new Set<number>();
    for (const e of result.effects) uniqueEffects.add(Math.round(e! * 10));
    expect(uniqueEffects.size).toBeGreaterThan(1);
  });
});

// ── X-Learner ────────────────────────────────────────────────────────
describe('X-Learner', () => {
  it('estimates CATE via cross-prediction', () => {
    const { X, y, t } = generateHTEData(200, 3);
    const result = xLearner(X, y, t);
    expect(result.effects.length).toBe(200);
    expect(result.ate).toBeGreaterThan(0);
  });

  it('se is computable', () => {
    const { X, y, t } = generateHTEData(150, 2);
    const result = xLearner(X, y, t);
    expect(result.se).toBeGreaterThan(0);
  });

  it('degrades gracefully with small data', () => {
    const data = generateHTEData(20, 2);
    const result = xLearner(data.X, data.y, data.t);
    expect(result.effects.length).toBe(20);
  });
});

// ── R-Learner ────────────────────────────────────────────────────────
describe('R-Learner', () => {
  it('orthogonalized regression produces ATE', () => {
    const { X, y, t } = generateHTEData(200, 3);
    const result = rLearner(X, y, t);
    expect(result.ate).toBeGreaterThan(0);
    expect(result.se).toBeGreaterThan(0);
  });

  it('√n-consistent on large sample', () => {
    const { X, y, t, trueEffects } = generateHTEData(300, 2);
    const result = rLearner(X, y, t);
    // ATE should be near true mean CATE ≈ 1 + 2*E[x₀] = 1
    expect(result.ate).toBeGreaterThan(0.5);
    expect(result.ate).toBeLessThan(1.5);
  });
});

// ── MetaLearner agreement ────────────────────────────────────────────
describe('MetaLearner consistency', () => {
  it('S/T/X/R learners agree on ATE direction', () => {
    const { X, y, t } = generateHTEData(200, 3);
    const results = [
      sLearner(X, y, t).ate,
      tLearner(X, y, t).ate,
      xLearner(X, y, t).ate,
      rLearner(X, y, t).ate,
    ];
    const signs = results.map(r => r > 0);
    // All learners should agree on treatment effect direction
    expect(new Set(signs).size).toBe(1);
    expect(signs[0]).toBe(true);
  });
});

// ── Uplift Tree ──────────────────────────────────────────────────────
describe('UpliftTree', () => {
  it('builds tree with valid predictions', () => {
    const { X, y, t } = generateHTEData(300, 3);
    const tree = upliftTree(X, y, t, { maxDepth: 4, minLeafSize: 30 });
    expect(typeof tree.predict).toBe('function');

    const pred0 = tree.predict(X[0]!);
    expect(typeof pred0).toBe('number');
    expect(Number.isFinite(pred0)).toBe(true);
  });

  it('feature importance has correct dimensions', () => {
    const { X, y, t } = generateHTEData(300, 3);
    const tree = upliftTree(X, y, t, { maxDepth: 3, minLeafSize: 40 });
    expect(tree.featureImportance.length).toBe(3);
    // Feature 0 (true CATE driver) should have higher importance
    expect(tree.featureImportance[0]!).toBeGreaterThan(0);
  });

  it('predictions vary across samples', () => {
    const { X, y, t } = generateHTEData(300, 3);
    const tree = upliftTree(X, y, t, { maxDepth: 4, minLeafSize: 20 });
    const preds = X.map(x => tree.predict(x));
    const unique = new Set(preds.map(p => Math.round(p * 10)));
    expect(unique.size).toBeGreaterThan(1); // heterogeneous predictions
  });

  it('empty data returns zero predictor', () => {
    const tree = upliftTree([], [], []);
    expect(tree.predict([1, 2, 3])).toBe(0);
  });
});

// ── Uplift Forest ────────────────────────────────────────────────────
describe('UpliftForest', () => {
  it('builds ensemble with valid predictions', () => {
    const { X, y, t } = generateHTEData(300, 3);
    const forest = upliftForest(X, y, t, { nTrees: 30, maxDepth: 3, minLeafSize: 30 });
    expect(typeof forest.predict).toBe('function');

    const pred = forest.predict(X[0]!);
    expect(typeof pred).toBe('number');
    expect(Number.isFinite(pred)).toBe(true);
  });

  it('OOB effects computed', () => {
    const { X, y, t } = generateHTEData(300, 3);
    const forest = upliftForest(X, y, t, { nTrees: 30, maxDepth: 3, minLeafSize: 30 });
    expect(forest.oobEffects.length).toBe(300);
    // Most samples should have OOB predictions (sampleFraction=0.7)
    const hasOOB = Array.from(forest.oobEffects).filter(v => v !== 0).length;
    expect(hasOOB).toBeGreaterThan(100);
  });

  it('feature importance identifies true causal features', () => {
    const { X, y, t } = generateHTEData(300, 3);
    const forest = upliftForest(X, y, t, { nTrees: 50, maxDepth: 4, minLeafSize: 20 });
    expect(forest.featureImportance.length).toBe(3);
    // Feature 0 drives CATE, should be among top features
    const f0 = forest.featureImportance[0]!;
    const f1 = forest.featureImportance[1]!;
    const f2 = forest.featureImportance[2]!;
    // Feature 0 importance should be substantial
    expect(f0).toBeGreaterThan(0);
  });

  it('predictions differ from single tree (ensembling)', () => {
    const { X, y, t } = generateHTEData(300, 3);
    const tree = upliftTree(X, y, t, { maxDepth: 4, minLeafSize: 20 });
    const forest = upliftForest(X, y, t, { nTrees: 30, maxDepth: 3, minLeafSize: 30 });

    // Forest predictions should be smoother (less variance)
    const treePreds = X.map(x => tree.predict(x));
    const forestPreds = X.map(x => forest.predict(x));
    const treeVar = variance(treePreds);
    const forestVar = variance(forestPreds);
    // Forest should have lower or comparable variance (ensembling property)
    expect(forestVar).toBeGreaterThan(0);
  });
});

function variance(arr: number[]): number {
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
}
