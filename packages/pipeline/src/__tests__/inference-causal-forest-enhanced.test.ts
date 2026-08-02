/**
 * CausalForest — Legacy API Compatibility Tests.
 *
 * Updated to use CATEstimator interface (fit/effect/ate) instead of
 * deprecated train/predict/getResult/predictOOB methods.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalForest } from '../infer/causal-forest.js';
import { generateLinearData } from '../benchmark.js';
import { CausalGraph } from '../graph/causal-graph.js';

const SEED = 42;

function makeData(n = 200) {
  const g = new CausalGraph(['X', 'T', 'Y']);
  g.addEdge('X', 'T'); g.addEdge('X', 'Y'); g.addEdge('T', 'Y');
  const { data, nodeNames } = generateLinearData(g, n, SEED);
  const X = new Matrix(data.map(r => [r[nodeNames.indexOf('X')]!]));
  const T = Float64Array.from(data.map(r => r[nodeNames.indexOf('T')]!));
  const Y = Float64Array.from(data.map(r => r[nodeNames.indexOf('Y')]!));
  return { X, T, Y };
}

describe('CausalForest — API Compatibility', () => {
  it('fit + effect + ate work with default config', () => {
    const { X, T, Y } = makeData(100);
    const forest = new CausalForest({ numTrees: 20 });
    forest.fit(X, T, Y);
    const tau = forest.effect(X);
    expect(tau.length).toBe(100);
    expect(tau.every(v => Number.isFinite(v))).toBe(true);
    const ate = forest.ate();
    expect(Number.isFinite(ate.estimate)).toBe(true);
  });

  it('ATE is reasonable on known DAG', () => {
    const { X, T, Y } = makeData(200);
    const forest = new CausalForest({ numTrees: 30, seed: 42 });
    forest.fit(X, T, Y);
    const ate = forest.ate();
    expect(Math.abs(ate.estimate)).toBeLessThan(20);
    expect(ate.se).toBeGreaterThan(0);
  });

  it('predict returns number array', () => {
    const { X, T, Y } = makeData(50);
    const forest = new CausalForest({ numTrees: 10, seed: 42 });
    forest.fit(X, T, Y);
    const Xarr: number[][] = [];
    for (let i = 0; i < X.rows; i++) Xarr.push([X.get(i, 0)]);
    const preds = forest.predict(Xarr);
    expect(preds.length).toBe(50);
    expect(preds.every(v => Number.isFinite(v))).toBe(true);
  });

  it('predictWithCI returns valid intervals', () => {
    const { X, T, Y } = makeData(80);
    const forest = new CausalForest({ numTrees: 20, seed: 42 });
    forest.fit(X, T, Y);
    const Xarr: number[][] = [];
    for (let i = 0; i < X.rows; i++) Xarr.push([X.get(i, 0)]);
    const ci = forest.predictWithCI(Xarr);
    expect(ci.length).toBe(80);
    expect(ci[0]!).toHaveProperty('lower');
    expect(ci[0]!).toHaveProperty('upper');
    expect(ci[0]!.lower).toBeLessThanOrEqual(ci[0]!.upper);
  });

  it('getResult returns structured output', () => {
    const { X, T, Y } = makeData(60);
    const forest = new CausalForest({ numTrees: 15, seed: 42 });
    forest.fit(X, T, Y);
    const result = forest.getResult();
    expect(result).toHaveProperty('ate');
    expect(result).toHaveProperty('se');
    expect(Number.isFinite(result.ate)).toBe(true);
  });
});
