import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalForest } from '../infer/causal-forest.js';
import { doubleMLATE } from '../infer/double-ml.js';
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

describe('CausalForest', () => {
  it('produces finite effects on known DAG', () => {
    const { X, T, Y } = makeData(200);
    const forest = new CausalForest({ numTrees: 30, seed: 42 });
    forest.fit(X, T, Y);
    expect(forest.effect(X).every(v => Number.isFinite(v))).toBe(true);
  });
  it('handles small datasets gracefully', () => {
    const { X, T, Y } = makeData(30);
    const forest = new CausalForest({ numTrees: 10, minNodeSize: 3, seed: 42 });
    forest.fit(X, T, Y);
    expect(Number.isFinite(forest.ate().estimate)).toBe(true);
  });
});

describe('doubleMLATE', () => {
  it('produces finite ATE', () => {
    const { X, T, Y } = makeData(200);
    const Xarr: number[][] = [];
    for (let i = 0; i < X.rows; i++) Xarr.push([X.get(i, 0)]);
    const result = doubleMLATE(Xarr, Array.from(Y), Array.from(T));
    expect(Number.isFinite(result.ate)).toBe(true);
    expect(result.se).toBeGreaterThan(0);
  });
});
