/**
 * Model evaluation + Shapley attribution tests.
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../graph/causal-graph.js';
import { StructuralCausalModel } from '../gcm/structural-causal-model.js';
import {
  evaluateMechanismR2, evaluateMSE,
  shapleyAttribute, bootstrapRCA,
} from '../gcm/model-evaluation.js';

function threeNodeDAG(): CausalGraph {
  const g = new CausalGraph(['X', 'Y', 'Z']);
  g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
  return g;
}

function generateData(n: number): { data: number[][]; nodeMap: Map<string, number> } {
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 2 - 1;
    const y = x * 0.7 + Math.random() * 0.3;
    const z = y * 0.5 + Math.random() * 0.2;
    data.push([x, y, z]);
  }
  return { data, nodeMap: new Map([['X', 0], ['Y', 1], ['Z', 2]]) };
}

describe('evaluateMechanismR2', () => {
  it('returns R² for all nodes', () => {
    const g = threeNodeDAG();
    const scm = new StructuralCausalModel(g);
    const { data, nodeMap } = generateData(200);
    scm.train(data);
    const r2s = evaluateMechanismR2(scm, data, nodeMap);
    expect(r2s.size).toBe(3);
    expect(r2s.get('Y')).toBeGreaterThan(0.1);
    expect(r2s.get('Z')).toBeGreaterThan(0.1);
    expect(r2s.get('X')).toBe(0); // root node
  });

  it('handles small data gracefully', () => {
    const g = threeNodeDAG();
    const scm = new StructuralCausalModel(g);
    const data = [[1, 2, 3]];
    const r2s = evaluateMechanismR2(scm, data, new Map([['X', 0], ['Y', 1], ['Z', 2]]));
    expect(r2s.get('X')).toBeDefined();
  });
});

describe('evaluateMSE', () => {
  it('returns finite MSE values', () => {
    const g = threeNodeDAG();
    const scm = new StructuralCausalModel(g);
    const { data, nodeMap } = generateData(100);
    scm.train(data);
    const mses = evaluateMSE(scm, data, nodeMap);
    expect(mses.size).toBe(3);
    for (const [, v] of mses) expect(Number.isFinite(v)).toBe(true);
  });
});

describe('shapleyAttribute', () => {
  it('returns ranked root causes', () => {
    const g = threeNodeDAG();
    const scm = new StructuralCausalModel(g);
    const { data } = generateData(300);
    scm.train(data);

    // Create anomalous observation
    const obs: Record<string, number> = { X: 1.5, Y: 3.0, Z: 5.0 };
    const results = shapleyAttribute(scm, obs, 3);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]!.rank).toBe(1);
    expect(results[0]!.confidence).toBeGreaterThan(0);
  });

  it('returns empty for trivial graphs', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const scm = new StructuralCausalModel(g);
    const data: number[][] = [];
    for (let i = 0; i < 50; i++) {
      const a = Math.random();
      data.push([a, a * 0.5 + Math.random() * 0.1]);
    }
    scm.train(data);
    const results = shapleyAttribute(scm, { A: 1, B: 2 }, 5);
    expect(Array.isArray(results)).toBe(true);
  });
});

describe('bootstrapRCA', () => {
  it('returns CIs for anomaly scores', () => {
    const g = threeNodeDAG();
    const scm = new StructuralCausalModel(g);
    const { data } = generateData(200);
    scm.train(data);

    const observations: Record<string, number>[] = [];
    for (let i = 0; i < 20; i++) {
      const x = Math.random() * 2 + 1; // shifted distribution
      const y = x * 0.7 + Math.random() * 0.5;
      const z = y * 0.5 + Math.random() * 0.4;
      observations.push({ X: x, Y: y, Z: z });
    }

    const cis = bootstrapRCA(scm, observations, 50);
    expect(cis.size).toBe(3);
    for (const [, ci] of cis) {
      expect(ci.ciLow).toBeLessThanOrEqual(ci.mean);
      expect(ci.ciHigh).toBeGreaterThanOrEqual(ci.mean);
    }
  });

  it('handles single observation gracefully', () => {
    const g = threeNodeDAG();
    const scm = new StructuralCausalModel(g);
    const { data } = generateData(100);
    scm.train(data);

    const cis = bootstrapRCA(scm, [{ X: 1, Y: 2, Z: 3 }]);
    // With n=1 observation, bootstrap produces unreliable CI — may return empty map
    expect(cis instanceof Map).toBe(true);
  });
});

// ── Edge cases ────────────────────────────────────────────────────────

describe('model evaluation edge cases', () => {
  it('evaluateMSE handles NaN in data', () => {
    const g = threeNodeDAG();
    const scm = new StructuralCausalModel(g);
    const data = [[1, 2, NaN], [3, NaN, 5]];
    const mses = evaluateMSE(scm, data, new Map([['X', 0], ['Y', 1], ['Z', 2]]));
    expect(mses.size).toBe(3);
  });

  it('shapleyAttribute handles single-node graph', () => {
    const g = new CausalGraph(['A']);
    const scm = new StructuralCausalModel(g);
    scm.train([[1], [2], [1.5]]);
    const results = shapleyAttribute(scm, { A: 5 }, 3);
    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBe(0); // no non-root nodes
  });
});
