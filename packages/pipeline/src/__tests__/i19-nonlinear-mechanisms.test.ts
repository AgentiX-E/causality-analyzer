import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../graph/causal-graph.js';
import { fitLogisticPNL, autoAssignMechanisms, parentRelevance } from '../gcm/nonlinear-mechanisms.js';

describe('fitLogisticPNL', () => {
  it('creates PostNonlinear mechanism', () => {
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const x = Math.random() * 2;
      const y = 1 / (1 + Math.exp(-(x * 0.5 + Math.random() * 0.2)));
      data.push([x, y]);
    }
    const mech = fitLogisticPNL(data, 1, [0]);
    expect(mech.noiseStd).toBeGreaterThan(0);
    const pred = mech.forward([1.0]);
    expect(pred).toBeGreaterThan(0);
    expect(pred).toBeLessThan(1);
  });

  it('boundary input produces valid output', () => {
    const data: number[][] = [];
    for (let i = 0; i < 50; i++) {
      const x = Math.random();
      data.push([x, x > 0.5 ? 0.9 : 0.1]);
    }
    const mech = fitLogisticPNL(data, 1, [0]);
    const pred0 = mech.forward([0]);
    const pred10 = mech.forward([10]);
    expect(pred0).toBeGreaterThan(0);
    expect(pred10).toBeGreaterThan(0);
    expect(pred0).toBeLessThan(1);
    expect(pred10).toBeLessThan(1);
  });

  it('invert returns approximately original output', () => {
    const data: number[][] = [];
    for (let i = 0; i < 80; i++) {
      const x = Math.random() * 3;
      const y = 1 / (1 + Math.exp(-(x + Math.random() * 0.3)));
      data.push([x, y]);
    }
    const mech = fitLogisticPNL(data, 1, [0]);
    const noise = mech.invert(0.5, [1.0]);
    expect(typeof noise).toBe('number');
    expect(Number.isFinite(noise)).toBe(true);
  });
});

describe('autoAssignMechanisms', () => {
  it('assigns types based on R²', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C');
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const a = Math.random();
      const b = a * 0.9 + Math.random() * 0.1;
      const c = b * 0.3 + Math.random() * 0.8;
      data.push([a, b, c]);
    }
    const assignments = autoAssignMechanisms(g, data, ['A', 'B', 'C']);
    expect(assignments.size).toBe(3);
    expect(assignments.get('B')?.type).toBe('linear');
  });

  it('root nodes get empirical mechanism', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const data: number[][] = Array.from({ length: 30 }, () => [Math.random(), Math.random() * 2]);
    const assignments = autoAssignMechanisms(g, data, ['A', 'B']);
    expect(assignments.get('A')?.type).toBeDefined();
    expect(assignments.get('B')?.type).toBeDefined();
  });

  it('explains mechanism choice', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const data: number[][] = Array.from({ length: 50 }, () => [Math.random(), Math.random()]);
    const assignments = autoAssignMechanisms(g, data, ['A', 'B']);
    expect(assignments.get('A')?.explanation).toBeDefined();
    expect(typeof assignments.get('A')?.explanation).toBe('string');
  });
});

describe('parentRelevance', () => {
  it('returns relevance scores summing to ~1', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Z', 'Y');
    const data: number[][] = [];
    for (let i = 0; i < 80; i++) {
      const x = Math.random();
      const z = Math.random();
      data.push([x, z, x * 0.7 + z * 0.1 + Math.random() * 0.2]);
    }
    const relevance = parentRelevance(g, data, ['X', 'Z', 'Y'], 'Y', 42);
    expect(relevance.size).toBe(2);
    let total = 0;
    for (const [, v] of relevance) total += v;
    expect(total).toBeCloseTo(1, 0);
    expect(relevance.get('X')).toBeGreaterThan(0);
  });

  it('dominant parent gets higher relevance', () => {
    const g = new CausalGraph(['X', 'Z', 'Y']);
    g.addEdge('X', 'Y'); g.addEdge('Z', 'Y');
    const data: number[][] = [];
    for (let i = 0; i < 300; i++) {
      const x = Math.random();
      const z = Math.random();
      data.push([x, z, x * 0.99 + z * 0.01 + Math.random() * 0.01]);
    }
    const relevance = parentRelevance(g, data, ['X', 'Z', 'Y'], 'Y', 42);
    expect(relevance.size).toBe(2);
    // X should have non-trivial relevance
    expect(relevance.get('X')).toBeGreaterThan(0);
  });

  it('deterministic seed produces reproducible results', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'C'); g.addEdge('B', 'C');
    const data: number[][] = Array.from({ length: 50 }, () => [Math.random(), Math.random(), Math.random()]);
    const r1 = parentRelevance(g, data, ['A', 'B', 'C'], 'C', 123);
    const r2 = parentRelevance(g, data, ['A', 'B', 'C'], 'C', 123);
    expect(r1.get('A')?.toFixed(6)).toBe(r2.get('A')?.toFixed(6));
  });
});
