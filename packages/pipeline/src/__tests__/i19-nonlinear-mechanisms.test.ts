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
    expect(assignments.get('C')?.type).toBeDefined();
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
});
