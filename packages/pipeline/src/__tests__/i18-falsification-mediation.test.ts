import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../graph/causal-graph.js';
import { falsifyGraph, lmcFalsification } from '../gcm/graph-falsification.js';
import { naturalDirectEffect, arrowStrength } from '../infer/mediation.js';

// ── Graph Falsification ─────────────────────────────────────────────

describe('falsifyGraph', () => {
  function synthData(): { data: Matrix; graph: CausalGraph; names: string[] } {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const data = new Matrix(100, 3);
    for (let r = 0; r < 100; r++) {
      const x = Math.random(); data.set(r, 0, x);
      data.set(r, 1, x * 0.7 + Math.random() * 0.3);
      data.set(r, 2, data.get(r, 1) * 0.5 + Math.random() * 0.2);
    }
    return { data, graph: g, names: ['X', 'Y', 'Z'] };
  }

  it('returns results without crashing', () => {
    const { data, graph, names } = synthData();
    const result = falsifyGraph(graph, data, names);
    expect(typeof result.falsified).toBe('boolean');
    expect(typeof result.pValue).toBe('number');
  });

  it('includes explanation', () => {
    const { data, graph, names } = synthData();
    const result = falsifyGraph(graph, data, names);
    expect(result.explanation.length).toBeGreaterThan(0);
  });
});

describe('lmcFalsification', () => {
  it('returns per-node results', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const data = new Matrix(50, 2);
    for (let r = 0; r < 50; r++) { data.set(r, 0, Math.random()); data.set(r, 1, data.get(r, 0) * 0.5 + Math.random() * 0.2); }
    const results = lmcFalsification(g, data, ['A', 'B']);
    expect(results.size).toBe(2);
    expect(results.get('A')?.violated).toBe(false); // no parents
  });
});

// ── Mediation ────────────────────────────────────────────────────────

describe('naturalDirectEffect', () => {
  it('computes NDE/NIE for mediated relationship', () => {
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const x = Math.random() * 2;
      const m = x * 0.6 + Math.random() * 0.3;
      const y = x * 0.2 + m * 0.5 + Math.random() * 0.1;
      data.push([x, m, y]);
    }
    const result = naturalDirectEffect(data, 0, 2, 1);
    expect(result.totalEffect).toBeGreaterThan(0);
    expect(result.nde).toBeGreaterThan(0);
    expect(result.nie).toBeGreaterThan(0);
  });
});

describe('arrowStrength', () => {
  it('returns normalized strengths', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z'); g.addEdge('X', 'Z');
    const data: number[][] = [];
    for (let i = 0; i < 80; i++) {
      const x = Math.random(); data.push([x, x * 0.7 + Math.random() * 0.2, x * 0.3 + Math.random() * 0.2]);
    }
    const strengths = arrowStrength(g, data, ['X', 'Y', 'Z']);
    expect(strengths.size).toBeGreaterThanOrEqual(3);
    for (const [, v] of strengths) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});
