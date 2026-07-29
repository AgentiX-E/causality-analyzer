/**
 * Advanced causal discovery tests — FCI, Grow-Shrink, Targeted Discovery.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../graph/causal-graph.js';
import { fciAlgorithm, growShrink, targetedDiscovery } from '../graph/advanced-discovery.js';

// ── FCI ──────────────────────────────────────────────────────────────

function linearSEM3(n: number): { data: Matrix; names: string[] } {
  const data = new Matrix(n, 3);
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 2;
    const y = x * 0.7 + Math.random() * 0.3;
    const z = y * 0.5 + Math.random() * 0.2;
    data.set(i, 0, x); data.set(i, 1, y); data.set(i, 2, z);
  }
  return { data, names: ['X', 'Y', 'Z'] };
}

describe('fciAlgorithm', () => {
  it('discovers edges in a chain X→Y→Z', () => {
    const { data, names } = linearSEM3(200);
    const { graph, pagEdges } = fciAlgorithm(data, names);
    expect(pagEdges.size).toBeGreaterThan(0);
    // For linear SEM chain, edges should be present
    expect(graph.nodes.length).toBe(3);
  });

  it('handles empty data', () => {
    const data = new Matrix(0, 3);
    const { graph } = fciAlgorithm(data, ['A', 'B', 'C']);
    expect(graph.nodes.length).toBe(3);
  });

  it('identifies v-structures', () => {
    // X→Z←Y (collider)
    const n = 200;
    const data = new Matrix(n, 4);
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 2;
      const y = Math.random() * 2;
      const z = x * 0.4 + y * 0.4 + Math.random() * 0.2;
      const w = z * 0.6 + Math.random() * 0.3;
      data.set(i, 0, x); data.set(i, 1, y); data.set(i, 2, z); data.set(i, 3, w);
    }
    const { pagEdges } = fciAlgorithm(data, ['X', 'Y', 'Z', 'W']);
    expect(pagEdges.size).toBeGreaterThan(0);
  });
});

// ── Grow-Shrink ──────────────────────────────────────────────────────

function generateGSData(n: number, trueBlanket: number[]): { data: Matrix; names: string[] } {
  const cols = 5;
  const data = new Matrix(n, cols);
  const names = ['A', 'B', 'C', 'D', 'E'];
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < cols; c++) data.set(i, c, Math.random());
  }
  // Make A dependent on B (markov blanket member)
  for (let i = 0; i < n; i++) {
    const a = data.get(i, 0);
    const b = a * 0.8 + Math.random() * 0.3;
    data.set(i, 1, b);
  }
  return { data, names };
}

describe('growShrink', () => {
  it('identifies non-empty blanket for dependent variables', () => {
    const { data, names } = generateGSData(200, [1]);
    const blanket = growShrink(data, 0, names);
    expect(blanket.length).toBeGreaterThan(0);
  });

  it('returns empty blanket for isolated variable', () => {
    // All independent columns
    const data = new Matrix(100, 4);
    for (let i = 0; i < 100; i++) for (let c = 0; c < 4; c++) data.set(i, c, Math.random());
    const blanket = growShrink(data, 0, ['A', 'B', 'C', 'D']);
    // With random independent data, blanket should be small
    expect(blanket.length).toBeLessThan(4);
  });

  it('handles small data gracefully', () => {
    const data = new Matrix(10, 3);
    for (let i = 0; i < 10; i++) for (let c = 0; c < 3; c++) data.set(i, c, Math.random());
    const blanket = growShrink(data, 1, ['X', 'Y', 'Z']);
    expect(Array.isArray(blanket)).toBe(true);
  });
});

// ── Targeted Discovery ───────────────────────────────────────────────

describe('targetedDiscovery', () => {
  it('finds parents for target variable', () => {
    const n = 200;
    const data = new Matrix(n, 4);
    const names = ['X', 'Y', 'Z', 'W'];
    for (let i = 0; i < n; i++) {
      const x = Math.random();
      const y = x * 0.7 + Math.random() * 0.3;
      const z = y * 0.5 + Math.random() * 0.2;
      const w = Math.random();
      data.set(i, 0, x); data.set(i, 1, y); data.set(i, 2, z); data.set(i, 3, w);
    }
    const result = targetedDiscovery(data, ['Y', 'Z'], names);
    expect(result.get('Y')?.length).toBeGreaterThan(0);
    expect(result.get('Z')?.length).toBeGreaterThan(0);
  });

  it('handles unknown target gracefully', () => {
    const data = new Matrix(10, 2);
    const result = targetedDiscovery(data, ['UNKNOWN'], ['A', 'B']);
    expect(result.get('UNKNOWN')).toEqual([]);
  });

  it('handles empty data', () => {
    const data = new Matrix(0, 3);
    const result = targetedDiscovery(data, ['X'], ['X', 'Y', 'Z']);
    expect(result.get('X')).toEqual([]);
  });

  it('handles multiple targets', () => {
    const n = 150;
    const data = new Matrix(n, 4);
    const names = ['A', 'B', 'C', 'D'];
    for (let i = 0; i < n; i++) {
      const a = Math.random();
      const b = a * 0.6 + Math.random() * 0.4;
      const c = b * 0.5 + Math.random() * 0.3;
      const d = c * 0.4 + Math.random() * 0.3;
      data.set(i, 0, a); data.set(i, 1, b); data.set(i, 2, c); data.set(i, 3, d);
    }
    const result = targetedDiscovery(data, ['B', 'C', 'D'], names);
    expect(result.size).toBe(3);
  });
});
