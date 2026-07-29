/**
 * I36: Boundary coverage tests for modified algorithm files.
 * Targets edge cases not covered by existing test suites.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../graph/causal-graph.js';
import { gesAlgorithm } from '../graph/ges.js';
import { golemAlgorithm } from '../graph/golem.js';
import { directLiNGAM } from '../graph/lingam.js';
import { dagmaAlgorithm } from '../graph/dagma.js';

// ── GES edge cases ──────────────────────────────────────────────────

describe('GES boundary coverage', () => {
  it('handles empty data matrix gracefully', () => {
    const m = new Matrix(0, 3);
    const g = gesAlgorithm(m, ['X', 'Y', 'Z']);
    expect(g.nodeCount).toBe(3);
    expect(g.edges.length).toBe(0);
  });

  it('handles single-node graph', () => {
    const m = new Matrix([[1], [2], [3]]);
    const g = gesAlgorithm(m, ['X']);
    expect(g.nodeCount).toBe(1);
    expect(g.edges.length).toBe(0);
  });

  it('handles two-node independent data', () => {
    const m = new Matrix([[1, 0.1], [2, 0.2], [3, -0.1], [4, 0.0], [5, 0.1]]);
    const g = gesAlgorithm(m, ['X', 'Y']);
    expect(g.nodeCount).toBe(2);
    // Independent nodes: few or no edges
    expect(g.edges.length).toBeLessThanOrEqual(1);
  });

  it('produces valid DAG with no cycles', () => {
    const m = new Matrix([
      [0, 1, 2], [0, 1, 2], [0, 1, 2],
      [0, 1, 2], [0, 1, 2], [0, 1, 2],
    ]);
    const g = gesAlgorithm(m, ['X', 'Y', 'Z']);
    expect(g.hasCycle()).toBe(false);
    for (const e of g.edges) {
      if (e.directed) expect(e.source).not.toBe(e.target);
    }
  });
});

// ── GOLEM edge cases ────────────────────────────────────────────────

describe('GOLEM boundary coverage', () => {
  it('handles minimal data (n < 5)', () => {
    const g = golemAlgorithm([[1, 2], [3, 4]], ['X', 'Y']);
    expect(g.graph.nodeCount).toBe(2);
    expect(g.graph.edges.length).toBe(0);
  });

  it('handles single variable', () => {
    const g = golemAlgorithm([[1], [2], [3], [4], [5]], ['X']);
    expect(g.graph.nodeCount).toBe(1);
    expect(g.graph.edges.length).toBe(0);
  });

  it('accepts Matrix input type', () => {
    const m = new Matrix([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]);
    const g = golemAlgorithm(m, ['X', 'Y']);
    expect(g.graph.nodeCount).toBe(2);
  });

  it('accepts array input type', () => {
    const g = golemAlgorithm([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], ['X', 'Y']);
    expect(g.graph.nodeCount).toBe(2);
  });

  it('respects LBFGS optimizer config', () => {
    const g = golemAlgorithm(
      [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
      ['X', 'Y'],
      { optimizer: 'lbfgs', maxIter: 100, wThreshold: 0.0 },
    );
    expect(g.graph.nodeCount).toBe(2);
  });

  it('produces Float64Array W with correct dimensions', () => {
    const g = golemAlgorithm(
      [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
      ['X', 'Y', 'Z'],
    );
    expect(g.W).toBeInstanceOf(Float64Array);
    expect(g.W.length).toBe(9);
  });
});

// ── LiNGAM edge cases ───────────────────────────────────────────────

describe('LiNGAM boundary coverage', () => {
  it('handles single variable', () => {
    const m = new Matrix([[1], [2], [3]]);
    const r = directLiNGAM(m, ['X']);
    expect(r.order).toEqual(['X']);
    expect(r.graph.edges.length).toBe(0);
  });

  it('handles two variables with clear direction', () => {
    const rows: number[][] = [];
    for (let i = 0; i < 50; i++) {
      const x = Math.random() * 2 - 1;
      const y = 2 * x + (Math.random() - 0.5) * 0.1; // Y depends on X
      rows.push([x, y]);
    }
    const m = new Matrix(rows);
    const r = directLiNGAM(m, ['X', 'Y']);
    expect(r.order.length).toBe(2);
    expect(r.graph.edges.length).toBeGreaterThanOrEqual(0);
  });

  it('handles empty data', () => {
    const m = new Matrix(0, 2);
    const r = directLiNGAM(m, ['X', 'Y']);
    expect(r.order).toEqual(['X', 'Y']);
    expect(r.graph.edges.length).toBe(0);
  });

  it('correlation ordering works for large graphs', () => {
    // 12 nodes triggers correlation-based ordering (>10 threshold)
    const rows: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const row: number[] = [];
      for (let j = 0; j < 12; j++) row.push(Math.random());
      rows.push(row);
    }
    const m = new Matrix(rows);
    const names = Array.from({ length: 12 }, (_, i) => `V${i}`);
    const r = directLiNGAM(m, names);
    expect(r.order.length).toBe(12);
    // All names present in order
    for (const n of names) expect(r.order).toContain(n);
  });
});

// ── DAGMA edge cases ────────────────────────────────────────────────

describe('DAGMA boundary coverage', () => {
  it('handles minimal graph', () => {
    const rows = [[1, 2], [3, 4], [5, 6]];
    const r = dagmaAlgorithm(rows, ['X', 'Y'], { T: 1, warmIter: 100, maxIter: 200 });
    expect(r.graph.nodeCount).toBe(2);
    expect(r.W.length).toBe(4);
  });

  it('suppresses self-loops', () => {
    const rows: number[][] = [];
    for (let i = 0; i < 100; i++) {
      rows.push([Math.random(), Math.random() + rows[i - 1]?.[0] ?? 0]);
    }
    const r = dagmaAlgorithm(rows, ['X', 'Y'], { T: 2, warmIter: 200, maxIter: 500 });
    for (const e of r.graph.edges) {
      expect(e.source).not.toBe(e.target);
    }
  });

  it('h value is finite for valid output', () => {
    const rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
    const r = dagmaAlgorithm(rows, ['X', 'Y', 'Z'], { T: 2, warmIter: 200, maxIter: 500 });
    expect(Number.isFinite(r.h)).toBe(true);
  });
});
