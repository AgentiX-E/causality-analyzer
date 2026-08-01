/**
 * GCM Auto-Assign Mechanism Tests.
 *
 * Validates automatic mechanism selection based on data characteristics.
 */
import { describe, it, expect } from 'vitest';
import { autoAssignMechanisms } from '../../src/gcm/auto-assign.js';
import { CausalGraph } from '../../src/graph/causal-graph.js';

describe('Auto-Assign Mechanisms', () => {
  it('assigns constant for nodes with no data column', () => {
    const g = new CausalGraph(['X', 'Y']);
    const nodeIndex = new Map([['X', 0]]); // Y has no column
    const data = [[1.0, 2.0]];

    const result = autoAssignMechanisms(g, data, nodeIndex);
    expect(result.assignments.length).toBe(2);
    const yAssign = result.assignments.find(a => a.node === 'Y')!;
    expect(yAssign.mechanism).toBe('constant');
  });

  it('assigns discrete for binary variable', () => {
    const g = new CausalGraph(['X']);
    const nodeIndex = new Map([['X', 0]]);
    const data = Array.from({ length: 50 }, () => [Math.random() > 0.5 ? 1 : 0]);

    const result = autoAssignMechanisms(g, data, nodeIndex);
    const xAssign = result.assignments[0]!;
    expect(xAssign.mechanism).toBe('discrete');
  });

  it('assigns anm for low-skew continuous data', () => {
    const g = new CausalGraph(['X']);
    const nodeIndex = new Map([['X', 0]]);
    // Normal(0, 1) — low skew
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const u1 = Math.max(1e-10, Math.random());
      const u2 = Math.random();
      data.push([Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)]);
    }

    const result = autoAssignMechanisms(g, data, nodeIndex);
    const xAssign = result.assignments[0]!;
    expect(['anm', 'pnl']).toContain(xAssign.mechanism);
    expect(xAssign.confidence).toBeGreaterThan(0);
    expect(xAssign.confidence).toBeLessThanOrEqual(1);
  });

  it('assigns pnl for high-skew data', () => {
    const g = new CausalGraph(['X']);
    const nodeIndex = new Map([['X', 0]]);
    // Deterministic Pareto-like data — guaranteed high positive skew (~4.0+)
    const data: number[][] = [];
    for (let i = 0; i < 200; i++) {
      // 95% small values [0, 2), 5% extreme outliers [80, 180]
      data.push([i % 20 === 0 ? 80 + i : (i * 0.01)]);
    }

    const result = autoAssignMechanisms(g, data, nodeIndex);
    const xAssign = result.assignments[0]!;
    expect(xAssign.mechanism).toBe('pnl');
  });

  it('handles constant value gracefully', () => {
    const g = new CausalGraph(['X']);
    const nodeIndex = new Map([['X', 0]]);
    const data = Array.from({ length: 10 }, () => [5.0]);

    const result = autoAssignMechanisms(g, data, nodeIndex);
    const xAssign = result.assignments[0]!;
    expect(xAssign.mechanism).toBe('constant');
  });

  it('reports summary counts correctly', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    const nodeIndex = new Map([['X', 0], ['Y', 1], ['Z', 2]]);
    const data = Array.from({ length: 50 }, () => [
      Math.random() > 0.5 ? 1 : 0,     // X: binary
      Math.random(),                    // Y: continuous
      Math.exp(Math.random() * 2),      // Z: high skew
    ]);

    const result = autoAssignMechanisms(g, data, nodeIndex);
    expect(result.summary.totalNodes).toBe(3);
    expect(result.summary.discreteCount).toBe(1);
    expect(result.summary.pnlCount + result.summary.anmCount + result.summary.constantCount + result.summary.discreteCount).toBe(3);
  });

  it('flags hasFallbacks for truly ambiguous data', () => {
    const g = new CausalGraph(['X', 'Y']);
    // X has no data column — should trigger constant assignment
    const nodeIndex = new Map([['Y', 0]]);
    const data = [[1.0], [2.0]];

    const result = autoAssignMechanisms(g, data, nodeIndex);
    // X has no data → constant → fallback
    expect(result.hasFallbacks).toBe(true);
  });

  it('respects custom thresholds', () => {
    const g = new CausalGraph(['X']);
    const nodeIndex = new Map([['X', 0]]);
    // 5 unique values (binary-like)
    const data = Array.from({ length: 100 }, () => [Math.floor(Math.random() * 5)]);

    // Default threshold 10 — 5 unique values should be 'discrete'
    const r1 = autoAssignMechanisms(g, data, nodeIndex);
    expect(r1.assignments[0]!.mechanism).toBe('discrete');

    // Custom threshold 3 — 5 unique values should NOT be 'discrete'
    const r2 = autoAssignMechanisms(g, data, nodeIndex, { discreteThreshold: 3 });
    expect(r2.assignments[0]!.mechanism).not.toBe('discrete');
  });

  it('reports parent information', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Z');
    g.addEdge('Y', 'Z');
    const nodeIndex = new Map([['X', 0], ['Y', 1], ['Z', 2]]);
    const data = Array.from({ length: 50 }, () => [Math.random(), Math.random(), Math.random()]);

    const result = autoAssignMechanisms(g, data, nodeIndex);
    const zAssign = result.assignments.find(a => a.node === 'Z')!;
    expect(zAssign.parents).toContain('X');
    expect(zAssign.parents).toContain('Y');
  });
});
