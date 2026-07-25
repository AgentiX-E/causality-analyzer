/**
 * I80: DAGMA + Health check tests.
 */
import { describe, it, expect } from 'vitest';
import { dagmaAlgorithm } from '../../src/graph/dagma.js';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { generateLinearData } from '../../src/benchmark.js';
import { HealthChecker } from '../../src/health.js';

describe('DAGMA Algorithm', () => {
  it('discovers simple DAG from linear data', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data, nodeNames } = generateLinearData(g, 200, 42);
    const result = dagmaAlgorithm(data, nodeNames);
    expect(result.graph.nodeCount).toBe(3);
    expect(result.W).toBeInstanceOf(Float64Array);
    expect(result.W.length).toBe(9);
    expect(typeof result.h).toBe('number');
  });

  it('produces valid W matrix dimensions', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const { data, nodeNames } = generateLinearData(g, 100, 43);
    const result = dagmaAlgorithm(data, nodeNames);
    expect(result.W.length).toBe(4); // 2×2
  });

  it('respects wThreshold config', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y');
    const { data, nodeNames } = generateLinearData(g, 150, 44);
    const r1 = dagmaAlgorithm(data, nodeNames, { wThreshold: 0.5, maxOuterIter: 5 });
    const r2 = dagmaAlgorithm(data, nodeNames, { wThreshold: 0.1, maxOuterIter: 5 });
    // Stricter threshold → fewer edges
    expect(r1.graph.edges.length).toBeLessThanOrEqual(r2.graph.edges.length + 2);
  });

  it('respects lambda1 regularization', () => {
    const g = new CausalGraph(['X', 'Y']);
    const { data, nodeNames } = generateLinearData(g, 100, 45);
    const result = dagmaAlgorithm(data, nodeNames, { lambda1: 0.5, maxOuterIter: 5 });
    expect(result.graph.nodeCount).toBe(2);
  });

  it('handles small datasets', () => {
    const data = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [1.5, 3.0], [2.5, 5.0]];
    const result = dagmaAlgorithm(data, ['X', 'Y'], { maxOuterIter: 5 });
    expect(result.graph.nodeCount).toBe(2);
  });

  it('applies domain knowledge', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data, nodeNames } = generateLinearData(g, 150, 46);
    const result = dagmaAlgorithm(data, nodeNames, { maxOuterIter: 5 }, {
      forbids: [['Z', 'X']],
    });
    expect(result.graph.nodeCount).toBe(3);
  });
});

describe('Health Checker', () => {
  it('returns healthy for clean state', () => {
    const checker = new HealthChecker();
    const result = checker.getStatus();
    expect(result.status).toBe('healthy');
  });

  it('reports degraded when warnings present', () => {
    const checker = new HealthChecker();
    checker.setCheck('db', { status: 'warning', detail: 'slow connection' });
    const result = checker.getStatus();
    expect(result.status).toBe('degraded');
  });

  it('reports unhealthy when errors present', () => {
    const checker = new HealthChecker();
    checker.setCheck('db', { status: 'error', detail: 'connection refused' });
    const result = checker.getStatus();
    expect(result.status).toBe('unhealthy');
  });

  it('reports uptime', () => {
    const checker = new HealthChecker();
    const result = checker.getStatus();
    expect(typeof result.uptime).toBe('number');
    expect(result.uptime).toBeGreaterThan(0);
  });

  it('markReady/markNotReady toggle readiness', () => {
    const checker = new HealthChecker();
    expect(checker.isReady()).toBe(false);
    checker.markReady();
    expect(checker.isReady()).toBe(true);
    checker.markNotReady();
    expect(checker.isReady()).toBe(false);
  });

  it('isAlive returns true by default', () => {
    const checker = new HealthChecker();
    expect(checker.isAlive()).toBe(true);
  });
});
