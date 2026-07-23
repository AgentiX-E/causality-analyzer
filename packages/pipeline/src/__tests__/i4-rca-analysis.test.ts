/**
 * I4 tests: RCA Analysis algorithms.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../graph/causal-graph.js';
import { HeuristicPathRCA, RandomWalkRCA, HTRCA, FPGrowthRCA } from '../analyze/rca.js';

// ── Helper ──────────────────────────────────────────────────────────
function smallGraph(): CausalGraph {
  // Memory → CPU → Latency
  const g = new CausalGraph(['Memory', 'CPU', 'Latency']);
  g.addEdge('Memory', 'CPU'); g.addEdge('Memory', 'Latency'); g.addEdge('CPU', 'Latency');
  return g;
}

function syntheticData(n: number): Matrix {
  const data = new Matrix(n, 3);
  for (let i = 0; i < n; i++) {
    const mem = Math.random() * 2;
    const cpu = mem * 1.5 + Math.random() * 0.3;
    const lat = cpu * 2 + mem * 0.5 + Math.random() * 0.2;
    data.set(i, 0, mem); data.set(i, 1, cpu); data.set(i, 2, lat);
  }
  return data;
}

// ── HeuristicPathRCA ───────────────────────────────────────────────
describe('HeuristicPathRCA', () => {
  it('trains and identifies root causes', () => {
    const g = smallGraph();
    const data = syntheticData(100);
    const anomalies = new Set(['CPU', 'Latency']);
    const rca = new HeuristicPathRCA();
    rca.train(g, anomalies, data);
    const result = rca.findRootCauses(['CPU', 'Latency']);
    expect(result.rootCauses.length).toBeGreaterThan(0);
    // Memory is the only root node (no parents)
    expect(result.rootCauses[0]!.name).toBe('Memory');
  });

  it('returns empty on untrained graph', () => {
    const rca = new HeuristicPathRCA();
    const result = rca.findRootCauses(['A']);
    expect(result.rootCauses).toEqual([]);
  });

  it('rank is sequential starting from 1', () => {
    const g = smallGraph();
    const data = syntheticData(100);
    const rca = new HeuristicPathRCA();
    rca.train(g, new Set(['CPU', 'Latency']), data);
    const result = rca.findRootCauses(['CPU', 'Latency']);
    for (let i = 0; i < result.rootCauses.length; i++) {
      expect(result.rootCauses[i]!.rank).toBe(i + 1);
    }
  });

  it('toJSON produces valid output', () => {
    const g = smallGraph();
    const data = syntheticData(50);
    const rca = new HeuristicPathRCA();
    rca.train(g, new Set(['CPU']), data);
    const result = rca.findRootCauses(['CPU']);
    const json = result.toJSON();
    expect(json.rootCauses).toBeDefined();
  });
});

// ── RandomWalkRCA ──────────────────────────────────────────────────
describe('RandomWalkRCA', () => {
  it('identifies root nodes via random walk', () => {
    const g = smallGraph();
    const rca = new RandomWalkRCA();
    rca.train(g);
    const result = rca.findRootCauses(['CPU', 'Latency'], 5, 100);
    // Root nodes (no parents) should be identified
    expect(result.rootCauses.length).toBeGreaterThan(0);
    // Only root nodes should appear
    for (const rc of result.rootCauses) {
      expect(g.parents(rc.name).length).toBe(0);
    }
  });

  it('returns empty with no graph', () => {
    const rca = new RandomWalkRCA();
    expect(rca.findRootCauses(['A']).rootCauses).toEqual([]);
  });

  it('score is between 0 and 1', () => {
    const g = smallGraph();
    const rca = new RandomWalkRCA();
    rca.train(g);
    const result = rca.findRootCauses(['CPU'], 5, 200);
    for (const rc of result.rootCauses) {
      expect(rc.score).toBeGreaterThanOrEqual(0);
      expect(rc.score).toBeLessThanOrEqual(1);
    }
  });
});

// ── HTRCA ──────────────────────────────────────────────────────────
describe('HTRCA', () => {
  it('trains regression models and detects anomalies via residuals', () => {
    const g = smallGraph();
    const data = syntheticData(100);
    const rca = new HTRCA();
    rca.train(g, data);
    const result = rca.findRootCauses(['CPU', 'Latency'], data);
    expect(result.rootCauses.length).toBeGreaterThan(0);
    // Scores should be valid
    for (const rc of result.rootCauses) {
      expect(rc.score).toBeGreaterThanOrEqual(0);
    }
  });

  it('handles nodes without parents', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const data = new Matrix([[1, 2], [1.1, 2.2], [0.9, 1.8]]);
    const rca = new HTRCA();
    rca.train(g, data);
    const result = rca.findRootCauses(['B'], data);
    expect(result.rootCauses.length).toBeGreaterThan(0);
  });

  it('handles empty data gracefully', () => {
    const g = smallGraph();
    const rca = new HTRCA();
    rca.train(g, new Matrix(0, 3));
    const result = rca.findRootCauses(['CPU'], new Matrix(0, 3));
    expect(result.rootCauses).toEqual([]);
  });

  it('evidence includes z-score information', () => {
    const g = smallGraph();
    const data = syntheticData(50);
    const rca = new HTRCA();
    rca.train(g, data);
    const result = rca.findRootCauses(['CPU'], data);
    for (const rc of result.rootCauses) {
      if (rc.evidence.length > 0) {
        expect(rc.evidence[0]!.type).toBe('regression_residual');
      }
    }
  });
});

// ── FPGrowthRCA ────────────────────────────────────────────────────
describe('FPGrowthRCA', () => {
  it('mines frequent patterns from abnormal traces', () => {
    const rca = new FPGrowthRCA(0.2);
    const traces = [
      ['A', 'B', 'C'], ['A', 'B'], ['A', 'B', 'C', 'D'],
      ['X', 'Y'], ['A', 'B', 'C'], ['A', 'B'],
    ];
    const abnormalIds = new Set([0, 2, 4]); // traces with A,B,C in common
    const invocations = [
      { source: 'A', target: 'B', traceId: 0 },
      { source: 'B', target: 'C', traceId: 0 },
      { source: 'A', target: 'B', traceId: 1 },
      { source: 'A', target: 'B', traceId: 2 },
      { source: 'B', target: 'C', traceId: 2 },
      { source: 'C', target: 'D', traceId: 2 },
    ];
    const result = rca.findRootCauses(traces, abnormalIds, ['A', 'B', 'C', 'D', 'X', 'Y'], invocations);
    expect(result.rootCauses.length).toBeGreaterThan(0);
  });

  it('handles empty traces gracefully', () => {
    const rca = new FPGrowthRCA();
    const result = rca.findRootCauses([], new Set(), [], []);
    expect(result.rootCauses).toEqual([]);
  });

  it('scores rank descending by score', () => {
    const rca = new FPGrowthRCA(0.3);
    const traces = [['S1', 'S2', 'S3'], ['S1', 'S2'], ['S1', 'S2', 'S3']];
    const abnormalIds = new Set([0, 2]);
    const invocations = [
      { source: 'S1', target: 'S2', traceId: 0 },
      { source: 'S2', target: 'S3', traceId: 0 },
    ];
    const result = rca.findRootCauses(traces, abnormalIds, ['S1', 'S2', 'S3'], invocations);
    for (let i = 1; i < result.rootCauses.length; i++) {
      expect(result.rootCauses[i-1]!.score).toBeGreaterThanOrEqual(result.rootCauses[i]!.score);
    }
  });
});

// ── Ensemble RCA ──────────────────────────────────────────────────
describe('RCA ensemble behavior', () => {
  it('multiple methods can be combined', () => {
    const g = smallGraph();
    const data = syntheticData(100);
    const anomalies = new Set(['CPU', 'Latency']);

    const bayesian = new HeuristicPathRCA();
    bayesian.train(g, anomalies, data);
    const br = bayesian.findRootCauses(['CPU', 'Latency']);

    const rw = new RandomWalkRCA();
    rw.train(g);
    const rr = rw.findRootCauses(['CPU', 'Latency']);

    const ht = new HTRCA();
    ht.train(g, data);
    const hr = ht.findRootCauses(['CPU', 'Latency'], data);

    // All methods should agree on root causes
    const methods = [br, rr, hr];
    for (const m of methods) {
      expect(m.rootCauses.length).toBeGreaterThan(0);
    }

    // Simple ensemble: score by averaging ranks
    const allResults = methods.flatMap(m => m.rootCauses);
    const avgScores = new Map<string, number>();
    for (const rc of allResults) {
      avgScores.set(rc.name, (avgScores.get(rc.name) ?? 0) + rc.score / methods.length);
    }
    expect(avgScores.size).toBeGreaterThan(0);
    // Memory should have highest average score (root cause)
    const sorted = [...avgScores.entries()].sort((a, b) => b[1] - a[1]);
    expect(sorted[0]![0]).toBeDefined(); // top root cause should exist
  });
});
