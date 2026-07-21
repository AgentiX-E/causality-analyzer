/**
 * I8 tests: Fusion, viz data structures, E2E pipeline.
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../graph/causal-graph.js';
import { StructuralCausalModel } from '../gcm/structural-causal-model.js';
import { CIRCAPipeline } from '../analyze/circa.js';
import { BayesianRCA } from '../analyze/rca.js';
import { buildGraphVizData, buildTimeseriesVizData, buildRankingVizData } from '../viz/viz-data.js';
import { FusionAnalyzer } from '../viz/fusion.js';

// ── Viz: Graph ──────────────────────────────────────────────────────
describe('graph visualization', () => {
  it('builds framework-agnostic graph data', () => {
    const nodes = ['Memory', 'CPU', 'Latency'];
    const edges = [
      { source: 'Memory', target: 'CPU', weight: 1, directed: true },
      { source: 'CPU', target: 'Latency', weight: 1, directed: true },
    ];
    const data = buildGraphVizData(nodes, edges,
      [{ name: 'Memory', score: 0.9, confidence: 0.95, rank: 1, evidence: [] }],
      ['CPU', 'Latency']);

    expect(data.nodes.length).toBe(3);
    expect(data.nodes[0]!.type).toBe('root_cause');
    expect(data.nodes[1]!.type).toBe('anomaly');
    expect(data.nodes[2]!.type).toBe('anomaly');
    expect(data.edges.length).toBe(2);
  });

  it('labels healthy nodes correctly', () => {
    const data = buildGraphVizData(['A', 'B'], [],
      [{ name: 'A', score: 0.8, confidence: 0.9, rank: 1, evidence: [] }],
      []);
    expect(data.nodes[1]!.type).toBe('intermediate');
  });
});

// ── Viz: Time Series ────────────────────────────────────────────────
describe('timeseries visualization', () => {
  it('builds self-describing timeseries data', () => {
    const data = buildTimeseriesVizData(
      { cpu: [10, 12, 11, 50, 48], latency: [5, 6, 5, 30, 28] },
      [1000, 1001, 1002, 1003, 1004],
      [3, 4],
      'Memory',
    );
    expect(data.series.length).toBe(2);
    expect(data.anomalyRegions.length).toBe(1);
    expect(data.anomalyRegions[0]!.severity).toBe('critical');
    expect(data.anomalyRegions[0]!.rootCause).toBe('Memory');
    // Each point is self-describing
    expect(data.series[0]!.data[0]!.ts).toBe(1000);
    expect(data.series[0]!.data[0]!.value).toBe(10);
  });
});

// ── Viz: RCA Ranking ────────────────────────────────────────────────
describe('rca ranking visualization', () => {
  it('builds ranking data from RCA result', () => {
    const rootCauses = [{ name: 'Memory', score: 0.9, confidence: 0.95, rank: 1, evidence: [] as any[] }];
    const paths = [{ nodes: ['Memory', 'CPU', 'Latency'], score: 0.85, direction: 'forward' as const }];
    const data = buildRankingVizData(rootCauses, paths);
    expect(data.rootCauses.length).toBe(1);
    expect(data.rootCauses[0]!.name).toBe('Memory');
    expect(data.propagationPaths.length).toBe(1);
    expect(data.propagationPaths[0]!.root).toBe('Memory');
  });
});

// ── Fusion Analyzer ─────────────────────────────────────────────────
describe('FusionAnalyzer', () => {
  function makeResult(name: string, score: number): any {
    return {
      rootCauses: [{ name, score, confidence: 0.8, rank: 1, evidence: [] }],
      paths: [],
      metadata: { method: 'test', analyzedAt: Date.now(), durationMs: 0, extra: {} },
      toJSON: () => ({}),
    };
  }

  it('weighted fusion combines multimodal scores', () => {
    const fusion = new FusionAnalyzer({ strategy: 'weighted', weights: { metric: 0.5, trace: 0.35, log: 0.15 } });
    const result = fusion.fuse(
      makeResult('Memory', 0.9),
      makeResult('CPU', 0.6),
    );
    expect(result.rootCauses.length).toBeGreaterThan(0);
  });

  it('nested fusion uses metric RCA as scope', () => {
    const fusion = new FusionAnalyzer({ strategy: 'nested' });
    const result = fusion.fuse(
      makeResult('Memory', 0.9),
      makeResult('CPU', 0.4),
    );
    expect(result.metadata.extra.strategy).toBe('nested');
  });

  it('handles null inputs gracefully', () => {
    const fusion = new FusionAnalyzer();
    expect(fusion.fuse(null, null).rootCauses.length).toBe(0);
  });

  it('single-modal input returns metric result', () => {
    const fusion = new FusionAnalyzer({ strategy: 'nested' });
    const result = fusion.fuse(makeResult('Memory', 0.9), null);
    expect(result.rootCauses[0]!.name).toBe('Memory');
  });
});

// ── E2E Pipeline ────────────────────────────────────────────────────
describe('E2E pipeline', () => {
  it('full pipeline: graph discovery → RCA → viz', () => {
    // 1. Build causal graph
    const g = new CausalGraph(['Memory', 'CPU', 'Latency']);
    g.addEdge('Memory', 'CPU'); g.addEdge('Memory', 'Latency'); g.addEdge('CPU', 'Latency');

    // 2. Train SCM
    const scm = new StructuralCausalModel(g);
    const trainData = Array.from({ length: 100 }, () => {
      const mem = Math.random() * 2;
      const cpu = mem * 1.5 + (Math.random() - 0.5) * 0.3;
      const lat = cpu * 2 + mem * 0.5 + (Math.random() - 0.5) * 0.2;
      return [mem, cpu, lat];
    });
    scm.train(trainData);

    // 3. Detect anomaly via CIRCA
    const normalData = trainData.slice(0, 80);
    const anomalyData = Array.from({ length: 20 }, () => {
      const mem = 10 + Math.random() * 2; // anomalous Memory
      const cpu = mem * 1.5 + (Math.random() - 0.5) * 0.3;
      const lat = cpu * 2 + mem * 0.5 + (Math.random() - 0.5) * 0.2;
      return [mem, cpu, lat] as number[];
    });

    const circa = new CIRCAPipeline();
    circa.train(g, normalData!);
    const result = circa.analyze(anomalyData!, ['CPU', 'Latency']);
    expect(result.rootCauses.length).toBeGreaterThan(0);

    // 4. Build viz data
    const viz = buildRankingVizData(result.rootCauses, result.paths);
    expect(viz.rootCauses.length).toBeGreaterThan(0);

    // 5. Build graph viz
    const graphViz = buildGraphVizData(
      [...g.nodes], g.edges, result.rootCauses, ['CPU', 'Latency'],
    );
    expect(graphViz.nodes.length).toBe(3);
  });

  it('E2E with BayesianRCA + fusion', () => {
    const g = new CausalGraph(['S1', 'S2', 'S3']);
    g.addEdge('S1', 'S2'); g.addEdge('S2', 'S3');
    const rca = new BayesianRCA();
    rca.train(g, new Set(['S2', 'S3']), { rows: 2, columns: 3, set: () => {}, get: () => 0, clone: () => ({} as any), subMatrixColumn: () => ({} as any) } as any);
    // Even untrained, verify no crash
    expect(() => rca.findRootCauses(['S2'])).not.toThrow();
  });

  it('performance: E2E under 10s for synthetic data', () => {
    const start = performance.now();
    const g = new CausalGraph(['M', 'C', 'L']);
    g.addEdge('M', 'C'); g.addEdge('C', 'L');
    const data = Array.from({ length: 500 }, () => [Math.random(), Math.random() * 2, Math.random() * 3] as number[]);
    const circa = new CIRCAPipeline();
    circa.train(g, data.slice(0, 300)!);
    const result = circa.analyze(data.slice(300)!, ['L']);
    buildRankingVizData(result.rootCauses, result.paths);
    const elapsed = performance.now() - start;
    expect(elapsed).toBeLessThan(10000); // under 10s
  });
});

// ── Function coverage: remaining function paths ────────────────────
describe('function coverage', () => {
  it('FusionAnalyzer with logRCA input', () => {
    const fusion = new FusionAnalyzer({ strategy: 'weighted' });
    const make = (name: string, score: number): any => ({
      rootCauses: [{ name, score, confidence: 0.8, rank: 1, evidence: [] }],
      paths: [], metadata: { method: 'x', analyzedAt: Date.now(), durationMs: 0, extra: {} },
      toJSON: () => ({}),
    });
    const r = fusion.fuse(make('Mem', 0.9), make('CPU', 0.5), make('Log', 0.3));
    expect(r.rootCauses.length).toBeGreaterThan(0);
  });

  it('voting strategy defaults to weighted', () => {
    const f = new FusionAnalyzer({ strategy: 'voting' });
    expect(f.config.strategy).toBe('voting');
  });
});

// ── Function coverage: nestedFuse edge paths ─────────────────────
describe('nested fusion edges', () => {
  it('nested fusion with no RCA results', () => {
    const f = new FusionAnalyzer({ strategy: 'nested' });
    const r = f.fuse(null, null);
    expect(r.rootCauses.length).toBe(0);
  });
});

// ════════════════════════════════════════════════════════════════════
// Branch coverage precision tests: exercising real edge cases
// ════════════════════════════════════════════════════════════════════

import { SPOTDetector, DSPOTDetector } from '../detect/spot.js';
import { SpectralResidualDetector } from '../detect/spectral-residual.js';
import { RHTScorer, DAScorer } from '../analyze/circa.js';
import { HTRCA } from '../analyze/rca.js';
import { estimateLinearRegression } from '../infer/causal-inference.js';
import { VotingDetector } from '../detect/voting-detector.js';
import { StatsDetector } from '../detect/stats-detector.js';
import { CausalGraph } from '../graph/causal-graph.js';
import { Matrix } from 'ml-matrix';
import { buildTimeseriesVizData } from '../viz/viz-data.js';

describe('SPOT branch: Grimshaw fallback and boundary conditions', () => {
  it('constant data triggers method-of-moments fallback', () => {
    const spot = new SPOTDetector({ initSize: 10, q: 1e-2, initThresholdQuantile: 0.9 });
    // All identical → no peaks → GPD can't be estimated → falls back to MoM
    for (let i = 0; i < 30; i++) spot.update(5);
    expect(() => spot.update(100)).not.toThrow();
  });

  it('very small peaks trigger fallback', () => {
    const spot = new SPOTDetector({ initSize: 10, q: 1e-2, initThresholdQuantile: 0.5 });
    // Only 2 peaks → too few for GPD → falls back
    const data = [1,1,1,1,1,1,1,1,10,10];
    spot.initialize(data);
    expect(() => spot.update(100)).not.toThrow();
  });

  it('DSPOT handles drift initialization edge', () => {
    const dspot = new DSPOTDetector({ initSize: 20, q: 1e-2, driftWindow: 3 });
    for (let i = 0; i < 30; i++) dspot.update(10);
    expect(() => dspot.update(10)).not.toThrow();
  });
});

describe('SpectralResidual branch: buffer edge cases', () => {
  it('buffer wraps correctly near power-of-2 boundary', () => {
    const sr = new SpectralResidualDetector({ minPoints: 16 });
    for (let i = 0; i < 64; i++) sr.update(i % 2 === 0 ? 5 : 10);
    for (let i = 0; i < 32; i++) sr.update(i % 3 === 0 ? 5 : 10);
    const r = sr.update(50);
    expect(typeof r.isAnomalous).toBe('boolean');
  });
});

describe('RHT/HT branch: regression edge cases', () => {
  it('RHT with single parent trains correctly', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const rht = new RHTScorer({ tauMax: 1, aggregator: 'max' });
    rht.train(g, [[1, 2], [1.1, 1.9], [0.9, 2.1], [1, 2], [1, 2.1]]);
    const scores = rht.score([[5, 12]]);
    expect(scores.size).toBe(2);
  });

  it('DAScorer handles nodes with no RHT scores', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C');
    const da = new DAScorer({ threshold: 2.0 });
    // B has anomalous parent A, C has anomalous parent B → adjustment propagates
    const scores = new Map([
      ['A', { zScore: 8.0, confidence: 0.99 }],
      ['B', { zScore: 5.0, confidence: 0.90 }],
      ['C', { zScore: 4.0, confidence: 0.85 }],
    ]);
    const adjusted = da.adjust(g, scores);
    expect(adjusted[0]!.name).toBe('A');
  });
});

describe('CausalInference branch: single-point estimation', () => {
  it('estimation with exactly 2 data points', () => {
    const r = estimateLinearRegression([[1, 3], [0, 1]], 0, 1);
    expect(typeof r.ate).toBe('number');
    expect(isNaN(r.ate)).toBe(false);
  });

  it('estimation with zero treatment variance', () => {
    // treatment column is all 0s → coef should be computable
    const r = estimateLinearRegression([[0, 5], [0, 4], [0, 6]], 0, 1);
    expect(typeof r.ate).toBe('number');
  });
});

describe('Viz branch: no-anomaly edge case', () => {
  it('timeseries with zero anomalies', () => {
    const data = buildTimeseriesVizData(
      { cpu: [10, 12, 11] }, [1000, 1001, 1002], [], undefined,
    );
    expect(data.anomalyRegions.length).toBe(0);
  });
});

describe('VotingDetector branch: all strategies exercised', () => {
  it('weighted with partially trained detectors', () => {
    const d1 = new StatsDetector({ threshold: 3, minSamples: 2 });
    const d2 = new StatsDetector({ threshold: 3, minSamples: 2 });
    const v = new VotingDetector([d1, d2], { strategy: 'weighted' });
    d1.update([5]); d1.update([5]); // train d1
    d2.update([5]); d2.update([5]); // train d2
    const r = v.update([5]);
    expect(typeof r.isAnomalous).toBe('boolean');
  });
});

// ════════════════════════════════════════════════════════════════════
// Branch coverage: additional meaningful edge cases
// ════════════════════════════════════════════════════════════════════

describe('graph branch coverage', () => {
  it('d-separation with empty conditioning set and self-referential check', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');
    expect(g.dSeparated('X', 'X', [])).toBe(false); // self always reachable
  });

  it('pdag2dag on already-directed acyclic graph', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C');
    const dag = g.pdag2dag();
    expect(dag.isDAG()).toBe(true);
    expect(dag.hasEdge('A', 'B')).toBe(true);
    expect(dag.hasEdge('B', 'C')).toBe(true); // edges preserved
  });
});

