/**
 * I5 tests: CIRCA algorithm (RHTScorer + DAScorer).
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../graph/causal-graph.js';
import { RHTScorer, DAScorer, CIRCAPipeline } from '../analyze/circa.js';

function smallGraph(): CausalGraph {
  // Memory → CPU → Latency (with Memory also directly → Latency)
  const g = new CausalGraph(['Memory', 'CPU', 'Latency']);
  g.addEdge('Memory', 'CPU'); g.addEdge('Memory', 'Latency'); g.addEdge('CPU', 'Latency');
  return g;
}

function generateData(graph: CausalGraph, nNormal: number, nAnomaly: number, rootCauseAnomaly: string): {
  normal: number[][]; anomaly: number[][];
} {
  const nodes = [...graph.nodes];
  const m = nodes.length;
  const order = graph.topologicalSort();

  function generate(n: number, injectAnomaly: boolean): number[][] {
    const data: number[][] = [];
    for (let r = 0; r < n; r++) {
      const values = new Array(m).fill(0);
      for (const name of order) {
        const idx = nodes.indexOf(name);
        let val = (Math.random() - 0.5) * 0.4; // noise
        for (const p of graph.parents(name)) {
          const pIdx = nodes.indexOf(p);
          val += values[pIdx]! * 1.5;
        }
        values[idx] = val;
      }
      // Inject anomaly at root cause
      if (injectAnomaly) {
        const rcIdx = nodes.indexOf(rootCauseAnomaly);
        values[rcIdx] += 8 + Math.random() * 2; // large mean shift
      }
      data.push(values);
    }
    return data;
  }

  return { normal: generate(nNormal, false), anomaly: generate(nAnomaly, true) };
}

// ── RHTScorer ──────────────────────────────────────────────────────
describe('RHTScorer', () => {
  it('trains regression models and produces z-scores', () => {
    const g = smallGraph();
    const { normal, anomaly } = generateData(g, 200, 50, 'Memory');
    const scorer = new RHTScorer({ tauMax: 3, aggregator: 'max' });
    scorer.train(g, normal);
    const scores = scorer.score(anomaly);
    expect(scores.size).toBe(3);
    // Memory (root cause) should have higher z-score
    const memScore = scores.get('Memory')!;
    expect(memScore.zScore).toBeGreaterThan(1);
  });

  it('returns empty map when not trained', () => {
    const scorer = new RHTScorer();
    expect(scorer.score([[1, 2, 3]]).size).toBe(0);
  });

  it('aggregates with mean method', () => {
    const g = smallGraph();
    const { normal, anomaly } = generateData(g, 100, 30, 'Memory');
    const scorer = new RHTScorer({ aggregator: 'mean' });
    scorer.train(g, normal);
    const scores = scorer.score(anomaly);
    expect(scores.size).toBeGreaterThan(0);
  });

  it('aggregates with sum method', () => {
    const g = smallGraph();
    const { normal, anomaly } = generateData(g, 80, 20, 'Memory');
    const scorer = new RHTScorer({ aggregator: 'sum' });
    scorer.train(g, normal);
    const scores = scorer.score(anomaly);
    expect(scores.size).toBeGreaterThan(0);
  });

  it('root node without parents gets valid model', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const normal = Array.from({ length: 50 }, () => [Math.random(), Math.random() * 2 + 1]);
    const anomaly = Array.from({ length: 10 }, () => [Math.random() + 5, Math.random() * 2 + 1]);
    const scorer = new RHTScorer();
    scorer.train(g, normal);
    const scores = scorer.score(anomaly);
    expect(scores.has('A')).toBe(true);
    expect(scores.has('B')).toBe(true);
  });
});

// ── DAScorer ──────────────────────────────────────────────────────
describe('DAScorer', () => {
  it('adjusts scores downward for nodes with anomalous parents', () => {
    const g = smallGraph();
    // Memory is root cause, CPU and Latency should get scores reduced
    const rhtScores = new Map<string, { zScore: number; confidence: number }>([
      ['Memory', { zScore: 8.0, confidence: 0.99 }],
      ['CPU', { zScore: 6.0, confidence: 0.95 }],
      ['Latency', { zScore: 5.0, confidence: 0.90 }],
    ]);
    const da = new DAScorer({ threshold: 3.0 });
    const adjusted = da.adjust(g, rhtScores);
    // Memory should rank highest after adjustment
    expect(adjusted[0]!.name).toBe('Memory');
    expect(adjusted.length).toBe(3);
    // Scores should be in descending order
    for (let i = 1; i < adjusted.length; i++) {
      expect(adjusted[i-1]!.score).toBeGreaterThanOrEqual(adjusted[i]!.score);
    }
  });

  it('disables adjustment when enabled=false', () => {
    const g = smallGraph();
    const rhtScores = new Map<string, { zScore: number; confidence: number }>([
      ['Memory', { zScore: 3.0, confidence: 0.8 }],
      ['CPU', { zScore: 6.0, confidence: 0.95 }],
      ['Latency', { zScore: 5.0, confidence: 0.90 }],
    ]);
    const da = new DAScorer({ enabled: false });
    const adjusted = da.adjust(g, rhtScores);
    // Without DA, highest z-score wins (CPU has 6.0)
    expect(adjusted[0]!.name).toBe('CPU');
  });

  it('handles empty scores gracefully', () => {
    const g = smallGraph();
    const da = new DAScorer();
    const adjusted = da.adjust(g, new Map());
    expect(adjusted.length).toBeGreaterThanOrEqual(0);
  });

  it('rank is sequential', () => {
    const g = smallGraph();
    const rhtScores = new Map<string, { zScore: number; confidence: number }>([
      ['Memory', { zScore: 8.0, confidence: 0.99 }],
      ['CPU', { zScore: 6.0, confidence: 0.95 }],
      ['Latency', { zScore: 5.0, confidence: 0.90 }],
    ]);
    const da = new DAScorer();
    const adjusted = da.adjust(g, rhtScores);
    for (let i = 0; i < adjusted.length; i++) {
      expect(adjusted[i]!.rank).toBe(i + 1);
    }
  });
});

// ── CIRCA Pipeline ─────────────────────────────────────────────────
describe('CIRCAPipeline', () => {
  it('end-to-end: identifies Memory as root cause', () => {
    const g = smallGraph();
    const { normal, anomaly } = generateData(g, 300, 60, 'Memory');
    const pipeline = new CIRCAPipeline({ tauMax: 3 }, { threshold: 3.0 });
    pipeline.train(g, normal);
    const result = pipeline.analyze(anomaly, ['CPU', 'Latency']);
    expect(result.rootCauses.length).toBeGreaterThan(0);
    // Top-1 recall: Memory should be top candidate
    expect(result.rootCauses[0]!.name).toBe('Memory');
  });

  it('returns empty result without training', () => {
    const pipeline = new CIRCAPipeline();
    const result = pipeline.analyze([[1, 2, 3]], ['A']);
    expect(result.rootCauses).toEqual([]);
  });

  it('builds propagation paths from root cause to anomalous nodes', () => {
    const g = smallGraph();
    const { normal, anomaly } = generateData(g, 200, 40, 'Memory');
    const pipeline = new CIRCAPipeline();
    pipeline.train(g, normal);
    const result = pipeline.analyze(anomaly, ['Latency']);
    expect(result.paths.length).toBeGreaterThan(0);
    // At least one path should start from a root cause
    expect(result.paths.some(p => g.parents(p.nodes[0]!).length === 0)).toBe(true);
  });

  it('multi-SLI: supports multiple anomalous SLIs', () => {
    const g = smallGraph();
    const { normal, anomaly } = generateData(g, 200, 40, 'CPU');
    const pipeline = new CIRCAPipeline();
    pipeline.train(g, normal);
    const result = pipeline.analyze(anomaly, ['CPU', 'Latency']);
    expect(result.rootCauses.length).toBeGreaterThan(0);
  });

  it('adaptive window: handles different failure durations', () => {
    const g = smallGraph();
    const { normal } = generateData(g, 100, 10, 'Memory');
    const pipeline = new CIRCAPipeline();
    pipeline.train(g, normal);
    // Test with short window (10 points) and long window (80 points)
    const shortAnomaly = Array.from({ length: 10 }, () => [Math.random() * 10, Math.random() * 2 + 1, Math.random() * 3 + 2]);
    const longAnomaly = Array.from({ length: 80 }, () => [Math.random() * 10, Math.random() * 2 + 1, Math.random() * 3 + 2]);
    const rShort = pipeline.analyze(shortAnomaly, ['CPU']);
    const rLong = pipeline.analyze(longAnomaly, ['CPU']);
    expect(rShort.rootCauses.length).toBeGreaterThan(0);
    expect(rLong.rootCauses.length).toBeGreaterThan(0);
  });
});
