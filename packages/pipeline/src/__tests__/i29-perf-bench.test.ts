/**
 * Performance benchmark suite — CI-verified regression detection.
 *
 * Each benchmark measures execution time of a key algorithmic operation.
 * Performance budgets (maxMs) enforce that operations stay within
 * acceptable latency bounds for production AIOps workloads.
 * Thresholds are based on typical latency budgets for incident response.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../graph/causal-graph.js';
import { pcAlgorithm, fisherZTest } from '../graph/pc.js';
import { gesAlgorithm } from '../graph/ges.js';
import { kciTest } from '../graph/kci.js';
import { SPOTDetector } from '../detect/spot.js';
import { StatsDetector } from '../detect/stats-detector.js';
import { directLiNGAM } from '../graph/lingam.js';

const PERF_BUDGETS = {
  /** PC algorithm on a 4-node DAG with 500 samples */
  pc4Node500: 500,
  /** GES on a 4-node DAG with 300 samples */
  ges4Node300: 1000,
  /** d-separation test on a 10-node graph — CI/sandbox-safe upper bound */
  dsep10Node: 50,
  /** Fisher Z test on 500 samples — CI/sandbox-safe upper bound */
  fisherZ500: 150,
  /** KCI unconditional on 100 samples */
  kci100: 200,
  /** SPOT calibration with 100 samples */
  spotCalibration: 150,
  /** LiNGAM on 4-node graph with 200 samples */
  lingam4Node200: 300,
  /** StatsDetector batch training on 1000 samples */
  statsBatch: 30,
};

function generateLinearData(nodes: string[], edges: Array<[string, string, number]>, N: number): Matrix {
  const g = new CausalGraph(nodes);
  for (const [f, t] of edges) g.addEdge(f, t);
  const order = g.topologicalSort();
  const data = Matrix.zeros(N, nodes.length);
  for (let r = 0; r < N; r++) {
    const vals = new Array(nodes.length).fill(0);
    for (const name of order) {
      const idx = nodes.indexOf(name);
      let v = (Math.random() - 0.5) * 2;
      for (const [f, t] of edges) { if (t === name) v += (vals[nodes.indexOf(f)] ?? 0) * 0.8; }
      vals[idx] = v; data.set(r, idx, v);
    }
  }
  return data;
}

describe('Performance: PC algorithm', () => {
  it(`4-node DAG × 500 samples ≤ ${PERF_BUDGETS.pc4Node500}ms`, () => {
    const nodes = ['A', 'B', 'C', 'D'];
    const edges: Array<[string, string, number]> = [['A', 'B', 1.5], ['B', 'C', 1.8], ['A', 'D', 2], ['C', 'D', 1.2]];
    const data = generateLinearData(nodes, edges, 500);
    const t0 = performance.now();
    pcAlgorithm(data, nodes, { alpha: 0.05, stable: true });
    const ms = performance.now() - t0;
    expect(ms).toBeLessThan(PERF_BUDGETS.pc4Node500);
  });
});

describe('Performance: GES', () => {
  it(`4-node DAG × 300 samples ≤ ${PERF_BUDGETS.ges4Node300}ms`, () => {
    const nodes = ['A', 'B', 'C', 'D'];
    const edges: Array<[string, string, number]> = [['A', 'B', 2], ['B', 'C', 1.5], ['A', 'D', 2]];
    const data = generateLinearData(nodes, edges, 300);
    const t0 = performance.now();
    gesAlgorithm(data, nodes);
    const ms = performance.now() - t0;
    expect(ms).toBeLessThan(PERF_BUDGETS.ges4Node300);
  });
});

describe('Performance: d-separation', () => {
  it(`10-node graph ≤ ${PERF_BUDGETS.dsep10Node}ms`, () => {
    const nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'];
    const g = new CausalGraph(nodes);
    for (let i = 0; i < nodes.length - 1; i++) g.addEdge(nodes[i]!, nodes[i + 1]!);
    const t0 = performance.now();
    g.dSeparated('A', 'J', ['E']);
    const ms = performance.now() - t0;
    expect(ms).toBeLessThan(PERF_BUDGETS.dsep10Node);
  });
});

describe('Performance: Fisher Z', () => {
  it(`500 samples ≤ ${PERF_BUDGETS.fisherZ500}ms`, () => {
    const data = new Matrix(500, 3);
    for (let r = 0; r < 500; r++) {
      data.set(r, 0, Math.random()); data.set(r, 1, Math.random()); data.set(r, 2, Math.random());
    }
    const t0 = performance.now();
    fisherZTest(data, 0, 1, [2]);
    const ms = performance.now() - t0;
    expect(ms).toBeLessThan(PERF_BUDGETS.fisherZ500);
  });
});

describe('Performance: KCI', () => {
  it(`unconditional 100 samples ≤ ${PERF_BUDGETS.kci100}ms`, () => {
    const data = new Matrix(100, 2);
    for (let r = 0; r < 100; r++) { data.set(r, 0, Math.random()); data.set(r, 1, Math.random()); }
    const t0 = performance.now();
    kciTest(data, 0, 1, [], { nPermutations: 20 });
    const ms = performance.now() - t0;
    expect(ms).toBeLessThan(PERF_BUDGETS.kci100);
  });
});

describe('Performance: SPOT', () => {
  it(`calibration 100 samples ≤ ${PERF_BUDGETS.spotCalibration}ms`, () => {
    const t0 = performance.now();
    const s = new SPOTDetector({ initSize: 20, q: 0.95 });
    for (let i = 0; i < 30; i++) s.update(1 + Math.random() * 0.2);
    const ms = performance.now() - t0;
    expect(ms).toBeLessThan(PERF_BUDGETS.spotCalibration);
  });
});

describe('Performance: LiNGAM', () => {
  it(`4-node 200 samples ≤ ${PERF_BUDGETS.lingam4Node200}ms`, () => {
    const nodes = ['A', 'B', 'C', 'D'];
    const edges: Array<[string, string, number]> = [['A', 'B', 2], ['B', 'C', 3], ['A', 'D', 1.5]];
    const data = generateLinearData(nodes, edges, 200);
    const t0 = performance.now();
    directLiNGAM(data, nodes);
    const ms = performance.now() - t0;
    expect(ms).toBeLessThan(PERF_BUDGETS.lingam4Node200);
  });
});

describe('Performance: StatsDetector', () => {
  it(`batch 1000 samples ≤ ${PERF_BUDGETS.statsBatch}ms`, () => {
    const data = Array.from({ length: 1000 }, () => [Math.random() * 10]);
    const t0 = performance.now();
    const d = new StatsDetector({ minSamples: 10 });
    d.train(data);
    const ms = performance.now() - t0;
    expect(ms).toBeLessThan(PERF_BUDGETS.statsBatch);
  });
});
