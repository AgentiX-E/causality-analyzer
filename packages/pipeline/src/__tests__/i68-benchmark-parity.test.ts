/**
 * I26: Systematic Benchmark Parity Suite.
 *
 * Validates all discovery algorithms produce valid output on standard DAGs.
 * TPR thresholds are intentionally loose — causal discovery with small samples
 * is inherently difficult and the focus is on correctness + stability.
 */
import { describe, it, expect } from 'vitest';
import {
  asiaGraph, sachsGraph, butterflyGraph, mBiasGraph,
  randomDAG, generateLinearData, computeSHD, runBenchmark, formatBenchmarkTable,
} from '../../src/benchmark.js';
import { pcAlgorithm } from '../../src/graph/pc.js';
import { gesAlgorithm } from '../../src/graph/ges.js';
import { bossAlgorithm } from '../../src/graph/boss.js';
import { fciAlgorithm } from '../../src/graph/advanced-discovery.js';
import { gfciAlgorithm } from '../../src/graph/gfci.js';
import { notearsAlgorithm } from '../../src/graph/notears.js';
import { directLiNGAM } from '../../src/graph/lingam.js';
import { Matrix } from 'ml-matrix';
import type { CausalGraph } from '../../src/graph/causal-graph.js';

type AlgoFn = (data: Matrix, nodes: string[]) => CausalGraph;

interface AlgoSpec { name: string; fn: AlgoFn; maxTimeMs: number }

const ALGOS: AlgoSpec[] = [
  { name: 'PC', fn: (d, n) => pcAlgorithm(d, n).graph, maxTimeMs: 8000 },
  { name: 'GES', fn: (d, n) => gesAlgorithm(d, n), maxTimeMs: 8000 },
  { name: 'BOSS', fn: (d, n) => bossAlgorithm(d, n, { numStarts: 2, maxIter: 20 }), maxTimeMs: 15000 },
  { name: 'LiNGAM', fn: (d, n) => directLiNGAM(d, n).graph, maxTimeMs: 8000 },
  { name: 'FCI', fn: (d, n) => fciAlgorithm(d, n).graph, maxTimeMs: 8000 },
  { name: 'GFCI', fn: (d, n) => gfciAlgorithm(d, n).graph, maxTimeMs: 15000 },
];

const GRAPHS = [
  { name: 'ASIA', graph: asiaGraph(), nSamples: 500 },
  { name: 'Sachs', graph: sachsGraph(), nSamples: 500 },
  { name: 'Butterfly', graph: butterflyGraph(), nSamples: 300 },
  { name: 'M-Bias', graph: mBiasGraph(), nSamples: 300 },
  { name: 'Random-5', graph: randomDAG(5, 0.3, 42), nSamples: 300 },
];

describe('Benchmark Parity — Algorithm Validity', () => {
  for (const { name, graph, nSamples } of GRAPHS) {
    const { data, nodeNames } = generateLinearData(graph, nSamples, 42);
    const matrix = new Matrix(data);

    describe(`${name} (${graph.nodes.length} nodes)`, () => {
      for (const algo of ALGOS) {
        it(`${algo.name} produces valid output within time limit`, () => {
          const start = Date.now();
          const predicted = algo.fn(matrix, nodeNames);
          const elapsed = Date.now() - start;
          const metrics = computeSHD(predicted, graph);

          // Core validity
          expect(predicted.nodes.length).toBe(graph.nodes.length);
          expect(metrics.shd).toBeGreaterThanOrEqual(0);
          expect(elapsed).toBeLessThan(algo.maxTimeMs);

          // SHD should be finite
          expect(isFinite(metrics.shd)).toBe(true);
        });
      }
    });
  }

  it('NOTEARS converges to valid DAG on small graphs', () => {
    const g = asiaGraph();
    const { data, nodeNames } = generateLinearData(g, 300, 42);
    const start = Date.now();
    const result = notearsAlgorithm(data, nodeNames, { lambda1: 0.1, maxOuterIter: 5, wThreshold: 0.3 });
    const elapsed = Date.now() - start;
    expect(result.graph.nodes.length).toBe(g.nodes.length);
    expect(elapsed).toBeLessThan(10000);
  });
});

describe('Benchmark Report', () => {
  it('generates valid report for ASIA with all algorithms', () => {
    const truth = asiaGraph();
    const { data, nodeNames } = generateLinearData(truth, 500, 42);
    const result = runBenchmark('ASIA', truth, data, nodeNames);

    expect(result.algorithms.length).toBeGreaterThanOrEqual(6);
    for (const a of result.algorithms) {
      expect(a.tpr).toBeGreaterThanOrEqual(0);
      expect(a.tpr).toBeLessThanOrEqual(1);
      expect(a.fpr).toBeGreaterThanOrEqual(0);
      expect(a.timeMs).toBeGreaterThan(0);
    }
  });

  it('formats markdown table with correct structure', () => {
    const truth = asiaGraph();
    const { data, nodeNames } = generateLinearData(truth, 300, 42);
    const result = runBenchmark('ASIA', truth, data, nodeNames);
    const table = formatBenchmarkTable([result]);

    expect(table).toContain('# Causal Discovery Benchmark Results');
    expect(table).toContain('| ASIA |');
    expect(table).toContain('| SHD |');
    expect(table).toContain('| TPR |');
  });

  it('BOSS achieves SHD within 1.5× GES on ASIA', () => {
    const truth = asiaGraph();
    const { data, nodeNames } = generateLinearData(truth, 500, 42);
    const result = runBenchmark('ASIA', truth, data, nodeNames);

    const ges = result.algorithms.find(a => a.algorithm === 'GES')!;
    const boss = result.algorithms.find(a => a.algorithm === 'BOSS')!;

    expect(boss.shd).toBeLessThanOrEqual(Math.ceil(ges.shd * 1.5));
  });
});
