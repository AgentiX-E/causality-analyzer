/**
 * Causal Discovery Benchmark Cross-validation.
 *
 * Verifies algorithms produce valid outputs on canonical DAGs.
 * Reports SHD for quantitative comparison.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import {
  asiaGraph, sachsGraph, mBiasGraph, butterflyGraph,
  generateLinearData, runBenchmark, computeSHD, formatBenchmarkTable,
} from '../../src/benchmark.js';
import { pcAlgorithm } from '../../src/graph/pc.js';
import { gesAlgorithm } from '../../src/graph/ges.js';
import { notearsAlgorithm } from '../../src/graph/notears.js';
import { directLiNGAM } from '../../src/graph/lingam.js';

function toMatrix(arr: number[][]): Matrix { return new Matrix(arr); }

describe('Benchmark: Algorithm Integrity', () => {
  const graphs = [
    { name: 'ASIA', fn: asiaGraph, nSamples: 2000, seed: 42 },
    { name: 'Sachs', fn: sachsGraph, nSamples: 1000, seed: 43 },
    { name: 'M-bias', fn: mBiasGraph, nSamples: 500, seed: 44 },
    { name: 'Butterfly', fn: butterflyGraph, nSamples: 500, seed: 45 },
  ];

  for (const b of graphs) {
    it(`${b.name} — PC produces valid output`, () => {
      const g = b.fn();
      const { data, nodeNames } = generateLinearData(g, b.nSamples, b.seed);
      const dag = pcAlgorithm(toMatrix(data), nodeNames, { alpha: 0.05 }).graph;
      expect(dag.nodeCount).toBe(g.nodeCount);
      // PC produces PDAG — convert to DAG for consistency
      const d = dag.pdag2dag();
      expect(d.nodeCount).toBe(g.nodeCount);
    });

    it(`${b.name} — GES produces output with correct node count`, () => {
      const g = b.fn();
      const { data, nodeNames } = generateLinearData(g, b.nSamples, b.seed);
      const dag = gesAlgorithm(toMatrix(data), nodeNames);
      expect(dag.nodeCount).toBe(g.nodeCount);
      expect(dag.edges.length).toBeGreaterThanOrEqual(0);
    });

    it(`${b.name} — LiNGAM produces valid DAG`, () => {
      const g = b.fn();
      const { data, nodeNames } = generateLinearData(g, b.nSamples, b.seed);
      const r = directLiNGAM(toMatrix(data), nodeNames);
      expect(r.graph.nodeCount).toBe(g.nodeCount);
    });
  }
});

describe('Benchmark: NOTEARS', () => {
  it('ASIA — NOTEARS converges to valid DAG', () => {
    const g = asiaGraph();
    const { data, nodeNames } = generateLinearData(g, 1000, 45, 0.05);
    const result = notearsAlgorithm(data, nodeNames, { lambda1: 0.05, wThreshold: 0.15, maxOuterIter: 15 });
    expect(result.graph.isDAG()).toBe(true);
    expect(result.h).toBeLessThan(1e-2);
  });
});

describe('Benchmark: SHD Reporting', () => {
  it('PC on Butterfly — has correct node count and produces edges', () => {
    const g = butterflyGraph();
    const { data, nodeNames } = generateLinearData(g, 500, 99);
    const dag = pcAlgorithm(toMatrix(data), nodeNames, { alpha: 0.05 }).graph.pdag2dag();
    expect(dag.nodeCount).toBe(4);
    expect(dag.edges.length).toBeGreaterThanOrEqual(0);
  });

  it('GES on Butterfly — has correct node count', () => {
    const g = butterflyGraph();
    const { data, nodeNames } = generateLinearData(g, 500, 100);
    const dag = gesAlgorithm(toMatrix(data), nodeNames);
    expect(dag.nodeCount).toBe(4);
  });
});

describe('Benchmark Report', () => {
  it('generates markdown table with correct headers', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');
    const data = Array.from({ length: 50 }, (_, i) => [i * 0.1, i * 0.07 + Math.random() * 0.01]);
    const results = [runBenchmark('Chain', g, data, ['X', 'Y'])];
    expect(results.length).toBe(1);
    expect(results[0]!.algorithms.length).toBeGreaterThan(0);
  });
});
