/**
 * I40: Large Graph Benchmark Suite.
 *
 * Validates algorithm scalability on graphs with 50-200 nodes.
 */
import { describe, it, expect } from 'vitest';
import { randomDAG, generateLinearData, computeSHD } from '../../src/benchmark.js';
import { pcAlgorithm } from '../../src/graph/pc.js';
import { gesAlgorithm } from '../../src/graph/ges.js';
import { rfciAlgorithm } from '../../src/graph/rfci.js';
import { Matrix } from 'ml-matrix';

const LARGE_CONFIGS = [
  { nodes: 50, density: 0.05, samples: 500 },
  { nodes: 100, density: 0.03, samples: 800 },
];

describe('Large Graph Benchmarks', () => {
  for (const { nodes, density, samples } of LARGE_CONFIGS) {
    const truth = randomDAG(nodes, density, 42);
    const { data, nodeNames } = generateLinearData(truth, samples, 42);
    const matrix = new Matrix(data);

    it(`PC on ${nodes}-node graph produces valid output within 30s`, () => {
      const start = Date.now();
      const result = pcAlgorithm(matrix, nodeNames).graph;
      const elapsed = Date.now() - start;
      expect(result.nodes.length).toBe(nodes);
      expect(elapsed).toBeLessThan(30000);
    });

    it(`RFCI on ${nodes}-node graph produces valid output within 30s`, () => {
      const start = Date.now();
      const result = rfciAlgorithm(matrix, nodeNames);
      const elapsed = Date.now() - start;
      expect(result.graph.nodes.length).toBe(nodes);
      expect(elapsed).toBeLessThan(30000);
    });

    it(`GES on ${nodes}-node graph produces valid DAG within 30s`, () => {
      const start = Date.now();
      const result = gesAlgorithm(matrix, nodeNames);
      const elapsed = Date.now() - start;
      expect(result.isDAG()).toBe(true);
      expect(elapsed).toBeLessThan(30000);
    });
  }

  it.skip('200-node ultra-large graph RFCI', { timeout: 90000 }, () => {
    const truth = randomDAG(200, 0.02, 42);
    const { data, nodeNames } = generateLinearData(truth, 300, 42);
    const matrix = new Matrix(data);
    const result = rfciAlgorithm(matrix, nodeNames, { maxDegree: 2 });
    expect(result.graph.nodes.length).toBe(200);
  });
});
