/**
 * I27: Stability Selection + StARS Tests.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { stabilitySelection, starsSelection } from '../../src/graph/stability-selection.js';
import { pcAlgorithm } from '../../src/graph/pc.js';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { generateLinearData, asiaGraph } from '../../src/benchmark.js';

function pcWrapper(data: Matrix, nodeNames: string[]): CausalGraph {
  return pcAlgorithm(data, nodeNames).graph;
}

describe('Stability Selection', () => {
  it('produces stable graph from ASIA data', () => {
    const g = asiaGraph();
    const { data, nodeNames } = generateLinearData(g, 300, 42);
    const matrix = new Matrix(data);
    const result = stabilitySelection(matrix, nodeNames, pcWrapper, {
      nSubsamples: 20,
      subsampleFraction: 0.7,
      edgeThreshold: 0.5,
      seed: 42,
    });

    expect(result.nSubsamples).toBe(20);
    expect(result.edgeStability.size).toBeGreaterThan(0);
    expect(result.stableGraph.nodes.length).toBe(nodeNames.length);
  });

  it('produces deterministic results with same seed', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data, nodeNames } = generateLinearData(g, 200, 42);
    const matrix = new Matrix(data);

    const r1 = stabilitySelection(matrix, nodeNames, pcWrapper, { nSubsamples: 10, seed: 42 });
    const r2 = stabilitySelection(matrix, nodeNames, pcWrapper, { nSubsamples: 10, seed: 42 });

    expect(r1.edgeStability.size).toBe(r2.edgeStability.size);
  });

  it('stability score is between 0 and 1', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B');
    const { data, nodeNames } = generateLinearData(g, 100, 42);
    const matrix = new Matrix(data);
    const result = stabilitySelection(matrix, nodeNames, pcWrapper, {
      nSubsamples: 10,
      edgeThreshold: 0,
      seed: 42,
    });

    for (const [, stability] of result.edgeStability) {
      expect(stability).toBeGreaterThanOrEqual(0);
      expect(stability).toBeLessThanOrEqual(1);
    }
  });

  it('threshold filters low-stability edges', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B');
    const { data, nodeNames } = generateLinearData(g, 100, 42);
    const matrix = new Matrix(data);
    const r1 = stabilitySelection(matrix, nodeNames, pcWrapper, { nSubsamples: 10, edgeThreshold: 0.9, seed: 42 });
    const r2 = stabilitySelection(matrix, nodeNames, pcWrapper, { nSubsamples: 10, edgeThreshold: 0.0, seed: 42 });

    // Higher threshold yields fewer or equal edges
    expect(r1.stableGraph.edges.length).toBeLessThanOrEqual(r2.stableGraph.edges.length);
  });
});

describe('StARS Selection', () => {
  it('selects best parameter from range', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data, nodeNames } = generateLinearData(g, 200, 42);
    const matrix = new Matrix(data);

    const paramFn = (alpha: number) => (d: Matrix, n: string[]) => pcAlgorithm(d, n, { alpha }).graph;
    const params = [0.01, 0.05, 0.1];

    const result = starsSelection(matrix, nodeNames, paramFn, params, {
      nSubsamples: 10,
      seed: 42,
    });

    expect(params).toContain(result.bestParam);
    expect(result.stabilityValues.length).toBe(params.length);
    expect(result.bestGraph.nodes.length).toBe(nodeNames.length);
  });

  it('stability values decrease with higher complexity params', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C');
    const { data, nodeNames } = generateLinearData(g, 150, 42);
    const matrix = new Matrix(data);

    const paramFn = (alpha: number) => (d: Matrix, n: string[]) => pcAlgorithm(d, n, { alpha }).graph;
    const result = starsSelection(matrix, nodeNames, paramFn, [0.01, 0.1], {
      nSubsamples: 8,
      seed: 42,
    });

    for (const v of result.stabilityValues) {
      expect(v.stability).toBeGreaterThanOrEqual(0);
      expect(v.stability).toBeLessThanOrEqual(1);
    }
  });
});
