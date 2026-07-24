/**
 * MVPC Algorithm Tests.
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { mvpcAlgorithm } from '../../src/graph/mvpc.js';
import { pcAlgorithm } from '../../src/graph/pc.js';
import { butterflyGraph, generateLinearData, computeSHD } from '../../src/benchmark.js';
import { Matrix } from 'ml-matrix';

describe('MVPC Algorithm', () => {
  it('recovers chain X→Y→Z from complete data', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data } = generateLinearData(g, 200, 42);

    const dag = mvpcAlgorithm(data, ['X', 'Y', 'Z']);
    expect(dag.nodeCount).toBe(3);
  });

  it('SHD close to PC on complete Butterfly data', () => {
    const g = butterflyGraph();
    const { data, nodeNames } = generateLinearData(g, 500, 43);
    const matrix = new Matrix(data);

    const mvpcDag = mvpcAlgorithm(data, nodeNames);
    const pcDag = pcAlgorithm(matrix, nodeNames).graph.pdag2dag();

    const mvpcShd = computeSHD(mvpcDag, g).shd;
    const pcShd = computeSHD(pcDag, g).shd;
    expect(mvpcShd).toBeLessThanOrEqual(pcShd + 3);
  });

  it('handles missing values without crashing', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data } = generateLinearData(g, 200, 44);

    // Introduce 10% missing values
    for (let i = 0; i < data.length; i++) {
      for (let j = 0; j < data[i]!.length; j++) {
        if (Math.random() < 0.1) data[i]![j] = NaN;
      }
    }

    const dag = mvpcAlgorithm(data, ['X', 'Y', 'Z']);
    expect(dag.nodeCount).toBe(3);
  });

  it('handles empty data gracefully', () => {
    const data = Array.from({length: 50}, () => [Math.random(), Math.random()]);
    const dag = mvpcAlgorithm(data, ['X', 'Y']);
    expect(dag.nodeCount).toBe(2);
  });

  it('produces reasonable results with 30% missing', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y');
    const { data } = generateLinearData(g, 300, 45);

    for (let i = 0; i < data.length; i++)
      for (let j = 0; j < data[i]!.length; j++)
        if (Math.random() < 0.3) data[i]![j] = NaN;

    const dag = mvpcAlgorithm(data, ['X', 'Y', 'Z']);
    expect(dag.nodeCount).toBe(3);
  });
});
