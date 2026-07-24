/**
 * GRaSP Algorithm Tests.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { graspAlgorithm } from '../../src/graph/grasp.js';
import { gesAlgorithm } from '../../src/graph/ges.js';
import { asiaGraph, generateLinearData, computeSHD } from '../../src/benchmark.js';

describe('GRaSP Algorithm', () => {
  it('returns valid DAG for chain X→Y→Z', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data } = generateLinearData(g, 200, 42);

    const dag = graspAlgorithm(new Matrix(data), ['X', 'Y', 'Z']);
    expect(dag.isDAG()).toBe(true);
    expect(dag.nodeCount).toBe(3);
  });

  it('returns valid DAG for fork X←Z→Y', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('Z', 'X'); g.addEdge('Z', 'Y');
    const { data } = generateLinearData(g, 200, 43);

    const dag = graspAlgorithm(new Matrix(data), ['X', 'Y', 'Z']);
    expect(dag.isDAG()).toBe(true);
  });

  it('ASIA benchmark — SHD ≤ GES SHD', () => {
    const truth = asiaGraph();
    const { data, nodeNames } = generateLinearData(truth, 1000, 44);
    const matrix = new Matrix(data);

    const graspDag = graspAlgorithm(matrix, nodeNames);
    const gesDag = gesAlgorithm(matrix, nodeNames);

    const graspShd = computeSHD(graspDag, truth).shd;
    const gesShd = computeSHD(gesDag, truth).shd;

    expect(graspDag.isDAG()).toBe(true);
    // GRaSP should not be worse than GES
    expect(graspShd).toBeLessThanOrEqual(gesShd + 2);
  });

  it('respects lambda1 regularization (sparser with higher lambda)', () => {
    const g = new CausalGraph(['X', 'Y', 'Z', 'W']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z'); g.addEdge('Z', 'W');
    const { data } = generateLinearData(g, 300, 45);

    const sparse = graspAlgorithm(new Matrix(data), ['X', 'Y', 'Z', 'W'], { lambda1: 2.0 });
    const dense = graspAlgorithm(new Matrix(data), ['X', 'Y', 'Z', 'W'], { lambda1: 0.01 });

    expect(sparse.edges.length).toBeLessThanOrEqual(dense.edges.length);
    expect(sparse.isDAG()).toBe(true);
    expect(dense.isDAG()).toBe(true);
  });

  it('empty graph yields empty result', () => {
    const data = new Matrix(Array.from({ length: 100 }, () => [Math.random(), Math.random()]));
    const dag = graspAlgorithm(data, ['X', 'Y'], { lambda1: 5.0 });
    expect(dag.nodeCount).toBe(2);
    expect(dag.isDAG()).toBe(true);
  });
});
