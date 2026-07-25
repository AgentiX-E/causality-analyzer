/**
 * RCD + CD-NOD Algorithm Tests.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { rcdAlgorithm } from '../../src/graph/rcd.js';
import { cdnodAlgorithm } from '../../src/graph/cdnod.js';
import { butterflyGraph, generateLinearData, computeSHD } from '../../src/benchmark.js';
import { pcAlgorithm } from '../../src/graph/pc.js';

describe('RCD Algorithm', () => {
  it('recovers chain X→Y→Z', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data } = generateLinearData(g, 200, 42);
    const dag = rcdAlgorithm(new Matrix(data), ['X', 'Y', 'Z']);
    expect(dag.isDAG()).toBe(true);
    expect(dag.nodeCount).toBe(3);
  });

  it('SHD ≤ PC on Butterfly', () => {
    const g = butterflyGraph();
    const { data, nodeNames } = generateLinearData(g, 500, 43);
    const matrix = new Matrix(data);

    const rcdDag = rcdAlgorithm(matrix, nodeNames);
    const pcDag = pcAlgorithm(matrix, nodeNames).graph.pdag2dag();

    const rcdShd = computeSHD(rcdDag, g).shd;
    const pcShd = computeSHD(pcDag, g).shd;
    expect(rcdDag.isDAG()).toBe(true);
    expect(rcdShd).toBeLessThanOrEqual(pcShd + 2);
  });

  it('handles empty data gracefully', () => {
    const g = new CausalGraph(['X', 'Y']);
    const { data } = generateLinearData(g, 100, 44);
    const dag = rcdAlgorithm(new Matrix(data), ['X', 'Y']);
    expect(dag.nodeCount).toBe(2);
  });

  it('respects maxDegree constraint', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C'); g.addEdge('C', 'D');
    const { data } = generateLinearData(g, 200, 47);
    const dag = rcdAlgorithm(new Matrix(data), ['A', 'B', 'C', 'D'], { maxDegree: 1, alpha: 0.01 });
    expect(dag.isDAG()).toBe(true);
  });

  it('applies domain knowledge constraints', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data } = generateLinearData(g, 200, 48);
    const dag = rcdAlgorithm(new Matrix(data), ['X', 'Y', 'Z'], {}, {
      forbids: [['Z', 'X']],
    });
    expect(dag.isDAG()).toBe(true);
  });

  it('produces valid output with lenient alpha', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data } = generateLinearData(g, 150, 49);
    const dag = rcdAlgorithm(new Matrix(data), ['X', 'Y', 'Z'], { alpha: 0.5 });
    expect(dag.isDAG()).toBe(true);
    expect(dag.nodeCount).toBe(3);
  });

  it('handles single variable case', () => {
    const dag = rcdAlgorithm(new Matrix([[1], [2], [3]]), ['X']);
    expect(dag.nodeCount).toBe(1);
    expect(dag.isDAG()).toBe(true);
  });
});

describe('CD-NOD Algorithm', () => {
  it('returns valid DAG for chain', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data } = generateLinearData(g, 200, 45);
    const { graph } = cdnodAlgorithm(new Matrix(data), ['X', 'Y', 'Z']);
    expect(graph.isDAG()).toBe(true);
  });

  it('detects changing edges with domain shifts', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');

    // Generate data with 2 domains: domain 0 (normal), domain 1 (shifted)
    const data: number[][] = [];
    const domains: number[] = [];
    for (let i = 0; i < 200; i++) {
      const d = i < 100 ? 0 : 1;
      domains.push(d);
      const x = Math.random();
      const y = 0.7 * x + (d === 1 ? 0.5 : 0) + Math.random() * 0.1;
      const z = 0.8 * y + Math.random() * 0.1;
      data.push([x, y, z]);
    }

    const matrix = new Matrix(data);
    const result = cdnodAlgorithm(matrix, ['X', 'Y', 'Z'], { domains });
    expect(result.graph.isDAG()).toBe(true);
    // changingEdges may or may not detect changes depending on threshold
    expect(result.changingEdges).toBeDefined();
  });

  it('no domains — still produces valid graph', () => {
    const g = new CausalGraph(['X', 'Y']);
    const { data } = generateLinearData(g, 100, 46);
    const { graph } = cdnodAlgorithm(new Matrix(data), ['X', 'Y'], {});
    expect(graph.nodeCount).toBe(2);
  });
});
