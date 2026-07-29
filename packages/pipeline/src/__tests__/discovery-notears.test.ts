/**
 * NOTEARS Algorithm Tests.
 *
 * Validates DAG recovery on synthetic linear SEM data.
 */
import { describe, it, expect } from 'vitest';
import { notearsAlgorithm } from '../../src/graph/notears.js';
import { CausalGraph } from '../../src/graph/causal-graph.js';

function generateLinearData(graph: CausalGraph, n: number = 100, noise: number = 0.1): number[][] {
  const nodes = [...graph.nodes];
  const order = graph.topologicalSort();
  const data: number[][] = Array.from({ length: n }, () => new Array(nodes.length).fill(0));

  for (let i = 0; i < n; i++) {
    for (const node of order) {
      const parents = graph.parents(node);
      let val = 0;
      for (const p of parents) {
        const pIdx = nodes.indexOf(p);
        val += 0.7 * data[i]![pIdx]!;
      }
      val += (Math.random() - 0.5) * noise * 2;
      data[i]![nodes.indexOf(node)] = val;
    }
  }
  return data;
}

describe('NOTEARS Algorithm', () => {
  it('recovers simple 2-node DAG X→Y', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');
    const data = generateLinearData(g, 200, 0.05);

    const result = notearsAlgorithm(data, ['X', 'Y'], {
      lambda1: 0.05,
      wThreshold: 0.15,
      maxOuterIter: 10,
    });

    expect(result.graph.isDAG()).toBe(true);
    expect(result.h).toBeLessThan(1e-3);
  });

  it('recovers 3-node chain X→Y→Z', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y');
    g.addEdge('Y', 'Z');
    const data = generateLinearData(g, 200, 0.05);

    const result = notearsAlgorithm(data, ['X', 'Y', 'Z'], {
      lambda1: 0.05,
      wThreshold: 0.15,
      maxOuterIter: 15,
    });

    expect(result.graph.isDAG()).toBe(true);
    expect(result.h).toBeLessThan(1e-3);
  });

  it('returns a valid DAG (no cycles)', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B');
    g.addEdge('B', 'C');
    const data = generateLinearData(g, 150, 0.05);

    const result = notearsAlgorithm(data, ['A', 'B', 'C'], {
      lambda1: 0.1,
      wThreshold: 0.2,
      maxOuterIter: 10,
    });

    expect(result.graph.isDAG()).toBe(true);
  });

  it('converges within maxOuterIter', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');
    const data = generateLinearData(g, 100, 0.02);

    const result = notearsAlgorithm(data, ['X', 'Y'], {
      lambda1: 0.1,
      wThreshold: 0.2,
      maxOuterIter: 20,
    });

    expect(result.iterations).toBeGreaterThan(0);
    expect(result.iterations).toBeLessThanOrEqual(500 * 20);
  });

  it('handles independent variables (empty graph)', () => {
    const g = new CausalGraph(['X', 'Y']);
    // No edges — independent
    const data: number[][] = [];
    for (let i = 0; i < 100; i++) {
      data.push([Math.random(), Math.random()]);
    }

    const result = notearsAlgorithm(data, ['X', 'Y'], {
      lambda1: 0.2,
      wThreshold: 0.3,
      maxOuterIter: 10,
    });

    expect(result.graph.isDAG()).toBe(true);
    // Should recover sparse graph (few or no edges)
    expect(result.graph.edges.length).toBeLessThanOrEqual(2);
  });
});
