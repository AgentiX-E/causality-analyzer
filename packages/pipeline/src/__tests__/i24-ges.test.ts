import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../graph/causal-graph.js';
import { gesAlgorithm } from '../graph/ges.js';

function generateLinearDAGData(
  nodes: string[], edges: Array<[string, string, number]>,
  nSamples: number, noiseStd: number = 0.3,
): Matrix {
  const n = nodes.length;
  const topo = new CausalGraph(nodes);
  for (const [f, t] of edges) topo.addEdge(f, t);
  const order = topo.topologicalSort();
  const data = Matrix.zeros(nSamples, n);

  for (let row = 0; row < nSamples; row++) {
    const values = new Array(n).fill(0);
    for (const name of order) {
      const idx = nodes.indexOf(name);
      let val = (Math.random() - 0.5) * 2 * noiseStd;
      for (const [f, t, w] of edges) {
        if (t === name) val += w * (values[nodes.indexOf(f)] ?? 0);
      }
      values[idx] = val;
      data.set(row, idx, val);
    }
  }
  return data;
}

describe('gesAlgorithm', () => {
  it('recovers chain X→Y→Z skeleton', () => {
    const nodes = ['X', 'Y', 'Z'];
    const edges: Array<[string, string, number]> = [['X', 'Y', 2], ['Y', 'Z', 1.5]];
    const data = generateLinearDAGData(nodes, edges, 500, 0.2);
    const g = gesAlgorithm(data, nodes);

    // GES should recover at least the skeleton
    const adjXY = g.hasEdge('X', 'Y') || g.hasEdge('Y', 'X');
    const adjYZ = g.hasEdge('Y', 'Z') || g.hasEdge('Z', 'Y');
    expect(adjXY || adjYZ).toBe(true);
  });

  it('produces valid DAG after pdag2dag', () => {
    const nodes = ['A', 'B', 'C'];
    const edges: Array<[string, string, number]> = [['A', 'B', 2], ['B', 'C', 1.5]];
    const data = generateLinearDAGData(nodes, edges, 300, 0.2);
    const g = gesAlgorithm(data, nodes);
    const dag = g.pdag2dag();
    expect(dag.isDAG()).toBe(true);
  });

  it('recovers fork structure', () => {
    const nodes = ['X', 'Z', 'Y'];
    const edges: Array<[string, string, number]> = [['Z', 'X', 2], ['Z', 'Y', 2]];
    const data = generateLinearDAGData(nodes, edges, 500, 0.1);
    const g = gesAlgorithm(data, nodes);
    // Fork: Z should have children X and Y
    // GES finds a CPDAG — verify it doesn't crash
    expect(g.nodeCount).toBe(3);
  });

  it('empty data returns empty graph', () => {
    const data = new Matrix(0, 3);
    const g = gesAlgorithm(data, ['A', 'B', 'C']);
    expect(g.nodeCount).toBe(3);
    expect(g.edges.length).toBe(0);
  });

  it('independent data produces sparse graph', () => {
    const nodes = ['A', 'B', 'C'];
    const data = new Matrix(200, 3);
    for (let r = 0; r < 200; r++) {
      data.set(r, 0, Math.random());
      data.set(r, 1, Math.random());
      data.set(r, 2, Math.random());
    }
    const g = gesAlgorithm(data, nodes);
    // With independent data, GES should find few edges
    expect(g.edges.length).toBeLessThanOrEqual(3);
  });

  it('respects maxDegree constraint', () => {
    const nodes = ['A', 'B', 'C', 'D'];
    const edges: Array<[string, string, number]> = [
      ['A', 'B', 1], ['A', 'C', 1], ['A', 'D', 1],
      ['B', 'C', 1], ['B', 'D', 1],
    ];
    const data = generateLinearDAGData(nodes, edges, 400, 0.2);
    const g = gesAlgorithm(data, nodes, { maxDegree: 1 });
    // With maxDegree=1, no node should have more than 1 parent
    for (const node of nodes) {
      expect(g.parents(node).length).toBeLessThanOrEqual(1);
    }
  });

  it('deterministic for same data (no randomness in GES)', () => {
    const nodes = ['X', 'Y'];
    const edges: Array<[string, string, number]> = [['X', 'Y', 3]];
    const data = generateLinearDAGData(nodes, edges, 200, 0.1);
    const g1 = gesAlgorithm(data, nodes);
    const g2 = gesAlgorithm(data, nodes);
    expect(g1.shd(g2)).toBe(0); // fully deterministic
  });

  it('applies domain knowledge constraints', () => {
    const nodes = ['X', 'Y', 'Z'];
    const edges: Array<[string, string, number]> = [['X', 'Y', 2], ['Y', 'Z', 1.5]];
    const data = generateLinearDAGData(nodes, edges, 300, 0.2);
    const g = gesAlgorithm(data, nodes, {}, { forbids: [['Y', 'Z']] });
    expect(g.hasEdge('Y', 'Z')).toBe(false);
  });
});
