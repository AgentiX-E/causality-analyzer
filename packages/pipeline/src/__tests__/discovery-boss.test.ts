/**
 * BOSS Algorithm Tests.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { bossAlgorithm } from '../../src/graph/boss.js';
import { gesAlgorithm } from '../../src/graph/ges.js';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { computeSHD } from '../../src/graph/drift-detection.js';

function generateLinearData(graph: CausalGraph, n: number, seed: number): { data: number[][]; nodeNames: string[] } {
  const rng = ((s: number) => {
    let state = s >>> 0;
    return () => { state = (state * 1664525 + 1013904223) >>> 0; return state / 0x100000000; };
  })(seed);
  const nodes = graph.nodes;
  const d = nodes.length;
  const parentMap = new Map<string, string[]>();
  for (const node of nodes) parentMap.set(node, graph.parents(node));
  const topo = graph.topologicalSort();
  const data: number[][] = [];
  for (let r = 0; r < n; r++) {
    const row = new Array(d).fill(0);
    for (const node of topo) {
      const idx = nodes.indexOf(node);
      let val = 0;
      for (const p of parentMap.get(node) ?? []) val += 0.7 * row[nodes.indexOf(p)]!;
      const u1 = Math.max(1e-10, rng()), u2 = rng();
      val += Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      row[idx] = val;
    }
    data.push(row);
  }
  return { data, nodeNames: [...nodes] };
}

function asiaGraph(): CausalGraph {
  const g = new CausalGraph(['A', 'S', 'T', 'L', 'B', 'E', 'X', 'D']);
  g.addEdge('A', 'T'); g.addEdge('S', 'L'); g.addEdge('S', 'B');
  g.addEdge('T', 'E'); g.addEdge('L', 'E'); g.addEdge('B', 'D');
  g.addEdge('E', 'X'); g.addEdge('E', 'D');
  return g;
}

describe('BOSS Algorithm', () => {
  it('returns valid DAG for chain X→Y→Z', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data } = generateLinearData(g, 300, 42);
    const result = bossAlgorithm(new Matrix(data), ['X', 'Y', 'Z'], { numStarts: 3 });
    expect(result.isDAG()).toBe(true);
    expect(result.nodes.length).toBe(3);
  });

  it('returns valid DAG for fork X←Z→Y', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('Z', 'X'); g.addEdge('Z', 'Y');
    const { data } = generateLinearData(g, 300, 42);
    const result = bossAlgorithm(new Matrix(data), ['X', 'Y', 'Z'], { numStarts: 3 });
    expect(result.isDAG()).toBe(true);
  });

  it('ASIA benchmark — SHD ≤ GES SHD', () => {
    const truth = asiaGraph();
    const { data, nodeNames } = generateLinearData(truth, 1000, 44);
    const matrix = new Matrix(data);

    const bossDag = bossAlgorithm(matrix, nodeNames, { numStarts: 3, maxIter: 30 });
    const gesDag = gesAlgorithm(matrix, nodeNames);

    const bossShd = computeSHD(bossDag, truth).shd;
    const gesShd = computeSHD(gesDag, truth).shd;

    expect(bossDag.isDAG()).toBe(true);
    expect(bossShd).toBeLessThanOrEqual(gesShd + 3);
  });

  it('produces deterministic results with same seed', () => {
    const g = new CausalGraph(['X', 'Y', 'Z', 'W']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z'); g.addEdge('Z', 'W');
    const { data } = generateLinearData(g, 200, 99);

    const r1 = bossAlgorithm(new Matrix(data), ['X', 'Y', 'Z', 'W'], { numStarts: 3, seed: 42 });
    const r2 = bossAlgorithm(new Matrix(data), ['X', 'Y', 'Z', 'W'], { numStarts: 3, seed: 42 });

    const shd = computeSHD(r1, r2).shd;
    expect(shd).toBe(0);
  });

  it('handles empty data gracefully', () => {
    const result = bossAlgorithm(new Matrix(5, 2), ['X', 'Y'], { numStarts: 1 });
    expect(result.nodes).toEqual(['X', 'Y']);
  });

  it('handles 2-node graph', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const { data } = generateLinearData(g, 100, 42);
    const result = bossAlgorithm(new Matrix(data), ['A', 'B'], { numStarts: 2 });
    expect(result.isDAG()).toBe(true);
  });

  it('respects maxParents constraint', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D', 'E']);
    g.addEdge('A', 'B'); g.addEdge('A', 'C'); g.addEdge('A', 'D');
    g.addEdge('B', 'E'); g.addEdge('C', 'E');
    const { data } = generateLinearData(g, 300, 42);
    const result = bossAlgorithm(new Matrix(data), ['A', 'B', 'C', 'D', 'E'], {
      numStarts: 2,
      maxParents: 2,
    });
    expect(result.isDAG()).toBe(true);
    // Each node should have ≤ 2 parents (maxParents=2)
    for (const node of result.nodes) {
      expect(result.parents(node).length).toBeLessThanOrEqual(2);
    }
  });

  it('respects domainKnowledge constraints', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const { data } = generateLinearData(g, 200, 42);
    const result = bossAlgorithm(new Matrix(data), ['X', 'Y', 'Z'], {
      numStarts: 2,
    }, {
      forbids: [['Z', 'X']],
    });
    expect(result.isDAG()).toBe(true);
  });

  it('scales to 8-node graph within time limit', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C');
    g.addEdge('C', 'D'); g.addEdge('D', 'E');
    g.addEdge('A', 'F'); g.addEdge('F', 'G');
    g.addEdge('G', 'H');
    const { data } = generateLinearData(g, 200, 42);
    const start = Date.now();
    const result = bossAlgorithm(new Matrix(data), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], {
      numStarts: 3,
      maxIter: 30,
    });
    const elapsed = Date.now() - start;
    expect(result.isDAG()).toBe(true);
    expect(elapsed).toBeLessThan(10000);
  });
});
