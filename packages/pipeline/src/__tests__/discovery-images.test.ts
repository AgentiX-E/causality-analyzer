/**
 * I77: IMaGES multi-dataset causal discovery tests.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { imagesAlgorithm } from '../../src/graph/images.js';

describe('IMaGES Algorithm', () => {
  it('returns empty graph for empty datasets', () => {
    const result = imagesAlgorithm([], ['X', 'Y', 'Z']);
    expect(result.nodes.length).toBe(3);
    expect(result.edges.length).toBe(0);
  });

  it('produces valid CPDAG on single dataset', () => {
    const data = new Matrix([
      [1, 2, 3],
      [2, 4, 6],
      [3, 6, 9],
      [4, 8, 12],
      [5, 10, 15],
    ]);
    const result = imagesAlgorithm([data], ['X', 'Y', 'Z']);
    expect(result.nodes.length).toBe(3);
    // Should produce a DAG (via pdag2dag)
    expect(result.isDAG()).toBe(true);
  });

  it('produces valid CPDAG on multiple datasets', () => {
    const d1 = new Matrix([
      [1, 2, 3],
      [2, 4, 6],
      [3, 6, 9],
    ]);
    const d2 = new Matrix([
      [1, 2.5, 3.1],
      [2, 4.1, 6.2],
      [3, 6.3, 9.1],
    ]);
    const result = imagesAlgorithm([d1, d2], ['X', 'Y', 'Z']);
    expect(result.nodes.length).toBe(3);
    expect(result.isDAG()).toBe(true);
  });

  it('respects maxDegree constraint', () => {
    const data = new Matrix([
      [1, 2, 3, 4],
      [2, 4, 6, 8],
      [3, 6, 9, 12],
      [4, 8, 12, 16],
      [5, 10, 15, 20],
    ]);
    const result = imagesAlgorithm([data], ['A', 'B', 'C', 'D'], { maxDegree: 1 });
    expect(result.isDAG()).toBe(true);
    // Each node should have at most 1 parent
    for (const node of result.nodes) {
      expect(result.parents(node).length).toBeLessThanOrEqual(1);
    }
  });

  it('handles domain knowledge constraints', () => {
    const data = new Matrix([
      [1, 2, 3],
      [2, 4, 6],
      [3, 6, 9],
    ]);
    const result = imagesAlgorithm([data], ['X', 'Y', 'Z'], {}, {
      forbids: [['Y', 'X']],
    });
    expect(result.isDAG()).toBe(true);
    expect(result.hasEdge('Y', 'X')).toBe(false);
  });

  it('produces consistent results across runs', () => {
    const data = new Matrix([
      [1, 2],
      [2, 4],
      [3, 6],
      [4, 8],
    ]);
    const r1 = imagesAlgorithm([data], ['A', 'B']);
    const r2 = imagesAlgorithm([data], ['A', 'B']);
    expect(r1.edges.length).toBe(r2.edges.length);
  });
});
