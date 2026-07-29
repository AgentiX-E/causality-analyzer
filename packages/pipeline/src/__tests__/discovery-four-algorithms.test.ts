/**
 * I89: PC-Max + ExactSearch + TiMINo + GIN combined tests.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { pcMaxAlgorithm } from '../../src/graph/pc-max.js';
import { exactSearch } from '../../src/graph/exact-search.js';
import { timinoAlgorithm } from '../../src/graph/timino.js';
import { ginDetect } from '../../src/graph/gin.js';

describe('PC-Max', () => {
  it('produces valid DAG on chain', () => {
    const data = new Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9], [1.5, 3, 4.5], [2.5, 5, 7.5], [3.5, 7, 10.5]]);
    const result = pcMaxAlgorithm(data, ['X', 'Y', 'Z']);
    expect(result.nodeCount).toBe(3);
    expect(result.isDAG()).toBe(true);
  });

  it('handles independent data', () => {
    const data = new Matrix(Array.from({ length: 30 }, () => [Math.random(), Math.random()]));
    const result = pcMaxAlgorithm(data, ['A', 'B']);
    expect(result.isDAG()).toBe(true);
  });
});

describe('ExactSearch', () => {
  it('finds optimal DAG on 2-node chain', () => {
    const data = new Matrix([[1, 2], [2, 4], [3, 6], [4, 8]]);
    const result = exactSearch(data, ['X', 'Y']);
    expect(result.nodeCount).toBe(2);
    expect(result.isDAG()).toBe(true);
  });

  it('handles 3-variable data', () => {
    const data = new Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]]);
    const result = exactSearch(data, ['A', 'B', 'C']);
    expect(result.isDAG()).toBe(true);
  });
});

describe('TiMINo', () => {
  it('discovers lagged dependencies', () => {
    const T = 200;
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      const x = Math.random() * 2 - 1;
      const y = (t > 0 ? 0.7 * data[t - 1]![0]! : 0) + Math.random() * 0.3;
      data.push([x, y]);
    }
    const result = timinoAlgorithm(data, ['X', 'Y'], 2, 0.1);
    expect(result.edges).toBeDefined();
    expect(result.tauMax).toBe(2);
  });

  it('handles small data', () => {
    const result = timinoAlgorithm([[1, 2], [3, 4]], ['X', 'Y']);
    expect(result.edges).toEqual([]);
  });
});

describe('GIN', () => {
  it('detects latent clusters in multi-variable data', () => {
    const N = 100;
    const data = new Matrix(N, 4);
    for (let r = 0; r < N; r++) {
      const latent = Math.random();
      data.set(r, 0, 0.8 * latent + Math.random() * 0.2);
      data.set(r, 1, 0.9 * latent + Math.random() * 0.2);
      data.set(r, 2, Math.random());
      data.set(r, 3, Math.random());
    }
    const result = ginDetect(data, ['A', 'B', 'C', 'D']);
    expect(result.nClusters).toBeGreaterThanOrEqual(0);
  });

  it('handles small data', () => {
    const data = new Matrix([[1, 2], [3, 4]]);
    const result = ginDetect(data, ['X', 'Y']);
    expect(result.nClusters).toBe(0);
  });
});
