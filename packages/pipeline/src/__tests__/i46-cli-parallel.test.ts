/**
 * I6-I7 conformance: Chunked PC + Parallel infrastructure + CLI benchmark.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { chunkedPC } from '../parallel.js';

describe('Chunked PC', () => {
  it('handles small data directly', async () => {
    const nodes = ['A', 'B', 'C'];
    const data = new Matrix(100, 3);
    for (let r = 0; r < 100; r++) {
      data.set(r, 0, Math.random() * 5);
      data.set(r, 1, 2 * data.get(r, 0) + (Math.random() - 0.5));
      data.set(r, 2, (Math.random() - 0.5) * 3);
    }
    const result = await chunkedPC(data, nodes, 200, 50);
    expect(result.graph.nodes.length).toBe(3);
    expect(result.convergence).toBe(1);
  });

  it('handles chunked data with voting', async () => {
    const N = 600, d = 4;
    const nodes = ['W', 'X', 'Y', 'Z'];
    const data = new Matrix(N, d);
    for (let r = 0; r < N; r++) {
      const w = Math.random() * 5;
      data.set(r, 0, w);
      data.set(r, 1, 1.5 * w + (Math.random() - 0.5) * 0.3);
      data.set(r, 2, 2 * data.get(r, 1) + (Math.random() - 0.5) * 0.2);
      data.set(r, 3, (Math.random() - 0.5) * 2);
    }
    const result = await chunkedPC(data, nodes, 200, 50, 0.05);
    expect(result.graph.nodes.length).toBe(4);
    expect(result.graph.edges.length).toBeGreaterThan(0);
  });

  it('convergence ratio is between 0 and 1', async () => {
    const nodes = ['A', 'B'];
    const data = new Matrix(500, 2);
    for (let r = 0; r < 500; r++) {
      data.set(r, 0, Math.random() * 5);
      data.set(r, 1, (Math.random() - 0.5) * 3);
    }
    const result = await chunkedPC(data, nodes, 200, 80);
    expect(result.convergence).toBeGreaterThan(0);
    expect(result.convergence).toBeLessThanOrEqual(1);
  });
});

describe('Parallel infrastructure exports', () => {
  it('parallelMap is a function', async () => {
    const { parallelMap } = await import('../parallel.js');
    expect(typeof parallelMap).toBe('function');
  });

  it('chunkedPC is a function', async () => {
    const { chunkedPC } = await import('../parallel.js');
    expect(typeof chunkedPC).toBe('function');
  });
});
