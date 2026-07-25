/**
 * I78: Latent cluster discovery (TSC) tests.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { discoverClusters } from '../../src/graph/latent-clusters.js';

describe('discoverClusters (TSC)', () => {
  it('returns empty for small data (d < 4)', () => {
    const data = new Matrix([[1, 2, 3], [2, 4, 6]]);
    const result = discoverClusters(data);
    expect(result.nClusters).toBe(0);
    expect(result.assignments.size).toBe(0);
    expect(result.clusterSizes).toEqual([]);
  });

  it('returns empty for small samples (n < 10)', () => {
    const data = new Matrix([
      [1, 2, 3, 4],
      [2, 3, 4, 5],
    ]);
    const result = discoverClusters(data);
    expect(result.nClusters).toBe(0);
  });

  it('produces cluster result on sufficient data', () => {
    // Generate data with 4 variables that are partially correlated
    const rows: number[][] = [];
    for (let i = 0; i < 30; i++) {
      const base = Math.random();
      rows.push([
        base * 2 + Math.random(),
        base * 1.8 + Math.random(),
        base * 2.2 + Math.random(),
        Math.random() * 3,
      ]);
    }
    const data = new Matrix(rows);
    const result = discoverClusters(data);
    // Should produce some valid result
    expect(result.assignments).toBeInstanceOf(Map);
    expect(Array.isArray(result.clusterSizes)).toBe(true);
  });

  it('returns consistent cluster structure', () => {
    const rows: number[][] = [];
    for (let i = 0; i < 50; i++) {
      const latent = Math.sin(i * 0.1) + 1;
      rows.push([
        latent * 0.8 + (Math.random() - 0.5) * 0.5,
        latent * 1.2 + (Math.random() - 0.5) * 0.5,
        Math.random() * 5,
        Math.random() * 5,
      ]);
    }
    const data = new Matrix(rows);
    const result = discoverClusters(data, 0.1);
    expect(result.nClusters).toBeGreaterThanOrEqual(0);
    // All cluster assignments should be valid indices
    for (const [_, clusterId] of result.assignments) {
      expect(clusterId).toBeLessThan(result.nClusters);
      expect(clusterId).toBeGreaterThanOrEqual(0);
    }
  });

  it('cluster sizes sum matches assignments size', () => {
    const rows: number[][] = [];
    for (let i = 0; i < 40; i++) {
      const l1 = Math.random();
      rows.push([l1 * 2, l1 * 2.1 + 0.1, l1 * 1.9 - 0.1, Math.random() * 4, Math.random() * 4]);
    }
    const data = new Matrix(rows);
    const result = discoverClusters(data, 0.2);
    const totalSize = result.clusterSizes.reduce((s, v) => s + v, 0);
    expect(totalSize).toBe(result.assignments.size);
  });
});
