import { describe, it, expect } from 'vitest';
import { dagmaAlgorithm } from '../graph/dagma.js';
import { asiaGraph, generateLinearData } from '../benchmark.js';

describe('DAGMA Debug', () => {
  it('produces non-zero weights on ASIA', { timeout: 45000 }, () => {
    const truth = asiaGraph();
    const { data, nodeNames } = generateLinearData(truth, 5000, 42);
    const r = dagmaAlgorithm(data, nodeNames, { T: 2, warmIter: 500, maxIter: 1000 });
    const W = r.W; const d = nodeNames.length;
    let maxAbs = 0;
    for (let i = 0; i < d * d; i++) maxAbs = Math.max(maxAbs, Math.abs(W[i]!));
    console.log('Edges:', r.graph.edges.length, 'h:', r.h, 'max|W|:', maxAbs);
    // DAGMA should produce some non-zero weights
    expect(r.graph.edges.length + maxAbs).toBeGreaterThan(0);
  });

  it('produces edges with very low thresholds', { timeout: 45000 }, () => {
    const truth = asiaGraph();
    const { data, nodeNames } = generateLinearData(truth, 5000, 42);
    const r = dagmaAlgorithm(data, nodeNames, { lambda1: 0.0001, wThreshold: 0.01, T: 2, warmIter: 500, maxIter: 1000 });
    console.log('Low threshold — edges:', r.graph.edges.length, 'h:', r.h);
    expect(r.graph.edges.length).toBeGreaterThan(0);
  });
});
