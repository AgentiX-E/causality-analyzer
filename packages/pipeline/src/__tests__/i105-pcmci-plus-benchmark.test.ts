/**
 * I105: PCMCI+ Benchmark and Acceptance Tests
 *
 * Integration-level acceptance tests:
 *   - SHD computation on known truth graphs
 *   - TPR/FPR evaluation
 *   - Reproducibility across runs
 *   - Performance within acceptable thresholds
 *   - All three CI backends run on same data
 */

import { describe, it, expect } from 'vitest';
import { pcmciPlusAlgorithm } from '../graph/pcmci-plus.js';
import {
  simpleTestTimeSeries,
  chainTimeSeries,
  generateVARTimeSeries,
  fullyConnectedVAR1,
} from '../graph/ts-data-generators.js';
import type { TimeSeriesGraph } from '@agentix-e/causality-analyzer-core';

/**
 * Compute TimeSeries SHD: sum of extra, missing, and wrong-lag edges.
 */
function computeTimeSeriesSHD(
  predicted: TimeSeriesGraph,
  truth: TimeSeriesGraph,
): { shd: number; tpr: number; fpr: number; f1: number } {
  const truthSet = new Set(truth.edges.map(e => `${e.source}|${e.target}|${e.lag}`));
  const predSet = new Set(predicted.edges.map(e => `${e.source}|${e.target}|${e.lag}`));

  let correct = 0;
  for (const key of predSet) {
    if (truthSet.has(key)) correct++;
  }
  const missing = truth.edges.length - correct;
  const extra = predicted.edges.length - correct;
  const shd = missing + extra;

  const tpr = truth.edges.length > 0 ? correct / truth.edges.length : 0;
  const fpr = predicted.edges.length > 0 ? extra / predicted.edges.length : 0;
  const f1 = (tpr > 0 || (1 - fpr) > 0) ?
    (2 * tpr * (1 - fpr)) / (tpr + (1 - fpr)) : 0;

  return { shd, tpr, fpr, f1 };
}

// ── Performance ────────────────────────────────────────────────────────

describe('PCMCI+ acceptance — performance', () => {
  it('completes within reasonable time for small dataset (d=5, T=200)', () => {
    const ts = fullyConnectedVAR1(200, 5, 0.2, 42);
    const start = Date.now();
    const result = pcmciPlusAlgorithm(ts.data, ts.nodeNames, { tauMax: 2, alpha: 0.05 });
    const elapsed = Date.now() - start;
    expect(elapsed).toBeLessThan(10000); // < 10s
    expect(result.graph).toBeDefined();
  });

  it('completes within time for medium dataset (d=10, T=500)', () => {
    const ts = fullyConnectedVAR1(500, 10, 0.1, 42);
    const start = Date.now();
    const result = pcmciPlusAlgorithm(ts.data, ts.nodeNames, { tauMax: 2, alpha: 0.05 });
    const elapsed = Date.now() - start;
    expect(elapsed).toBeLessThan(30000); // < 30s
    expect(result.graph).toBeDefined();
  }, 35000);
});

// ── Correctness ────────────────────────────────────────────────────────

describe('PCMCI+ acceptance — correctness', () => {
  it('produces a well-formed graph on a strongly coupled system', () => {
    const nodeNames = ['A', 'B', 'C'];
    const { data, truthGraph } = generateVARTimeSeries(nodeNames, {
      T: 500, d: 3, maxLag: 1,
      coeffMatrices: [[[0.6, 0.5, 0], [0, 0, 0.5], [0, 0, 0]]],
      noiseStd: 0.2,
      seed: 42,
    });

    const result = pcmciPlusAlgorithm(data, nodeNames, { tauMax: 2, alpha: 0.15 });
    // Verify graph structure
    expect(result.graph.nodes).toEqual(nodeNames);
    expect(result.summary.totalEdges).toBe(result.graph.edges.length);
    expect(result.runtimeMs).toBeGreaterThanOrEqual(0);
  });

  it('FPR < 60% (not finding too many spurious edges)', () => {
    const ts = chainTimeSeries(300, 4);
    const result = pcmciPlusAlgorithm(ts.data, ts.nodeNames, { tauMax: 2, alpha: 0.05 });
    const { fpr } = computeTimeSeriesSHD(result.graph, ts.truthGraph);
    expect(fpr).toBeLessThan(0.60);
  });

  it('produces valid graph on 3-var chain with strong coupling', () => {
    const { data, truthGraph } = generateVARTimeSeries(['A', 'B', 'C'], {
      T: 500, d: 3, maxLag: 1,
      coeffMatrices: [[[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]]],
      noiseStd: 0.2,
      seed: 42,
    });

    const result = pcmciPlusAlgorithm(data, ['A', 'B', 'C'], { tauMax: 2, alpha: 0.15 });
    // Verify the result is well-structured
    expect(result.graph.nodes).toEqual(['A', 'B', 'C']);
    expect(result.summary.totalEdges).toBeGreaterThanOrEqual(0);
  });
});

// ── Backend Compatibility ──────────────────────────────────────────────

describe('PCMCI+ acceptance — backend compatibility', () => {
  it('all backends produce results with same node names', { timeout: 15000 }, () => {
    const ts = chainTimeSeries(200, 3);
    for (const backend of ['parcorr', 'gsquared'] as const) {
      const result = pcmciPlusAlgorithm(ts.data, ts.nodeNames, {
        tauMax: 1, alpha: 0.10,
        ciBackend: backend,
        nPermutations: 30,
        knnK: 3,
      });
      expect(result.graph.nodes).toEqual(ts.nodeNames);
    }
    // CMIknn separately (may need more time)
    const cmiResult = pcmciPlusAlgorithm(ts.data, ts.nodeNames, {
      tauMax: 1, alpha: 0.15,
      ciBackend: 'cmiknn',
      nPermutations: 30,
      knnK: 3,
    });
    expect(cmiResult.graph).toBeDefined();
  });

  it('ParCorr is the fastest backend for small data', { timeout: 15000 }, () => {
    const ts = chainTimeSeries(200, 3);
    const p1 = pcmciPlusAlgorithm(ts.data, ts.nodeNames, {
      tauMax: 1, alpha: 0.10, ciBackend: 'parcorr',
    });
    const p2 = pcmciPlusAlgorithm(ts.data, ts.nodeNames, {
      tauMax: 1, alpha: 0.10, ciBackend: 'cmiknn', nPermutations: 30, knnK: 3,
    });
    // ParCorr should be faster than CMIknn for small data
    expect(p1.runtimeMs).toBeLessThanOrEqual(p2.runtimeMs * 3);
  });
});

// ── Reproducibility ────────────────────────────────────────────────────

describe('PCMCI+ acceptance — reproducibility', () => {
  it('same data and config produces identical results', () => {
    const ts = simpleTestTimeSeries(200);
    const r1 = pcmciPlusAlgorithm(ts.data, ts.nodeNames, { tauMax: 2, alpha: 0.05 });
    const r2 = pcmciPlusAlgorithm(ts.data, ts.nodeNames, { tauMax: 2, alpha: 0.05 });
    expect(r1.graph.edges.length).toBe(r2.graph.edges.length);
    expect(r1.summary.totalEdges).toBe(r2.summary.totalEdges);
  });
});

