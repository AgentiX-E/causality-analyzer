/**
 * I7: Branch coverage gap-filling tests.
 *
 * Targets the lowest-coverage branches across the infer pipeline.
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../graph/causal-graph.js';
import {
  adjustBackdoor, findBackdoorSet, estimateFrontdoor,
  estimateIV, estimatePropensityScore, estimatePSMatching,
  estimateDoublyRobust,
} from '../infer/effect-estimation.js';
import { estimateLinearRegression, refuteBootstrap, refuteDataSubset, refutePlaceboTreatment } from '../infer/causal-inference.js';
import { summarizeRefutation, runRefutationBatch } from '../infer/refutation-base.js';
import type { Refuter, RefutationBatch } from '../infer/refutation-base.js';
import type { RefutationResult } from '../infer/causal-inference.js';
import { findBackdoorAdjustmentSet } from '../infer/backdoor.js';
import { createRNG } from '@agentix-e/causality-analyzer-core';

// ── Refutation Base ──────────────────────────────────────────────────

describe('Refutation Base', () => {
  class TestRefuter implements Refuter {
    readonly method = 'test';
    refute(_data: number[][], _tIdx: number, _oIdx: number): RefutationResult {
      return { originalEstimate: 1.0, newEstimate: 0.9, pValue: 0.34, isRobust: true };
    }
  }

  class FailingRefuter implements Refuter {
    readonly method = 'fail';
    refute(_data: number[][], _tIdx: number, _oIdx: number): RefutationResult {
      return { originalEstimate: 1.0, newEstimate: 0.1, pValue: 0.001, isRobust: false };
    }
  }

  it('summarizeRefutation handles empty array', () => {
    expect(summarizeRefutation([])).toContain('No refutations');
  });

  it('summarizeRefutation with robust result', () => {
    const batch: RefutationBatch[] = [
      { method: 'test', originalEstimate: 1.0, newEstimate: 0.9, pValue: 0.34, isRobust: true },
    ];
    expect(summarizeRefutation(batch)).toContain('1/1');
  });

  it('summarizeRefutation with mixed results', () => {
    const batch: RefutationBatch[] = [
      { method: 'a', originalEstimate: 1.0, newEstimate: 0.9, pValue: 0.34, isRobust: true },
      { method: 'b', originalEstimate: 1.0, newEstimate: 0.1, pValue: 0.001, isRobust: false },
    ];
    expect(summarizeRefutation(batch)).toContain('1/2');
  });

  it('runRefutationBatch collects results', () => {
    const refuters: Refuter[] = [new TestRefuter(), new FailingRefuter()];
    const data = [[0, 1], [0, 2], [1, 3], [1, 4]];
    expect(runRefutationBatch(refuters, data, 0, 1).length).toBe(2);
  });
});

// ── Effect Estimation Edge Cases ─────────────────────────────────────

describe('Effect Estimation Edge Cases', () => {
  const basicData = [[0, 1], [0, 2], [1, 3], [1, 4], [0, 1.5], [1, 3.5]];

  it('adjustBackdoor with binary treatment and no confounders', () => {
    const g = new CausalGraph(['T', 'Y']);
    g.addEdge('T', 'Y');
    const r = adjustBackdoor(g, 'T', 'Y', basicData, new Map([['T', 0], ['Y', 1]]));
    expect(r.ate).toBeGreaterThan(0);
    expect(r.adjustors).toEqual([]);
  });

  it('estimateFrontdoor with single mediator', () => {
    const g = new CausalGraph(['X', 'M', 'Y']);
    g.addEdge('X', 'M'); g.addEdge('M', 'Y');
    const data = [[0, 1, 2], [0, 1.1, 2.1], [1, 3, 5], [1, 3.2, 5.3], [0, 0.9, 1.8], [1, 2.8, 4.9]];
    const r = estimateFrontdoor(g, 'X', 'Y', data, new Map([['X', 0], ['M', 1], ['Y', 2]]), ['M']);
    expect(typeof r.ate).toBe('number');
  });

  it('estimateIV with 2SLS', () => {
    const g = new CausalGraph(['Z', 'T', 'Y', 'U']);
    g.addEdge('Z', 'T'); g.addEdge('T', 'Y'); g.addEdge('U', 'T'); g.addEdge('U', 'Y');
    const data = [[1, 0.5, 1, 0.1], [0, 0.2, 0.5, 0.1], [1, 0.6, 1.2, 0.2], [0, 0.1, 0.3, 0.2], [1, 0.5, 1.1, 0.1], [0, 0.3, 0.7, 0.1]];
    const r = estimateIV(g, 'T', 'Y', data, new Map([['Z', 0], ['T', 1], ['Y', 2], ['U', 3]]), ['Z']);
    expect(typeof r.ate).toBe('number');
  });

  it('estimatePropensityScore returns Float64Array of scores', () => {
    const data = [[0.5, 0, 1], [0.3, 0, 0.8], [0.8, 1, 3], [0.2, 0, 0.6]];
    const scores = estimatePropensityScore(data, 1, [0]);
    expect(scores).toBeInstanceOf(Float64Array);
    expect(scores.length).toBe(4);
  });

  it('estimatePSMatching with caliper', () => {
    const g = new CausalGraph(['C', 'T', 'Y']);
    g.addEdge('C', 'T'); g.addEdge('C', 'Y'); g.addEdge('T', 'Y');
    const data = [[0.5, 0, 1], [0.3, 0, 0.8], [0.8, 1, 3], [0.2, 0, 0.6]];
    const r = estimatePSMatching(g, 'T', 'Y', data, new Map([['C', 0], ['T', 1], ['Y', 2]]), ['C']);
    expect(typeof r.ate).toBe('number');
  });

  it('estimateDoublyRobust with confounders', () => {
    const g = new CausalGraph(['C', 'T', 'Y']);
    g.addEdge('C', 'T'); g.addEdge('C', 'Y'); g.addEdge('T', 'Y');
    const data = [[0.5, 0, 1], [0.3, 0, 0.8], [0.8, 1, 3], [0.2, 0, 0.6], [0.9, 1, 3.5], [0.4, 0, 1.0]];
    const r = estimateDoublyRobust(g, 'T', 'Y', data, new Map([['C', 0], ['T', 1], ['Y', 2]]), ['C']);
    expect(typeof r.ate).toBe('number');
  });
});

// ── Causal Inference (index-based API) ──────────────────────────────

describe('Causal Inference Index API', () => {
  const data = [[0, 1], [0, 2], [1, 3], [1, 4], [0, 1.5], [1, 3.5]];

  it('estimateLinearRegression with no covariates', () => {
    const r = estimateLinearRegression(data, 0, 1, []);
    expect(typeof r.ate).toBe('number');
    expect(r.model).toBeDefined();
  });

  it('refuteBootstrap returns valid refutation result', () => {
    const r = refuteBootstrap(data, 0, 1, 20, 42);
    expect(typeof r.originalEstimate).toBe('number');
    expect(typeof r.newEstimate).toBe('number');
    expect(typeof r.isRobust).toBe('boolean');
  });

  it('refuteDataSubset returns valid refutation result', () => {
    const r = refuteDataSubset(data, 0, 1, 42);
    expect(typeof r.originalEstimate).toBe('number');
    expect(typeof r.newEstimate).toBe('number');
  });

  it('refutePlaceboTreatment returns valid refutation result', () => {
    const r = refutePlaceboTreatment(data, 0, 1, 42);
    expect(typeof r.originalEstimate).toBe('number');
    expect(typeof r.newEstimate).toBe('number');
  });
});

// ── Backdoor Variant Coverage ────────────────────────────────────────

describe('Backdoor Adjustment Variants', () => {
  it('findBackdoorAdjustmentSet on chain graph', () => {
    const g = new CausalGraph(['X', 'M', 'Y']);
    g.addEdge('X', 'M'); g.addEdge('M', 'Y');
    // X→M→Y: no backdoor paths → empty set
    const s = findBackdoorAdjustmentSet(g, 'X', 'Y');
    expect(Array.isArray(s)).toBe(true);
  });

  it('findBackdoorAdjustmentSet on fork graph', () => {
    const g = new CausalGraph(['C', 'T', 'Y']);
    g.addEdge('C', 'T'); g.addEdge('C', 'Y');
    // C→T, C→Y: C is a confounder → [C]
    const s = findBackdoorAdjustmentSet(g, 'T', 'Y');
    expect(s.length).toBe(1);
    expect(s[0]).toBe('C');
  });

  it('findBackdoorAdjustmentSet on collider graph', () => {
    const g = new CausalGraph(['T', 'M', 'Y']);
    g.addEdge('T', 'M'); g.addEdge('Y', 'M');
    // T→M←Y: M is a collider → empty set
    const s = findBackdoorAdjustmentSet(g, 'T', 'Y');
    expect(Array.isArray(s)).toBe(true);
  });
});
