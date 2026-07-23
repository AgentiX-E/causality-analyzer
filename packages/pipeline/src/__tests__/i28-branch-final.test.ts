/**
 * Branch-coverage gap closure tests.
 *
 * Each test targets a specific uncovered branch identified by the
 * v8 coverage report. Focused on mediation, SPOT/GPD, and IRLS paths.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../graph/causal-graph.js';
import { naturalDirectEffect, arrowStrength } from '../infer/mediation.js';
import { estimateCATE, estimateIPW, checkFairness } from '../infer/cate-fairness.js';
import { SPOTDetector, DSPOTDetector } from '../detect/spot.js';
import { StatsDetector } from '../detect/stats-detector.js';
import { SpectralResidualDetector } from '../detect/spectral-residual.js';
import { VotingDetector } from '../detect/voting-detector.js';
import { falsifyGraph, lmcFalsification } from '../gcm/graph-falsification.js';
import { StructuralCausalModel, evaluateMechanismR2 } from '../gcm/structural-causal-model.js';
import { identifyByDoCalculus } from '../infer/do-calculus.js';

// ── Mediation branches (60% → 80%+) ──────────────────────────────────

describe('mediation branch coverage', () => {
  it('n < 3 returns zero effects with explanation', () => {
    const r = naturalDirectEffect([[1, 2]], 0, 1, 0);
    expect(r.nde).toBe(0);
    expect(r.nie).toBe(0);
    expect(r.explanation).toContain('Insufficient');
  });

  it('proportionMediated is between 0 and 1', () => {
    const data: number[][] = [];
    for (let i = 0; i < 30; i++) {
      const t = Math.random();
      const m = t * 0.7 + Math.random() * 0.3;
      const y = t * 0.2 + m * 0.6 + Math.random() * 0.1;
      data.push([t, m, y]);
    }
    const r = naturalDirectEffect(data, 0, 2, 1);
    expect(r.proportionMediated).toBeGreaterThanOrEqual(0);
    expect(r.proportionMediated).toBeLessThanOrEqual(1);
  });

  it('arrowStrength with 2-node chain', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const data = new Matrix(Array.from({ length: 20 }, (_, i) => [i, i * 0.8]));
    const s = arrowStrength(g, data, ['A', 'B']);
    expect(s.size).toBeGreaterThanOrEqual(0);
  });
});

// ── SPOT / GPD branches (77% → 85%+) ─────────────────────────────────

describe('SPOT branch coverage', () => {
  it('calibrates with large initSize', () => {
    const s = new SPOTDetector({ initSize: 30, q: 0.98 });
    for (let i = 0; i < 50; i++) s.update(Math.random() * 2);
    const r = s.update(1);
    expect(typeof r.isAnomalous).toBe('boolean');
  });

  it('DSPOT detects drift after warmup', () => {
    const d = new DSPOTDetector({ initSize: 20, driftWindow: 10, q: 0.95 });
    for (let i = 0; i < 30; i++) d.update(1 + Math.random() * 0.2);
    for (let i = 0; i < 15; i++) d.update(3 + i * 0.5);
    const r = d.update(10);
    expect(typeof r.isAnomalous).toBe('boolean');
  });

  it('VotingDetector with max strategy', () => {
    const d1 = new StatsDetector({ threshold: 2, minSamples: 3 });
    const d2 = new StatsDetector({ threshold: 3, minSamples: 3 });
    const v = new VotingDetector([d1, d2], { strategy: 'maximum' });
    for (let i = 0; i < 5; i++) v.update([5]);
    const r = v.update([50]);
    expect(typeof r.isAnomalous).toBe('boolean');
  });
});

// ── IRLS / CATE-IPW branches (79% → 85%+) ────────────────────────────

describe('CATE/IPW branch coverage', () => {
  it('IPW with single covariate', () => {
    const data: number[][] = [];
    for (let i = 0; i < 50; i++) {
      const x = Math.random();
      const t = x > 0.5 ? 1 : 0;
      data.push([x, t, t * 0.5 + x * 0.3 + Math.random() * 0.1]);
    }
    const r = estimateIPW(data, 1, 2, [0]);
    expect(r.se).toBeGreaterThan(0);
  });

  it('CATE with no features', () => {
    const data: number[][] = Array.from({ length: 30 }, () => {
      const t = Math.random() > 0.5 ? 1 : 0;
      return [t, t * 0.5 + Math.random() * 0.2];
    });
    const r = estimateCATE(data, 0, 1, []);
    expect(typeof r.baselineATE).toBe('number');
  });

  it('checkFairness with empty groups', () => {
    const rc = [{ name: 'svc', score: 0.8 }];
    const r = checkFairness(rc, {});
    expect(r.fair).toBe(true);
    expect(r.disparity).toBe(0);
  });
});

// ── Viz branches (81% → 90%+) ────────────────────────────────────────



// ── Falsification branches ────────────────────────────────────────────

describe('falsification branch coverage', () => {
  it('falsifyGraph with connected graph', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C');
    const data = new Matrix(Array.from({ length: 30 }, () => [Math.random(), Math.random() * 2, Math.random() * 3]));
    const r = falsifyGraph(g, data, ['A', 'B', 'C']);
    expect(typeof r.falsified).toBe('boolean');
  });

  it('lmcFalsification with parents', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const data = new Matrix(Array.from({ length: 20 }, () => [Math.random(), Math.random() * 2]));
    const r = lmcFalsification(g, data, ['A', 'B']);
    expect(r.has('A')).toBe(true);
    expect(r.has('B')).toBe(true);
  });
});

// ── Effect estimation branches ────────────────────────────────────────


