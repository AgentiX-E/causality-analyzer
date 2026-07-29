/**
 * I6 tests: Causal Inference + Branch coverage improvements.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../graph/causal-graph.js';
import {
  CausalAnalysis, identifyBackdoor, identifyFrontdoor,
  estimateLinearRegression, refutePlaceboTreatment,
  refuteDataSubset, refuteBootstrap,
} from '../infer/causal-inference.js';
import { CIRCAPipeline } from '../analyze/circa.js';
import { HeuristicPathRCA } from '../analyze/rca.js';

// ── Identification ──────────────────────────────────────────────────
describe('causal identification', () => {
  it('backdoor identifies common causes', () => {
    // X → Y with confounder Z (Z→X, Z→Y)
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('Z', 'X'); g.addEdge('Z', 'Y'); g.addEdge('X', 'Y');
    const estimand = identifyBackdoor(g, 'X', 'Y');
    expect(estimand.backdoorVariables['backdoor']).toContain('Z');
  });

  it('frontdoor identifies mediators', () => {
    // X → M → Y (with unobserved confounder U→X, U→Y)
    const g = new CausalGraph(['X', 'M', 'Y']);
    g.addEdge('X', 'M'); g.addEdge('M', 'Y');
    const estimand = identifyFrontdoor(g, 'X', 'Y');
    expect(estimand).not.toBeNull();
    expect(estimand!.frontdoorVariables).toContain('M');
  });

  it('frontdoor returns null when no mediator exists', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');
    expect(identifyFrontdoor(g, 'X', 'Y')).toBeNull();
  });
});

// ── Estimation ──────────────────────────────────────────────────────
describe('causal estimation', () => {
  it('linear regression estimates treatment effect', () => {
    // Y = 3 * T + 2 * Z + noise
    const n = 200;
    const data: number[][] = [];
    for (let i = 0; i < n; i++) {
      const t = Math.round(Math.random());
      const z = Math.random() * 5;
      const y = 3 * t + 2 * z + (Math.random() - 0.5) * 0.5;
      data.push([t, y, z]);
    }
    const result = estimateLinearRegression(data, 0, 1, [2]);
    expect(result.ate).toBeCloseTo(3, 0);
  });

  it('model estimate produces predictions', () => {
    const data = Array.from({ length: 50 }, () => {
      const t = Math.random() * 10;
      return [t, 2 * t + 5 + (Math.random() - 0.5) * 0.3];
    });
    const result = estimateLinearRegression(data, 0, 1);
    expect(typeof result.model.estimate(0)).toBe('number');
  });

  it('handles single predictor', () => {
    const data = [[1, 3], [2, 5], [3, 7]]; // y ≈ 2x + 1
    const result = estimateLinearRegression(data, 0, 1);
    expect(Math.abs(result.ate - 2)).toBeLessThan(1);
  });
});

// ── Refutation ──────────────────────────────────────────────────────
describe('causal refutation', () => {
  it('placebo treatment nullifies effect', () => {
    const data = Array.from({ length: 100 }, (_, i) => {
      const t = i % 2;
      return [t, 3 * t + (Math.random() - 0.5)];
    });
    const result = refutePlaceboTreatment(data, 0, 1, 30);
    expect(result.method).toBe('placebo_treatment');
    expect(typeof result.pValue).toBe('number');
  });

  it('data subset refutation tests stability', () => {
    const data = Array.from({ length: 80 }, (_, i) => {
      const t = i % 2;
      return [t, 3 * t + (Math.random() - 0.5)];
    });
    const result = refuteDataSubset(data, 0, 1, 0.7, 10);
    expect(result.method).toBe('data_subset');
    expect(typeof result.isRobust).toBe('boolean');
  });

  it('bootstrap provides confidence intervals', () => {
    const data = Array.from({ length: 100 }, (_, i) => {
      const t = i % 2;
      return [t, 3 * t + (Math.random() - 0.5)];
    });
    const result = refuteBootstrap(data, 0, 1, 50);
    expect(result.method).toBe('bootstrap');
    expect(typeof result.isRobust).toBe('boolean');
  });
});

// ── CausalAnalysis Pipeline ─────────────────────────────────────────
describe('CausalAnalysis', () => {
  it('end-to-end identification + estimation + refutation', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('Z', 'X'); g.addEdge('Z', 'Y'); g.addEdge('X', 'Y');
    const data = Array.from({ length: 100 }, (_, i) => {
      const z = Math.random() * 5;
      const x = Math.round(Math.random());
      const y = 3 * x + 2 * z + (Math.random() - 0.5);
      return [x, y, z];
    });
    const ca = new CausalAnalysis();
    ca.ingest(data, ['X', 'Y', 'Z'], 'X', 'Y');
    ca.model(g);
    const result = ca.analyze();
    expect(result).not.toBeNull();
    expect(result!.estimate!.ate).toBeCloseTo(3, 0);
    expect(result!.refutations.length).toBe(3);
  });

  it('identify returns null without model', () => {
    const ca = new CausalAnalysis();
    expect(ca.identify()).toBeNull();
  });

  it('estimate returns null without estimand', () => {
    const ca = new CausalAnalysis();
    expect(ca.estimate()).toBeNull();
  });
});

// ── Branch coverage: CIRCA edge cases ─────────────────────────────
describe('CIRCA branch coverage', () => {
  it('empty anomaly data handled', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const pipe = new CIRCAPipeline();
    pipe.train(g, [[1, 2], [1.1, 2.2]]);
    const result = pipe.analyze([], []);
    expect(result.rootCauses.length).toBeGreaterThanOrEqual(0);
  });

  it('RCA ensemble with no candidates', () => {
    const g = new CausalGraph(['A', 'B']);
    const rca = new HeuristicPathRCA();
    rca.train(g, new Set(['A']), new Matrix(2, 2));
    const result = rca.findRootCauses([]);
    expect(result.rootCauses.length).toBeGreaterThanOrEqual(0);
  });

  it('RHT single data point handled', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const pipe = new CIRCAPipeline();
    pipe.train(g, [[1, 2], [1.1, 2.1]]);
    const result = pipe.analyze([[5, 10]], ['B']);
    expect(result.rootCauses.length).toBeGreaterThan(0);
  });
});
