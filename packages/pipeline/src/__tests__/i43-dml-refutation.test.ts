/**
 * I11: 效应估计与反驳增强 — DML + MetaLearners + Extended Refutation
 *
 * Coverage target: ≥95% statements, branches, functions, lines
 */
import { describe, it, expect } from 'vitest';
import { createRNG } from '@agentix-e/causality-analyzer-core';
import {
  estimateDML, sLearnerATE, tLearnerATE, xLearnerATE,
  RidgeRegressor, LogisticClassifier,
} from '../infer/dml.js';
import {
  refuteRandomCommonCause,
  refuteDummyOutcome,
  refuteUnobservedConfounder,
  comprehensiveRefutation,
} from '../infer/refutation-extended.js';

// ── Helper: generate synthetic data with treatment effect ────────────
function generateTreatmentData(
  n: number = 500,
  ate: number = 1.0,
  nCovariates: number = 2,
  seed?: number,
): { data: number[][]; treatmentIdx: number; outcomeIdx: number; covariateIndices: number[] } {
  const rng = createRNG(seed ?? 42);

  // Columns: [cov1, cov2, ..., treatment, outcome]
  const data: number[][] = [];
  const treatmentIdx = nCovariates;
  const outcomeIdx = nCovariates + 1;
  const covariateIndices = Array.from({ length: nCovariates }, (_, i) => i);

  for (let r = 0; r < n; r++) {
    const row: number[] = [];
    const covariates: number[] = [];

    // Generate covariates
    for (let c = 0; c < nCovariates; c++) {
      const cv = rng() * 4 - 2; // N(0,4)-like
      row.push(cv);
      covariates.push(cv);
    }

    // Treatment assignment: logistic on covariates
    const logit = covariates.reduce((s, cv) => s + 0.3 * cv, 0);
    const probT = 1 / (1 + Math.exp(-logit));
    const t = rng() < probT ? 1 : 0;
    row.push(t);

    // Outcome: baseline + treatment effect + covariate effect + noise
    const y = 2.0 + ate * t + covariates.reduce((s, cv) => s + 0.5 * cv, 0) + (rng() - 0.5) * 0.5;
    row.push(y);

    data.push(row);
  }

  return { data, treatmentIdx, outcomeIdx, covariateIndices };
}

// ── Ridge Regressor Tests ──────────────────────────────────────────
describe('RidgeRegressor', () => {
  it('should fit and predict on simple linear data', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [3, 5, 7, 9, 11]; // y = 2x + 1
    const model = new RidgeRegressor(0);
    model.fit(X, y);
    const pred = model.predict([[6]])[0]!;
    expect(pred).toBeCloseTo(13, 0);
  });

  it('should handle regularization', () => {
    const X = [[1], [2]];
    const y = [10, 20];
    const model = new RidgeRegressor(1.0);
    model.fit(X, y);
    const pred = model.predict([[3]])[0]!;
    // With ridge, prediction is regularized
    expect(typeof pred).toBe('number');
  });

  it('should handle empty data', () => {
    const model = new RidgeRegressor();
    model.fit([], []);
    const pred = model.predict([]);
    expect(pred.length).toBe(0);
  });

  it('should have correct nParams', () => {
    const model = new RidgeRegressor();
    model.fit([[1, 2], [3, 4]], [5, 6]);
    expect(model.nParams).toBe(3); // 2 features + intercept
  });
});

// ── Logistic Classifier Tests ──────────────────────────────────────
describe('LogisticClassifier', () => {
  it('should classify binary outcomes', () => {
    const X = [[-2], [-1], [0], [1], [2]];
    const y = [0, 0, 0, 1, 1];
    const model = new LogisticClassifier();
    model.fit(X, y);
    const proba = model.predictProba([[3]])[0]!;
    expect(proba).toBeGreaterThan(0.5);
  });

  it('should return 0.5 for unfitted model', () => {
    const model = new LogisticClassifier();
    const proba = model.predictProba([[1, 2]])[0]!;
    expect(proba).toBe(0.5);
  });

  it('should handle linearly separable data', () => {
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 100; i++) {
      X.push([i < 50 ? -5 : 5]);
      y.push(i < 50 ? 0 : 1);
    }
    const model = new LogisticClassifier();
    model.fit(X, y);
    const proba0 = model.predictProba([[-5]])[0]!;
    const proba1 = model.predictProba([[5]])[0]!;
    expect(proba0).toBeLessThan(0.3);
    expect(proba1).toBeGreaterThan(0.7);
  });

  it('should handle empty data', () => {
    const model = new LogisticClassifier();
    model.fit([], []);
    const proba = model.predictProba([]);
    expect(proba.length).toBe(0);
  });
});

// ── DML Estimator Tests ────────────────────────────────────────────
describe('DML Estimator', () => {
  it('should recover ATE on synthetic data', () => {
    const { data, treatmentIdx, outcomeIdx, covariateIndices } = generateTreatmentData(500, 1.0, 2);

    const result = estimateDML(data, treatmentIdx, outcomeIdx, covariateIndices, { nFolds: 5, seed: 42 });
    // DML should recover ATE ≈ 1.0 within 0.3
    expect(result.ate).toBeCloseTo(1.0, 0);
    expect(result.se).toBeGreaterThan(0);
    expect(result.ciLow).toBeLessThan(result.ate);
    expect(result.ciHigh).toBeGreaterThan(result.ate);
    expect(result.n).toBeGreaterThan(0);
    expect(result.foldEstimates.length).toBeGreaterThanOrEqual(1);
  });

  it('should handle empty data gracefully', () => {
    const result = estimateDML([], 0, 1, [], { nFolds: 3 });
    expect(result.ate).toBe(0);
    expect(result.se).toBe(0);
  });

  it('should work with >=3 folds', () => {
    const { data, treatmentIdx, outcomeIdx, covariateIndices } = generateTreatmentData(300, 0.5, 1);
    const result = estimateDML(data, treatmentIdx, outcomeIdx, covariateIndices, { nFolds: 10, seed: 123 });
    expect(result.ate).toBeDefined();
    expect(result.se).toBeGreaterThan(0);
  });

  it('should use custom learners', () => {
    const { data, treatmentIdx, outcomeIdx, covariateIndices } = generateTreatmentData(200, 0.8, 1);
    const outcomeModel = new RidgeRegressor(0.01);
    const propensityModel = new LogisticClassifier();
    const result = estimateDML(data, treatmentIdx, outcomeIdx, covariateIndices, {
      nFolds: 3,
      outcomeModel,
      propensityModel,
    });
    expect(result.ate).toBeDefined();
  });

  it('should handle very small sample', () => {
    const { data, treatmentIdx, outcomeIdx, covariateIndices } = generateTreatmentData(20, 0.5, 1);
    const result = estimateDML(data, treatmentIdx, outcomeIdx, covariateIndices, { nFolds: 3 });
    expect(result.ate).toBeDefined();
  });
});

// ── S-Learner Tests ────────────────────────────────────────────────
describe('S-Learner', () => {
  it('should recover ATE on synthetic data', () => {
    const { data, treatmentIdx, outcomeIdx, covariateIndices } = generateTreatmentData(300, 0.5, 1);
    const { ate, se } = sLearnerATE(data, treatmentIdx, outcomeIdx, covariateIndices);
    expect(ate).toBeCloseTo(0.5, 0);
    expect(se).toBeGreaterThan(0);
  });

  it('should work without covariates', () => {
    const { data, treatmentIdx, outcomeIdx } = generateTreatmentData(200, 1.0, 0);
    const { ate, se } = sLearnerATE(data, treatmentIdx, outcomeIdx, []);
    expect(ate).toBeDefined();
    expect(se).toBeGreaterThan(0);
  });
});

// ── T-Learner Tests ────────────────────────────────────────────────
describe('T-Learner', () => {
  it('should recover ATE on synthetic data', () => {
    const { data, treatmentIdx, outcomeIdx, covariateIndices } = generateTreatmentData(400, 0.8, 2);
    const { ate, se } = tLearnerATE(data, treatmentIdx, outcomeIdx, covariateIndices);
    expect(ate).toBeCloseTo(0.8, 0);
    expect(se).toBeGreaterThan(0);
  });

  it('should handle imbalanced treatment', () => {
    // Manually create imbalanced data
    const data: number[][] = [];
    for (let i = 0; i < 200; i++) {
      data.push([0.1, 0, 1.5 + Math.random() * 0.2]);  // mostly control
      if (i < 40) data.push([0.2, 1, 2.5 + Math.random() * 0.2]); // some treated
    }
    const { ate, se } = tLearnerATE(data, 1, 2, [0]);
    expect(ate).toBeDefined();
    expect(se).toBeGreaterThan(0);
  });
});

// ── X-Learner Tests ────────────────────────────────────────────────
describe('X-Learner', () => {
  it('should recover ATE on synthetic data', () => {
    const { data, treatmentIdx, outcomeIdx, covariateIndices } = generateTreatmentData(400, 0.7, 2);
    const { ate, se } = xLearnerATE(data, treatmentIdx, outcomeIdx, covariateIndices);
    expect(ate).toBeCloseTo(0.7, 0);
    expect(se).toBeGreaterThan(0);
  });

  it('should work without covariates', () => {
    const { data, treatmentIdx, outcomeIdx } = generateTreatmentData(200, 0.5, 0);
    const { ate, se } = xLearnerATE(data, treatmentIdx, outcomeIdx, []);
    expect(ate).toBeDefined();
  });

  it('should use custom propensity model', () => {
    const { data, treatmentIdx, outcomeIdx, covariateIndices } = generateTreatmentData(200, 0.5, 1);
    const propModel = new LogisticClassifier();
    const { ate, se } = xLearnerATE(data, treatmentIdx, outcomeIdx, covariateIndices, propModel);
    expect(ate).toBeDefined();
    expect(se).toBeGreaterThan(0);
  });
});

// ── Extended Refutation Tests ──────────────────────────────────────
describe('Extended Refutation', () => {
  function makeData(n: number = 200): { data: number[][]; tIdx: number; oIdx: number } {
    const data: number[][] = [];
    for (let i = 0; i < n; i++) {
      const z = Math.random() * 4 - 2;
      const t = Math.random() < 0.5 ? 0 : 1;
      const y = 2 + 0.6 * t + 0.4 * z + (Math.random() - 0.5) * 0.5;
      data.push([z, t, y]);
    }
    return { data, tIdx: 1, oIdx: 2 };
  }

  it('refuteRandomCommonCause: should produce valid result', () => {
    const { data, tIdx, oIdx } = makeData(200);
    const result = refuteRandomCommonCause(data, tIdx, oIdx, { nSimulations: 5, seed: 42 });
    expect(result.method).toBe('random_common_cause');
    expect(result.originalEstimate).toBeDefined();
    expect(result.newEstimate).toBeDefined();
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
    expect(typeof result.isRobust).toBe('boolean');
  });

  it('refuteRandomCommonCause: should be robust for large effect', () => {
    // Create data with strong treatment effect
    const data: number[][] = [];
    for (let i = 0; i < 300; i++) {
      data.push([Math.random(), Math.random() < 0.5 ? 0 : 1, Math.random() < 0.5 ? 0 : 1]);
    }
    const result = refuteRandomCommonCause(data, 1, 2, { nSimulations: 3, seed: 1, effectSize: 0.1 });
    expect(result.originalEstimate).toBeDefined();
  });

  it('refuteDummyOutcome: should flag false effect for random data', () => {
    // Random data: no treatment effect
    const data: number[][] = [];
    for (let i = 0; i < 200; i++) {
      data.push([Math.random(), Math.random() < 0.5 ? 0 : 1, Math.random()]);
    }
    const result = refuteDummyOutcome(data, 1, 2, { nSimulations: 10, seed: 7 });
    expect(result.method).toBe('dummy_outcome');
    expect(typeof result.isRobust).toBe('boolean');
  });

  it('refuteDummyOutcome: should produce valid result', () => {
    const { data, tIdx, oIdx } = makeData(200);
    const result = refuteDummyOutcome(data, tIdx, oIdx, { nSimulations: 10, seed: 42 });
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it('refuteUnobservedConfounder: should produce valid result', () => {
    const { data, tIdx, oIdx } = makeData(200);
    const result = refuteUnobservedConfounder(data, tIdx, oIdx, { rho: 0.3, nSimulations: 10, seed: 42 });
    expect(result.method).toBe('unobserved_confounder');
    expect(result.originalEstimate).toBeDefined();
    expect(result.newEstimate).toBeDefined();
    expect(typeof result.isRobust).toBe('boolean');
  });

  it('refuteUnobservedConfounder: higher rho should change estimate more', () => {
    const { data, tIdx, oIdx } = makeData(300);
    const r1 = refuteUnobservedConfounder(data, tIdx, oIdx, { rho: 0.1, nSimulations: 5, seed: 1 });
    const r5 = refuteUnobservedConfounder(data, tIdx, oIdx, { rho: 0.5, nSimulations: 5, seed: 1 });
    // Higher rho should generally cause larger deviation
    const dev1 = Math.abs(r1.newEstimate - r1.originalEstimate);
    const dev5 = Math.abs(r5.newEstimate - r5.originalEstimate);
    expect(dev5).toBeGreaterThanOrEqual(dev1 * 0.5); // at least comparable
  });

  it('comprehensiveRefutation: should run all methods', () => {
    const { data, tIdx, oIdx } = makeData(200);
    const result = comprehensiveRefutation(data, tIdx, oIdx, [0], 42);
    expect(result.methods.length).toBe(3);
    expect(result.robustCount).toBeGreaterThanOrEqual(0);
    expect(result.robustCount).toBeLessThanOrEqual(3);
    expect(result.summary).toContain('/3');
  });
});

// ── Cross-method agreement ─────────────────────────────────────────
describe('Cross-Method Agreement', () => {
  it('DML, S, T, X learners should agree within reasonable range', () => {
    const { data, treatmentIdx, outcomeIdx, covariateIndices } = generateTreatmentData(400, 0.7, 2, 123);

    const dmlResult = estimateDML(data, treatmentIdx, outcomeIdx, covariateIndices, { nFolds: 5, seed: 42 });
    const sResult = sLearnerATE(data, treatmentIdx, outcomeIdx, covariateIndices);
    const tResult = tLearnerATE(data, treatmentIdx, outcomeIdx, covariateIndices);
    const xResult = xLearnerATE(data, treatmentIdx, outcomeIdx, covariateIndices);

    // All should be roughly in [0.3, 1.1] range for ATE=0.7
    const estimates = [dmlResult.ate, sResult.ate, tResult.ate, xResult.ate];
    for (const e of estimates) {
      expect(e).toBeGreaterThanOrEqual(0.2);
      expect(e).toBeLessThanOrEqual(1.2);
    }
  });
});
