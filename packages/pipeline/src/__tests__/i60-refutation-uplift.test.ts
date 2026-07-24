/**
 * Refutation Extensions + Uplift Tests.
 *
 * Validates:
 * - Random Common Cause refuter
 * - Dummy Outcome refuter
 * - Uplift evaluation (Qini curve, AUUC, uplift@k)
 */
import { describe, it, expect } from 'vitest';
import { refuteRandomCommonCause, refuteDummyOutcome } from '../../src/infer/refutation-extensions.js';
import { evaluateUplift, upliftAtK, compareUpliftModels } from '../../src/infer/uplift.js';
import type { UpliftObservation } from '../../src/infer/uplift.js';

// Simple linear regression estimator for testing refutation
function makeEstimator(data: number[][]) {
  const n = data.length;
  // ATE = mean(Y | T=1) - mean(Y | T=0)
  let tSum = 0, tN = 0, cSum = 0, cN = 0;
  for (const row of data) {
    if ((row[1] ?? 0) > 0.5) { tSum += row[2] ?? 0; tN++; }
    else { cSum += row[2] ?? 0; cN++; }
  }
  return {
    ate: (tN > 0 ? tSum / tN : 0) - (cN > 0 ? cSum / cN : 0),
    se: 0.1,
  };
}

function generateTestData(n = 100, treatmentEffect = 1.0): number[][] {
  // T ~ Bernoulli(0.5), Y = T * effect + ε, ε ~ N(0, 1)
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const t = Math.random() > 0.5 ? 1 : 0;
    // Simple random normal via Box-Muller
    const u1 = Math.max(1e-10, Math.random());
    const u2 = Math.random();
    const eps = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    data.push([0, t, t * treatmentEffect + eps]); // [dummy, treatment, outcome]
  }
  return data;
}

describe('Refutation Extensions', () => {
  describe('Random Common Cause', () => {
    it('returns valid RefutationResult structure', () => {
      const data = generateTestData(100, 0.5);
      const result = refuteRandomCommonCause(data, 1, 2, makeEstimator, {
        numSimulations: 20,
        seed: 42,
      });
      expect(result.method).toBe('random_common_cause');
      expect(typeof result.originalEstimate).toBe('number');
      expect(typeof result.newEstimate).toBe('number');
      expect(result.pValue).toBeGreaterThanOrEqual(0);
      expect(result.pValue).toBeLessThanOrEqual(1);
      expect(typeof result.isRobust).toBe('boolean');
    });

    it('produces reasonable results with seed', () => {
      const data = generateTestData(100, 0.8);
      const r1 = refuteRandomCommonCause(data, 1, 2, makeEstimator, { numSimulations: 20, seed: 42 });
      const r2 = refuteRandomCommonCause(data, 1, 2, makeEstimator, { numSimulations: 20, seed: 42 });
      // Same seed should produce same results
      expect(r1.newEstimate).toBe(r2.newEstimate);
      expect(r1.pValue).toBe(r2.pValue);
    });

    it('handles different confound strengths', () => {
      const data = generateTestData(100, 1.0);
      const r1 = refuteRandomCommonCause(data, 1, 2, makeEstimator, {
        numSimulations: 10,
        seed: 42,
        confoundStrength: 0.1,
      });
      const r2 = refuteRandomCommonCause(data, 1, 2, makeEstimator, {
        numSimulations: 10,
        seed: 42,
        confoundStrength: 0.5,
      });
      // Stronger confounder should not crash
      expect(typeof r2.originalEstimate).toBe('number');
    });
  });

  describe('Dummy Outcome', () => {
    it('returns valid RefutationResult structure', () => {
      const data = generateTestData(100, 0.5);
      const result = refuteDummyOutcome(data, 1, 2, makeEstimator, {
        numSimulations: 20,
        seed: 42,
      });
      expect(result.method).toBe('dummy_outcome');
      expect(typeof result.isRobust).toBe('boolean');
    });

    it('is reproducible with seed', () => {
      const data = generateTestData(100, 0.5);
      const r1 = refuteDummyOutcome(data, 1, 2, makeEstimator, { numSimulations: 20, seed: 42 });
      const r2 = refuteDummyOutcome(data, 1, 2, makeEstimator, { numSimulations: 20, seed: 42 });
      expect(r1.newEstimate).toBe(r2.newEstimate);
    });
  });
});

describe('Uplift Modeling', () => {
  function makeObservations(): UpliftObservation[] {
    // Synthetic: treatment effect = score (higher score = higher uplift)
    const obs: UpliftObservation[] = [];
    for (let i = 0; i < 100; i++) {
      const score = (i % 10) / 9; // 0 to 1
      const treatment = Math.random() > 0.5 ? 1 : 0;
      const outcome = treatment * score + Math.random() * 0.5;
      obs.push({ score, treatment, outcome });
    }
    return obs;
  }

  it('evaluates uplift curve on valid data', () => {
    const obs = makeObservations();
    const evaluation = evaluateUplift(obs);
    expect(typeof evaluation.auuc).toBe('number');
    expect(evaluation.curve.length).toBe(obs.length);
    expect(evaluation.curve[0]!.proportion).toBeGreaterThan(0);
    expect(evaluation.curve[obs.length - 1]!.proportion).toBeCloseTo(1, 5);
  });

  it('uplitAtK returns finite values', () => {
    const obs = makeObservations();
    const u10 = upliftAtK(obs, 0.1);
    const u20 = upliftAtK(obs, 0.2);
    expect(isFinite(u10)).toBe(true);
    expect(isFinite(u20)).toBe(true);
  });

  it('handles empty observations gracefully', () => {
    const evaluation = evaluateUplift([]);
    expect(evaluation.auuc).toBe(0);
    expect(evaluation.curve).toEqual([]);
  });

  it('compareUpliftModels returns ratio', () => {
    const obsA = makeObservations();
    const obsB = makeObservations();
    const ratio = compareUpliftModels(obsA, obsB);
    expect(isFinite(ratio)).toBe(true);
  });

  it('auuc is higher for better model', () => {
    // Create observations where treatment perfectly aligns with score
    const goodObs: UpliftObservation[] = [];
    for (let i = 0; i < 50; i++) {
      const score = i / 49;
      goodObs.push({ score, treatment: 1, outcome: score * 2 });
      goodObs.push({ score, treatment: 0, outcome: 0 });
    }

    const badObs: UpliftObservation[] = [];
    for (let i = 0; i < 50; i++) {
      badObs.push({ score: Math.random(), treatment: Math.random() > 0.5 ? 1 : 0, outcome: Math.random() });
    }

    const goodEval = evaluateUplift(goodObs, false);
    const badEval = evaluateUplift(badObs, false);
    expect(goodEval.auuc).toBeGreaterThan(badEval.auuc);
  });

  it('qini coefficient is finite', () => {
    const obs = makeObservations();
    const evaluation = evaluateUplift(obs);
    expect(isFinite(evaluation.qiniCoefficient)).toBe(true);
    expect(isFinite(evaluation.auuc)).toBe(true);
  });

  it('single-group degenerate case returns zeros', () => {
    const obs: UpliftObservation[] = [
      { score: 0.9, treatment: 1, outcome: 1 },
      { score: 0.5, treatment: 1, outcome: 0 },
    ];
    const evaluation = evaluateUplift(obs);
    expect(evaluation.auuc).toBe(0);
    expect(evaluation.curve).toEqual([]);
  });
});
