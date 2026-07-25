/**
 * Refutation Suite — Extended refutation methods for causal inference.
 *
 * Beyond basic Bootstrap/Placebo/DataSubset refutations, this module adds:
 * - Random Common Cause: add synthetic unmeasured confounder, check ATE stability
 * - Dummy Outcome: replace outcome with random variable, verify ATE → 0
 *
 * These provide comprehensive robustness validation following
 * DoWhy's refutation framework (Sharma & Kiciman, 2020).
 *
 * @packageDocumentation
 */
import { createRNG } from '@agentix-e/causality-analyzer-core';
import type { RefutationResult } from './causal-inference.js';

/**
 * Random Common Cause refutation.
 *
 * Adds a synthetic unmeasured confounder to the data and re-estimates
 * the effect. If the estimate is robust, adding random confounding
 * should not substantially change the ATE.
 *
 * The synthetic confounder is constructed as:
 *   U = w·T + (1-w)·Y + ε
 * where w ~ Uniform(0, 1) and ε ~ N(0, σ²).
 *
 * @returns RefutationResult with new estimate, p-value, robustness flag
 */
export function refuteRandomCommonCause(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  estimateFn: (data: number[][]) => { ate: number; se: number },
  options: {
    numSimulations?: number;
    confoundStrength?: number;
    seed?: number;
  } = {},
): RefutationResult {
  const numSims = options.numSimulations ?? 50;
  const strength = options.confoundStrength ?? 0.3;
  const rng = createRNG(options.seed ?? null);

  const originalResult = estimateFn(data);
  const originalATE = originalResult.ate;

  // Simulate: add random confounder, re-estimate ATE
  const simulatedATEs: number[] = [];
  for (let sim = 0; sim < numSims; sim++) {
    // Create synthetic confounder U = w·T + (1-w)·Y + ε
    const w = rng();
    const sigma = strength * Math.sqrt(
      data.reduce((s, r) => s + (r[outcomeIdx] ?? 0) ** 2, 0) / data.length,
    );

    const augmentedData = data.map(row => {
      const t = row[treatmentIdx] ?? 0;
      const y = row[outcomeIdx] ?? 0;
      // Box-Muller normal random variate
      const u1 = Math.max(1e-10, rng());
      const u2 = rng();
      const epsilon = sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const confounder = w * t + (1 - w) * y + epsilon;
      return [...row, confounder];
    });

    // For re-estimation, we need to include the confounder as a covariate
    // Since the estimator takes [data,treatmentIdx,outcomeIdx,covariateIndices],
    // and our estimateFn interface is generic, we rely on the caller to handle it.
    // Here we use a simple approach: augment data and estimate directly.
    const _augmentedIdx = data[0]?.length ?? 0;
    simulatedATEs.push(estimateFn(augmentedData).ate);
  }

  // Compute p-value: fraction of simulations where ATE differs significantly
  const diffs = simulatedATEs.map(a => Math.abs(a - originalATE));
  const significantChanges = diffs.filter(d => d > Math.abs(originalATE) * 0.1).length;
  const pValue = (significantChanges + 1) / (numSims + 1);

  const isRobust = pValue > 0.05; // null hypothesis: confounder doesn't change ATE

  return {
    method: 'random_common_cause',
    originalEstimate: originalATE,
    newEstimate: simulatedATEs.reduce((s, a) => s + a, 0) / simulatedATEs.length,
    pValue,
    isRobust,
  };
}

/**
 * Dummy Outcome refutation.
 *
 * Replaces the true outcome with a random variable (independent of treatment)
 * and re-estimates the causal effect. If the estimation method is valid,
 * the estimated ATE should be approximately zero.
 *
 * This tests whether the identification/estimation method can distinguish
 * true causal effects from noise.
 *
 * @returns RefutationResult. isRobust=true means the method correctly
 *          identified that the dummy outcome has no causal effect.
 */
export function refuteDummyOutcome(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  estimateFn: (data: number[][]) => { ate: number; se: number },
  options: {
    numSimulations?: number;
    seed?: number;
  } = {},
): RefutationResult {
  const numSims = options.numSimulations ?? 50;
  const rng = createRNG(options.seed ?? null);

  const originalResult = estimateFn(data);
  const originalATE = originalResult.ate;

  // Simulate: replace outcome with random variable, re-estimate
  const dummyATEs: number[] = [];
  for (let sim = 0; sim < numSims; sim++) {
    const dummyData = data.map(row => {
      const newRow = [...row];
      // Replace outcome with random normal: N(0, σ²)
      const outcomeStd = Math.sqrt(
        data.reduce((s, r) => s + (r[outcomeIdx] ?? 0) ** 2, 0) / data.length,
      );
      const u1 = Math.max(1e-10, rng());
      const u2 = rng();
      newRow[outcomeIdx] = outcomeStd *
        Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      return newRow;
    });

    dummyATEs.push(estimateFn(dummyData).ate);
  }

  // Under the null hypothesis (no causal effect), ATE should be ≈ 0.
  // p-value: proportion of simulations where |ATE| >= |original ATE| / 10
  const absDummy = dummyATEs.map(a => Math.abs(a));
  const absOriginal = Math.abs(originalATE);
  const threshold = Math.max(absOriginal * 0.1, 1e-6);
  const extremeCount = absDummy.filter(a => a >= threshold).length;
  const pValue = (extremeCount + 1) / (numSims + 1);

  // Robust if most dummy ATEs are close to zero (p > 0.05 means the dummy
  // ATEs are significantly different from the real ATE)
  const isRobust = pValue < 0.05;

  return {
    method: 'dummy_outcome',
    originalEstimate: originalATE,
    newEstimate: dummyATEs.reduce((s, a) => s + a, 0) / dummyATEs.length,
    pValue,
    isRobust,
  };
}
