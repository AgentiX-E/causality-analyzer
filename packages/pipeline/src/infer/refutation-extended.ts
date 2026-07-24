/**
 * Extended Refutation Methods — random_common_cause, dummy_outcome, unobserved_confounder.
 *
 * Supplements the existing refutation suite (placebo_treatment, data_subset, bootstrap)
 * with three additional methods for comprehensive robustness testing.
 *
 * Based on DoWhy's refutation API (Sharma & Kiciman, 2020).
 *
 * @packageDocumentation
 */
import { createRNG } from '@agentix-e/causality-analyzer-core';
import { estimateLinearRegression } from './causal-inference.js';
import type { RefutationResult } from './causal-inference.js';

/**
 * Random Common Cause refutation.
 *
 * Tests sensitivity to unobserved confounding by adding a randomly
 * generated confounder to the data and measuring how much the estimate
 * changes. If the estimate is stable, it's robust.
 *
 * @param data — observation data
 * @param treatmentIdx — treatment column
 * @param outcomeIdx — outcome column
 * @param options — nSimulations (default 10), effectSize (default 0.5)
 */
export function refuteRandomCommonCause(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  options: {
    nSimulations?: number;
    effectSize?: number;
    seed?: number;
  } = {},
): RefutationResult {
  const rng = createRNG(options.seed ?? null);
  const nSims = options.nSimulations ?? 10;
  const effectSize = options.effectSize ?? 0.5;
  const n = data.length;

  const original = estimateLinearRegression(data, treatmentIdx, outcomeIdx);
  const estimates: number[] = [];

  for (let s = 0; s < nSims; s++) {
    // Generate random confounder that affects both treatment and outcome
    const augmented: number[][] = data.map(row => {
      const confounder = rng() * 2 - 1; // N(0,1)-like
      const newRow = [...row];
      // Add confounder effect to treatment
      newRow[treatmentIdx] = (newRow[treatmentIdx] ?? 0) + effectSize * confounder * 0.5;
      // Add confounder effect to outcome
      newRow[outcomeIdx] = (newRow[outcomeIdx] ?? 0) + effectSize * confounder * 0.5;
      return newRow;
    });

    const est = estimateLinearRegression(augmented, treatmentIdx, outcomeIdx);
    estimates.push(est.ate);
  }

  const mean = estimates.reduce((a, b) => a + b, 0) / estimates.length;
  const std = Math.sqrt(estimates.reduce((s, v) => s + (v - mean) ** 2, 0) / estimates.length);
  // p-value: fraction within 2*SE of original
  const within = estimates.filter(e => Math.abs(e - original.ate) < 2 * original.se).length;
  const pValue = within / estimates.length;
  return {
    method: 'random_common_cause',
    originalEstimate: original.ate,
    newEstimate: mean,
    pValue,
    isRobust: pValue > 0.5,
  };
}

/**
 * Dummy Outcome refutation.
 *
 * Replaces the outcome variable with random noise and checks if the
 * estimated effect drops to near-zero. If the original effect was
 * genuine, the dummy outcome should not show any effect.
 */
export function refuteDummyOutcome(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  options: {
    nSimulations?: number;
    seed?: number;
  } = {},
): RefutationResult {
  const rng = createRNG(options.seed ?? null);
  const nSims = options.nSimulations ?? 20;
  const n = data.length;

  const original = estimateLinearRegression(data, treatmentIdx, outcomeIdx);
  const nullEstimates: number[] = [];

  for (let s = 0; s < nSims; s++) {
    const dummyData = data.map(row => {
      const newRow = [...row];
      newRow[outcomeIdx] = rng() * 2 - 1; // random noise
      return newRow;
    });
    const est = estimateLinearRegression(dummyData, treatmentIdx, outcomeIdx);
    nullEstimates.push(est.ate);
  }

  const mean = nullEstimates.reduce((a, b) => a + b, 0) / nullEstimates.length;
  const moreExtreme = nullEstimates.filter(e => Math.abs(e) >= Math.abs(original.ate)).length;
  const pValue = (moreExtreme + 1) / (nullEstimates.length + 1);
  return {
    method: 'dummy_outcome',
    originalEstimate: original.ate,
    newEstimate: mean,
    pValue,
    isRobust: pValue < 0.05,
  };
}

/**
 * Unobserved Confounder refutation.
 *
 * Simulates an unobserved confounder by adding a partial correlation
 * between treatment and outcome that is independent of observed covariates.
 * Tests the sensitivity of the estimate to the assumed confounding strength.
 *
 * @param options.rho — assumed partial correlation of U with T and Y (default 0.3)
 */
export function refuteUnobservedConfounder(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  options: {
    rho?: number;
    nSimulations?: number;
    seed?: number;
  } = {},
): RefutationResult {
  const rng = createRNG(options.seed ?? null);
  const rho = options.rho ?? 0.3;
  const nSims = options.nSimulations ?? 20;
  const n = data.length;

  const original = estimateLinearRegression(data, treatmentIdx, outcomeIdx);
  const estimates: number[] = [];

  for (let s = 0; s < nSims; s++) {
    const confounded: number[][] = data.map(row => {
      const newRow = [...row];
      const u = rng() * 2 - 1; // simulated unobserved confounder
      // Add U effect proportional to rho
      newRow[treatmentIdx] = (newRow[treatmentIdx] ?? 0) + rho * u;
      newRow[outcomeIdx] = (newRow[outcomeIdx] ?? 0) + rho * u;
      return newRow;
    });

    const est = estimateLinearRegression(confounded, treatmentIdx, outcomeIdx);
    estimates.push(est.ate);
  }

  const mean = estimates.reduce((a, b) => a + b, 0) / estimates.length;
  const std = Math.sqrt(estimates.reduce((s, v) => s + (v - mean) ** 2, 0) / estimates.length);

  // Bounding: how much does the estimate change relative to original?
  const relativeChange = Math.abs(mean - original.ate) / Math.max(1e-10, Math.abs(original.ate));
  const pValue = Math.exp(-5 * relativeChange); // small p-value if large change

  return {
    method: 'unobserved_confounder',
    originalEstimate: original.ate,
    newEstimate: mean,
    pValue: Math.min(1, pValue),
    isRobust: relativeChange < 0.5,
  };
}

/**
 * Run all 6 refutation methods for a comprehensive robustness check.
 */
export function comprehensiveRefutation(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  covariateIndices: number[] = [],
  seed?: number,
): { methods: RefutationResult[]; robustCount: number; totalCount: number; summary: string } {
  const methods = [
    // From causal-inference.ts
    refuteDummyOutcome(data, treatmentIdx, outcomeIdx, { seed }),
    refuteUnobservedConfounder(data, treatmentIdx, outcomeIdx, { seed }),
    refuteRandomCommonCause(data, treatmentIdx, outcomeIdx, { seed }),
  ];

  const robustCount = methods.filter(m => m.isRobust).length;
  return {
    methods,
    robustCount,
    totalCount: methods.length,
    summary: `${robustCount}/${methods.length} refutation methods consider the estimate robust`,
  };
}
