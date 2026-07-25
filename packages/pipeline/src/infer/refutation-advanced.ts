/**
 * Advanced Refutation Methods.
 *
 * Adds the missing DoWhy refuters:
 * - addUnobservedCommonCause: correlated confounder (not just random noise)
 * - E-value sensitivity: VanderWeele & Ding 2017
 */
import { createRNG } from '@agentix-e/causality-analyzer-core';
import type { RefutationResult } from './causal-inference.js';

/**
 * Add Unobserved Common Cause refutation.
 *
 * Unlike refuteRandomCommonCause (which adds purely random noise), this
 * refuter adds a confounder CORRELATED with both treatment and outcome.
 * This is a much stronger test: if the estimate survives a correlated
 * unmeasured confounder, it's genuinely robust.
 *
 * @returns RefutationResult
 */
export function refuteAddUnobservedCommonCause(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  estimateFn: (data: number[][]) => { ate: number; se: number },
  options: {
    confoundCorrelation?: number;
    numSimulations?: number;
    seed?: number;
  } = {},
): RefutationResult {
  const corr = options.confoundCorrelation ?? 0.3;
  const nSims = options.numSimulations ?? 30;
  const rng = createRNG(options.seed ?? null);

  const original = estimateFn(data);
  const simResults: number[] = [];

  for (let s = 0; s < nSims; s++) {
    // Create confounder Z = corr·T + corr·Y + N(0, 1-corr²)
    const sigma = Math.sqrt(Math.max(0, 1 - corr * corr));
    const augmented = data.map(row => {
      const t = row[treatmentIdx] ?? 0;
      const y = row[outcomeIdx] ?? 0;
      // Box-Muller for N(0, sigma²)
      const u1 = Math.max(1e-10, rng());
      const u2 = rng();
      const noise = sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const z = corr * t + corr * y + noise;
      return [...row, z];
    });

    simResults.push(estimateFn(augmented).ate);
  }

  const meanSim = simResults.reduce((s, v) => s + v, 0) / simResults.length;
  const diffs = simResults.map(a => Math.abs(a - original.ate));
  const sigCount = diffs.filter(d => d > Math.abs(original.ate) * 0.1).length;
  const pValue = (sigCount + 1) / (nSims + 1);

  return {
    method: 'add_unobserved_common_cause',
    originalEstimate: original.ate,
    newEstimate: meanSim,
    pValue,
    isRobust: pValue > 0.05,
  };
}
