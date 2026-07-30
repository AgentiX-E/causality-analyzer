/**
 * CI Test Backend — dispatch layer for conditional independence tests.
 *
 * Provides a unified interface across ParCorr, CMIknn, and Gsquared
 * backends, enabling PCMCI+ to swap CI test methods without code changes.
 *
 * @packageDocumentation
 */

import type { CIBackend, CITestResult } from '@agentix-e/causality-analyzer-core';
import { parCorrTest } from './parcorr.js';
import { cmiknnTest, type CMIknnConfig } from './cmiknn.js';
import { gsquaredCITest, type GsquaredConfig } from './gsquared-ci.js';

export type { CIBackend, CITestResult };

/**
 * Run a conditional independence test with the specified backend.
 *
 * @param data - (n × totalCols) design matrix
 * @param xCol - column index for variable X
 * @param yCol - column index for variable Y
 * @param condCols - column indices for conditioning set Z
 * @param backend - which CI test to use
 * @param params - backend-specific parameters (k for CMIknn, nPermutations, seed)
 * @returns CITestResult with p-value and test statistic
 */
export function ciTest(
  data: number[][],
  xCol: number,
  yCol: number,
  condCols: number[],
  backend: CIBackend,
  params?: { knnK?: number; nPermutations?: number; seed?: number },
): CITestResult {
  switch (backend) {
    case 'parcorr':
      return parCorrTest(data, xCol, yCol, condCols);

    case 'cmiknn': {
      const cmiConfig: CMIknnConfig = {
        seed: params?.seed ?? 42,
      };
      if (params?.knnK !== undefined) cmiConfig.k = params.knnK;
      if (params?.nPermutations !== undefined) cmiConfig.nPermutations = params.nPermutations;
      return cmiknnTest(data, xCol, yCol, condCols, cmiConfig);
    }

    case 'gsquared': {
      const gsConfig: GsquaredConfig = {};
      if (params?.nPermutations !== undefined) gsConfig.nPermutations = params.nPermutations;
      if (params?.seed !== undefined) gsConfig.seed = params.seed;
      return gsquaredCITest(data, xCol, yCol, condCols, gsConfig);
    }

    default: {
      // Exhaustive check — TypeScript ensures this is unreachable
      const _exhaustive: never = backend;
      throw new Error(`Unknown CI backend: ${String(_exhaustive)}`);
    }
  }
}
