/**
 * Refutation Base — unified interface for all refutation methods.
 *
 * Defines the Refuter interface and a RefutationRunner for
 * batch execution, mirroring DoWhy's design patterns.
 */
import type { RefutationResult } from './causal-inference.js';

export interface Refuter {
  readonly method: string;
  refute(data: number[][], treatmentIdx: number, outcomeIdx: number): Promise<RefutationResult> | RefutationResult;
}

export interface RefutationBatch {
  method: string;
  originalEstimate: number;
  newEstimate: number;
  pValue: number;
  isRobust: boolean;
}

/**
 * Run multiple refuters in batch and collect results.
 * Returns a summary with per-refuter details.
 */
export function runRefutationBatch(
  refuters: Refuter[],
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
): RefutationBatch[] {
  const results: RefutationBatch[] = [];

  for (const refuter of refuters) {
    const r = refuter.refute(data, treatmentIdx, outcomeIdx) as RefutationResult;
    results.push({
      method: refuter.method,
      originalEstimate: r.originalEstimate,
      newEstimate: r.newEstimate,
      pValue: r.pValue,
      isRobust: r.isRobust,
    });
  }

  return results;
}

/**
 * Summarize refutation batch into human-readable string.
 * Mirrors DoWhy's CausalRefutation.__str__ output.
 */
export function summarizeRefutation(results: RefutationBatch[]): string {
  if (results.length === 0) return 'No refutations performed.';

  const lines: string[] = ['Refutation Results:', '━━━━━━━━━━━━━━━━━━━━━'];
  for (const r of results) {
    const status = r.isRobust ? '✅ ROBUST' : '⚠️ SENSITIVE';
    lines.push(
      `${r.method}: ${status}\n` +
      `  Original: ${r.originalEstimate.toFixed(4)} → New: ${r.newEstimate.toFixed(4)}\n` +
      `  p-value: ${r.pValue.toFixed(4)}`,
    );
  }

  const robustCount = results.filter(r => r.isRobust).length;
  lines.push(`━━━━━━━━━━━━━━━━━━━━━`);
  lines.push(`Passed: ${robustCount}/${results.length} refutation tests`);

  return lines.join('\n');
}
