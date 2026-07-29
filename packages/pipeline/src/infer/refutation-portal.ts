/**
 * Unified Refutation Portal — one-shot execution of all 7 refutation
 * methods with automatic HTML report generation.
 *
 * Provides runAllRefutations() which executes every refutation method
 * on the same data and produces a structured result set, and
 * generateRefutationReport() to format the results as an HTML report.
 *
 * @packageDocumentation
 */

import type { RefutationResult } from './causal-inference.js';
import { refutePlaceboTreatment, refuteDataSubset, refuteBootstrap } from './causal-inference.js';
import { refuteAddUnobservedCommonCause } from './refutation-advanced.js';
import { refuteRandomCommonCause, refuteDummyOutcome } from './refutation-extensions.js';

// ── Types ────────────────────────────────────────────────────────────────

/** Function that computes ATE given data, returning ATE with standard error */
export type ATEEstimator = (data: number[][]) => { ate: number; se: number };

/** Result of running all refutations in a single batch. */
export interface RefutationPortalResult {
  /** Original ATE estimate from the provided estimator */
  readonly originalATE: number;
  /** Per-method refutation results */
  readonly results: ReadonlyArray<RefutationResult>;
  /** Fraction of methods indicating robustness */
  readonly robustFraction: number;
  /** Overall verdict */
  readonly verdict: 'robust' | 'sensitive' | 'inconclusive';
  /** Total execution time (ms) */
  readonly runtimeMs: number;
}

/** Structured report output. */
export interface RefutationReport {
  readonly markdown: string;
  readonly html: string;
  readonly json: RefutationPortalResult;
}

// ── 7th Refutation Method ───────────────────────────────────────────────

/**
 * Simulated outcome refutation — the 7th and final refutation method.
 *
 * Replaces the outcome with data simulated from a known DGP where the
 * true treatment effect is zero. The estimated ATE should approach zero
 * if the estimation method is well-specified.
 *
 * @param data - original data matrix
 * @param treatmentIdx - treatment column index
 * @param outcomeIdx - outcome column index
 * @param estimateFn - ATE estimation function
 * @param options - simulation settings
 * @returns refutation result
 */
export function refuteSimulatedOutcome(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  estimateFn: ATEEstimator,
  options: {
    numSimulations?: number;
    seed?: number;
  } = {},
): RefutationResult {
  const nSims = options.numSimulations ?? 100;
  const seed = options.seed ?? 42;

  let state = seed;
  const rng = (): number => { state = (state * 1664525 + 1013904223) & 0x7FFFFFFF; return state / 0x7FFFFFFF; };

  // Original estimate
  const original = estimateFn(data);
  const n = data.length;
  const estimates: number[] = [];

  for (let s = 0; s < nSims; s++) {
    const simData = data.map(row => {
      const nr = [...row];
      const u1 = Math.max(1e-10, rng());
      const u2 = rng();
      const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      nr[outcomeIdx] = noise;
      return nr;
    });
    const { ate } = estimateFn(simData);
    estimates.push(ate);
  }

  const newEstimate = estimates.reduce((a, b) => a + b, 0) / nSims;
  const isRobust = Math.abs(newEstimate) <= 2 * Math.sqrt(
    estimates.reduce((s, v) => s + (v - newEstimate) ** 2, 0) / nSims,
  );

  return {
    method: 'Simulated Outcome',
    originalEstimate: original.ate,
    newEstimate,
    pValue: Math.abs(newEstimate) < 0.01 ? 0 : 0.5,
    isRobust,
  };
}

// ── Unified Portal ──────────────────────────────────────────────────────

/**
 * Run all 7 refutation methods on the same data.
 *
 * Methods:
 *  1. Placebo Treatment — scramble treatment labels
 *  2. Data Subset — re-estimate on random 80% subsamples
 *  3. Bootstrap — bootstrap resampling CI
 *  4. Random Common Cause — add synthetic independent confounder
 *  5. Dummy Outcome — replace outcome with random noise
 *  6. Unobserved Confounder — add correlated confounder
 *  7. Simulated Outcome — test estimator on known zero-effect DGP
 *
 * @param data - (n × p) data matrix
 * @param treatmentIdx - treatment column index
 * @param outcomeIdx - outcome column index
 * @param estimateFn - ATE estimator function
 * @param options - configuration
 * @returns unified result
 */
export function runAllRefutations(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  estimateFn: ATEEstimator,
  options: {
    numSimulations?: number;
    seed?: number;
  } = {},
): RefutationPortalResult {
  const startTime = Date.now();
  const seed = options.seed ?? 42;
  const nSims = options.numSimulations ?? 100;
  const original = estimateFn(data);
  const results: RefutationResult[] = [];

  // 1. Placebo Treatment
  try { results.push(refutePlaceboTreatment(data, treatmentIdx, outcomeIdx, nSims, seed)); }
  catch { results.push({ method: 'Placebo Treatment', originalEstimate: original.ate, newEstimate: original.ate, pValue: 1, isRobust: false }); }

  // 2. Data Subset (80%)
  try { results.push(refuteDataSubset(data, treatmentIdx, outcomeIdx, 0.8, nSims, seed)); }
  catch { results.push({ method: 'Data Subset', originalEstimate: original.ate, newEstimate: original.ate, pValue: 1, isRobust: false }); }

  // 3. Bootstrap
  try { results.push(refuteBootstrap(data, treatmentIdx, outcomeIdx, nSims, seed)); }
  catch { results.push({ method: 'Bootstrap', originalEstimate: original.ate, newEstimate: original.ate, pValue: 1, isRobust: false }); }

  // 4. Random Common Cause
  try { results.push(refuteRandomCommonCause(data, treatmentIdx, outcomeIdx, estimateFn, { numSimulations: nSims, seed })); }
  catch { results.push({ method: 'Random Common Cause', originalEstimate: original.ate, newEstimate: original.ate, pValue: 1, isRobust: false }); }

  // 5. Dummy Outcome
  try { results.push(refuteDummyOutcome(data, treatmentIdx, outcomeIdx, estimateFn, { numSimulations: nSims, seed })); }
  catch { results.push({ method: 'Dummy Outcome', originalEstimate: original.ate, newEstimate: original.ate, pValue: 1, isRobust: false }); }

  // 6. Unobserved Confounder
  try { results.push(refuteAddUnobservedCommonCause(data, treatmentIdx, outcomeIdx, estimateFn, { numSimulations: nSims })); }
  catch { results.push({ method: 'add_unobserved_common_cause', originalEstimate: original.ate, newEstimate: original.ate, pValue: 1, isRobust: false }); }

  // 7. Simulated Outcome
  try { results.push(refuteSimulatedOutcome(data, treatmentIdx, outcomeIdx, estimateFn, { numSimulations: nSims, seed })); }
  catch { results.push({ method: 'Simulated Outcome', originalEstimate: original.ate, newEstimate: original.ate, pValue: 1, isRobust: false }); }

  const robustCount = results.filter(r => r.isRobust).length;
  const robustFraction = robustCount / 7;
  const verdict = robustFraction >= 5/7 ? 'robust' : robustFraction >= 3/7 ? 'inconclusive' : 'sensitive';

  return { originalATE: original.ate, results, robustFraction, verdict, runtimeMs: Date.now() - startTime };
}

/**
 * Generate HTML and Markdown report from refutation results.
 */
export function generateRefutationReport(result: RefutationPortalResult): RefutationReport {
  const rows = result.results.map(r =>
    `| ${r.method} | ${r.newEstimate.toFixed(4)} | ${r.isRobust ? '✅' : '❌'} |`
  ).join('\n');

  const markdown = `# Refutation Report\n\n**Original ATE:** ${result.originalATE.toFixed(4)}\n**Verdict:** ${result.verdict.toUpperCase()} (${result.results.filter(r => r.isRobust).length}/7 robust)\n**Runtime:** ${result.runtimeMs}ms\n\n| Method | New Estimate | Robust |\n|--------|-------------|--------|\n${rows}`;

  const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>Refutation Report</title><style>body{font-family:system-ui,sans-serif;max-width:800px;margin:2rem auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}th{background:#f5f5f5}.robust{color:green}.v${result.verdict}{font-weight:700;padding:0.5rem;border-radius:4px}</style></head><body><h1>Refutation Report</h1><p><b>Original ATE:</b> ${result.originalATE.toFixed(4)} | <b>Verdict:</b> ${result.verdict.toUpperCase()} (${result.results.filter(r => r.isRobust).length}/7)</p><table><tr><th>Method</th><th>New Estimate</th><th>Robust</th></tr>${result.results.map(r => `<tr><td>${r.method}</td><td>${r.newEstimate.toFixed(4)}</td><td class="${r.isRobust ? 'robust' : ''}">${r.isRobust ? '✅' : '❌'}</td></tr>`).join('')}</table></body></html>`;

  return { markdown, html, json: result };
}
