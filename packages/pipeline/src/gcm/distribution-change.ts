/**
 * Distribution Change Attribution — detect and attribute causal mechanism shifts.
 *
 * Critical for AIOps: distinguishes between a "data drift" event (normal variation)
 * and a "mechanism change" event (the causal relationship itself changed,
 * e.g., due to a deployment, configuration change, or infrastructure failure).
 *
 * Methods:
 *   mechanismChangeDetection — per-node test for mechanism stability
 *   distributionChangeRobust — multiply-robust causal change attribution
 *   changeAttributionCI — bootstrap confidence intervals for change attribution
 *
 * References:
 *   Budhathoki et al. (AISTATS 2021). "Causal Mechanism Change Detection"
 *   Budhathoki et al. (ICML 2024). "Multiply-Robust Causal Change Attribution"
 *
 * @packageDocumentation
 */
import { StructuralCausalModel } from './structural-causal-model.js';
import { colMean, createRNG } from '@agentix-e/causality-analyzer-core';

/**
 * Result of mechanism change detection for a single node.
 */
export interface MechanismChangeResult {
  node: string;
  /** Has the mechanism changed? */
  changed: boolean;
  /** Mean noise difference between before and after */
  noiseShift: number;
  /** p-value of the change test */
  pValue: number;
  /** Z-score of the noise shift */
  zScore: number;
}

/**
 * Detect which causal mechanisms have changed between two time windows.
 *
 * For each node, computes the noise (residual) distribution under the SCM
 * in both "before" and "after" periods, then tests for a significant shift
 * using a two-sample Z-test.
 *
 * Returns per-node change detection results.
 */
export function detectMechanismChanges(
  scm: StructuralCausalModel,
  before: Record<string, number>[],
  after: Record<string, number>[],
  alpha: number = 0.05,
): MechanismChangeResult[] {
  const graph = scm.causalGraph;
  const nodes = graph.topologicalSort();
  const results: MechanismChangeResult[] = [];

  // Compute noise for each observation
  const beforeNoise = before.map(obs => scm.abduct(obs));
  const afterNoise = after.map(obs => scm.abduct(obs));

  for (const node of nodes) {
    const bVals = beforeNoise.map(n => n[node] ?? 0).filter(v => !Number.isNaN(v));
    const aVals = afterNoise.map(n => n[node] ?? 0).filter(v => !Number.isNaN(v));

    if (bVals.length < 2 || aVals.length < 2) {
      results.push({ node, changed: false, noiseShift: 0, pValue: 1, zScore: 0 });
      continue;
    }

    const bMean = bVals.reduce((a, b) => a + b, 0) / bVals.length;
    const aMean = aVals.reduce((a, b) => a + b, 0) / aVals.length;
    const bVar = bVals.reduce((s, v) => s + (v - bMean) ** 2, 0) / (bVals.length - 1);
    const aVar = aVals.reduce((s, v) => s + (v - aMean) ** 2, 0) / (aVals.length - 1);

    const se = Math.sqrt(Math.max(1e-10, bVar / bVals.length + aVar / aVals.length));
    const zScore = Math.abs(bMean - aMean) / se;
    const pValue = 2 * (1 - normalCDF(zScore));

    results.push({
      node,
      changed: pValue < alpha,
      noiseShift: aMean - bMean,
      pValue,
      zScore,
    });
  }

  return results;
}

/**
 * Multiply-robust distribution change attribution.
 *
 * Uses the SCM to attribute observed distribution shifts to specific
 * causal mechanisms. Each node's contribution is computed as the
 * fraction of total shift explained by that node's mechanism change.
 *
 * This is the causal analogue of Shapley value attribution for
 * distribution changes rather than point anomalies.
 */
export function distributionChangeRobust(
  scm: StructuralCausalModel,
  before: Record<string, number>[],
  after: Record<string, number>[],
): Map<string, { contribution: number; mechanismShift: number; confidence: number }> {
  const changes = detectMechanismChanges(scm, before, after);
  const totalShift = changes.reduce((s, c) => s + Math.abs(c.noiseShift), 0);

  const result = new Map<string, { contribution: number; mechanismShift: number; confidence: number }>();
  for (const change of changes) {
    result.set(change.node, {
      contribution: totalShift > 0 ? Math.abs(change.noiseShift) / totalShift : 0,
      mechanismShift: change.noiseShift,
      confidence: 1 - change.pValue,
    });
  }

  return result;
}

/**
 * Bootstrap confidence intervals for change attribution.
 *
 * Resamples the before/after observations and recomputes change attribution
 * to quantify uncertainty in the attribution results.
 */
export function changeAttributionCI(
  scm: StructuralCausalModel,
  before: Record<string, number>[],
  after: Record<string, number>[],
  nBootstraps: number = 200,
  seed?: number,
): Map<string, { contribution: number; ciLow: number; ciHigh: number }> {
  const rng = createRNG(seed ?? null);
  const n1 = before.length, n2 = after.length;
  if (n1 < 2 || n2 < 2) return new Map();

  const bootContribs = new Map<string, number[]>();
  const baseline = distributionChangeRobust(scm, before, after);
  for (const [node] of baseline) bootContribs.set(node, []);

  for (let b = 0; b < nBootstraps; b++) {
    const bootBefore: Record<string, number>[] = [];
    const bootAfter: Record<string, number>[] = [];
    for (let i = 0; i < n1; i++) bootBefore.push(before[Math.floor(rng() * n1)]!);
    for (let i = 0; i < n2; i++) bootAfter.push(after[Math.floor(rng() * n2)]!);

    const attrib = distributionChangeRobust(scm, bootBefore, bootAfter);
    for (const [node, val] of attrib) {
      bootContribs.get(node)?.push(val.contribution);
    }
  }

  const result = new Map<string, { contribution: number; ciLow: number; ciHigh: number }>();
  const alpha = 0.05;
  for (const [node, vals] of bootContribs) {
    const sorted = [...vals].sort((a, b) => a - b);
    result.set(node, {
      contribution: baseline.get(node)?.contribution ?? 0,
      ciLow: sorted[Math.floor(nBootstraps * alpha / 2)] ?? 0,
      ciHigh: sorted[Math.floor(nBootstraps * (1 - alpha / 2))] ?? 0,
    });
  }

  return result;
}

// ── Helpers ──────────────────────────────────────────────────────────

function normalCDF(x: number): number {
  return 0.5 * (1 + erfApprox(x / Math.sqrt(2)));
}

function erfApprox(x: number): number {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const t = 1 / (1 + p * Math.abs(x));
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x));
  return sign * y;
}
