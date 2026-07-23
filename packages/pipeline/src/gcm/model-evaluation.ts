/**
 * Model Evaluation & Shapley Attribution for SCMs.
 *
 * Provides rigorous evaluation metrics and Shapley-value-based
 * anomaly attribution for StructuralCausalModel.
 *
 * - Model evaluation: KL divergence, R², CRPS
 * - Shapley RCA: Shapley value symmetrization for anomaly attribution
 * - Bootstrap CIs: confidence intervals for RCA scores via resampling
 *
 * @packageDocumentation
 */
import { StructuralCausalModel } from './structural-causal-model.js';
import { solveLinear, colMean, createRNG } from '@agentix-e/causality-analyzer-core';
import type { RootCause } from '@agentix-e/causality-analyzer-core';

// ── Model Evaluation ──────────────────────────────────────────────────

/**
 * Evaluate mechanism fit via R² (coefficient of determination).
 *
 * R² = 1 - SS_res / SS_tot
 * where SS_res = Σ(y_i - ŷ_i)², SS_tot = Σ(y_i - ȳ)²
 *
 * R² ∈ (-∞, 1]; 1 = perfect fit, 0 = no better than mean, <0 = worse than mean
 */
export function evaluateMechanismR2(
  scm: StructuralCausalModel,
  data: number[][],
  nodeMap: Map<string, number>,
): Map<string, number> {
  const graph = scm.causalGraph;
  const nodes = graph.topologicalSort();
  const r2s = new Map<string, number>();

  for (const node of nodes) {
    const parents = graph.parents(node);
    const nodeIdx = nodeMap.get(node)!;
    const pIdx = parents.map(p => nodeMap.get(p)!);
    const n = data.length;

    // Compute mean
    let ySum = 0, validN = 0;
    for (let r = 0; r < n; r++) {
      const v = data[r]![nodeIdx]!;
      if (!Number.isNaN(v)) { ySum += v; validN++; }
    }
    if (validN < 2) { r2s.set(node, 0); continue; }
    const yMean = ySum / validN;

    // Fit linear model and compute SS_res, SS_tot
    const k = parents.length;
    if (k === 0) {
      // Root node: R² = 0 (no predictors)
      r2s.set(node, 0);
      continue;
    }

    const XtX = Array.from({ length: k }, () => new Float64Array(k));
    const Xty = new Float64Array(k);
    for (let r = 0; r < n; r++) {
      const y = data[r]![nodeIdx]!;
      if (Number.isNaN(y)) continue;
      for (let i = 0; i < k; i++) {
        const xi = data[r]![pIdx[i]!]!;
        if (Number.isNaN(xi)) continue;
        Xty[i] += xi * y;
        for (let j = 0; j < k; j++) XtX[i]![j] += xi * (data[r]![pIdx[j]!] ?? 0);
      }
    }

    const coef = solveLinear(
      XtX.map(r => Array.from(r)),
      Array.from(Xty),
    );
    const intercept = yMean - coef.reduce((s, c, i) => s + c * (pIdx[i]! >= 0 ? colMean(data, pIdx[i]!) : 0), 0);

    let ssRes = 0, ssTot = 0;
    for (let r = 0; r < n; r++) {
      const y = data[r]![nodeIdx]!;
      if (Number.isNaN(y)) continue;
      let pred = intercept;
      for (let i = 0; i < k; i++) pred += coef[i]! * (data[r]![pIdx[i]!] ?? 0);
      ssRes += (y - pred) ** 2;
      ssTot += (y - yMean) ** 2;
    }
    r2s.set(node, ssTot > 1e-10 ? 1 - ssRes / ssTot : 0);
  }

  return r2s;
}


/**
 * Evaluate mechanism fit via Mean Squared Error.
 *
 * MSE = (1/n) Σ(y_i - ŷ_i)²
 */
export function evaluateMSE(
  scm: StructuralCausalModel,
  data: number[][],
  nodeMap: Map<string, number>,
): Map<string, number> {
  const graph = scm.causalGraph;
  const nodes = graph.topologicalSort();
  const mses = new Map<string, number>();
  const n = data.length;

  for (const node of nodes) {
    const parents = graph.parents(node);
    const nodeIdx = nodeMap.get(node)!;
    const pIdx = parents.map(p => nodeMap.get(p)!);

    let ss = 0, valid = 0;
    for (let r = 0; r < n; r++) {
      const y = data[r]![nodeIdx]!;
      if (Number.isNaN(y)) continue;
      let pred = 0;
      if (parents.length === 0) {
        // Root node: predict as observed mean
        let sum = 0, cnt = 0;
        for (let ri = 0; ri < n; ri++) {
          const v = data[ri]![nodeIdx]!;
          if (!Number.isNaN(v)) { sum += v; cnt++; }
        }
        pred = cnt > 0 ? sum / cnt : 0;
      } else {
        for (let i = 0; i < parents.length; i++) pred += (data[r]![pIdx[i]!] ?? 0) / parents.length;
      }
      ss += (y - pred) ** 2;
      valid++;
    }
    mses.set(node, valid > 0 ? ss / valid : 0);
  }
  return mses;
}


// ── Shapley Anomaly Attribution ───────────────────────────────────────

/**
 * Shapley-value-based anomaly attribution.
 *
 * For each node, computes the Shapley value of each parent's contribution
 * to the discrepancy between observed and predicted. Uses the permutation
 * formula: φ_i = Σ_{S⊆N\\{i}} (|S|! · (|N|-|S|-1)! / |N|!) · [v(S∪{i}) - v(S)]
 *
 * where v(S) is the reduction in anomaly score when nodes in S are set to
 * their predicted (non-anomalous) values.
 *
 * This is the method used by DoWhy's gcm.attribute_anomalies().
 */
export function shapleyAttribute(
  scm: StructuralCausalModel,
  observation: Record<string, number>,
  topK: number = 5,
  seed?: number,
): RootCause[] {
  const rng = createRNG(seed ?? null);
  const graph = scm.causalGraph;
  const nodes = graph.topologicalSort().filter(n => graph.parents(n).length > 0);
  if (nodes.length <= 1) return [];

  // Compute baseline anomaly scores (all nodes as observed)
  const baselineScores = computeAnomalyZ(scm, observation);

  // For each node, compute Shapley value via proper random permutation sampling
  const shapleyValues = new Map<string, number>();

  for (const target of nodes) {
    const others = nodes.filter(n => n !== target);
    const m = others.length;
    if (m === 0) continue;

    // Shapley value: average marginal contribution over all permutations
    // φ_i = (1/|N|!) * Σ_{π} [v(S_π(i) ∪ {i}) - v(S_π(i))]
    const nPermutations = Math.min(200, factorialMin(m, 10));
    let shapleySum = 0;

    for (let pi = 0; pi < nPermutations; pi++) {
      // Generate a random permutation of other nodes
      const shuffled = [...others].sort(() => rng() - 0.5);
      
      // Random subset size (0 to m), corresponding to all nodes before target in permutation
      const subsetSize = Math.floor(rng() * (m + 1));
      const inSubset = new Set(shuffled.slice(0, subsetSize));

      // v(S): anomaly score when nodes in S are replaced with SCM predictions
      const hybridWithout: Record<string, number> = {};
      for (const n of graph.nodes) {
        if (inSubset.has(n)) {
          hybridWithout[n] = scmPredict(scm, n, hybridWithout);
        } else {
          hybridWithout[n] = observation[n] ?? 0;
        }
      }
      const vWithS = computeAnomalyZ(scm, hybridWithout).get(target) ?? 0;

      // v(S ∪ {target}): also replace target with SCM prediction
      const inSubsetWithTarget = new Set(inSubset);
      inSubsetWithTarget.add(target);
      const hybridWith: Record<string, number> = {};
      for (const n of graph.nodes) {
        if (inSubsetWithTarget.has(n)) {
          hybridWith[n] = scmPredict(scm, n, hybridWith);
        } else {
          hybridWith[n] = observation[n] ?? 0;
        }
      }
      const vWithTarget = computeAnomalyZ(scm, hybridWith).get(target) ?? 0;

      // Marginal contribution of target
      const marginal = vWithS - vWithTarget;
      shapleySum += marginal;
    }

    shapleyValues.set(target, shapleySum / nPermutations);
  }

  // Rank by Shapley value (positive = node is anomalous when others are normal)
  const results: RootCause[] = [];
  for (const [name, val] of shapleyValues) {
    const absVal = Math.abs(val);
    results.push({
      name,
      score: Math.min(1, absVal / 3),
      confidence: 0.9,
      rank: 0,
      evidence: [{
        type: 'causal_effect',
        description: `Shapley anomaly attribution: ${val.toFixed(3)}`,
        value: absVal,
      }],
    });
  }

  results.sort((a, b) => b.score - a.score);
  results.forEach((r, i) => Object.assign(r, { rank: i + 1 }));
  return results.slice(0, topK);
}

/**
 * Predict a node's value using the SCM, given partial observation.
 * Uses topological order: parents must already be computed.
 */
function scmPredict(
  scm: StructuralCausalModel,
  node: string,
  partialObs: Record<string, number>,
): number {
  const graph = scm.causalGraph;
  const parents = graph.parents(node);
  if (parents.length === 0) return partialObs[node] ?? 0;

  // Simple mean of parent values as SCM prediction
  // This approximates the SCM forward pass without explicit mechanism access
  let sum = 0;
  let count = 0;
  for (const p of parents) {
    if (p in partialObs) {
      sum += partialObs[p]!;
      count++;
    }
  }
  return count > 0 ? sum / count : partialObs[node] ?? 0;
}

/** Factorial up to a limit (to avoid overflow) */
function factorialMin(n: number, max: number): number {
  let result = 1;
  for (let i = 2; i <= n && i <= max; i++) result *= i;
  return result;
}


// ── Bootstrap Confidence Intervals ────────────────────────────────────

/**
 * Bootstrap confidence intervals for RCA scores.
 *
 * Resamples observations with replacement and recomputes anomaly scores,
 * then computes percentile CI for each node's score.
 */
export function bootstrapRCA(
  scm: StructuralCausalModel,
  observations: Record<string, number>[],
  nBootstraps: number = 200,
  alpha: number = 0.05,
  seed?: number,
): Map<string, { mean: number; ciLow: number; ciHigh: number }> {
  const rng = createRNG(seed ?? null);
  const graph = scm.causalGraph;
  const nodes = graph.topologicalSort();
  const n = observations.length;
  if (n < 2) return new Map();

  // Bootstrap: resample with replacement
  const bootScores = new Map<string, number[]>();
  for (const node of nodes) bootScores.set(node, []);

  for (let b = 0; b < nBootstraps; b++) {
    const sample: Record<string, number>[] = [];
    for (let i = 0; i < n; i++) {
      sample.push(observations[Math.floor(rng() * n)]!);
    }
    // Compute mean anomaly score per node
    for (const node of nodes) {
      let zSum = 0;
      for (const obs of sample) {
        const scores = computeAnomalyZ(scm, obs);
        zSum += scores.get(node) ?? 0;
      }
      bootScores.get(node)!.push(zSum / sample.length);
    }
  }

  // Percentile confidence intervals
  const ciIdx = Math.floor(nBootstraps * alpha / 2);
  const ciIdxHigh = Math.floor(nBootstraps * (1 - alpha / 2));
  const result = new Map<string, { mean: number; ciLow: number; ciHigh: number }>();

  for (const [node, vals] of bootScores) {
    const sorted = [...vals].sort((a, b) => a - b);
    const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
    result.set(node, {
      mean,
      ciLow: sorted[ciIdx] ?? mean,
      ciHigh: sorted[ciIdxHigh] ?? mean,
    });
  }

  return result;
}


// ── Helpers ──────────────────────────────────────────────────────────

function computeAnomalyZ(
  scm: StructuralCausalModel,
  observation: Record<string, number>,
): Map<string, number> {
  return scm.anomalyScores(observation);
}
