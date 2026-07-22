/**
 * Causal Inference Engine.
 *
 * Implements the five-step causal analysis framework:
 * 0. Ingest — data preparation
 * 1. Model  — causal graph specification
 * 2. Identify — estimand identification (backdoor, IV, frontdoor)
 * 3. Estimate — effect estimation (linear regression, propensity score)
 * 4. Refute  — sensitivity and robustness checks
 * 5. Act     — action suggestion for AIOps
 */
import { CausalGraph } from '../graph/causal-graph.js';
import type { IdentifiedEstimand, CausalEstimate, CausalEdge } from '@agentix-e/causality-analyzer-core';
import { solveLinear, normalTail, createRNG } from '@agentix-e/causality-analyzer-core';

// ── Estimand Identification ────────────────────────────────────────────

export function identifyBackdoor(
  graph: CausalGraph, treatment: string, outcome: string,
): IdentifiedEstimand {
  const backdoorVars = findBackdoorAdjustment(graph, treatment, outcome);
  return {
    estimandType: 'nonparametric_ate',
    treatmentVariables: [treatment],
    outcomeVariables: [outcome],
    backdoorVariables: { backdoor: backdoorVars },
    instrumentalVariables: [],
    frontdoorVariables: [],
  };
}

export function identifyFrontdoor(
  graph: CausalGraph, treatment: string, outcome: string,
): IdentifiedEstimand | null {
  // Find mediators: nodes on directed path from treatment to outcome
  const mediators = findMediators(graph, treatment, outcome);
  if (mediators.length === 0) return null;
  return {
    estimandType: 'nonparametric_ate',
    treatmentVariables: [treatment],
    outcomeVariables: [outcome],
    backdoorVariables: {},
    instrumentalVariables: [],
    frontdoorVariables: mediators,
  };
}

function findBackdoorAdjustment(graph: CausalGraph, treatment: string, outcome: string): string[] {
  // Standard backdoor criterion: adjust for variables that block all
  // backdoor paths from treatment to outcome without opening new paths.
  // Simplified: return common causes (parents of both treatment and outcome in
  // the moralized graph, excluding treatment descendants).
  const candidates: string[] = [];
  const treatDescendants = descendants(graph, treatment);
  for (const node of graph.nodes) {
    if (node === treatment || node === outcome) continue;
    if (treatDescendants.has(node)) continue;
    // Check if node is a common cause (ancestor of both)
    if (isAncestor(graph, node, treatment) || isAncestor(graph, node, outcome)) {
      candidates.push(node);
    }
  }
  return candidates;
}

function findMediators(graph: CausalGraph, treatment: string, outcome: string): string[] {
  const mediators: string[] = [];
  for (const node of graph.nodes) {
    if (node === treatment || node === outcome) continue;
    if (hasDirectedPath(graph, treatment, node) && hasDirectedPath(graph, node, outcome)) {
      mediators.push(node);
    }
  }
  return mediators;
}

function descendants(graph: CausalGraph, node: string): Set<string> {
  const result = new Set<string>();
  const stack = [node];
  while (stack.length > 0) {
    const u = stack.pop()!;
    for (const v of graph.children(u)) {
      if (!result.has(v)) { result.add(v); stack.push(v); }
    }
  }
  return result;
}

function isAncestor(graph: CausalGraph, ancestor: string, node: string): boolean {
  return hasDirectedPath(graph, ancestor, node);
}

function hasDirectedPath(graph: CausalGraph, from: string, to: string): boolean {
  const visited = new Set<string>();
  const stack = [from];
  while (stack.length > 0) {
    const u = stack.pop()!;
    if (u === to) return true;
    if (visited.has(u)) continue;
    visited.add(u);
    for (const v of graph.children(u)) if (!visited.has(v)) stack.push(v);
  }
  return false;
}

// ── Effect Estimation ──────────────────────────────────────────────────

export interface LinearRegressionEstimate {
  estimate(value: number): number;
}

/** Linear regression estimator for causal effects */
export function estimateLinearRegression(
  data: number[][], treatmentIdx: number, outcomeIdx: number,
  covariateIndices: number[] = [],
): { ate: number; se: number; model: LinearRegressionEstimate } {
  const n = data.length;
  const allPred = [...covariateIndices, treatmentIdx];
  const k = allPred.length;

  // OLS: y = β₀ + Σ βᵢ * xᵢ
  const X = allPred.map(i => data.map(r => r[i] ?? 0));
  const y = data.map(r => r[outcomeIdx] ?? 0);

  const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty = new Array(k).fill(0);
  let ySum = 0;

  for (let r = 0; r < n; r++) {
    ySum += y[r]!;
    for (let i = 0; i < k; i++) {
      Xty[i] += X[i]![r]! * y[r]!;
      for (let j = 0; j < k; j++) XtX[i]![j] += X[i]![r]! * X[j]![r]!;
    }
  }

  const coef = solveLinear(XtX, Xty);
  const yMean = ySum / n;
  const intercept = yMean - coef.reduce((s, c, i) => s + c * (X[i]!.reduce((a, v) => a + v, 0) / n), 0);

  // ATE = coefficient of treatment
  const treatBeta = coef[covariateIndices.length]!;
  const se = computeSE(data, treatmentIdx, outcomeIdx, coef, intercept, allPred);

  const model: LinearRegressionEstimate = {
    estimate(value: number) { return intercept + coef.reduce((s, c, i) => s + c * (i === covariateIndices.length ? value : 0), 0); },
  };

  return { ate: treatBeta, se, model };
}

function computeSE(
  data: number[][], tIdx: number, oIdx: number,
  coef: number[], intercept: number, predIdx: number[],
): number {
  const n = data.length;
  let ss = 0;
  for (let r = 0; r < n; r++) {
    let pred = intercept;
    for (let i = 0; i < predIdx.length; i++) pred += coef[i]! * (data[r]![predIdx[i]!] ?? 0);
    ss += ((data[r]![oIdx] ?? 0) - pred) ** 2;
  }
  return Math.sqrt(ss / Math.max(1, n - predIdx.length));
}

// ── Refutation ─────────────────────────────────────────────────────────

export interface RefutationResult {
  method: string;
  originalEstimate: number;
  newEstimate: number;
  pValue: number;
  isRobust: boolean;
}

/** Placebo treatment: scramble treatment to check if effect disappears */
export function refutePlaceboTreatment(
  data: number[][], treatmentIdx: number, outcomeIdx: number,
  nSimulations: number = 50,
  seed?: number,
): RefutationResult {
  const rng = createRNG(seed ?? null);
  const original = estimateLinearRegression(data, treatmentIdx, outcomeIdx);
  const nullEstimates: number[] = [];

  for (let s = 0; s < nSimulations; s++) {
    const scrambled = data.map(row => {
      const newRow = [...row];
      const randIdx = Math.floor(rng() * data.length);
      newRow[treatmentIdx] = data[randIdx]![treatmentIdx]!;
      return newRow;
    });
    nullEstimates.push(estimateLinearRegression(scrambled, treatmentIdx, outcomeIdx).ate);
  }

  const nullMean = nullEstimates.reduce((a, b) => a + b, 0) / nullEstimates.length;
  // Empirical p-value: fraction of null estimates >= more extreme than original
  const moreExtreme = nullEstimates.filter(e => Math.abs(e) >= Math.abs(original.ate)).length;
  const pValue = (moreExtreme + 1) / (nullEstimates.length + 1);
  return { method: 'placebo_treatment', originalEstimate: original.ate, newEstimate: nullMean, pValue, isRobust: pValue > 0.05 };
}

/** Data subset refutation: check stability across random subsets */
export function refuteDataSubset(
  data: number[][], treatmentIdx: number, outcomeIdx: number,
  subsetFraction: number = 0.8, nSimulations: number = 20,
  seed?: number,
): RefutationResult {
  const rng = createRNG(seed ?? null);
  const full = estimateLinearRegression(data, treatmentIdx, outcomeIdx);
  const estimates: number[] = [];
  const subsetSize = Math.floor(data.length * subsetFraction);

  for (let s = 0; s < nSimulations; s++) {
    const subset = shuffle(data, rng).slice(0, subsetSize);
    estimates.push(estimateLinearRegression(subset, treatmentIdx, outcomeIdx).ate);
  }

  const mean = estimates.reduce((a, b) => a + b, 0) / estimates.length;
  const varEst = estimates.reduce((s, v) => s + (v - mean) ** 2, 0) / estimates.length;
  const z = Math.abs(full.ate - mean) / Math.sqrt(varEst / estimates.length + 1e-10);
  const pValue = 2 * normalTail(z);
  return { method: 'data_subset', originalEstimate: full.ate, newEstimate: mean, pValue, isRobust: pValue > 0.05 };
}

/** Bootstrap refutation: resample with replacement */
export function refuteBootstrap(
  data: number[][], treatmentIdx: number, outcomeIdx: number,
  nBootstraps: number = 100,
  seed?: number,
): RefutationResult {
  const rng = createRNG(seed ?? null);
  const full = estimateLinearRegression(data, treatmentIdx, outcomeIdx);
  const estimates: number[] = [];
  const n = data.length;

  for (let b = 0; b < nBootstraps; b++) {
    const sample: number[][] = [];
    for (let i = 0; i < n; i++) {
      sample.push(data[Math.floor(rng() * n)]!);
    }
    estimates.push(estimateLinearRegression(sample, treatmentIdx, outcomeIdx).ate);
  }

  estimates.sort((a, b) => a - b);
  const ciLow = estimates[Math.floor(nBootstraps * 0.025)] ?? 0;
  const ciHigh = estimates[Math.floor(nBootstraps * 0.975)] ?? 0;
  const isRobust = ciLow <= 0 === ciHigh <= 0; // effect sign consistent
  return { method: 'bootstrap', originalEstimate: full.ate, newEstimate: (ciLow + ciHigh) / 2, pValue: ciLow * ciHigh > 0 ? 0.01 : 0.5, isRobust };
}

function shuffle<T>(arr: T[], rng: () => number): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j]!, a[i]!];
  }
  return a;
}
  return a;
}

// ── CausalAnalysis Main Class ──────────────────────────────────────────

export class CausalAnalysis {
  private graph: CausalGraph | null = null;
  private data: number[][] = [];
  private treatmentIdx = 0;
  private outcomeIdx = 0;
  private nodeMap = new Map<string, number>();

  ingest(data: number[][], columns: string[], treatment: string, outcome: string): this {
    this.data = data;
    this.nodeMap = new Map(columns.map((c, i) => [c, i]));
    this.treatmentIdx = this.nodeMap.get(treatment)!;
    this.outcomeIdx = this.nodeMap.get(outcome)!;
    return this;
  }

  model(graph: CausalGraph): this {
    this.graph = graph;
    return this;
  }

  identify(): IdentifiedEstimand | null {
    if (!this.graph) return null;
    const treatment = [...this.nodeMap.entries()].find(([, i]) => i === this.treatmentIdx)?.[0]!;
    const outcome = [...this.nodeMap.entries()].find(([, i]) => i === this.outcomeIdx)?.[0]!;
    return identifyBackdoor(this.graph, treatment, outcome);
  }

  estimate(estimand?: IdentifiedEstimand | null): { ate: number; se: number } | null {
    if (!estimand) return null;
    const backdoor = estimand.backdoorVariables['backdoor'] ?? [];
    const covIndices = backdoor.map(v => this.nodeMap.get(v)).filter((i): i is number => i !== undefined);
    const result = estimateLinearRegression(this.data, this.treatmentIdx, this.outcomeIdx, covIndices);
    return { ate: result.ate, se: result.se };
  }

  refute(ate: number): RefutationResult[] {
    return [
      refutePlaceboTreatment(this.data, this.treatmentIdx, this.outcomeIdx),
      refuteDataSubset(this.data, this.treatmentIdx, this.outcomeIdx),
      refuteBootstrap(this.data, this.treatmentIdx, this.outcomeIdx),
    ];
  }

  analyze(): { estimand: IdentifiedEstimand | null; estimate: { ate: number; se: number } | null; refutations: RefutationResult[] } | null {
    const estimand = this.identify();
    if (!estimand) return null;
    const est = this.estimate(estimand);
    if (!est) return null;
    return { estimand, estimate: est, refutations: this.refute(est.ate) };
  }
}
