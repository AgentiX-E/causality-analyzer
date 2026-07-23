/**
 * Graphical Causal Model (GCM) with Counterfactual Reasoning.
 *
 * Implements Pearl's three-step counterfactual framework:
 * 1. Abduction — infer exogenous noise from observed data
 * 2. Action   — apply do-operator (mutilate graph)
 * 3. Prediction — forward-simulate with inferred noise
 *
 * Supports two mechanism types:
 * - AdditiveNoise: X = β₀ + Σ βᵢ·paᵢ + ε (linear)
 * - PostNonlinear: X = g(β₀ + Σ βᵢ·paᵢ + ε) where g = sigmoid (nonlinear)
 *
 * Auto-assignment selects the best mechanism per node via R² comparison.
 * Also provides: anomaly attribution, distribution change detection,
 * interventional/counterfactual sampling, and CausalModel unified API.
 */
import { CausalGraph } from '../graph/causal-graph.js';
import type { RootCause } from '@agentix-e/causality-analyzer-core';
import { solveLinear, normalTail, createRNG } from '@agentix-e/causality-analyzer-core';

// ── Causal Mechanism ──────────────────────────────────────────────────
interface CausalMechanism {
  forward(parentValues: number[]): number;
  invert(x: number, parentValues: number[]): number;
  readonly noiseStd: number;
  readonly nodeName: string;
}

class AdditiveNoiseMechanism implements CausalMechanism {
  readonly nodeName: string;
  private coef: number[] = [];
  private intercept = 0;
  readonly noiseStd: number;

  constructor(nodeName: string, coef: number[], intercept: number, noiseStd: number) {
    this.nodeName = nodeName; this.coef = coef; this.intercept = intercept; this.noiseStd = noiseStd;
  }

  forward(parentValues: number[]): number {
    let result = this.intercept;
    for (let i = 0; i < this.coef.length; i++) result += this.coef[i]! * (parentValues[i] ?? 0);
    return result;
  }

  invert(x: number, parentValues: number[]): number {
    return x - this.forward(parentValues);
  }
}

/**
 * PostNonlinear mechanism: X = g(β₀ + Σ βᵢ·paᵢ + ε)
 * where g is sigmoid scaled to output range.
 */
class PostNonlinearMechanism implements CausalMechanism {
  readonly nodeName: string;
  private coef: number[] = [];
  private intercept = 0;
  readonly noiseStd: number;
  private yMin: number;
  private yMax: number;

  constructor(
    nodeName: string, coef: number[], intercept: number,
    noiseStd: number, yMin: number, yMax: number,
  ) {
    this.nodeName = nodeName; this.coef = coef; this.intercept = intercept;
    this.noiseStd = noiseStd; this.yMin = yMin; this.yMax = yMax;
  }

  private sigmoid(z: number): number {
    if (z > 15) return 1; if (z < -15) return 0;
    return 1 / (1 + Math.exp(-z));
  }

  private inverseSigmoid(y: number): number {
    const eps = 1e-8, range = this.yMax - this.yMin;
    if (range < 1e-8) return 0;
    const s = Math.max(eps, Math.min(1 - eps, (y - this.yMin) / range));
    return Math.log(s / (1 - s));
  }

  forward(parentValues: number[]): number {
    let z = this.intercept;
    for (let i = 0; i < this.coef.length; i++) z += this.coef[i]! * (parentValues[i] ?? 0);
    return this.yMin + this.sigmoid(z) * (this.yMax - this.yMin);
  }

  invert(x: number, parentValues: number[]): number {
    const gInv = this.inverseSigmoid(x);
    let fpa = this.intercept;
    for (let i = 0; i < this.coef.length; i++) fpa += this.coef[i]! * (parentValues[i] ?? 0);
    return gInv - fpa;
  }
}

export type MechanismType = 'additive_noise' | 'post_nonlinear';

// ── Structural Causal Model ──────────────────────────────────────────
export class StructuralCausalModel {
  private graph: CausalGraph;
  private mechanisms = new Map<string, CausalMechanism>();
  private nodeOrder: string[] = [];

  constructor(graph: CausalGraph) {
    this.graph = graph;
    this.nodeOrder = graph.topologicalSort();
  }

  /**
   * Train mechanisms from data. Uses auto-assign when no manual types given.
   * @param mechanismTypes — optional per-node mechanism override
   */
  train(data: number[][], mechanismTypes?: Record<string, MechanismType>): void {
    const nodes = [...this.graph.nodes];
    const nodeMap = new Map(nodes.map((n, i) => [n, i]));
    const n = data.length;

    for (const node of this.nodeOrder) {
      const parents = this.graph.parents(node);
      const nodeIdx = nodeMap.get(node)!;
      const pIdx = parents.map(p => nodeMap.get(p)!);
      const k = parents.length;

      if (k === 0) {
        const vals = data.map(r => r[nodeIdx]!).filter(v => !Number.isNaN(v));
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const ss = vals.reduce((s, v) => s + (v - mean) ** 2, 0);
        const std = Math.sqrt(ss / Math.max(1, vals.length - 1));
        this.mechanisms.set(node, new AdditiveNoiseMechanism(node, [], mean, std || 1));
        continue;
      }

      const validRows: number[] = [];
      for (let r = 0; r < n; r++) {
        if (Number.isNaN(data[r]![nodeIdx]!)) continue;
        if (pIdx.some(i => Number.isNaN(data[r]![i]!))) continue;
        validRows.push(r);
      }
      if (validRows.length < k + 2) {
        this.mechanisms.set(node, new AdditiveNoiseMechanism(node, new Array(k).fill(0), 0, 1));
        continue;
      }

      const specifiedType = mechanismTypes?.[node];
      const isAdditive = specifiedType === 'additive_noise' ||
        (specifiedType !== 'post_nonlinear' && this._chooseAdditive(data, validRows, nodeIdx, pIdx));

      if (isAdditive) {
        this._trainAdditive(data, validRows, node, nodeIdx, pIdx);
      } else {
        this._trainPostNonlinear(data, validRows, node, nodeIdx, pIdx);
      }
    }
  }

  private _chooseAdditive(data: number[][], rows: number[], nodeIdx: number, pIdx: number[]): boolean {
    const { ssRes } = this._fitLinear(data, rows, nodeIdx, pIdx);
    let ssTot = 0, ySum = 0;
    for (const r of rows) ySum += data[r]![nodeIdx]!;
    const yMean = ySum / rows.length;
    for (const r of rows) ssTot += (data[r]![nodeIdx]! - yMean) ** 2;
    return ssTot > 1e-10 ? 1 - ssRes / ssTot >= 0.9 : true;
  }

  private _fitLinear(data: number[][], rows: number[], nodeIdx: number, pIdx: number[]): { coef: number[]; intercept: number; ssRes: number } {
    const k = pIdx.length, n = rows.length;
    const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
    const Xty = new Array(k).fill(0);
    let ySum = 0;
    for (const r of rows) {
      const y = data[r]![nodeIdx]!; ySum += y;
      for (let i = 0; i < k; i++) {
        const xi = data[r]![pIdx[i]!]!; Xty[i] += xi * y;
        for (let j = 0; j < k; j++) XtX[i]![j] += xi * data[r]![pIdx[j]!]!;
      }
    }
    const coef = solveLinear(XtX, Xty);
    const yMean = ySum / n;
    const xMeans = pIdx.map((_, i) => { let s = 0; for (const r of rows) s += data[r]![pIdx[i]!]!; return s / n; });
    const intercept = yMean - coef.reduce((s, c, i) => s + c * (xMeans[i] ?? 0), 0);
    let ssRes = 0;
    for (const r of rows) {
      let pred = intercept;
      for (let i = 0; i < k; i++) pred += coef[i]! * data[r]![pIdx[i]!]!;
      ssRes += (data[r]![nodeIdx]! - pred) ** 2;
    }
    return { coef, intercept, ssRes };
  }

  private _trainAdditive(data: number[][], rows: number[], node: string, nodeIdx: number, pIdx: number[]): void {
    const k = pIdx.length;
    const { coef, intercept, ssRes } = this._fitLinear(data, rows, nodeIdx, pIdx);
    const std = Math.sqrt(ssRes / Math.max(1, rows.length - k - 1)) || 1;
    this.mechanisms.set(node, new AdditiveNoiseMechanism(node, coef, intercept, std));
  }

  private _trainPostNonlinear(data: number[][], rows: number[], node: string, nodeIdx: number, pIdx: number[]): void {
    const k = pIdx.length, n = rows.length;
    let yMin = Infinity, yMax = -Infinity;
    for (const r of rows) { const v = data[r]![nodeIdx]!; if (v < yMin) yMin = v; if (v > yMax) yMax = v; }
    const range = (yMax - yMin) || 1, eps = 1e-8;

    const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
    const Xty = new Array(k).fill(0);
    let zSum = 0;
    for (const r of rows) {
      const s = Math.max(eps, Math.min(1 - eps, (data[r]![nodeIdx]! - yMin) / range));
      const z = Math.log(s / (1 - s)); zSum += z;
      for (let i = 0; i < k; i++) {
        const xi = data[r]![pIdx[i]!]!; Xty[i] += xi * z;
        for (let j = 0; j < k; j++) XtX[i]![j] += xi * data[r]![pIdx[j]!]!;
      }
    }
    const coef = solveLinear(XtX, Xty);
    const zMean = zSum / n;
    const xMeans = pIdx.map((_, i) => { let s = 0; for (const r of rows) s += data[r]![pIdx[i]!]!; return s / n; });
    const intercept = zMean - coef.reduce((s, c, i) => s + c * (xMeans[i] ?? 0), 0);

    let ss = 0;
    for (const r of rows) {
      let pred = intercept;
      for (let i = 0; i < k; i++) pred += coef[i]! * data[r]![pIdx[i]!]!;
      const s = Math.max(eps, Math.min(1 - eps, (data[r]![nodeIdx]! - yMin) / range));
      ss += (Math.log(s / (1 - s)) - pred) ** 2;
    }
    const std = Math.sqrt(ss / Math.max(1, n - k - 1)) || 0.1;
    this.mechanisms.set(node, new PostNonlinearMechanism(node, coef, intercept, std, yMin, yMax));
  }

  // ── Counterfactual ──────────────────────────────────────────────
  abduct(observation: Record<string, number>): Record<string, number> {
    const noise: Record<string, number> = {};
    for (const node of this.nodeOrder) {
      const mech = this.mechanisms.get(node);
      if (!mech) continue;
      noise[node] = mech.invert(observation[node] ?? 0, this.graph.parents(node).map(p => observation[p] ?? 0));
    }
    return noise;
  }

  counterfactual(noise: Record<string, number>, intervention: Record<string, number>): Record<string, number> {
    const result: Record<string, number> = { ...intervention };
    for (const node of this.nodeOrder) {
      if (node in intervention) continue;
      const mech = this.mechanisms.get(node);
      if (!mech) continue;
      result[node] = mech.forward(this.graph.parents(node).map(p => result[p] ?? 0)) + (noise[node] ?? 0);
    }
    return result;
  }

  counterfactualSamples(noise: Record<string, number>, intervention: Record<string, number>, nSamples: number, seed?: number): Record<string, number>[] {
    const rng = createRNG(seed ?? null);
    const noiseStd = new Map<string, number>();
    for (const [node, mech] of this.mechanisms) noiseStd.set(node, mech.noiseStd);
    const samples: Record<string, number>[] = [];
    for (let s = 0; s < nSamples; s++) {
      const rn: Record<string, number> = {};
      for (const node of this.nodeOrder) {
        const u1 = Math.max(rng(), 1e-10), u2 = Math.max(rng(), 1e-10);
        const g = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        rn[node] = (noise[node] ?? 0) + g * (noiseStd.get(node) ?? 0.1) * 0.3;
      }
      samples.push(this.counterfactual(rn, intervention));
    }
    return samples;
  }

  interventionalSamples(intervention: Record<string, number>, nSamples: number, seed?: number): Record<string, number>[] {
    const rng = createRNG(seed ?? null);
    const noiseStd = new Map<string, number>();
    for (const [node, mech] of this.mechanisms) noiseStd.set(node, mech.noiseStd);
    const samples: Record<string, number>[] = [];
    for (let s = 0; s < nSamples; s++) {
      const result: Record<string, number> = { ...intervention };
      for (const node of this.nodeOrder) {
        if (node in intervention) continue;
        const mech = this.mechanisms.get(node);
        if (!mech) continue;
        const u1 = Math.max(rng(), 1e-10), u2 = Math.max(rng(), 1e-10);
        const g = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        result[node] = mech.forward(this.graph.parents(node).map(p => result[p] ?? 0)) + g * (noiseStd.get(node) ?? 0.1);
      }
      samples.push(result);
    }
    return samples;
  }

  // ── Anomaly Attribution ─────────────────────────────────────────
  anomalyScores(observation: Record<string, number>): Map<string, number> {
    const scores = new Map<string, number>();
    for (const node of this.nodeOrder) {
      const mech = this.mechanisms.get(node);
      if (!mech) continue;
      const predicted = mech.forward(this.graph.parents(node).map(p => observation[p] ?? 0));
      const z = Math.abs((observation[node] ?? 0) - predicted) / (mech.noiseStd || 1);
      scores.set(node, z);
    }
    return scores;
  }

  attributeAnomalies(observation: Record<string, number>, topK: number = 3): RootCause[] {
    const scores = this.anomalyScores(observation);
    const results: RootCause[] = [];
    for (const [name, z] of scores) {
      results.push({
        name, score: Math.min(1, z / 5), confidence: 1 - 2 * normalTail(z),
        rank: 0, evidence: [{ type: 'causal_effect', description: `SCM noise z-score: ${z.toFixed(2)}`, value: z }],
      });
    }
    results.sort((a, b) => b.score - a.score);
    results.forEach((r, i) => (r as { rank: number }).rank = i + 1);
    return results.slice(0, topK);
  }

  detectDistributionChange(before: Record<string, number>[], after: Record<string, number>[]): { changed: boolean; meanShift: number; pValue: number } {
    const bm = this._meanNoise(before), am = this._meanNoise(after);
    const diffs = this.nodeOrder.map(n => Math.abs((am[n] ?? 0) - (bm[n] ?? 0)));
    const meanDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
    return { changed: meanDiff > 1.0, meanShift: meanDiff, pValue: Math.exp(-meanDiff) };
  }

  private _meanNoise(data: Record<string, number>[]): Record<string, number> {
    const means: Record<string, number> = {};
    for (const node of this.nodeOrder) means[node] = 0;
    if (data.length === 0) return means;
    for (const d of data) {
      const n = this.abduct(d);
      for (const node of this.nodeOrder) means[node] += n[node] ?? 0;
    }
    for (const node of this.nodeOrder) means[node] /= data.length;
    return means;
  }

  get causalGraph(): CausalGraph { return this.graph; }
}

// ── CATE → RCA Bridge ─────────────────────────────────────────────
export function cateToRCA(treatmentEffects: Map<string, number>): RootCause[] {
  const results: RootCause[] = [];
  let rank = 1;
  const sorted = [...treatmentEffects.entries()].sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  for (const [name, effect] of sorted) {
    results.push({
      name, score: Math.min(1, Math.abs(effect) / 10),
      confidence: 0.8, rank: rank++,
      evidence: [{ type: 'causal_effect', description: `CATE: ${effect.toFixed(3)}`, value: Math.abs(effect) }],
    });
  }
  return results;
}

// ── CausalModel — Unified Causal Analysis API ────────────────────────
/**
 * Unified causal analysis API (DoWhy-compatible four-step pattern):
 * 1. model — causal graph
 * 2. identify — estimand
 * 3. estimate — effect
 * 4. refute — robustness
 *
 * Plus SCM counterfactual/interventional queries.
 */
export class CausalModel {
  private graph: CausalGraph;
  private scm: StructuralCausalModel | null = null;

  constructor(graph: CausalGraph) {
    this.graph = graph;
  }

  fitSCM(data: number[][], mechanismTypes?: Record<string, MechanismType>): this {
    this.scm = new StructuralCausalModel(this.graph);
    this.scm.train(data, mechanismTypes);
    return this;
  }

  counterfactual(observation: Record<string, number>, intervention: Record<string, number>): Record<string, number> | null {
    if (!this.scm) return null;
    return this.scm.counterfactual(this.scm.abduct(observation), intervention);
  }

  counterfactualSamples(observation: Record<string, number>, intervention: Record<string, number>, nSamples: number, seed?: number): Record<string, number>[] | null {
    if (!this.scm) return null;
    return this.scm.counterfactualSamples(this.scm.abduct(observation), intervention, nSamples, seed);
  }

  interventionalSamples(intervention: Record<string, number>, nSamples: number, seed?: number): Record<string, number>[] | null {
    if (!this.scm) return null;
    return this.scm.interventionalSamples(intervention, nSamples, seed);
  }

  attributeAnomalies(observation: Record<string, number>, topK?: number): RootCause[] | null {
    if (!this.scm) return null;
    return this.scm.attributeAnomalies(observation, topK);
  }

  get scmInstance(): StructuralCausalModel | null { return this.scm; }
  get causalGraph(): CausalGraph { return this.graph; }
}
