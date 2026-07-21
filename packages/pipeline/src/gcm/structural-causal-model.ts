/**
 * Graphical Causal Model (GCM) with Counterfactual Reasoning.
 *
 * Implements Pearl's three-step counterfactual framework:
 * 1. Abduction — infer exogenous noise from observed data
 * 2. Action   — apply do-operator (mutilate graph)
 * 3. Prediction — forward-simulate with inferred noise
 *
 * Also provides anomaly attribution and distribution change detection.
 */
import { CausalGraph } from '../graph/causal-graph.js';
import type { CausalityAnalyzerConfig, DetectionResult, RootCause, RCAResult } from '@agentix-e/causality-analyzer-core';
import { solveLinear, normalTail } from '@agentix-e/causality-analyzer-core';

// ── Causal Mechanism ──────────────────────────────────────────────────
/** A learnable causal mechanism: X = f(parents) + noise */
interface CausalMechanism {
  /** Forward: compute E[X | parents] */
  forward(parentValues: number[]): number;
  /** Inverse: given X and parents, recover noise */
  invert(x: number, parentValues: number[]): number;
  /** Standard deviation of noise (for anomaly scoring) */
  readonly noiseStd: number;
  readonly nodeName: string;
}

class AdditiveNoiseMechanism implements CausalMechanism {
  readonly nodeName: string;
  private coef: number[] = [];
  private intercept = 0;
  readonly noiseStd: number;

  constructor(nodeName: string, coef: number[], intercept: number, noiseStd: number) {
    this.nodeName = nodeName;
    this.coef = coef;
    this.intercept = intercept;
    this.noiseStd = noiseStd;
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

// ── Structural Causal Model ──────────────────────────────────────────
export class StructuralCausalModel {
  private graph: CausalGraph;
  private mechanisms = new Map<string, CausalMechanism>();
  private nodeOrder: string[] = [];

  constructor(graph: CausalGraph) {
    this.graph = graph;
    this.nodeOrder = graph.topologicalSort();
  }

  /** Train mechanisms from data: X_i = f(pa(X_i)) + ε */
  train(data: number[][]): void {
    const nodes = [...this.graph.nodes];
    const nodeMap = new Map(nodes.map((n, i) => [n, i]));
    const n = data.length;

    for (const node of this.nodeOrder) {
      const parents = this.graph.parents(node);
      const nodeIdx = nodeMap.get(node)!;
      const pIdx = parents.map(p => nodeMap.get(p)!);
      const k = parents.length;

      if (k === 0) {
        // Root node: X = μ + ε
        const vals = data.map(r => r[nodeIdx]!).filter(v => !Number.isNaN(v));
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const ss = vals.reduce((s, v) => s + (v - mean) ** 2, 0);
        const std = Math.sqrt(ss / Math.max(1, vals.length - 1));
        this.mechanisms.set(node, new AdditiveNoiseMechanism(node, [], mean, std || 1));

      } else {
        // OLS: X_i = β₀ + Σ βⱼ * paⱼ + ε
        const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
        const Xty = new Array(k).fill(0);
        let ySum = 0, validN = 0;
        for (let r = 0; r < n; r++) {
          const y = data[r]![nodeIdx]!;
          if (Number.isNaN(y)) continue;
          ySum += y; validN++;
          for (let i = 0; i < k; i++) {
            const xi = data[r]![pIdx[i]!]!;
            if (Number.isNaN(xi)) continue;
            Xty[i] += xi * y;
            for (let j = 0; j < k; j++) {
              const xj = data[r]![pIdx[j]!]!;
              if (!Number.isNaN(xj)) XtX[i]![j] += xi * xj;
            }
          }
        }
        const coef = solveLinear(XtX, Xty);
        const intercept = validN > 0 ? ySum / validN - coef.reduce((s, c, i) => s + c * ((validN > 0 ? Xty[i]! / validN : 0)), 0) : 0;
        let ss = 0;
        for (let r = 0; r < n; r++) {
          let pred = intercept;
          for (let i = 0; i < k; i++) pred += coef[i]! * (data[r]![pIdx[i]!] ?? 0);
          ss += ((data[r]![nodeIdx]! ?? 0) - pred) ** 2;
        }
        const std = Math.sqrt(ss / Math.max(1, n - k - 1)) || 1;
        this.mechanisms.set(node, new AdditiveNoiseMechanism(node, coef, intercept, std));
      }
    }
  }

  // ── Counterfactual ──────────────────────────────────────────────

  /** Abduction: infer noise from observation */
  abduct(observation: Record<string, number>): Record<string, number> {
    const noise: Record<string, number> = {};
    for (const node of this.nodeOrder) {
      const mech = this.mechanisms.get(node);
      if (!mech) continue;
      const parentValues = this.graph.parents(node).map(p => observation[p] ?? 0);
      noise[node] = mech.invert(observation[node] ?? 0, parentValues);
    }
    return noise;
  }

  /** Action + Prediction: apply intervention and forward-simulate */
  counterfactual(
    noise: Record<string, number>,
    intervention: Record<string, number>,
  ): Record<string, number> {
    const result: Record<string, number> = { ...intervention };
    for (const node of this.nodeOrder) {
      if (node in intervention) continue; // skip intervened nodes
      const mech = this.mechanisms.get(node);
      if (!mech) continue;
      const parentValues = this.graph.parents(node).map(p => result[p] ?? 0);
      result[node] = mech.forward(parentValues) + (noise[node] ?? 0);
    }
    return result;
  }

  // ── Anomaly Attribution ─────────────────────────────────────────

  /** Score each node by how anomalous its current value is under the SCM */
  anomalyScores(observation: Record<string, number>): Map<string, number> {
    const scores = new Map<string, number>();
    for (const node of this.nodeOrder) {
      const mech = this.mechanisms.get(node);
      if (!mech) continue;
      const parentValues = this.graph.parents(node).map(p => observation[p] ?? 0);
      const predicted = mech.forward(parentValues);
      const residual = (observation[node] ?? 0) - predicted;
      const z = Math.abs(residual) / (mech.noiseStd || 1);
      scores.set(node, z);
    }
    return scores;
  }

  /** Attribute anomalies: find root nodes with highest anomaly scores */
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

  // ── Distribution Change Detection ────────────────────────────────

  /** Detect if the observation's anomaly pattern indicates distribution change */
  detectDistributionChange(before: Record<string, number>[], after: Record<string, number>[]): { changed: boolean; meanShift: number; pValue: number } {
    const beforeMeans = this.computeMeanNoise(before);
    const afterMeans = this.computeMeanNoise(after);
    const diffs: number[] = [];
    for (const node of this.nodeOrder) {
      diffs.push(Math.abs((afterMeans[node] ?? 0) - (beforeMeans[node] ?? 0)));
    }
    const meanDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
    const changed = meanDiff > 1.0;
    return { changed, meanShift: meanDiff, pValue: Math.exp(-meanDiff) };
  }

  private computeMeanNoise(data: Record<string, number>[]): Record<string, number> {
    const means: Record<string, number> = {};
    for (const node of this.nodeOrder) means[node] = 0;
    if (data.length === 0) return means;
    const noises = data.map(d => this.abduct(d));
    for (const node of this.nodeOrder) {
      let sum = 0;
      for (const n of noises) sum += n[node] ?? 0;
      means[node] = sum / data.length;
    }
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

