/**
 * CIRCA Algorithm: RHTScorer + DAScorer
 *
 * Based on "Causal Inference-Based Root Cause Analysis for Online Service
 * Systems with Intervention Recognition" (Li et al., KDD 2022).
 *
 * RHTScorer: Regression-based Hypothesis Testing — fits regression
 * models X = f(parents(X)) on normal data, then scores nodes by
 * residual deviation during failure.
 *
 * DAScorer: Descendant Adjustment — corrects scores for anomaly
 * propagation effects. Nodes whose anomalous parents explain their
 * anomaly get their scores reduced. Nodes with anomalous children
 * whose scores exceed threshold get bonus points.
 */
import { CausalGraph } from '../graph/causal-graph.js';
import type { RootCause, RootCausePath, RCAResult, Evidence } from '@agentix-e/causality-analyzer-core';
import { solveLinear, normalTail } from '@agentix-e/causality-analyzer-core';

// ── Types ─────────────────────────────────────────────────────────────
export interface RHTConfig {
  /** Maximum time lag for parent inclusion (τ_max) */
  tauMax: number;
  /** Aggregation method for residual z-scores */
  aggregator: 'max' | 'mean' | 'sum';
}

export interface DAConfig {
  /** Threshold for considering a node as "potentially anomalous" */
  threshold: number;
  /** Whether to apply descendant adjustment */
  enabled: boolean;
}

interface NodeModel {
  coef: number[];
  intercept: number;
  residualStd: number;
  parentIndices: number[];
}

// ── RHTScorer ───────────────────────────────────────────────────────
export class RHTScorer {
  private graph: CausalGraph | null = null;
  private models = new Map<string, NodeModel>();
  readonly config: RHTConfig;

  constructor(config: Partial<RHTConfig> = {}) {
    this.config = { tauMax: config.tauMax ?? 5, aggregator: config.aggregator ?? 'max' };
  }

  /** Train regression models for each node given its parents */
  train(graph: CausalGraph, normalData: number[][]): void {
    this.graph = graph;
    const nodes = [...graph.nodes];
    const nodeMap = new Map(nodes.map((n, i) => [n, i]));

    for (const node of nodes) {
      const parents = graph.parents(node);
      const nodeIdx = nodeMap.get(node)!;
      const pIdx = parents.map(p => nodeMap.get(p)!);
      const n = normalData.length;

      if (parents.length === 0) {
        // Root node: no predictors
        const vals = normalData.map(r => r[nodeIdx]!).filter(v => !Number.isNaN(v));
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const ss = vals.reduce((s, v) => s + (v - mean) ** 2, 0);
        this.models.set(node, {
          coef: [], intercept: mean,
          residualStd: Math.sqrt(ss / Math.max(1, vals.length - 1)),
          parentIndices: [],
        });
        continue;
      }

      // OLS regression: X = β₀ + Σ βᵢ * parentᵢ + ε
      const k = parents.length;
      const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
      const Xty = new Array(k).fill(0);
      let ySum = 0, validN = 0;

      for (let r = 0; r < n; r++) {
        const y = normalData[r]![nodeIdx]!;
        if (Number.isNaN(y)) continue;
        ySum += y; validN++;
        for (let i = 0; i < k; i++) {
          const xi = normalData[r]![pIdx[i]!]!;
          if (Number.isNaN(xi)) continue;
          Xty[i] += xi * y;
          for (let j = 0; j < k; j++) {
            const xj = normalData[r]![pIdx[j]!]!;
            if (!Number.isNaN(xj)) XtX[i]![j] += xi * xj;
          }
        }
      }

      const coef = solveLinear(XtX, Xty);
      const yMean = validN > 0 ? ySum / validN : 0;
      const intercept = yMean - coef.reduce((s, c, i) => s + c * ((validN > 0 ? Xty[i]! / validN : 0)), 0);

      // Residual standard deviation
      let ss = 0, cnt = 0;
      for (let r = 0; r < n; r++) {
        let pred = intercept;
        for (let i = 0; i < k; i++) pred += coef[i]! * (normalData[r]![pIdx[i]!] ?? 0);
        const residual = (normalData[r]![nodeIdx]! ?? 0) - pred;
        if (!Number.isNaN(residual)) { ss += residual ** 2; cnt++; }
      }
      const residualStd = Math.sqrt(ss / Math.max(1, cnt - k - 1)) || 1e-6;

      this.models.set(node, { coef, intercept, residualStd, parentIndices: pIdx });
    }
  }

  /** Score nodes by residual z-score deviation in failure window */
  score(anomalyData: number[][]): Map<string, { zScore: number; confidence: number }> {
    const scores = new Map<string, { zScore: number; confidence: number }>();
    if (!this.graph) return scores;

    const nodes = [...this.graph.nodes];
    const nodeMap = new Map(nodes.map((n, i) => [n, i]));
    const n = anomalyData.length;

    for (const node of nodes) {
      const model = this.models.get(node);
      if (!model) continue;
      const nodeIdx = nodeMap.get(node)!;

      const zScores: number[] = [];
      for (let r = 0; r < n; r++) {
        let pred = model.intercept;
        for (let i = 0; i < model.parentIndices.length; i++) {
          pred += model.coef[i]! * (anomalyData[r]![model.parentIndices[i]!] ?? 0);
        }
        const residual = (anomalyData[r]![nodeIdx]! ?? 0) - pred;
        zScores.push(Math.abs(residual) / model.residualStd);
      }

      const z = this.aggregate(zScores);
      const conf = 1 - 2 * normalTail(Math.abs(z));
      scores.set(node, { zScore: z, confidence: Math.max(0, Math.min(1, conf)) });
    }

    return scores;
  }

  private aggregate(values: number[]): number {
    if (values.length === 0) return 0;
    switch (this.config.aggregator) {
      case 'max': return Math.max(...values);
      case 'mean': return values.reduce((a, b) => a + b, 0) / values.length;
      case 'sum': return values.reduce((a, b) => a + b);
    }
  }
}

// ── DAScorer ────────────────────────────────────────────────────────
export class DAScorer {
  readonly config: DAConfig;

  constructor(config: Partial<DAConfig> = {}) {
    this.config = { threshold: config.threshold ?? 3.0, enabled: config.enabled ?? true };
  }

  /**
   * Apply descendant adjustment to RHT scores.
   * Returns adjusted scores sorted by key: (score, -topological_layer).
   */
  adjust(graph: CausalGraph, rhtScores: Map<string, { zScore: number; confidence: number }>): RootCause[] {
    const nodes = [...graph.nodes];
    // Topological layers
    const layers = this.topologicalLayers(graph);
    const nodeLayer = new Map<string, number>();
    layers.forEach((layer, lvl) => layer.forEach(n => nodeLayer.set(n, lvl)));

    // Build child score map (bottom-up)
    const childScores = new Map<string, number>();
    for (let l = layers.length - 1; l >= 0; l--) {
      for (const node of layers[l]!) {
        let childScore = 0;
        for (const child of graph.children(node)) {
          const childRHT = rhtScores.get(child);
          if (childRHT && childRHT.zScore >= this.config.threshold) {
            childScore = Math.max(childScore, childRHT.zScore);
          }
        }
        childScores.set(node, childScore);
      }
    }

    // Compute adjusted scores
    const results: RootCause[] = [];
    for (const node of nodes) {
      const rht = rhtScores.get(node) ?? { zScore: 0, confidence: 0 };
      let adjustedScore = rht.zScore;

      if (this.config.enabled) {
        // If node has anomalous parents, reduce its score
        for (const parent of graph.parents(node)) {
          const parentRHT = rhtScores.get(parent);
          if (parentRHT && parentRHT.zScore >= this.config.threshold) {
            adjustedScore -= parentRHT.zScore * 0.5;
          }
        }
        // Bonus for having anomalous children
        const childScore = childScores.get(node) ?? 0;
        if (childScore > 0) adjustedScore += childScore * 0.3;
        adjustedScore = Math.max(0, adjustedScore);
      }

      const evidence: Evidence[] = [{
        type: 'regression_residual',
        description: `RHT z-score: ${rht.zScore.toFixed(2)}, DA adjusted: ${adjustedScore.toFixed(2)}`,
        value: adjustedScore,
      }];

      results.push({
        name: node,
        score: Math.min(1, adjustedScore / 10),
        confidence: rht.confidence,
        rank: 0,
        evidence,
      });
    }

    // Sort by adjusted score, tie-break by topological layer (root-first)
    results.sort((a, b) => {
      if (Math.abs(b.score - a.score) > 0.001) return b.score - a.score;
      return (nodeLayer.get(a.name) ?? 0) - (nodeLayer.get(b.name) ?? 0);
    });
    results.forEach((r, i) => (r as { rank: number }).rank = i + 1);

    return results;
  }

  private topologicalLayers(graph: CausalGraph): string[][] {
    const nodes = [...graph.nodes];
    const inDegree = new Map(nodes.map(n => [n, graph.parents(n).length]));
    const layers: string[][] = [];
    const queue = nodes.filter(n => inDegree.get(n) === 0);

    while (queue.length > 0) {
      layers.push([...queue]);
      const next: string[] = [];
      for (const u of queue) {
        for (const v of graph.children(u)) {
          const deg = (inDegree.get(v) ?? 1) - 1;
          inDegree.set(v, deg);
          if (deg === 0) next.push(v);
        }
      }
      queue.length = 0;
      queue.push(...next);
    }
    return layers;
  }
}

// ── CIRCA Pipeline ──────────────────────────────────────────────────
export class CIRCAPipeline {
  private rht: RHTScorer;
  private da: DAScorer;
  private graph: CausalGraph | null = null;

  constructor(rhtConfig?: Partial<RHTConfig>, daConfig?: Partial<DAConfig>) {
    this.rht = new RHTScorer(rhtConfig);
    this.da = new DAScorer(daConfig);
  }

  train(graph: CausalGraph, normalData: number[][]): void {
    this.graph = graph;
    this.rht.train(graph, normalData);
  }

  analyze(anomalyData: number[][], anomalousNodes: string[]): RCAResult {
    if (!this.graph) return { rootCauses: [], paths: [], metadata: { method: 'circa', analyzedAt: Date.now(), durationMs: 0, extra: {} }, toJSON() { return { rootCauses: [], paths: [] }; } };

    const rhtScores = this.rht.score(anomalyData);
    const rootCauses = this.da.adjust(this.graph, rhtScores);

    // Build paths from root causes to anomalous nodes
    const paths: RootCausePath[] = [];
    for (const rc of rootCauses.slice(0, 5)) {
      for (const anom of anomalousNodes) {
        const path = this.shortestPath(rc.name, anom);
        if (path.length > 0) {
          paths.push({ nodes: path, score: rc.score, direction: 'forward' });
        }
      }
    }

    return {
      rootCauses: rootCauses.slice(0, 5),
      paths,
      metadata: { method: 'circa', analyzedAt: Date.now(), durationMs: 0, extra: {} },
      toJSON() { return { rootCauses, paths, metadata: this.metadata }; },
    };
  }

  private shortestPath(from: string, to: string): string[] {
    if (from === to || !this.graph) return [];
    const dist = new Map<string, number>();
    const prev = new Map<string, string | null>();
    for (const n of this.graph.nodes) { dist.set(n, Infinity); prev.set(n, null); }
    dist.set(from, 0);
    const q = [from];
    while (q.length > 0) {
      const u = q.shift()!;
      if (u === to) break;
      for (const v of this.graph!.children(u)) {
        if ((dist.get(v) ?? Infinity) > (dist.get(u) ?? Infinity) + 1) { dist.set(v, dist.get(u)! + 1); prev.set(v, u); q.push(v); }
      }
    }
    if (!prev.get(to) && from !== to) return [];
    const path = [to];
    let cur: string | null = to;
    while ((cur = prev.get(cur!) ?? null)) path.unshift(cur);
    return path;
  }
}

