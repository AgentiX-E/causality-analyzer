/**
 * Generic RCA Agent — Causal Discovery + RCA + LLM Reasoning.
 *
 * The core agent pipeline is benchmark-agnostic:
 *   1. Causal discovery (PC algorithm, adaptive alpha) → service dependency graph
 *   2. Anomaly detection (MAD-based z-score, configurable detector)
 *   3. RCA ranking (HeuristicPath, configurable method)
 *   4. LLM reasoning (DeepSeek with graph + ranking context)
 *
 * Multi-fault support via iterative residual analysis:
 *   diagnoseMulti() identifies top cause, removes its effect, and re-ranks.
 *
 * Benchmark-specific concerns (data format, prediction format) are
 * handled by adapter functions passed to runners.
 *
 * @packageDocumentation
 */

import { Matrix } from 'ml-matrix';
import { pcAlgorithm } from '../graph/pc.js';
import { HeuristicPathRCA } from '../analyze/rca.js';
import { BOCDDetector } from '../detect/bocd.js';
import { MultiSourceRanker, type MultiSourceInput, type FusionRankEntry } from '../analyze/multi-source-ranker.js';
import type { CausalGraph } from '../graph/causal-graph.js';

// ── Types ────────────────────────────────────────────────────────────

/** Anomaly detection method */
export type AnomalyDetector = 'mad' | 'zscore';

/** RCA ranking method */
export type RCAMethod = 'heuristic-path';

export interface RCAgentConfig {
  /** LLM API key (read from env, never in code) */
  apiKey?: string;
  /** LLM model name (default: deepseek-chat) */
  model?: string;
  /** LLM base URL (default: https://api.deepseek.com) */
  baseUrl?: string;
  /** Anomaly detection method (default: mad) */
  anomalyDetector?: AnomalyDetector;
  /** Fraction of data tail used for anomaly detection (default: 0.2) */
  anomalyTailFraction?: number;
  /** MAD-based modified z-score threshold (default: 3.5). For zscore detector, use 0.3. */
  anomalyThreshold?: number;
  /** PC algorithm alpha — auto-computed if omitted (adaptive based on sample size) */
  pcAlpha?: number;
  /** Enable multi-fault iterative removal (default: false) */
  multiFault?: boolean;
}

export interface GraphNode {
  name: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

export interface CausalGraphResult {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface RCARankEntry {
  component: string;
  score: number;
  isRoot: boolean;
}

export interface RCADiagnosis {
  graph: CausalGraphResult;
  ranking: RCARankEntry[];
  anomalousServices: string[];
}

export interface RCAPrediction {
  component: string;
  datetime?: string;
  reason: string;
  rawLLMResponse: string;
}

// ── Adaptive PC Alpha ─────────────────────────────────────────────────

/**
 * Compute adaptive PC alpha based on sample size.
 *
 * On small datasets (≤200 points), CI tests are underpowered with α=0.05,
 * producing too few edges. On large datasets (≥500 points), α=0.05 is
 * appropriately strict. We linearly interpolate in between.
 */
function adaptiveAlpha(nSamples: number): number {
  if (nSamples >= 500) return 0.05;
  if (nSamples <= 200) return 0.10;
  // Linear interpolation: 0.10 at 200 → 0.05 at 500
  return 0.10 - 0.05 * (nSamples - 200) / 300;
}

// ── Anomaly Detection: MAD-based ──────────────────────────────────────

/**
 * Median Absolute Deviation (MAD).
 *
 * More robust than standard deviation for anomaly detection
 * because the median is unaffected by the anomalies themselves.
 *
 * Modified z-score formula (Iglewicz & Hoaglin 1993):
 *   M_i = 0.6745 * (x_i - median(X)) / MAD
 *
 * Threshold: |M_i| > 3.5 → anomalous (standard for MAD-based detection).
 */
function computeMAD(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const median = n % 2 === 1
    ? sorted[Math.floor(n / 2)]!
    : (sorted[n / 2 - 1]! + sorted[n / 2]!) / 2;
  const deviations = values.map(v => Math.abs(v - median));
  deviations.sort((a, b) => a - b);
  const m = deviations.length;
  const mad = m % 2 === 1
    ? deviations[Math.floor(m / 2)]!
    : (deviations[m / 2 - 1]! + deviations[m / 2]!) / 2;
  return mad === 0 ? 1e-10 : mad;
}

function madAnomalyDetection(
  data: Matrix,
  serviceNames: string[],
  tailFraction: number,
  threshold: number,
): string[] {
  const n = data.rows;
  if (n < 10) return serviceNames.slice(0, Math.min(3, serviceNames.length));

  const tail = Math.max(5, Math.floor(n * tailFraction));
  const anomalous: string[] = [];

  for (let j = 0; j < serviceNames.length; j++) {
    const col: number[] = [];
    for (let i = 0; i < n; i++) col.push(data.get(i, j));

    const median = [...col].sort((a, b) => a - b)[Math.floor(col.length / 2)]!;
    const mad = computeMAD(col);

    // Compute mean shift in the tail window
    const recentMean = col.slice(-tail).reduce((s, v) => s + v, 0) / tail;
    const modifiedZ = 0.6745 * (recentMean - median) / mad;

    if (Math.abs(modifiedZ) > threshold) {
      anomalous.push(serviceNames[j]!);
    }
  }

  return anomalous.length > 0 ? anomalous : serviceNames.slice(0, Math.min(3, serviceNames.length));
}

// ── Agent ─────────────────────────────────────────────────────────────

export class RCAgent {
  readonly config: Required<Omit<RCAgentConfig, 'pcAlpha'>> & { pcAlpha: number | null };
  private _lastDiscoveredGraph: CausalGraph | null = null;

  constructor(config: RCAgentConfig = {}) {
    this.config = {
      apiKey: config.apiKey ?? (typeof process !== 'undefined' ? process.env['DEEPSEEK_API_KEY'] ?? '' : ''),
      model: config.model ?? 'deepseek-chat',
      baseUrl: config.baseUrl ?? 'https://api.deepseek.com',
      anomalyDetector: config.anomalyDetector ?? 'mad',
      anomalyTailFraction: config.anomalyTailFraction !== undefined ? config.anomalyTailFraction : 0.2,
      anomalyThreshold: config.anomalyThreshold !== undefined ? config.anomalyThreshold : 3.5,
      pcAlpha: config.pcAlpha ?? null,
      multiFault: config.multiFault ?? false,
    };
  }

  /**
   * Return the graph from the most recent discover() call.
   * Falls back to a fresh discovery if none has been run.
   */
  get lastGraph(): CausalGraph | null {
    return this._lastDiscoveredGraph;
  }

  // ── Step 1: Causal Discovery ─────────────────────────────────────

  /**
   * Run PC causal discovery on service metric data.
   *
   * Uses adaptive alpha: larger on small datasets (underpowered CI tests),
   * stricter on large datasets.
   *
   * @param data — Matrix where columns are services, rows are time points
   * @param serviceNames — Column labels
   * @returns Causal graph with edges representing dependency relationships
   */
  discover(data: Matrix, serviceNames: string[]): CausalGraph {
    const alpha = this.config.pcAlpha ?? adaptiveAlpha(data.rows);
    const result = pcAlgorithm(data, serviceNames, {
      alpha,
      stable: true,
    });
    this._lastDiscoveredGraph = result.graph;
    return result.graph;
  }

  // ── Step 2: Anomaly Detection ────────────────────────────────────

  /**
   * Detect anomalous services using the configured detector.
   *
   * Available detectors:
   *   - 'mad':  Median Absolute Deviation (robust, default, threshold=3.5)
   *   - 'zscore': Simple z-score on tail window (threshold=0.3)
   */
  detectAnomalies(data: Matrix, serviceNames: string[]): string[] {
    switch (this.config.anomalyDetector) {
      case 'mad':
        return madAnomalyDetection(data, serviceNames, this.config.anomalyTailFraction, this.config.anomalyThreshold);
      case 'zscore':
        return zscoreAnomalyDetection(data, serviceNames, this.config.anomalyTailFraction, this.config.anomalyThreshold);
    }
  }

  // ── Step 3: RCA Ranking ──────────────────────────────────────────

  /**
   * Rank root cause candidates via HeuristicPath propagation scoring.
   * Root nodes (no parents) with high downstream anomaly get top scores.
   */
  rank(graph: CausalGraph, anomalousServices: string[], data: Matrix): RCARankEntry[] {
    const rca = new HeuristicPathRCA();
    rca.train(graph, new Set(anomalousServices), data);
    const result = rca.findRootCauses(anomalousServices);

    return result.rootCauses.map(rc => ({
      component: rc.name,
      score: rc.score,
      isRoot: graph.parents(rc.name).length === 0,
    }));
  }

  // ── Step 4: Full Diagnosis ───────────────────────────────────────

  /**
   * Run full causal diagnosis: discover → detect anomalies → rank.
   */
  diagnose(data: Matrix, serviceNames: string[], givenAnomalous?: string[]): RCADiagnosis {
    const graph = this.discover(data, serviceNames);
    const anomalousServices = givenAnomalous ?? this.detectAnomalies(data, serviceNames);
    const ranking = this.rank(graph, anomalousServices, data);

    return {
      graph: {
        nodes: [...graph.nodes].map(n => ({ name: n })),
        edges: graph.edges.map(e => ({
          source: e.source,
          target: e.target,
          weight: e.weight ?? 1.0,
        })),
      },
      ranking,
      anomalousServices,
    };
  }

  /**
   * Multi-fault diagnosis with iterative residual analysis.
   *
   * Identifies the top cause, removes its downstream effect from the
   * anomaly set, and re-ranks. Repeated up to `maxFaults` times or
   * until no significant residual anomalies remain.
   */
  diagnoseMulti(
    data: Matrix,
    serviceNames: string[],
    givenAnomalous?: string[],
    maxFaults: number = 3,
  ): RCADiagnosis[] {
    const results: RCADiagnosis[] = [];
    const graph = this.discover(data, serviceNames);
    const remainingAnomalies = new Set(givenAnomalous ?? this.detectAnomalies(data, serviceNames));

    for (let iteration = 0; iteration < maxFaults && remainingAnomalies.size > 0; iteration++) {
      const ranking = this.rank(graph, [...remainingAnomalies], data);

      results.push({
        graph: {
          nodes: [...graph.nodes].map(n => ({ name: n })),
          edges: graph.edges.map(e => ({
            source: e.source,
            target: e.target,
            weight: e.weight ?? 1.0,
          })),
        },
        ranking,
        anomalousServices: [...remainingAnomalies],
      });

      // Remove the top-ranked component and its descendants from remaining anomalies
      if (ranking.length > 0 && iteration < maxFaults - 1) {
        const top = ranking[0]!;
        remainingAnomalies.delete(top.component);

        // Also remove immediate downstream effects
        const children = graph.children(top.component);
        for (const child of children) {
          remainingAnomalies.delete(child);
        }

        // Stop if residual is too small
        if (remainingAnomalies.size === 0) break;
      }
    }

    return results;
  }

  // ── Step 5: LLM Reasoning ────────────────────────────────────────

  async reason(diagnosis: RCADiagnosis, contextDescription: string): Promise<RCAPrediction> {
    const prompt = this.buildPrompt(diagnosis, contextDescription);
    const rawResponse = await this.callLLM(prompt);
    return this.parseResponse(rawResponse);
  }

  private buildPrompt(diagnosis: RCADiagnosis, description: string): string {
    const graphDesc = diagnosis.graph.edges.length > 0
      ? diagnosis.graph.edges.slice(0, 15).map(e => `  ${e.source} → ${e.target} (weight=${e.weight.toFixed(2)})`).join('\n')
      : '  No dependencies discovered';

    const rankingDesc = diagnosis.ranking.length > 0
      ? diagnosis.ranking.slice(0, 5).map((r, i) => `  ${i + 1}. ${r.component} (score=${r.score.toFixed(3)}, root=${r.isRoot})`).join('\n')
      : '  No ranking available';

    const anomalyDesc = diagnosis.anomalousServices.length > 0
      ? diagnosis.anomalousServices.join(', ')
      : 'None detected';

    return `You are a root cause analysis expert for software systems.

# Incident Description
${description}

# Causal Dependency Graph (PC algorithm — directed edges represent service calls)
${graphDesc}

# Root Cause Ranking (HeuristicPath — propagation-based scoring, root services score higher)
${rankingDesc}

# Anomalous Services
${anomalyDesc}

# Task
Based on the causal graph and RCA ranking above, identify the most likely root cause.
The service with the highest RCA score and no anomalous ancestors is typically the root.

Output strictly as JSON (no markdown, no extra text):
{"root_cause_component": "SERVICE_NAME", "root_cause_reason": "brief explanation"}`;
  }

  private async callLLM(prompt: string): Promise<string> {
    if (!this.config.apiKey) {
      return JSON.stringify({
        root_cause_component: 'LLM_UNAVAILABLE',
        root_cause_reason: 'No API key configured',
      });
    }

    try {
      const response = await fetch(`${this.config.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.apiKey}`,
        },
        body: JSON.stringify({
          model: this.config.model,
          messages: [
            { role: 'system', content: 'You output only valid JSON. No markdown, no explanation.' },
            { role: 'user', content: prompt },
          ],
          temperature: 0.0,
          max_tokens: 300,
        }),
      });

      if (!response.ok) {
        return JSON.stringify({
          root_cause_component: 'API_ERROR',
          root_cause_reason: `HTTP ${response.status}`,
        });
      }

      const json = await response.json() as { choices: Array<{ message: { content: string } }> };
      return json.choices[0]?.message?.content ?? '{}';
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'unknown';
      return JSON.stringify({
        root_cause_component: 'API_ERROR',
        root_cause_reason: msg,
      });
    }
  }

  private parseResponse(raw: string): RCAPrediction {
    try {
      const jsonMatch = raw.match(/\{[\s\S]*\}/);
      const parsed = jsonMatch ? JSON.parse(jsonMatch[0]) : JSON.parse(raw);
      return {
        component: parsed.root_cause_component ?? 'UNKNOWN',
        datetime: parsed.root_cause_occurrence_datetime,
        reason: parsed.root_cause_reason ?? 'No reason provided',
        rawLLMResponse: raw,
      };
    } catch {
      return {
        component: 'UNKNOWN',
        reason: `Failed to parse LLM response: ${raw.slice(0, 100)}`,
        rawLLMResponse: raw,
      };
    }
  }

  // ── Step 6: Multi-Source Diagnosis (v3) ──────────────────────────

  /**
   * Multi-source diagnosis: BOCPD + CUSUM + HeuristicPathRCA → fused ranking.
   *
   * Leverages the full v3 pipeline: CUSUM changepoint detection on all
   * metrics, PC causal discovery, HeuristicPathRCA propagation, and
   * MultiSourceRanker for weighted fusion.
   *
   * @param data — Matrix (rows=time, cols=services), pre-aggregated per service
   * @param serviceNames — Column labels
   * @param options — Optional configuration
   */
  diagnoseV3(
    data: Matrix,
    serviceNames: string[],
    options?: {
      fusionWeights?: Partial<{ bocpd: number; cusum: number; heuristicPath: number; logError: number }>;
    },
  ): RCADiagnosis {
    const n = data.rows;
    const d = serviceNames.length;
    if (n < 2 || d === 0) {
      return { graph: { nodes: [], edges: [] }, ranking: [], anomalousServices: [] };
    }

    // ── CUSUM detection on all services ──
    const cusumDetector = new BOCDDetector({ threshold: 5.0, driftParam: 0.5 });
    const cusumResultsRaw = cusumDetector.detectAllColumns(data, serviceNames);

    // ── Filter: only services with significant shift (> 1.5σ) or detected changepoint ──
    const significantResults = cusumResultsRaw.filter(
      r => r.changepoint.detected || r.magnitudeShift > 1.5,
    );

    // If no significant signals found, fall back to top-3 by magnitude
    const activeResults = significantResults.length > 0
      ? significantResults
      : cusumResultsRaw.slice(0, Math.min(3, cusumResultsRaw.length));

    const anomalousServices = activeResults.map(r => r.service);

    // ── BOCPD-style timing + magnitude (only for active services) ──
    const bocpdSignals = activeResults.map(r => ({
      service: r.service,
      changepointIndex: r.changepoint.mostLikelyIndex,
      magnitudeShift: r.magnitudeShift,
      confidence: r.changepoint.confidence,
    }));

    const cusumSignals = activeResults.map(r => ({
      service: r.service,
      maxCusum: r.changepoint.maxCusum,
      magnitudeShift: r.magnitudeShift,
    }));

    // ── PC causal discovery + HeuristicPathRCA ──
    const graph = this.discover(data, serviceNames);
    const hpResult = new HeuristicPathRCA();
    hpResult.train(graph, new Set(anomalousServices), data);
    const hpOutput = hpResult.findRootCauses(anomalousServices);
    const hpSignals = hpOutput.rootCauses.map(rc => ({
      component: rc.name,
      score: rc.score,
      isRoot: graph.parents(rc.name).length === 0,
    }));

    // ── Multi-source fusion ──
    // When no traces/logs available (metric-only), heuristicPath is primary signal
    const defaultWeights = { bocpd: 0.25, cusum: 0.20, heuristicPath: 0.45, logError: 0.10 };
    const ranker = new MultiSourceRanker({ ...defaultWeights, ...options?.fusionWeights });
    const fused: FusionRankEntry[] = ranker.rank({
      bocpdResults: bocpdSignals,
      cusumResults: cusumSignals,
      heuristicPathRanking: hpSignals,
      logErrors: [],
      serviceNames,
      totalTimesteps: n,
    });

    const ranking: RCARankEntry[] = fused.map(f => ({
      component: f.component,
      score: f.score,
      isRoot: f.isRoot,
    }));

    return {
      graph: {
        nodes: serviceNames.map(s => ({ name: s })),
        edges: graph.edges.map(e => ({
          source: e.source,
          target: e.target,
          weight: e.weight ?? 1.0,
        })),
      },
      ranking,
      anomalousServices: activeResults.map(r => r.service),
    };
  }
}

// ── Fallback: Simple Z-score Detector ──────────────────────────────────

function zscoreAnomalyDetection(
  data: Matrix,
  serviceNames: string[],
  tailFraction: number,
  threshold: number,
): string[] {
  const n = data.rows;
  const tail = Math.max(1, Math.floor(n * tailFraction));

  const anomalous: string[] = [];
  for (let j = 0; j < serviceNames.length; j++) {
    const fullCol: number[] = [];
    for (let i = 0; i < n; i++) fullCol.push(data.get(i, j));
    const recentCol = fullCol.slice(-tail);

    const fullMean = fullCol.reduce((s, v) => s + v, 0) / fullCol.length;
    const fullStd = Math.sqrt(fullCol.reduce((s, v) => s + (v - fullMean) ** 2, 0) / fullCol.length) || 1;
    const recentMean = recentCol.reduce((s, v) => s + v, 0) / recentCol.length;
    const zScore = (recentMean - fullMean) / fullStd;

    if (Math.abs(zScore) > threshold) {
      anomalous.push(serviceNames[j]!);
    }
  }
  return anomalous.length > 0 ? anomalous : serviceNames.slice(0, Math.min(3, serviceNames.length));
}
