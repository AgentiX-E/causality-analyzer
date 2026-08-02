/**
 * Generic RCA Agent — Causal Discovery + RCA + LLM Reasoning.
 *
 * The core agent pipeline is benchmark-agnostic:
 *   1. Causal discovery (PC algorithm) → service dependency graph
 *   2. Anomaly detection (z-score on recent window)
 *   3. RCA ranking (HeuristicPath propagation scoring)
 *   4. LLM reasoning (DeepSeek with graph + ranking context)
 *
 * Benchmark-specific concerns (data format, prediction format) are
 * handled by adapter functions passed as constructor parameters.
 *
 * @packageDocumentation
 */

import { Matrix } from 'ml-matrix';
import { pcAlgorithm } from '../graph/pc.js';
import { HeuristicPathRCA } from '../analyze/rca.js';
import type { CausalGraph } from '../graph/causal-graph.js';

// ── Types ────────────────────────────────────────────────────────────

export interface RCAgentConfig {
  /** LLM API key (read from env, never in code) */
  apiKey?: string;
  /** LLM model name */
  model?: string;
  /** LLM base URL */
  baseUrl?: string;
  /** Fraction of data tail used for anomaly detection (default: 0.2) */
  anomalyTailFraction?: number;
  /** Z-score threshold for anomaly detection (default: 0.3) */
  anomalyThreshold?: number;
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

// ── Agent ─────────────────────────────────────────────────────────────

export class RCAgent {
  private readonly config: Required<RCAgentConfig>;

  constructor(config: RCAgentConfig = {}) {
    this.config = {
      apiKey: config.apiKey ?? (typeof process !== 'undefined' ? process.env['DEEPSEEK_API_KEY'] ?? '' : ''),
      model: config.model ?? 'deepseek-chat',
      baseUrl: config.baseUrl ?? 'https://api.deepseek.com',
      anomalyTailFraction: config.anomalyTailFraction ?? 0.2,
      anomalyThreshold: config.anomalyThreshold ?? 0.3,
    };
  }

  // ── Step 1: Causal Discovery ─────────────────────────────────────

  /**
   * Run PC causal discovery on service metric data.
   *
   * @param data — Matrix where columns are services, rows are time points
   * @param serviceNames — Column labels
   * @returns Causal graph with edges representing dependency relationships
   */
  discover(data: Matrix, serviceNames: string[]): CausalGraph {
    const result = pcAlgorithm(data, serviceNames, {
      alpha: 0.05,
      stable: true,
    });
    return result.graph;
  }

  // ── Step 2: Anomaly Detection ────────────────────────────────────

  /**
   * Detect anomalous services via z-score on the tail of the data.
   * Services with |z-score| > threshold on the recent window are flagged.
   */
  detectAnomalies(data: Matrix, serviceNames: string[]): string[] {
    const n = data.rows;
    const tail = Math.max(1, Math.floor(n * this.config.anomalyTailFraction));
    const threshold = this.config.anomalyThreshold;

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
    return anomalous;
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

  // ── Step 4: Full Diagnosis (1+2+3) ───────────────────────────────

  /**
   * Run full causal diagnosis: discover → detect anomalies → rank.
   * Returns structured evidence ready for LLM reasoning.
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

  // ── Step 5: LLM Reasoning ────────────────────────────────────────

  /**
   * Call LLM with causal evidence to produce root cause prediction.
   *
   * @param diagnosis — Output from diagnose()
   * @param contextDescription — Natural language description of the incident (from benchmark)
   * @returns Structured prediction including component, reason, and raw response
   */
  async reason(diagnosis: RCADiagnosis, contextDescription: string): Promise<RCAPrediction> {
    const prompt = this.buildPrompt(diagnosis, contextDescription);
    const rawResponse = await this.callLLM(prompt);
    return this.parseResponse(rawResponse);
  }

  /** Build the LLM prompt with causal graph and RCA ranking context */
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

  /** Call the LLM API */
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

  /** Parse LLM JSON response into structured prediction */
  private parseResponse(raw: string): RCAPrediction {
    try {
      // Try to extract JSON from response (may have markdown wrapping)
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
}
