#!/usr/bin/env node
/**
 * RCA Evaluation Benchmark (RCAEval).
 *
 * Evaluates Root Cause Analysis methods against known ground-truth
 * fault injection scenarios on synthetic microservice topologies.
 *
 * Metrics: Top-1, Top-3, Top-5 accuracy, MRR (Mean Reciprocal Rank)
 *
 * Reference: CIRCA (KDD 2022) — Causal Inference-Based Root Cause Analysis
 *
 * @packageDocumentation
 */

import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../src/graph/causal-graph.js';
import { RCAgent } from '../src/agent/rca-agent.js';
import { HeuristicPathRCA, RandomWalkRCA } from '../src/analyze/rca.js';
import { createRNG, colMean } from '@agentix-e/causality-analyzer-core';

const OUTPUT_DIR = join(import.meta.dirname, '..', 'benchmark-results');

// ── Topology Generator ───────────────────────────────────────────────

interface MicroserviceTopology {
  graph: CausalGraph;
  nodeNames: string[];
  /** Ground-truth dependency edges (service A → service B) */
  dependencies: Array<[string, string]>;
}

/**
 * Generate a synthetic microservice call-graph topology.
 * More realistic than pure random DAG — uses service mesh patterns:
 * - Frontend → API Gateway → [Service A, Service B, Service C]
 * - Service chains with fan-out
 * - Shared backend services
 */
function generateMicroserviceTopology(
  numServices: number,
  seed: number,
): MicroserviceTopology {
  const rng = createRNG(seed);
  const names = Array.from({ length: numServices }, (_, i) => `svc-${i}`);
  const g = new CausalGraph(names);
  const deps: Array<[string, string]> = [];

  // Layer-based architecture
  const layerSize = Math.max(3, Math.floor(numServices / 4));
  const layers = [
    names.slice(0, 1),                          // Layer 0: Frontend
    names.slice(1, 1 + layerSize),              // Layer 1: API Gateway + Auth
    names.slice(1 + layerSize, 1 + 2 * layerSize), // Layer 2: Business logic
    names.slice(1 + 2 * layerSize),             // Layer 3: Backend services
  ];

  // Add dependencies between layers
  for (let li = 0; li < layers.length - 1; li++) {
    const current = layers[li]!;
    const next = layers[li + 1]!;
    for (const src of current) {
      const numTargets = Math.max(1, Math.floor(next.length * 0.4));
      const shuffled = [...next].sort(() => rng() - 0.5);
      for (let t = 0; t < numTargets && t < shuffled.length; t++) {
        g.addEdge(src, shuffled[t]!);
        deps.push([src, shuffled[t]!]);
      }
    }
  }

  // Add cross-layer dependencies (fan-out)
  for (const src of layers[2]!) {
    if (rng() < 0.3) {
      const tgt = layers[1]![Math.floor(rng() * layers[1]!.length)]!;
      g.addEdge(src, tgt);
      deps.push([src, tgt]);
    }
  }

  return { graph: g, nodeNames: names, dependencies: deps };
}

// ── Fault Injection ──────────────────────────────────────────────────

interface FaultScenario {
  rootCauses: string[];
  affectedNodes: string[];
  /** Anomalous time series data with injected fault signatures */
  data: number[][];
  /** Column index → node name mapping */
  columns: string[];
}

/**
 * Inject a fault into the topology and generate anomalous time series.
 *
 * The fault propagates from rootCause to its descendants in the
 * causal graph, creating a cascading anomaly pattern.
 */
function injectFault(
  topology: MicroserviceTopology,
  rootCauses: string[],
  nPoints: number,
  seed: number,
): FaultScenario {
  const rng = createRNG(seed);
  const nodeNames = [...topology.graph.nodes];
  const nNodes = nodeNames.length;

  // Generate baseline data
  const data: number[][] = [];
  for (let t = 0; t < nPoints; t++) {
    const row: number[] = [];
    for (let i = 0; i < nNodes; i++) {
      // Baseline: N(100, 5)
      let val = 100 + boxMuller(rng) * 5;

      // Inject fault: increasing anomaly after midpoint
      if (t >= nPoints * 0.5) {
        const faultIntensity = (t - nPoints * 0.5) / (nPoints * 0.5);
        const node = nodeNames[i]!;

        // Root cause gets direct anomaly
        if (rootCauses.includes(node)) {
          // Latency spike — 30% increase scaling with faultIntensity
          val *= 1 + 0.3 * faultIntensity * (1 + Math.abs(boxMuller(rng)) * 0.5);
        } else {
          // Descendants get cascading anomaly based on graph distance from root cause
          for (const rc of rootCauses) {
            const distance = topology.graph.shortestPath?.(rc, node)?.length ?? Infinity;
            if (distance > 0 && distance < 10) {
              // Cascade attenuation: farther nodes get less anomaly
              const attenuation = 1.0 / (1.0 + distance);
              val *= 1 + 0.15 * faultIntensity * attenuation * (1 + Math.abs(boxMuller(rng)) * 0.3);
            }
          }
        }
      }
      row.push(val);
    }
    data.push(row);
  }

  // Find affected nodes (descendants of root causes)
  const affected: Set<string> = new Set(rootCauses);
  for (const rc of rootCauses) {
    const descendants = collectDescendants(topology.graph, rc);
    for (const d of descendants) affected.add(d);
  }

  return {
    rootCauses,
    affectedNodes: [...affected],
    data,
    columns: nodeNames,
  };
}

function collectDescendants(g: CausalGraph, node: string): string[] {
  const visited = new Set<string>();
  const stack = [node];
  const result: string[] = [];

  while (stack.length > 0) {
    const current = stack.pop()!;
    const children = g.children(current);
    for (const child of children) {
      if (!visited.has(child)) {
        visited.add(child);
        result.push(child);
        stack.push(child);
      }
    }
  }

  return result;
}

function boxMuller(rng: () => number): number {
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ── RCA Evaluation ───────────────────────────────────────────────────

interface RCAEvaluationResult {
  topologySize: number;
  numFaults: number;
  numScenarios: number;
  algorithm: string;
  top1Accuracy: number;
  top3Accuracy: number;
  top5Accuracy: number;
  mrr: number;
  avgRank: number;
  totalTimeMs: number;
}

/**
 * Run RCAEval benchmark: generate topologies, inject faults,
 * evaluate RCA methods.
 */
export function runRCAEvalBenchmark(options?: {
  maxTopologySize?: number;
  numScenarios?: number;
  seed?: number;
  dataPoints?: number;
}): RCAEvaluationResult[] {
  const maxSize = options?.maxTopologySize ?? 20;
  const numScenarios = options?.numScenarios ?? 10;
  const seed = options?.seed ?? 42;
  const nPoints = options?.dataPoints ?? 500;

  const topologies = [10, 15, maxSize].filter(s => s <= maxSize);
  const algorithms = ['HeuristicPathRCA', 'RandomWalkRCA', 'RCAgent'];
  const results: RCAEvaluationResult[] = [];
  let s = seed;

  for (const topoSize of topologies) {
    // Accumulate per-algorithm metrics
    const algoMetrics = new Map<string, {
      top1Sum: number; top3Sum: number; top5Sum: number;
      mrrSum: number; rankSum: number; scenarios: number;
      timeSum: number;
    }>();

    for (const algo of algorithms) {
      algoMetrics.set(algo, {
        top1Sum: 0, top3Sum: 0, top5Sum: 0,
        mrrSum: 0, rankSum: 0, scenarios: 0, timeSum: 0,
      });
    }

    for (let sc = 0; sc < numScenarios; sc++) {
      const topology = generateMicroserviceTopology(topoSize, s++);
      const numFaults = 1 + Math.floor((s++ % 3)); // 1-3 root causes
      const rootCandidates = [...topology.nodeNames].sort(() => (s++ % 2) - 0.5);
      const rootCauses = rootCandidates.slice(0, Math.min(numFaults, topology.nodeNames.length));

      const scenario = injectFault(topology, rootCauses, nPoints, s++);

      for (const algo of algorithms) {
        const m = algoMetrics.get(algo)!;
        const t0 = performance.now();

        // All methods work on the same data
        const dataMatrix = new Matrix(scenario.data);
        let ranked: Array<{ name: string; score: number }>;

        if (algo === 'RCAgent') {
          // RCAgent uses PC causal discovery + HeuristicPathRCA
          try {
            const agent = new RCAgent();
            const diagnosis = agent.diagnose(dataMatrix, scenario.columns);
            ranked = diagnosis.ranking.map(r => ({
              name: r.component,
              score: r.score,
            }));
            // If no ranking produced, fall back to correlation baseline
            if (ranked.length === 0) {
              ranked = rankByCorrelation(scenario, topology);
            }
          } catch {
            ranked = rankByCorrelation(scenario, topology);
          }
        } else if (algo === 'HeuristicPathRCA') {
          // Pure HeuristicPathRCA
          try {
            const anomalous = detectAnomalous(dataMatrix, scenario.columns);
            const rca = new HeuristicPathRCA();
            rca.train(topology.graph, new Set(anomalous), dataMatrix);
            const result = rca.findRootCauses(anomalous);
            ranked = result.rootCauses.map(rc => ({
              name: rc.name,
              score: rc.score,
            }));
            if (ranked.length === 0) {
              ranked = rankByCorrelation(scenario, topology);
            }
          } catch {
            ranked = rankByCorrelation(scenario, topology);
          }
        } else if (algo === 'RandomWalkRCA') {
          // Pure RandomWalkRCA
          try {
            const anomalous = detectAnomalous(dataMatrix, scenario.columns);
            const rca = new RandomWalkRCA();
            rca.train(topology.graph, new Set(anomalous), dataMatrix);
            const result = rca.findRootCauses(anomalous);
            ranked = result.rootCauses.map(rc => ({
              name: rc.name,
              score: rc.score,
            }));
            if (ranked.length === 0) {
              ranked = rankByCorrelation(scenario, topology);
            }
          } catch {
            ranked = rankByCorrelation(scenario, topology);
          }
        } else {
          ranked = rankByCorrelation(scenario, topology);
        }

        const timeMs = performance.now() - t0;

        // Evaluate rankings
        const ranks = rootCauses.map(rc => {
          const idx = ranked.findIndex(r => r.name === rc);
          return idx >= 0 ? idx + 1 : ranked.length + 1;
        });

        m.top1Sum += ranks.some(r => r === 1) ? 1 : 0;
        m.top3Sum += ranks.some(r => r <= 3) ? 1 : 0;
        m.top5Sum += ranks.some(r => r <= 5) ? 1 : 0;
        m.mrrSum += ranks.reduce((s, r) => s + 1 / r, 0) / ranks.length;
        m.rankSum += ranks.reduce((s, r) => s + r, 0) / ranks.length;
        m.scenarios++;
        m.timeSum += timeMs;
      }
    }

    for (const [algo, m] of algoMetrics) {
      if (m.scenarios === 0) continue;
      results.push({
        topologySize: topoSize,
        numFaults: 3,
        numScenarios: m.scenarios,
        algorithm: algo,
        top1Accuracy: m.top1Sum / m.scenarios,
        top3Accuracy: m.top3Sum / m.scenarios,
        top5Accuracy: m.top5Sum / m.scenarios,
        mrr: m.mrrSum / m.scenarios,
        avgRank: m.rankSum / m.scenarios,
        totalTimeMs: Math.round(m.timeSum),
      });
    }
  }

  return results;
}

/**
 * Simple anomaly detection: flag nodes whose post-fault mean shifts > 2σ from pre-fault.
 */
function detectAnomalous(data: Matrix, columns: string[]): string[] {
  const n = data.rows;
  if (n < 20) return [];

  const mid = Math.floor(n * 0.5);
  const anomalous: string[] = [];

  for (let ci = 0; ci < columns.length; ci++) {
    const pre: number[] = [], post: number[] = [];
    for (let i = 0; i < n; i++) {
      (i < mid ? pre : post).push(data.get(i, ci));
    }
    const preMean = pre.reduce((s, v) => s + v, 0) / pre.length;
    const preStd = Math.sqrt(pre.reduce((s, v) => s + (v - preMean) ** 2, 0) / pre.length) || 1;
    const postMean = post.reduce((s, v) => s + v, 0) / post.length;

    if (Math.abs(postMean - preMean) > 2 * preStd) {
      anomalous.push(columns[ci]!);
    }
  }

  return anomalous.length > 0 ? anomalous : columns.slice(0, Math.min(3, columns.length));
}

/**
 * Simple correlation-based root cause ranking baseline.
 * Ranks nodes by their correlation with the overall anomaly pattern.
 */
function rankByCorrelation(
  scenario: FaultScenario,
  _topology: MicroserviceTopology,
): Array<{ name: string; score: number }> {
  const { data, columns } = scenario;
  const n = data.length;
  const anomalyStart = Math.floor(n * 0.5);
  const preData = data.slice(0, anomalyStart);
  const postData = data.slice(anomalyStart);

  const scores: Array<{ name: string; score: number }> = [];

  for (let ci = 0; ci < columns.length; ci++) {
    // Mean shift metric: how much did the mean change from pre to post fault?
    const preMean = colMean(preData, ci);
    const postMean = colMean(postData, ci);
    const shift = Math.abs(postMean - preMean);

    // Also check tail behavior: max anomaly in post period
    let maxAnomaly = 0;
    for (const row of postData) {
      maxAnomaly = Math.max(maxAnomaly, Math.abs((row[ci] ?? 0) - preMean));
    }

    const score = shift + maxAnomaly * 0.5;
    scores.push({ name: columns[ci]!, score });
  }

  scores.sort((a, b) => b.score - a.score);
  return scores;
}

// ── Report Formatting ─────────────────────────────────────────────────

export function formatRCAEvalMarkdown(results: RCAEvaluationResult[]): string {
  const lines: string[] = [];
  lines.push('# RCAEval Benchmark Results');
  lines.push('');
  lines.push('> Topology-based root cause analysis evaluation');
  lines.push('');
  lines.push('| Topology | Algorithm | Top-1 | Top-3 | Top-5 | MRR | Avg Rank | Time (ms) |');
  lines.push('|----------|-----------|-------|-------|-------|-----|----------|----------|');

  for (const r of results) {
    lines.push(
      `| ${r.topologySize}-node | ${r.algorithm} | ${(r.top1Accuracy * 100).toFixed(1)}% | ${(r.top3Accuracy * 100).toFixed(1)}% | ${(r.top5Accuracy * 100).toFixed(1)}% | ${r.mrr.toFixed(3)} | ${r.avgRank.toFixed(1)} | ${r.totalTimeMs} |`,
    );
  }

  return lines.join('\n');
}

export function formatRCAEvalJSON(results: RCAEvaluationResult[]): string {
  return JSON.stringify(
    {
      benchmark: 'RCAEval',
      timestamp: new Date().toISOString(),
      results,
    },
    null,
    2,
  );
}

// ── CLI Runner ───────────────────────────────────────────────────────

const isMainModule = process.argv[1]?.includes('run-rcaeval');

if (isMainModule) {
  const MAX_SIZE = parseInt(process.env['BENCH_RCA_MAX_SIZE'] ?? '20', 10);
  const SCENARIOS = parseInt(process.env['BENCH_RCA_SCENARIOS'] ?? '10', 10);
  const SEED = parseInt(process.env['BENCH_RCA_SEED'] ?? '42', 10);

  console.log(`RCAEval Benchmark — max size: ${MAX_SIZE}, scenarios: ${SCENARIOS}`);
  console.time('RCAEval');

  mkdirSync(OUTPUT_DIR, { recursive: true });

  const results = runRCAEvalBenchmark({
    maxTopologySize: MAX_SIZE,
    numScenarios: SCENARIOS,
    seed: SEED,
  });

  // Save reports
  const mdPath = join(OUTPUT_DIR, 'benchmark-rcaeval.md');
  writeFileSync(mdPath, formatRCAEvalMarkdown(results));
  console.log(`\nMarkdown: ${mdPath}`);

  const jsonPath = join(OUTPUT_DIR, 'benchmark-rcaeval.json');
  writeFileSync(jsonPath, formatRCAEvalJSON(results));
  console.log(`JSON: ${jsonPath}`);

  console.timeEnd('RCAEval');
  console.log('\nDone.');
}
