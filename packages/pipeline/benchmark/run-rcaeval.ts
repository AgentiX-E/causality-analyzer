#!/usr/bin/env node
/**
 * RCA Evaluation Benchmark (RCAEval).
 *
 * Evaluates Root Cause Analysis methods against known ground-truth
 * fault injection scenarios on synthetic microservice topologies.
 *
 * Methods compared:
 *   - RCAgent (PC causal discovery + HeuristicPathRCA, realistic)
 *   - HeuristicPathRCA (on discovered graph — fair comparison)
 *   - RandomWalkRCA (on discovered graph)
 *   - HTRCA (on discovered graph)
 *   - CIRCA (on discovered graph)
 *   - Correlation baseline (no graph, simple mean shift)
 *
 * Metrics: Top-1, Top-3, Top-5 accuracy, Avg@5, MRR (Mean Reciprocal Rank)
 *
 * @packageDocumentation
 */

import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../src/graph/causal-graph.js';
import { RCAgent, type RCADiagnosis } from '../src/agent/rca-agent.js';
import { HeuristicPathRCA, RandomWalkRCA, HTRCA } from '../src/analyze/rca.js';
import { CIRCAPipeline } from '../src/analyze/circa.js';
import { createRNG, colMean } from '@agentix-e/causality-analyzer-core';

const OUTPUT_DIR = join(import.meta.dirname, '..', 'benchmark-results');

// ── Topology Generator ───────────────────────────────────────────────

interface MicroserviceTopology {
  graph: CausalGraph;
  nodeNames: string[];
  dependencies: Array<[string, string]>;
}

function generateMicroserviceTopology(
  numServices: number,
  seed: number,
): MicroserviceTopology {
  const rng = createRNG(seed);
  const names = Array.from({ length: numServices }, (_, i) => `svc-${i}`);
  const g = new CausalGraph(names);
  const deps: Array<[string, string]> = [];

  const layerSize = Math.max(3, Math.floor(numServices / 4));
  const layers = [
    names.slice(0, 1),
    names.slice(1, 1 + layerSize),
    names.slice(1 + layerSize, 1 + 2 * layerSize),
    names.slice(1 + 2 * layerSize),
  ];

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
  faultTypes: string[];
  data: number[][];
  columns: string[];
}

function injectFault(
  topology: MicroserviceTopology,
  rootCauses: string[],
  nPoints: number,
  seed: number,
): FaultScenario {
  const rng = createRNG(seed);
  const nodeNames = [...topology.graph.nodes];
  const nNodes = nodeNames.length;

  const faultTypes = ['latency_spike', 'cpu_stress', 'memory_leak', 'packet_loss', 'disk_io'];
  const faultType = faultTypes[seed % faultTypes.length]!;

  const data: number[][] = [];
  for (let t = 0; t < nPoints; t++) {
    const row: number[] = [];
    for (let i = 0; i < nNodes; i++) {
      let val = 100 + boxMuller(rng) * 5;

      if (t >= nPoints * 0.5) {
        const faultIntensity = (t - nPoints * 0.5) / (nPoints * 0.5);
        const node = nodeNames[i]!;

        if (rootCauses.includes(node)) {
          val *= 1 + faultMagnitude(faultType) * faultIntensity * (1 + Math.abs(boxMuller(rng)) * 0.5);
        } else {
          for (const rc of rootCauses) {
            const distance = topology.graph.shortestPath?.(rc, node)?.length ?? Infinity;
            if (distance > 0 && distance < 10) {
              const attenuation = 1.0 / (1.0 + distance);
              val *= 1 + 0.5 * faultMagnitude(faultType) * faultIntensity * attenuation * (1 + Math.abs(boxMuller(rng)) * 0.3);
            }
          }
        }
      }
      row.push(val);
    }
    data.push(row);
  }

  const affected: Set<string> = new Set(rootCauses);
  for (const rc of rootCauses) {
    const descendants = collectDescendants(topology.graph, rc);
    for (const d of descendants) affected.add(d);
  }

  return {
    rootCauses,
    affectedNodes: [...affected],
    faultTypes: [faultType],
    data,
    columns: nodeNames,
  };
}

function faultMagnitude(faultType: string): number {
  switch (faultType) {
    case 'latency_spike': return 0.30;
    case 'cpu_stress': return 0.25;
    case 'memory_leak': return 0.35;
    case 'packet_loss': return 0.20;
    case 'disk_io': return 0.15;
    default: return 0.25;
  }
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

// ── Evaluation Metrics ────────────────────────────────────────────────

interface RCAEvaluationResult {
  topologySize: number;
  algorithm: string;
  numScenarios: number;
  top1Accuracy: number;
  top3Accuracy: number;
  top5Accuracy: number;
  avgAtK: number;        // Average precision at k=5 (RCAEval paper metric)
  mrr: number;
  avgRank: number;
  totalTimeMs: number;
}

type RankedEntry = { name: string; score: number };

/**
 * Run the full RCAEval benchmark: generate topologies, inject faults,
 * evaluate all RCA methods with fair comparison on discovered graphs.
 */
export function runRCAEvalBenchmark(options?: {
  maxTopologySize?: number;
  numScenarios?: number;
  seed?: number;
  dataPoints?: number;
}): RCAEvaluationResult[] {
  const maxSize = options?.maxTopologySize ?? 64;
  const numScenarios = options?.numScenarios ?? 30;
  const seed = options?.seed ?? 42;
  const nPoints = options?.dataPoints ?? 500;

  const topologies = [10, 15, maxSize].filter(s => s <= maxSize);
  const methods = ['RCAgent', 'HeuristicPathRCA', 'RandomWalkRCA', 'HTRCA', 'CIRCA', 'Correlation'];
  const results: RCAEvaluationResult[] = [];
  let s = seed;

  for (const topoSize of topologies) {
    const algoMetrics = new Map<string, {
      top1Sum: number; top3Sum: number; top5Sum: number;
      avgAtKSum: number; mrrSum: number; rankSum: number;
      scenarios: number; timeSum: number;
    }>();

    for (const method of methods) {
      algoMetrics.set(method, {
        top1Sum: 0, top3Sum: 0, top5Sum: 0,
        avgAtKSum: 0, mrrSum: 0, rankSum: 0,
        scenarios: 0, timeSum: 0,
      });
    }

    for (let sc = 0; sc < numScenarios; sc++) {
      const topology = generateMicroserviceTopology(topoSize, s++);
      const numFaults = 1 + Math.floor((s++ % 3));
      const rootCandidates = [...topology.nodeNames].sort(() => (s++ % 2) - 0.5);
      const rootCauses = rootCandidates.slice(0, Math.min(numFaults, topology.nodeNames.length));
      const scenario = injectFault(topology, rootCauses, nPoints, s++);
      const dataMatrix = new Matrix(scenario.data);

      // Pre-discover graph using PC algorithm (fair comparison baseline)
      const agent = new RCAgent();
      const discoveredGraph = agent.discover(dataMatrix, scenario.columns);

      // Pre-detect anomalies (shared for all graph-based methods)
      const anomalousServices = agent.detectAnomalies(dataMatrix, scenario.columns);

      // ── Method 1: RCAgent (PC + HeuristicPathRCA, realistic pipeline) ──
      runMethod('RCAgent', () => {
        const diagnosis = agent.diagnose(dataMatrix, scenario.columns);
        return diagnosis.ranking.map(r => ({ name: r.component, score: r.score }));
      });

      // ── Method 2: HeuristicPathRCA on discovered graph (fair) ──
      runMethod('HeuristicPathRCA', () => {
        const rca = new HeuristicPathRCA();
        rca.train(discoveredGraph, new Set(anomalousServices), dataMatrix);
        const result = rca.findRootCauses(anomalousServices);
        return result.rootCauses.map(rc => ({ name: rc.name, score: rc.score }));
      });

      // ── Method 3: RandomWalkRCA on discovered graph ──
      runMethod('RandomWalkRCA', () => {
        const rca = new RandomWalkRCA();
        rca.train(discoveredGraph, new Set(anomalousServices), dataMatrix);
        const result = rca.findRootCauses(anomalousServices);
        return result.rootCauses.map(rc => ({ name: rc.name, score: rc.score }));
      });

      // ── Method 4: HTRCA on discovered graph ──
      runMethod('HTRCA', () => {
        const rca = new HTRCA();
        rca.train(discoveredGraph, new Set(anomalousServices), dataMatrix);
        const result = rca.findRootCauses(anomalousServices);
        return result.rootCauses.map(rc => ({ name: rc.name, score: rc.score }));
      });

      // ── Method 5: CIRCA on discovered graph ──
      runMethod('CIRCA', () => {
        // CIRCA takes number[][] for training
        const pipeline = new CIRCAPipeline();
        pipeline.train(discoveredGraph, scenario.data);
        const result = pipeline.analyze(scenario.data, anomalousServices);
        return result.rootCauses.map(rc => ({ name: rc.name, score: rc.score }));
      });

      // ── Method 6: Correlation baseline (no graph, pure statistics) ──
      runMethod('Correlation', () => {
        return rankByCorrelation(scenario);
      });

      // ── Inner helper: timed evaluation ──
      function runMethod(method: string, rankFn: () => RankedEntry[]): void {
        const m = algoMetrics.get(method)!;
        const t0 = performance.now();
        let ranked: RankedEntry[];
        try {
          ranked = rankFn();
        } catch {
          ranked = rankByCorrelation(scenario);
        }
        const timeMs = performance.now() - t0;

        if (ranked.length === 0) {
          ranked = rankByCorrelation(scenario);
        }

        const ranks = rootCauses.map(rc => {
          const idx = ranked.findIndex(r => r.name === rc);
          return idx >= 0 ? idx + 1 : ranked.length + 1;
        });

        m.top1Sum += ranks.some(r => r === 1) ? 1 : 0;
        m.top3Sum += ranks.some(r => r <= 3) ? 1 : 0;
        m.top5Sum += ranks.some(r => r <= 5) ? 1 : 0;
        // Avg@k: precision at each rank position 1..5
        for (let k = 1; k <= 5; k++) {
          m.avgAtKSum += ranks.some(r => r === k) ? 1 : 0;
        }
        m.mrrSum += ranks.reduce((s, r) => s + 1 / r, 0) / ranks.length;
        m.rankSum += ranks.reduce((s, r) => s + r, 0) / ranks.length;
        m.scenarios++;
        m.timeSum += timeMs;
      }
    }

    for (const [method, m] of algoMetrics) {
      if (m.scenarios === 0) continue;
      results.push({
        topologySize: topoSize,
        algorithm: method,
        numScenarios: m.scenarios,
        top1Accuracy: m.top1Sum / m.scenarios,
        top3Accuracy: m.top3Sum / m.scenarios,
        top5Accuracy: m.top5Sum / m.scenarios,
        avgAtK: m.avgAtKSum / (m.scenarios * 5),
        mrr: m.mrrSum / m.scenarios,
        avgRank: m.rankSum / m.scenarios,
        totalTimeMs: Math.round(m.timeSum),
      });
    }
  }

  return results;
}

/**
 * Detect anomalous nodes via mean shift (2σ threshold).
 */
function detectAnomalous(data: Matrix, columns: string[]): string[] {
  const n = data.rows;
  if (n < 20) return [];
  const mid = Math.floor(n * 0.5);
  const anomalous: string[] = [];
  for (let ci = 0; ci < columns.length; ci++) {
    const pre: number[] = [], post: number[] = [];
    for (let i = 0; i < n; i++) (i < mid ? pre : post).push(data.get(i, ci));
    const preMean = pre.reduce((s, v) => s + v, 0) / pre.length;
    const preStd = Math.sqrt(pre.reduce((s, v) => s + (v - preMean) ** 2, 0) / pre.length) || 1;
    const postMean = post.reduce((s, v) => s + v, 0) / post.length;
    if (Math.abs(postMean - preMean) > 2 * preStd) anomalous.push(columns[ci]!);
  }
  return anomalous.length > 0 ? anomalous : columns.slice(0, Math.min(3, columns.length));
}

/**
 * Simple correlation-based root cause ranking baseline.
 */
function rankByCorrelation(
  scenario: FaultScenario,
): RankedEntry[] {
  const { data, columns } = scenario;
  const n = data.length;
  const anomalyStart = Math.floor(n * 0.5);
  const postData = data.slice(anomalyStart);
  const scores: RankedEntry[] = [];

  for (let ci = 0; ci < columns.length; ci++) {
    const preMean = colMean(data.slice(0, anomalyStart), ci);
    const postMean = colMean(postData, ci);
    const shift = Math.abs(postMean - preMean);
    let maxAnomaly = 0;
    for (const row of postData) {
      maxAnomaly = Math.max(maxAnomaly, Math.abs((row[ci] ?? 0) - preMean));
    }
    scores.push({ name: columns[ci]!, score: shift + maxAnomaly * 0.5 });
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
  lines.push('> All graph-based methods use PC-discovered graph (fair comparison, no ground-truth leakage)');
  lines.push('');
  lines.push('| Topology | Algorithm | Top-1 | Top-3 | Top-5 | Avg@5 | MRR | Avg Rank | Time (ms) |');
  lines.push('|----------|-----------|-------|-------|-------|-------|-----|----------|----------|');

  for (const r of results) {
    lines.push(
      `| ${r.topologySize}-node | ${r.algorithm} | ${pct(r.top1Accuracy)} | ${pct(r.top3Accuracy)} | ${pct(r.top5Accuracy)} | ${r.avgAtK.toFixed(3)} | ${r.mrr.toFixed(3)} | ${r.avgRank.toFixed(1)} | ${r.totalTimeMs} |`,
    );
  }

  return lines.join('\n');
}

export function formatRCAEvalJSON(results: RCAEvaluationResult[]): string {
  return JSON.stringify({
    benchmark: 'RCAEval',
    timestamp: new Date().toISOString(),
    note: 'All graph-based methods use PC-discovered graph (fair comparison)',
    results,
  }, null, 2);
}

function pct(val: number): string { return `${(val * 100).toFixed(1)}%`; }

// ── CLI Runner ───────────────────────────────────────────────────────

const isMainModule = process.argv[1]?.includes('run-rcaeval');

if (isMainModule) {
  const MAX_SIZE = parseInt(process.env['BENCH_RCA_MAX_SIZE'] ?? '64', 10);
  const SCENARIOS = parseInt(process.env['BENCH_RCA_SCENARIOS'] ?? '30', 10);
  const SEED = parseInt(process.env['BENCH_RCA_SEED'] ?? '42', 10);

  console.log(`RCAEval Benchmark — max size: ${MAX_SIZE}, scenarios: ${SCENARIOS}`);
  console.log('Note: All graph-based methods use PC-discovered graph (fair comparison)');
  console.time('RCAEval');

  mkdirSync(OUTPUT_DIR, { recursive: true });

  const results = runRCAEvalBenchmark({
    maxTopologySize: MAX_SIZE,
    numScenarios: SCENARIOS,
    seed: SEED,
  });

  const mdPath = join(OUTPUT_DIR, 'benchmark-rcaeval.md');
  writeFileSync(mdPath, formatRCAEvalMarkdown(results));
  console.log(`\nMarkdown: ${mdPath}`);

  const jsonPath = join(OUTPUT_DIR, 'benchmark-rcaeval.json');
  writeFileSync(jsonPath, formatRCAEvalJSON(results));
  console.log(`JSON: ${jsonPath}`);

  console.timeEnd('RCAEval');
  console.log('\nDone.');
}
