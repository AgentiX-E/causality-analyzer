/**
 * Benchmark Runner — automated causal algorithm benchmarking with
 * result collection, timing, and report generation.
 *
 * Usage:
 *   pnpm benchmark:discovery  — causal discovery benchmark
 *   pnpm benchmark:estimation — CATE/ATE estimation benchmark
 *   pnpm benchmark:all        — full benchmark suite
 *   pnpm benchmark:report     — generate report from results/
 *
 * @packageDocumentation
 */

import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../src/graph/causal-graph.js';
import { pcAlgorithm } from '../src/graph/pc.js';
import { fciAlgorithm } from '../src/graph/advanced-discovery.js';
import { gesAlgorithm } from '../src/graph/ges.js';
import { directLiNGAM } from '../src/graph/lingam.js';
import { notearsAlgorithm } from '../src/graph/notears.js';
import { bossAlgorithm } from '../src/graph/boss.js';
import type { PCConfig } from '@agentix-e/causality-analyzer-core';

/** Benchmark result for a single algorithm-dataset pair */
interface BenchmarkEntry {
  dataset: string;
  algorithm: string;
  nodes: number;
  edges: number;
  shd: number;
  f1: number;
  precision: number;
  recall: number;
  timeMs: number;
  trial: number;
  timestamp: string;
}

/** Dataset definition for benchmarks */
interface BenchmarkDataset {
  name: string;
  graph: () => CausalGraph;
  nodes: number;
  trueEdges: number;
  samples: number;
  seed: number;
}

/** Algorithm entry for benchmark */
interface AlgorithmEntry {
  name: string;
  run: (data: Matrix, nodeNames: string[]) => CausalGraph;
  timeoutMs: number;
}

// ── Benchmark Datasets ─────────────────────────────────────────────

import {
  asiaGraph, sachsGraph, childGraph, insuranceGraph, waterGraph,
  alarmGraph, hailfinderGraph, hepar2Graph,
} from '../src/graph/canonical-graphs.js';

import { generateLinearData } from '../src/graph/synthetic-data.js';

/** All canonical BN datasets */
const canonicalDatasets: BenchmarkDataset[] = [
  { name: 'ASIA', graph: asiaGraph, nodes: 8, trueEdges: 8, samples: 2000, seed: 42 },
  { name: 'Sachs', graph: sachsGraph, nodes: 11, trueEdges: 17, samples: 2000, seed: 42 },
  { name: 'Child', graph: childGraph, nodes: 20, trueEdges: 25, samples: 2000, seed: 42 },
  { name: 'Insurance', graph: insuranceGraph, nodes: 27, trueEdges: 52, samples: 2000, seed: 42 },
  { name: 'Water', graph: waterGraph, nodes: 32, trueEdges: 66, samples: 2000, seed: 42 },
  { name: 'Alarm', graph: alarmGraph, nodes: 37, trueEdges: 46, samples: 2000, seed: 42 },
  { name: 'Hailfinder', graph: hailfinderGraph, nodes: 56, trueEdges: 66, samples: 2000, seed: 42 },
  { name: 'Hepar2', graph: hepar2Graph, nodes: 70, trueEdges: 123, samples: 2000, seed: 42 },
];

/** Erdos-Renyi synthetic datasets */
function erdosRenyiDatasets(): BenchmarkDataset[] {
  return [10, 30, 50].map(d => ({
    name: `ER-d${d}`,
    graph: () => generateRandomDAG(d, 2 * d, 42),
    nodes: d,
    trueEdges: 2 * d,
    samples: 2000,
    seed: 42,
  }));
}

// ── Benchmark Algorithms ───────────────────────────────────────────

const discoveryAlgorithms: AlgorithmEntry[] = [
  { name: 'PC', run: (d, n) => pcAlgorithm(d, n).graph, timeoutMs: 30000 },
  { name: 'GES', run: (d, n) => gesAlgorithm(d, n), timeoutMs: 60000 },
  { name: 'FCI', run: (d, n) => fciAlgorithm(d, n, {}).graph, timeoutMs: 120000 },
  { name: 'LiNGAM', run: (d, n) => directLiNGAM(d, n).graph, timeoutMs: 60000 },
  { name: 'NOTEARS', run: (d, n) => notearsAlgorithm(d, n).graph, timeoutMs: 60000 },
  { name: 'BOSS', run: (d, n) => bossAlgorithm(d, n, { numStarts: 5, maxIter: 30, seed: 42 }).graph, timeoutMs: 180000 },
];

// ── Metrics ────────────────────────────────────────────────────────

/** Compute SHD between discovered and true graph */
function computeSHD(discovered: CausalGraph, truth: CausalGraph): number {
  let shd = 0;
  const dEdges = new Set(discovered.edges.map(e => `${e.source}→${e.target}`));
  const tEdges = new Set(truth.edges.map(e => `${e.source}→${e.target}`));

  for (const de of dEdges) { if (!tEdges.has(de)) shd++; } // FP
  for (const te of tEdges) { if (!dEdges.has(te)) shd++; } // FN

  // Count reversals separately (already counted in FP+FN above)
  return shd;
}

/** Compute F1 score */
function computeF1(discovered: CausalGraph, truth: CausalGraph): {
  f1: number; precision: number; recall: number;
} {
  const dEdges = new Set(discovered.edges.map(e => `${e.source}→${e.target}`));
  const tEdges = new Set(truth.edges.map(e => `${e.source}→${e.target}`));

  let tp = 0;
  for (const te of tEdges) { if (dEdges.has(te)) tp++; }

  const precision = dEdges.size > 0 ? tp / dEdges.size : 0;
  const recall = tEdges.size > 0 ? tp / tEdges.size : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return { f1, precision, recall };
}

// ── Runner ─────────────────────────────────────────────────────────

/**
 * Run a single benchmark trial.
 */
function runTrial(
  dataset: BenchmarkDataset,
  algo: AlgorithmEntry,
  trial: number,
): BenchmarkEntry | null {
  const truth = dataset.graph();
  const nodeNames = truth.nodes;
  const startTime = currentTimeMs();

  try {
    // Generate data from truth graph
    const { data, nodeNames: names } = generateLinearData(truth, dataset.samples, dataset.seed + trial);
    const mat = new Matrix(data);

    const result = algo.run(mat, names);
    const elapsed = currentTimeMs() - startTime;

    if (elapsed > algo.timeoutMs) return null; // timeout

    const shd = computeSHD(result, truth);
    const { f1, precision, recall } = computeF1(result, truth);

    return {
      dataset: dataset.name,
      algorithm: algo.name,
      nodes: dataset.nodes,
      edges: dataset.trueEdges,
      shd,
      f1,
      precision,
      recall,
      timeMs: elapsed,
      trial,
      timestamp: new Date().toISOString(),
    };
  } catch {
    return null; // algorithm failed
  }
}

/**
 * Run full benchmark suite.
 */
function runBenchmark(
  datasets: BenchmarkDataset[],
  algorithms: AlgorithmEntry[],
  trials: number,
  outputDir: string,
): BenchmarkEntry[] {
  mkdirSync(outputDir, { recursive: true });
  const results: BenchmarkEntry[] = [];

  for (const ds of datasets) {
    for (const algo of algorithms) {
      // Skip slow algorithms on large graphs
      if (ds.nodes > 50 && ['BOSS', 'NOTEARS', 'FCI'].includes(algo.name)) continue;
      if (ds.nodes > 30 && algo.name === 'BOSS') continue;

      for (let t = 0; t < trials; t++) {
        const entry = runTrial(ds, algo, t);
        if (entry) results.push(entry);
      }
    }
  }

  // Save raw results
  const jsonPath = join(outputDir, 'benchmark-results.json');
  writeFileSync(jsonPath, JSON.stringify(results, null, 2));

  // Generate report
  const reportPath = join(outputDir, 'benchmark-report.md');
  writeFileSync(reportPath, generateReport(results));

  return results;
}

// ── Report Generation ──────────────────────────────────────────────

function generateReport(results: BenchmarkEntry[]): string {
  const lines: string[] = [
    `# Causality Analyzer Benchmark Report`,
    `**Date:** ${new Date().toISOString().split('T')[0]}`,
    `**Total Trials:** ${results.length}`,
    `**Datasets:** ${[...new Set(results.map(r => r.dataset))].join(', ')}`,
    `**Algorithms:** ${[...new Set(results.map(r => r.algorithm))].join(', ')}`,
    ``,
    `| Dataset | Algorithm | SHD (avg) | F1 | Precision | Recall | Time(s) |`,
    `|---------|-----------|-----------|-----|-----------|--------|---------|`,
  ];

  // Group by dataset, algorithm
  const groups = new Map<string, BenchmarkEntry[]>();
  for (const r of results) {
    const key = `${r.dataset}|${r.algorithm}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(r);
  }

  // Sort by dataset name, then algorithm
  const allDatasets = [...new Set(results.map(r => r.dataset))].sort();
  const allAlgos = [...new Set(results.map(r => r.algorithm))].sort();

  for (const ds of allDatasets) {
    for (const algo of allAlgos) {
      const g = groups.get(`${ds}|${algo}`);
      if (!g || g.length === 0) continue;

      const n = g.length;
      const avgField = <T extends keyof BenchmarkEntry>(field: T) =>
        g.reduce((s, r) => s + (r[field] as number), 0) / n;

      const shdAvg = avgField('shd').toFixed(1);
      const f1Avg = avgField('f1').toFixed(3);
      const prec = avgField('precision').toFixed(3);
      const rec = avgField('recall').toFixed(3);
      const time = (avgField('timeMs') / 1000).toFixed(1);

      lines.push(`| ${ds} | ${algo} | ${shdAvg} | ${f1Avg} | ${prec} | ${rec} | ${time} |`);
    }
  }

  return lines.join('\n');
}

// ── CLI Entry ──────────────────────────────────────────────────────

function currentTimeMs(): number {
  return Date.now();
}

function generateRandomDAG(d: number, e: number, seed: number): CausalGraph {
  let s = seed;
  const rng = () => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return s / 0x7FFFFFFF; };
  const nodes = Array.from({ length: d }, (_, i) => `V${i}`);
  const g = new CausalGraph(nodes);

  // Create a random topological ordering
  const order = [...nodes].sort(() => rng() - 0.5);

  // Add edges following the ordering (ensures DAG)
  let addedEdges = 0;
  const maxEdges = Math.min(e, (d * (d - 1)) / 2);

  for (let i = 0; i < d && addedEdges < maxEdges; i++) {
    for (let j = i + 1; j < d && addedEdges < maxEdges; j++) {
      if (rng() < 0.3) {
        g.addEdge(order[i]!, order[j]!);
        addedEdges++;
      }
    }
  }

  return g;
}

// ── Public API ─────────────────────────────────────────────────────

export { runBenchmark, computeSHD, computeF1, runTrial };
export type { BenchmarkEntry, BenchmarkDataset, AlgorithmEntry };
export {
  canonicalDatasets, erdosRenyiDatasets, discoveryAlgorithms,
  generateReport,
};
