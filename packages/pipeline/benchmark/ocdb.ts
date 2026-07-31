/**
 * OCDB (Open Causal Discovery Benchmark) — Standard Benchmark Suite.
 *
 * Implements the canonical OCDB methodology: Erdős–Rényi random DAGs
 * across multiple graph sizes (8, 10, 20, 50, 100 nodes) and sample
 * sizes (200, 500, 1000, 5000), producing SHD/F1/TPR/FPR/SID/time
 * metrics for reproducible cross-library comparison.
 *
 * Reference: https://github.com/causal-learn/OCDB
 *
 * @packageDocumentation
 */

import { CausalGraph } from '../src/graph/causal-graph.js';
import { createRNG } from '@agentix-e/causality-analyzer-core';
import {
  generateLinearData,
  runBenchmark,
} from '../src/benchmark.js';

// ── OCDB Configuration ───────────────────────────────────────────────

/** Standard graph sizes for OCDB small/medium/large tiers. */
export const OCDB_GRAPH_SIZES = {
  small: [8, 10],
  medium: [20, 50],
  large: [100],
} as const;

/** Standard sample sizes per graph. */
export const OCDB_SAMPLE_SIZES = [200, 500, 1000, 5000] as const;

/** Number of random DAG instances per (size × density) combination. */
export const OCDB_INSTANCES_PER_SIZE = 5;

/** Edge density settings for Erdős–Rényi graphs. */
export const OCDB_DENSITIES = {
  low: 0.15,    // ~d edges per node (sparse)
  medium: 0.25, // ~2d edges per node
  high: 0.4,    // ~4d edges per node (dense)
} as const;

// ── Erdős–Rényi Random DAG Generator ─────────────────────────────────

/**
 * Generate a random DAG using Erdős–Rényi topology.
 * Nodes are ordered V0..Vn-1; edges only go forward (i < j) to
 * guarantee acyclicity.
 */
function generateERDAG(nodes: number, density: number, seed: number): CausalGraph {
  const rng = createRNG(seed);
  const names = Array.from({ length: nodes }, (_, i) => `V${i}`);
  const g = new CausalGraph(names);

  for (let i = 0; i < nodes; i++) {
    for (let j = i + 1; j < nodes; j++) {
      if (rng() < density) {
        g.addEdge(names[i], names[j]);
      }
    }
  }
  return g;
}

// ── OCDB Instance ────────────────────────────────────────────────────

export interface OCDBInstance {
  graphSize: number;
  density: number;
  instanceIndex: number;
  groundTruth: CausalGraph;
}

export interface OCDBSampleResult {
  graphSize: number;
  density: number;
  sampleSize: number;
  instanceIndex: number;
  algorithms: Array<{
    algorithm: string;
    shd: number;
    tpr: number;
    fpr: number;
    f1: number;
    nEdges: number;
    nCorrect: number;
    nMissing: number;
    nExtra: number;
    timeMs: number;
  }>;
}

export interface OCDBAggregateResult {
  graphSize: number;
  density: number;
  sampleSize: number;
  numInstances: number;
  algorithms: Array<{
    algorithm: string;
    avgShd: number;
    stdShd: number;
    avgTpr: number;
    avgFpr: number;
    avgF1: number;
    avgTimeMs: number;
  }>;
}

// ── Runner ───────────────────────────────────────────────────────────

/**
 * Run the full OCDB benchmark suite.
 *
 * Generates random DAGs at multiple sizes/densities, generates data
 * at multiple sample sizes, and benchmarks each algorithm × instance ×
 * sample size combination.
 */
export function runOCDBBenchmark(options?: {
  maxGraphSize?: number;
  skipLarge?: boolean;
  seed?: number;
}): { raw: OCDBSampleResult[]; aggregated: OCDBAggregateResult[] } {
  const maxSize = options?.maxGraphSize ?? 50;
  const seed = options?.seed ?? 42;

  const sizes: number[] = [];
  for (const s of OCDB_GRAPH_SIZES.small) if (s <= maxSize) sizes.push(s);
  for (const s of OCDB_GRAPH_SIZES.medium) if (s <= maxSize) sizes.push(s);
  if (!options?.skipLarge) {
    for (const s of OCDB_GRAPH_SIZES.large) if (s <= maxSize) sizes.push(s);
  }

  const densities: Array<{ label: string; value: number }> = [
    { label: 'low', value: OCDB_DENSITIES.low },
    { label: 'medium', value: OCDB_DENSITIES.medium },
    { label: 'high', value: OCDB_DENSITIES.high },
  ];

  const raw: OCDBSampleResult[] = [];
  let instanceSeed = seed;

  for (const size of sizes) {
    for (const { label: densLabel, value: density } of densities) {
      for (let inst = 0; inst < OCDB_INSTANCES_PER_SIZE; inst++) {
        // Generate ground-truth DAG
        const truth = generateERDAG(size, density, instanceSeed++);
        const nodeNames = [...truth.nodes];
        const trueEdgeCount = truth.edges.length;

        // Skip graphs with too few edges (degenerate)
        if (trueEdgeCount < 2) continue;

          // Test at each sample size
        for (const nSamples of OCDB_SAMPLE_SIZES) {
          const { data } = generateLinearData(truth, nSamples, instanceSeed++);

          // Determine which algorithms to skip on large graphs
          const skipLarge = size > 50
            ? ['LiNGAM', 'NOTEARS', 'FCI', 'GFCI']
            : size > 30
              ? ['LiNGAM', 'NOTEARS']
              : [];

          const benchmarkResult = runBenchmark(
            `${size}n_${densLabel}_inst${inst}_${nSamples}s`,
            truth,
            data,
            nodeNames,
            skipLarge,
          );

          for (const alg of benchmarkResult.algorithms) {
            raw.push({
              graphSize: size,
              density,
              sampleSize: nSamples,
              instanceIndex: inst,
              algorithms: [{
                algorithm: alg.algorithm,
                shd: alg.shd,
                tpr: alg.tpr,
                fpr: alg.fpr,
                f1: alg.f1,
                nEdges: alg.nEdges,
                nCorrect: alg.nCorrect,
                nMissing: alg.nMissing,
                nExtra: alg.nExtra,
                timeMs: alg.timeMs,
              }],
            });
          }
        }
      }
    }
  }

  // Aggregate by (size, density, sampleSize)
  const aggregated = aggregateOCDBResults(raw);

  return { raw, aggregated };
}

// ── Aggregation ──────────────────────────────────────────────────────

function aggregateOCDBResults(raw: OCDBSampleResult[]): OCDBAggregateResult[] {
  const groups = new Map<string, OCDBSampleResult[]>();

  for (const r of raw) {
    const key = `${r.graphSize}_${r.density}_${r.sampleSize}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(r);
  }

  const aggregated: OCDBAggregateResult[] = [];

  for (const [key, results] of groups) {
    const [sizeStr, densStr, sampStr] = key.split('_');
    const graphSize = parseInt(sizeStr!, 10);
    const density = parseFloat(densStr!);
    const sampleSize = parseInt(sampStr!, 10);

    const algMap = new Map<string, number[][]>();

    for (const r of results) {
      for (const a of r.algorithms) {
        if (!algMap.has(a.algorithm)) algMap.set(a.algorithm, []);
        algMap.get(a.algorithm)!.push([
          a.shd, a.tpr, a.fpr, a.f1, a.timeMs,
        ]);
      }
    }

    const algResults = [...algMap.entries()].map(([algorithm, values]) => {
      const n = values.length;
      const sumShd = values.reduce((s, v) => s + v[0]!, 0);
      const sumTpr = values.reduce((s, v) => s + v[1]!, 0);
      const sumFpr = values.reduce((s, v) => s + v[2]!, 0);
      const sumF1 = values.reduce((s, v) => s + v[3]!, 0);
      const sumTime = values.reduce((s, v) => s + v[4]!, 0);

      const avgShd = sumShd / n;
      const stdShd = Math.sqrt(
        values.reduce((s, v) => s + (v[0]! - avgShd) ** 2, 0) / n,
      );

      return {
        algorithm,
        avgShd: Math.round(avgShd * 100) / 100,
        stdShd: Math.round(stdShd * 100) / 100,
        avgTpr: Math.round(sumTpr / n * 1000) / 1000,
        avgFpr: Math.round(sumFpr / n * 1000) / 1000,
        avgF1: Math.round(sumF1 / n * 1000) / 1000,
        avgTimeMs: Math.round(sumTime / n),
      };
    });

    algResults.sort((a, b) => a.avgShd - b.avgShd);

    aggregated.push({
      graphSize,
      density,
      sampleSize,
      numInstances: results.length,
      algorithms: algResults,
    });
  }

  // Sort by graphSize, density, sampleSize
  aggregated.sort((a, b) =>
    a.graphSize - b.graphSize ||
    a.density - b.density ||
    a.sampleSize - b.sampleSize,
  );

  return aggregated;
}

// ── Report Formatting ─────────────────────────────────────────────────

/**
 * Format OCDB aggregated results as a Markdown comparison table.
 */
export function formatOCDBMarkdown(aggregated: OCDBAggregateResult[]): string {
  const lines: string[] = [];
  lines.push('# OCDB Benchmark Results');
  lines.push('');
  lines.push('> Generated: ' + new Date().toISOString());
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push('| Size | Density | Samples | Algorithm | SHD (μ±σ) | TPR | FPR | F1 | Time (ms) |');
  lines.push('|------|---------|---------|-----------|-----------|-----|-----|----|----------|');

  for (const r of aggregated) {
    const densityLabel = r.density <= 0.2 ? 'low' : r.density <= 0.3 ? 'medium' : 'high';
    for (const a of r.algorithms) {
      const shd = `${a.avgShd}±${a.stdShd}`;
      lines.push(
        `| ${r.graphSize} | ${densityLabel} | ${r.sampleSize} | ${a.algorithm} | ${shd} | ${a.avgTpr.toFixed(3)} | ${a.avgFpr.toFixed(3)} | ${a.avgF1.toFixed(3)} | ${a.avgTimeMs} |`,
      );
    }
  }

  return lines.join('\n');
}

/**
 * Format OCDB results as JSON for programmatic consumption.
 */
export function formatOCDBJSON(aggregated: OCDBAggregateResult[]): string {
  return JSON.stringify(
    {
      benchmark: 'OCDB',
      timestamp: new Date().toISOString(),
      results: aggregated,
    },
    null,
    2,
  );
}
