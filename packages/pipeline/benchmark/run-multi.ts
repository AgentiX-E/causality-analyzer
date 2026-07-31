/**
 * Multi-trial Benchmark CLI — 5-trial mean ± std for competitive comparison.
 *
 * Run via: pnpm benchmark:discovery (uses 3 trials for fast iteration)
 *          pnpm benchmark:discovery:full (uses 5 trials for publication)
 */
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import {
  asiaGraph, sachsGraph, alarmGraph, childGraph,
  generateLinearData, computeSHD,
  type AlgorithmResult,
} from '../src/benchmark.js';
import { Matrix } from 'ml-matrix';
import { pcAlgorithm } from '../src/graph/pc.js';
import { gesAlgorithm } from '../src/graph/ges.js';
import { fciAlgorithm } from '../src/graph/advanced-discovery.js';
import { gfciAlgorithm } from '../src/graph/gfci.js';
import { bossAlgorithm } from '../src/graph/boss.js';
import { notearsAlgorithm } from '../src/graph/notears.js';
import { directLiNGAM } from '../src/graph/lingam.js';
import { _resetFisherZCache } from '@agentix-e/causality-analyzer-core';
import type { CausalGraph } from '../src/graph/causal-graph.js';

const OUTPUT_DIR = join(import.meta.dirname, '..', 'benchmark-results');
const N_TRIALS = parseInt(process.env.BENCH_TRIALS ?? '3', 10);

const DISCOVERY_DATASETS = [
  { name: 'ASIA', fn: asiaGraph, n: 2000 },
  { name: 'Sachs', fn: sachsGraph, n: 2000 },
  { name: 'Child', fn: childGraph, n: 2000 },
];

// ── Metrics aggregation ─────────────────────────────────────────────────

interface TrialResult { shd: number; f1: number; tpr: number; fpr: number; timeMs: number; }
interface AggregatedResult { name: string; mean: TrialResult; std: TrialResult; min: TrialResult; max: TrialResult; trials: number; }

function aggregate(trials: TrialResult[]): AggregatedResult | null {
  if (trials.length === 0) return null;
  const n = trials.length;
  const fields: (keyof TrialResult)[] = ['shd', 'f1', 'tpr', 'fpr', 'timeMs'];
  const mean: Record<string, number> = {};
  const std: Record<string, number> = {};
  const min: Record<string, number> = {};
  const max: Record<string, number> = {};
  for (const f of fields) {
    const vals = trials.map(t => t[f]);
    mean[f] = vals.reduce((a,b) => a+b, 0) / n;
    std[f] = n > 1 ? Math.sqrt(vals.reduce((a,b) => a + (b - mean[f]!) ** 2, 0) / (n - 1)) : 0;
    min[f] = Math.min(...vals);
    max[f] = Math.max(...vals);
  }
  return { name: '', mean: mean as TrialResult, std: std as TrialResult, min: min as TrialResult, max: max as TrialResult, trials: n };
}

// ── Algorithm runner ────────────────────────────────────────────────────

function runAlgo(
  name: string, fn: (d: Matrix, nodes: string[]) => CausalGraph,
  data: Matrix, nodeNames: string[], truth: CausalGraph,
  skipIfSlow: boolean,
): TrialResult | null {
  if (skipIfSlow) return null;
  const t0 = Date.now();
  try {
    const pred = fn(data, nodeNames);
    const { shd, f1, tpr, fpr } = computeSHD(pred, truth);
    return { shd, f1, tpr, fpr, timeMs: Date.now() - t0 };
  } catch {
    return null;
  }
}

// ── Main ─────────────────────────────────────────────────────────────────

const skipLarge: Record<string, number> = { LiNGAM: 30, NOTEARS: 50, FCI: 100, GFCI: 100 };

function runBenchmarks() {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const allResults: Array<{ dataset: string; algo: string } & AggregatedResult> = [];

  for (const ds of DISCOVERY_DATASETS) {
    console.log(`\n=== ${ds.name} (${ds.fn().nodes.length} nodes, ${N_TRIALS} trials) ===`);
    const truth = ds.fn();
    const truthEdges = truth.edges.length;

    const algoResults: Map<string, TrialResult[]> = new Map();
    const algos: Array<{ name: string; fn: (d: Matrix, nodes: string[]) => CausalGraph }> = [
      { name: 'PC', fn: (d, n) => pcAlgorithm(d, n, {}).graph },
      { name: 'GES', fn: (d, n) => gesAlgorithm(d, n) },
      { name: 'BOSS', fn: (d, n) => bossAlgorithm(d, n, { seed: 42 }) },
      { name: 'NOTEARS', fn: (d, n) =>
        notearsAlgorithm(Array.from({ length: d.rows }, (_, i): number[] =>
          Array.from({ length: d.columns }, (_, j): number => d.get(i, j))), n,
          { lambda1: 0.005, maxOuterIter: 100, tol: 1e-8, wThreshold: 0.1 }).graph },
      { name: 'LiNGAM', fn: (d, n) => directLiNGAM(d, n).graph },
      { name: 'FCI', fn: (d, n) => fciAlgorithm(d, n, {}).graph },
      { name: 'GFCI', fn: (d, n) => gfciAlgorithm(d, n).graph },
    ];

    for (const algo of algos) {
      const skipSlow = (skipLarge[algo.name] ?? 999) < truth.nodeCount;
      const trials: TrialResult[] = [];

      for (let t = 0; t < N_TRIALS; t++) {
        _resetFisherZCache(); // prevent cross-trial cache contamination
        const { data, nodeNames } = generateLinearData(truth, ds.n, 42 + t);
        const mat = new Matrix(data);
        const r = runAlgo(algo.name, algo.fn, mat, nodeNames, truth, skipSlow);
        if (r) trials.push(r);
      }

      const agg = aggregate(trials);
      if (agg) {
        agg.name = algo.name;
        allResults.push({ dataset: ds.name, algo: algo.name, ...agg });
        const shdStr = String(agg.mean.shd).padStart(3);
        const f1Str = agg.mean.f1.toFixed(3);
        const stdStr = N_TRIALS > 1 ? ` (±${agg.std.shd.toFixed(1)})` : '';
        console.log(`  ${algo.name.padEnd(8)} SHD=${shdStr}${stdStr} F1=${f1Str} [${agg.min.shd}–${agg.max.shd}]`);
      } else {
        console.log(`  ${algo.name.padEnd(8)} skipped`);
      }
    }
  }

  // ── Save ──
  writeFileSync(join(OUTPUT_DIR, 'benchmark-history.json'), JSON.stringify(allResults, null, 2));

  // Markdown report
  const lines = ['# Causality Analyzer — Multi-Trial Benchmark', `**Trials:** ${N_TRIALS}`, `**Date:** ${new Date().toISOString().split('T')[0]}`, '',
    '| Dataset | Algorithm | SHD (mean) | SHD (std) | SHD (range) | F1 | Time (ms) |',
    '|---------|-----------|------------|-----------|-------------|-----|-----------|'];
  for (const r of allResults) {
    lines.push(`| ${r.dataset} | ${r.algo} | ${r.mean.shd} | ±${r.std.shd.toFixed(1)} | ${r.min.shd}–${r.max.shd} | ${r.mean.f1.toFixed(3)} | ${Math.round(r.mean.timeMs)} |`);
  }
  writeFileSync(join(OUTPUT_DIR, 'benchmark-report.md'), lines.join('\n'));
  console.log(`\nReport saved to ${OUTPUT_DIR}/benchmark-report.md`);
}

runBenchmarks();
