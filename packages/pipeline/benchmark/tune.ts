/**
 * Hyperparameter Grid Search — automated CI tuning for v2.0.
 *
 * For each (algorithm, dataset, param_grid) combination, runs N trials
 * and selects the configuration with best mean SHD (discovery) or PEHE
 * (estimation).
 *
 * Run: pnpm benchmark:tune (all algorithms)
 *      pnpm benchmark:tune --algo=notears --dataset=asia
 */
import { writeFileSync, mkdirSync, readFileSync, existsSync } from 'fs';
import { join } from 'path';
import {
  asiaGraph, sachsGraph, childGraph,
  generateLinearData, computeSHD,
} from '../src/benchmark.js';
import { Matrix } from 'ml-matrix';
import { pcAlgorithm } from '../src/graph/pc.js';
import { gesAlgorithm } from '../src/graph/ges.js';
import { fciAlgorithm } from '../src/graph/advanced-discovery.js';
import { gfciAlgorithm } from '../src/graph/gfci.js';
import { bossAlgorithm } from '../src/graph/boss.js';
import { notearsAlgorithm } from '../src/graph/notears.js';
import type { CausalGraph } from '../src/graph/causal-graph.js';

const OUTPUT_DIR = join(import.meta.dirname, '..', 'benchmark-results');
const N_TRIALS = parseInt(process.env.BENCH_TUNING_TRIALS ?? '2', 10);

// ── Grid Definitions ───────────────────────────────────────────────────

interface ParamGrid {
  algo: string;
  params: Record<string, number[]>;
  fn: (d: Matrix, n: string[], params: Record<string, number>) => CausalGraph;
}

interface TuneDataset {
  name: string;
  graph: () => CausalGraph;
  samples: number;
}

const DATASETS: TuneDataset[] = [
  { name: 'ASIA', graph: asiaGraph, samples: 2000 },
  { name: 'Sachs', graph: sachsGraph, samples: 2000 },
  { name: 'Child', graph: childGraph, samples: 2000 },
];

const GRIDS: ParamGrid[] = [
  {
    algo: 'NOTEARS',
    params: {
      lambda1: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
      wThreshold: [0.05, 0.1, 0.15, 0.2],
    },
    fn: (d, n, p) => notearsAlgorithm(
      Array.from({ length: d.rows }, (_, i): number[] =>
        Array.from({ length: d.columns }, (_, j): number => d.get(i, j))),
      n,
      { lambda1: p.lambda1, maxOuterIter: 50, tol: 1e-8, wThreshold: p.wThreshold },
    ).graph,
  },
  {
    algo: 'GES',
    params: {
      penaltyDiscount: [1.0, 1.5, 2.0, 2.5, 3.0],
    },
    fn: (d, n, p) => gesAlgorithm(d, n, { penaltyDiscount: p.penaltyDiscount }),
  },
  {
    algo: 'BOSS',
    params: {
      numStarts: [3, 5, 10],
      maxParents: [4, 6, 8, -1],
    },
    fn: (d, n, p) => bossAlgorithm(d, n, { numStarts: p.numStarts, maxIter: 50, maxParents: p.maxParents, seed: 42 }),
  },
];

// ── Grid Search Engine ─────────────────────────────────────────────────

interface TuneResult {
  algo: string;
  dataset: string;
  params: Record<string, number>;
  shdMean: number;
  shdStd: number;
  f1Mean: number;
  trials: number;
}

function cartesianProduct(params: Record<string, number[]>): Record<string, number>[] {
  const keys = Object.keys(params);
  const values = Object.values(params);
  if (keys.length === 0) return [{}];

  const results: Record<string, number>[] = [];
  const indices = new Array(keys.length).fill(0);

  while (true) {
    const combo: Record<string, number> = {};
    for (let i = 0; i < keys.length; i++) {
      combo[keys[i]!] = values[i]![indices[i]!]!;
    }
    results.push(combo);

    let carry = keys.length - 1;
    while (carry >= 0) {
      indices[carry]!++;
      if (indices[carry]! < values[carry]!.length) break;
      indices[carry] = 0;
      carry--;
    }
    if (carry < 0) break;
  }
  return results;
}

function runTune(grid: ParamGrid, dataset: TuneDataset): TuneResult | null {
  const truth = dataset.graph();
  const combos = cartesianProduct(grid.params);
  let best: TuneResult | null = null;

  for (const combo of combos) {
    const shdValues: number[] = [];
    const f1Values: number[] = [];

    for (let t = 0; t < N_TRIALS; t++) {
      const { data, nodeNames } = generateLinearData(truth, dataset.samples, 42 + t);
      const mat = new Matrix(data);
      try {
        const pred = grid.fn(mat, nodeNames, combo);
        const m = computeSHD(pred, truth);
        shdValues.push(m.shd);
        f1Values.push(m.f1);
      } catch {
        // skip failed runs
      }
    }

    if (shdValues.length === 0) continue;

    const shdMean = shdValues.reduce((a, b) => a + b, 0) / shdValues.length;
    const shdStd = N_TRIALS > 1
      ? Math.sqrt(shdValues.reduce((a, b) => a + (b - shdMean) ** 2, 0) / (shdValues.length - 1))
      : 0;
    const f1Mean = f1Values.reduce((a, b) => a + b, 0) / f1Values.length;

    if (!best || shdMean < best.shdMean) {
      best = { algo: grid.algo, dataset: dataset.name, params: combo, shdMean, shdStd, f1Mean, trials: shdValues.length };
    }
  }

  return best;
}

// ── Main ─────────────────────────────────────────────────────────────────

function main() {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const results: TuneResult[] = [];
  const totalCombos = GRIDS.reduce((s, g) => s + cartesianProduct(g.params).length, 0) * DATASETS.length;

  console.log(`Grid Search: ${GRIDS.length} algorithms × ${DATASETS.length} datasets = ${totalCombos} configs × ${N_TRIALS} trials`);
  console.log('');

  for (const grid of GRIDS) {
    for (const ds of DATASETS) {
      const n = grid.params;
      const total = cartesianProduct(n).length;
      console.log(`${grid.algo} on ${ds.name}: ${total} configs × ${N_TRIALS} trials`);
      const best = runTune(grid, ds);
      if (best) {
        results.push(best);
        console.log(`  BEST: SHD=${best.shdMean.toFixed(1)} (±${best.shdStd.toFixed(1)}) F1=${best.f1Mean.toFixed(3)} params=${JSON.stringify(best.params)}`);
      }
    }
  }

  // ── Save ──
  const path = join(OUTPUT_DIR, 'tuning-results.json');
  writeFileSync(path, JSON.stringify(results, null, 2));
  console.log(`\nResults saved to ${path}`);

  // Markdown summary
  const lines = ['# Hyperparameter Tuning Results', `**Date:** ${new Date().toISOString().split('T')[0]}`, `**Trials:** ${N_TRIALS}`, '',
    '| Algorithm | Dataset | Best SHD | F1 | Best Params |',
    '|-----------|---------|----------|-----|-------------|'];
  for (const r of results.sort((a, b) => a.algo.localeCompare(b.algo) || a.dataset.localeCompare(b.dataset))) {
    lines.push(`| ${r.algo} | ${r.dataset} | ${r.shdMean.toFixed(1)} | ${r.f1Mean.toFixed(3)} | \`${JSON.stringify(r.params)}\` |`);
  }
  writeFileSync(join(OUTPUT_DIR, 'tuning-report.md'), lines.join('\n'));
}

main();
