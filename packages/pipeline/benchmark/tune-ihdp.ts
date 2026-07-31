/**
 * IHDP Forest Hyperparameter Tuning — CI grid search for v2.1.
 *
 * Searches forest nuisance model parameters (nTrees, maxDepth,
 * minLeafSize, sampleFraction) to minimize CausalForestDML PEHE.
 *
 * Run: pnpm benchmark:tune:ihdp
 *      BENCH_TUNING_TRIALS=3 pnpm benchmark:tune:ihdp (CI mode)
 */
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { generateIHDP, computePEHE, computeATEError } from './ihdp-data.js';
import { CausalForestDML } from '../src/infer/dml-estimators.js';
import { ForestDRLearner } from '../src/infer/dr-estimators.js';
import { CausalForest } from '../src/infer/causal-forest.js';
import type { NuisanceModel } from '../src/infer/double-ml.js';

const OUTPUT_DIR = join(import.meta.dirname, '..', 'benchmark-results');
const N_REPS = parseInt(process.env.BENCH_TUNING_TRIALS ?? '5', 10);

interface TuneResult {
  estimator: string;
  nTrees: number;
  maxDepth: number;
  minLeafSize: number;
  sampleFraction: number;
  pehe: number;
  ateError: number;
}

function forestNuisance(nTrees: number, maxDepth: number, minLeafSize: number, sampleFraction: number): NuisanceModel {
  return (trainX: number[][], trainY: number[], trainIdx: number[]) => {
    const forest = new CausalForest({
      nTrees, maxDepth, minLeafSize, seed: 42, sampleFraction,
    });
    forest.train(
      trainIdx.map((i: number) => trainX[i]!),
      trainIdx.map((i: number) => trainY[i]!),
      new Array(trainIdx.length).fill(1),
    );
    return (x: number[]) => forest.predictOne(x);
  };
}

function run(): void {
  mkdirSync(OUTPUT_DIR, { recursive: true });

  const grid = {
    nTrees: [30, 50, 100, 200],
    maxDepth: [8, 12, 15],
    minLeafSize: [3, 5, 10],
    sampleFraction: [0.5, 0.7],
  };

  const combos = cartesianProduct(grid);
  const totalRuns = combos.length * N_REPS * 2; // ×2 for two estimators
  console.log(`IHDP Forest Tuning: ${combos.length} configs × ${N_REPS} reps × 2 estimators = ${totalRuns} runs\n`);

  const results: TuneResult[] = [];

  // Pre-generate all IHDP repetitions for fair comparison
  const datasets = Array.from({ length: N_REPS }, (_, t) => generateIHDP(747, 25, 42 + t));

  for (const combo of combos) {
    const nuis = forestNuisance(combo.nTrees, combo.maxDepth, combo.minLeafSize, combo.sampleFraction);

    // CausalForestDML
    let peheSumCF = 0, ateSumCF = 0, okCF = 0;
    for (const ds of datasets) {
      try {
        const m = new CausalForestDML({
          nFolds: 3, outcomeModel: nuis, propensityModel: nuis,
          forestConfig: { nTrees: 50, minLeafSize: 5, seed: 42 },
        });
        m.fit(ds.X, ds.y, ds.t);
        const cate = m.effect(ds.X);
        peheSumCF += computePEHE(cate, ds.tauTrue);
        ateSumCF += computeATEError(cate, ds.tauTrue);
        okCF++;
      } catch { /* skip */ }
    }
    if (okCF > 0) {
      results.push({
        estimator: 'CausalForestDML',
        ...combo,
        pehe: peheSumCF / okCF,
        ateError: ateSumCF / okCF,
      });
    }

    // ForestDRLearner
    let peheSumDR = 0, okDR = 0;
    for (const ds of datasets) {
      try {
        const m = new ForestDRLearner({
          forestConfig: { nTrees: 30, minLeafSize: 5, seed: 42 },
        });
        m.fit(ds.X, ds.y, ds.t);
        const cate = new Array(ds.X.length).fill(m.ate);
        peheSumDR += computePEHE(cate, ds.tauTrue);
        okDR++;
      } catch { /* skip */ }
    }
    if (okDR > 0) {
      results.push({
        estimator: 'ForestDRLearner',
        ...combo,
        pehe: peheSumDR / okDR,
        ateError: 0,
      });
    }
  }

  // Find best per estimator
  for (const est of ['CausalForestDML', 'ForestDRLearner'] as const) {
    const subset = results.filter(r => r.estimator === est).sort((a, b) => a.pehe - b.pehe);
    if (subset.length === 0) continue;
    const best = subset[0]!;
    console.log(`${est}: PEHE=${best.pehe.toFixed(3)} nTrees=${best.nTrees} maxDepth=${best.maxDepth} minLeaf=${best.minLeafSize} sampleFrac=${best.sampleFraction}`);
  }

  writeFileSync(join(OUTPUT_DIR, 'tuning-ihdp.json'), JSON.stringify(results, null, 2));
  console.log(`\nResults saved to ${OUTPUT_DIR}/tuning-ihdp.json`);
}

function cartesianProduct(params: Record<string, number[]>): Record<string, number>[] {
  const keys = Object.keys(params);
  const values = Object.values(params);
  const results: Record<string, number>[] = [];
  const indices = new Array(keys.length).fill(0);
  while (true) {
    const combo: Record<string, number> = {};
    for (let i = 0; i < keys.length; i++) combo[keys[i]!] = values[i]![indices[i]!]!;
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

run();
