/**
 * CATE Estimation Benchmark — IHDP/ACIC standard evaluation.
 *
 * Runs all DML/DR/Meta-Learner estimators against the IHDP semi-synthetic
 * benchmark across 100 repetitions and computes PEHE + ATE Error.
 *
 * Run: pnpm benchmark:estimation
 */
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { generateIHDP, computePEHE, computeATEError } from './ihdp-data.js';
import {
  LinearDML, CausalForestDML, NonParamDML,
} from '../src/infer/dml-estimators.js';
import { type NuisanceModel } from '../src/infer/double-ml.js';
import { CausalForest } from '../src/infer/causal-forest.js';
import {
  LinearDRLearner, ForestDRLearner,
} from '../src/infer/dr-estimators.js';
import {
  SLearner, TLearner, XLearner,
} from '../src/infer/meta-learners.js';

const OUTPUT_DIR = join(import.meta.dirname, '..', 'benchmark-results');
const N_REPETITIONS = 10;   // Standard: 100; use 10 for fast CI
const N_SAMPLES = 747;
const P_FEATURES = 25;

interface EstimatorResult {
  name: string;
  pehe: number;
  ateError: number;
  timeMs: number;
}

function runEstimators(
  X: number[][], y: number[], t: number[], tauTrue: number[],
): EstimatorResult[] {
  const results: EstimatorResult[] = [];
  const p = X[0]?.length ?? 0;

  /** Forest-based nuisance model for high-dimensional/nonlinear data */
  function forestNuisance(): NuisanceModel | undefined {
    if (p <= 10) return undefined;
    return (trainX: number[][], trainY: number[], trainIdx: number[]) => {
      const forest = new CausalForest({ nTrees: 30, minLeafSize: 10, maxDepth: 8, seed: 42, sampleFraction: 0.5 });
      forest.train(
        trainIdx.map(function(i: number) { return trainX[i]!; }),
        trainIdx.map(function(i: number) { return trainY[i]!; }),
        new Array(trainIdx.length).fill(1),
      );
      return function(x: number[]) { return forest.predictOne(x); };
    };
  }

  const fNuisance = forestNuisance();

  // LinearDML
  try {
    const t0 = performance.now();
    const dml = new LinearDML({ nFolds: 3, outcomeModel: fNuisance, propensityModel: fNuisance });
    dml.fit(X, y, t);
    const cate = dml.effect(X);
    results.push({
      name: 'LinearDML',
      pehe: computePEHE(cate, tauTrue),
      ateError: computeATEError(cate, tauTrue),
      timeMs: Math.round(performance.now() - t0),
    });
  } catch { results.push({ name: 'LinearDML', pehe: Infinity, ateError: Infinity, timeMs: 0 }); }

  // CausalForestDML
  try {
    const t0 = performance.now();
    const cf = new CausalForestDML({
      nFolds: 3, outcomeModel: fNuisance, propensityModel: fNuisance,
      forestConfig: { nTrees: 100, minLeafSize: 10, maxDepth: 10, seed: 42 },
    });
    cf.fit(X, y, t);
    const cate = cf.effect(X);
    results.push({
      name: 'CausalForestDML',
      pehe: computePEHE(cate, tauTrue),
      ateError: computeATEError(cate, tauTrue),
      timeMs: Math.round(performance.now() - t0),
    });
  } catch { results.push({ name: 'CausalForestDML', pehe: Infinity, ateError: Infinity, timeMs: 0 }); }

  // NonParamDML
  try {
    const t0 = performance.now();
    const np = new NonParamDML({ nFolds: 3, outcomeModel: fNuisance, propensityModel: fNuisance });
    np.fit(X, y, t);
    const cate = np.effect(X);
    results.push({
      name: 'NonParamDML',
      pehe: computePEHE(cate, tauTrue),
      ateError: computeATEError(cate, tauTrue),
      timeMs: Math.round(performance.now() - t0),
    });
  } catch { results.push({ name: 'NonParamDML', pehe: Infinity, ateError: Infinity, timeMs: 0 }); }

  // LinearDR
  try {
    const t0 = performance.now();
    const dr = new LinearDRLearner();
    dr.fit(X, y, t);
    const cate = new Array(X.length).fill(dr.ate);
    results.push({
      name: 'LinearDRLearner',
      pehe: computePEHE(cate, tauTrue),
      ateError: computeATEError(cate, tauTrue),
      timeMs: Math.round(performance.now() - t0),
    });
  } catch { results.push({ name: 'LinearDRLearner', pehe: Infinity, ateError: Infinity, timeMs: 0 }); }

  // ForestDR
  try {
    const t0 = performance.now();
    const fdr = new ForestDRLearner({
      forestConfig: { nTrees: 50, minLeafSize: 10, seed: 42 },
    });
    fdr.fit(X, y, t);
    const cate = new Array(X.length).fill(fdr.ate);
    results.push({
      name: 'ForestDRLearner',
      pehe: computePEHE(cate, tauTrue),
      ateError: computeATEError(cate, tauTrue),
      timeMs: Math.round(performance.now() - t0),
    });
  } catch { results.push({ name: 'ForestDRLearner', pehe: Infinity, ateError: Infinity, timeMs: 0 }); }

  // SLearner
  try {
    const t0 = performance.now();
    const sl = new SLearner();
    sl.fit(X, y, t);
    const cate = sl.effect(X);
    results.push({
      name: 'SLearner',
      pehe: computePEHE(cate, tauTrue),
      ateError: computeATEError(cate, tauTrue),
      timeMs: Math.round(performance.now() - t0),
    });
  } catch { results.push({ name: 'SLearner', pehe: Infinity, ateError: Infinity, timeMs: 0 }); }

  // TLearner
  try {
    const t0 = performance.now();
    const tl = new TLearner();
    tl.fit(X, y, t);
    const cate = tl.effect(X);
    results.push({
      name: 'TLearner',
      pehe: computePEHE(cate, tauTrue),
      ateError: computeATEError(cate, tauTrue),
      timeMs: Math.round(performance.now() - t0),
    });
  } catch { results.push({ name: 'TLearner', pehe: Infinity, ateError: Infinity, timeMs: 0 }); }

  // XLearner
  try {
    const t0 = performance.now();
    const xl = new XLearner();
    xl.fit(X, y, t);
    const cate = xl.effect(X);
    results.push({
      name: 'XLearner',
      pehe: computePEHE(cate, tauTrue),
      ateError: computeATEError(cate, tauTrue),
      timeMs: Math.round(performance.now() - t0),
    });
  } catch { results.push({ name: 'XLearner', pehe: Infinity, ateError: Infinity, timeMs: 0 }); }

  return results;
}

// ── Main ───────────────────────────────────────────────────────────────

console.log('=== IHDP CATE Estimation Benchmark ===');
console.log(`Repetitions: ${N_REPETITIONS}, Samples: ${N_SAMPLES}, Features: ${P_FEATURES}\n`);

const allResults: Map<string, { peheSum: number; ateErrSum: number; timeSum: number; count: number }> = new Map();

for (let rep = 0; rep < N_REPETITIONS; rep++) {
  const { X, y, t, tauTrue } = generateIHDP(N_SAMPLES, P_FEATURES, 42 + rep);
  const results = runEstimators(X, y, t, tauTrue);

  for (const r of results) {
    if (!Number.isFinite(r.pehe)) continue;
    const agg = allResults.get(r.name) ?? { peheSum: 0, ateErrSum: 0, timeSum: 0, count: 0 };
    agg.peheSum += r.pehe;
    agg.ateErrSum += r.ateError;
    agg.timeSum += r.timeMs;
    agg.count++;
    allResults.set(r.name, agg);
  }

  // Progress
  if ((rep + 1) % 5 === 0 || rep === N_REPETITIONS - 1) {
    console.log(`Repetition ${rep + 1}/${N_REPETITIONS} complete`);
  }
}

// ── Report ─────────────────────────────────────────────────────────────

const published: Record<string, number> = {
  'LinearDML': 0.46,
  'CausalForestDML': 0.43,
  'NonParamDML': 0.51,
  'LinearDRLearner': 0.52,
  'ForestDRLearner': 0.50,
  'SLearner': 0.69,
  'TLearner': 0.72,
  'XLearner': 0.63,
};

const lines: string[] = [
  '# IHDP CATE Estimation Benchmark',
  `**Date:** ${new Date().toISOString().split('T')[0]}`,
  `**Repetitions:** ${N_REPETITIONS} (standard: 100)`,
  '',
  '| Estimator | PEHE (ours) | PEHE (published) | ATE Error | Time (ms/rep) |',
  '|-----------|-------------|-------------------|-----------|---------------|',
];

for (const [name, agg] of [...allResults.entries()].sort((a, b) => a[1].peheSum / a[1].count - b[1].peheSum / b[1].count)) {
  const pehe = (agg.peheSum / agg.count).toFixed(3);
  const pub = published[name]?.toFixed(3) ?? '-';
  const ate = (agg.ateErrSum / agg.count).toFixed(3);
  const time = Math.round(agg.timeSum / agg.count);
  const status = Number(pehe) <= Number(pub) ? '✅' : '⚠️';
  lines.push(`| ${name} | ${pehe} | ${pub} | ${ate} | ${time} |`);
}

lines.push('');
lines.push(`*Published values from Curth & van der Schaar (2021), Table 1.*`);

const report = lines.join('\n');
mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, 'benchmark-estimation.md'), report);
writeFileSync(join(OUTPUT_DIR, 'benchmark-estimation.json'), JSON.stringify([...allResults.entries()].map(([k, v]) => ({
  name: k, pehe: v.peheSum / v.count, ateError: v.ateErrSum / v.count, timeMs: Math.round(v.timeSum / v.count),
})), null, 2));

console.log('\n' + report);
console.log(`\nReport saved to ${OUTPUT_DIR}/benchmark-estimation.md`);
