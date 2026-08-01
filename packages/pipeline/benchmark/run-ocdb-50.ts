/**
 * Standalone 50-node OCDB runner with per-config error handling.
 * PC + GES only. BOSS/NOTEARS/LiNGAM/FCI/GFCI skipped.
 */
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { createRNG } from '@agentix-e/causality-analyzer-core';
import { CausalGraph } from '../src/graph/causal-graph.js';
import { generateLinearData, runBenchmark } from '../src/benchmark.js';
import { formatOCDBJSON } from './ocdb.js';

const SIZE = 50;
const SEED = 42;
const DENSITIES = [
  { label: 'low', v: 0.15 },
  { label: 'med', v: 0.25 },
  { label: 'high', v: 0.40 },
];
const SAMPLES = [200, 500, 1000, 5000];
const INSTANCES = 5;
const SKIP_ALGOS = ['BOSS', 'NOTEARS', 'LiNGAM', 'FCI', 'GFCI'];

const raw: any[] = [];
let s = SEED;

console.log('50-node OCDB -- PC + GES only\n');

for (const { label, v: density } of DENSITIES) {
  for (let inst = 0; inst < INSTANCES; inst++) {
    const rng = createRNG(s++);
    const names = Array.from({ length: SIZE }, (_, i) => 'V' + String(i));
    const truth = new CausalGraph(names);
    for (let i = 0; i < SIZE; i++) {
      for (let j = i + 1; j < SIZE; j++) {
        if (rng() < density) truth.addEdge(names[i]!, names[j]!);
      }
    }
    if (truth.edges.length < 2) continue;
    const nodeNames = [...truth.nodes];

    for (const nSamples of SAMPLES) {
      const desc = '50n_' + label + '_inst' + inst + '_' + nSamples + 's';
      console.log('  ' + desc + ' (algorithms: 2)');

      try {
        const { data } = generateLinearData(truth, nSamples, s++);
        const result = runBenchmark(desc, truth, data, nodeNames, SKIP_ALGOS);
        for (const alg of result.algorithms) {
          raw.push({
            graphSize: SIZE, density, sampleSize: nSamples, instanceIndex: inst,
            algorithm: alg.algorithm, shd: alg.shd, tpr: alg.tpr,
            fpr: alg.fpr, f1: alg.f1, timeMs: alg.timeMs,
          });
        }
      } catch (e: any) {
        console.error('  FAILED ' + desc + ': ' + e.message);
      }
    }
  }
}

// Aggregate
const groups = new Map<string, any[]>();
for (const r of raw) {
  const k = r.graphSize + '_' + r.sampleSize;
  if (!groups.has(k)) groups.set(k, []);
  groups.get(k)!.push(r);
}

const aggregated: any[] = [];
for (const [k, vals] of groups) {
  const [sizeStr, sampStr] = k.split('_');
  const algMap = new Map<string, number[][]>();
  for (const v of vals) {
    if (!algMap.has(v.algorithm)) algMap.set(v.algorithm, []);
    algMap.get(v.algorithm)!.push([v.shd, v.tpr, v.fpr, v.f1, v.timeMs]);
  }
  const algResults = [...algMap.entries()].map(([algo, vs]) => {
    const n = vs.length;
    const avg = (idx: number) => vs.reduce((s2, x) => s2 + x[idx]!, 0) / n;
    const std = (idx: number) => Math.sqrt(vs.reduce((s2, x) => s2 + (x[idx]! - avg(idx)) ** 2, 0) / n);
    return {
      algorithm: algo,
      avgShd: Math.round(avg(0) * 100) / 100,
      stdShd: Math.round(std(0) * 100) / 100,
      avgTpr: Math.round(avg(1) * 1000) / 1000,
      avgFpr: Math.round(avg(2) * 1000) / 1000,
      avgF1: Math.round(avg(3) * 1000) / 1000,
      avgTimeMs: Math.round(avg(4)),
    };
  });
  algResults.sort((a, b) => a.avgShd - b.avgShd);
  aggregated.push({
    graphSize: parseInt(sizeStr!, 10), sampleSize: parseInt(sampStr!, 10),
    numInstances: vals.length, algorithms: algResults,
  });
}

const outDir = join(import.meta.dirname, '..', 'benchmark-results');
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, 'benchmark-ocdb-50.json'), formatOCDBJSON(aggregated));

console.log('\nDone. ' + aggregated.length + ' configs, ' + raw.length + ' results.');
for (const r of aggregated) {
  const best = r.algorithms[0];
  console.log('  ' + r.graphSize + 'n ' + r.sampleSize + 's: ' + best.algorithm + ' SHD=' + best.avgShd + ' F1=' + best.avgF1.toFixed(3));
}
