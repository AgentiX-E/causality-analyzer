/**
 * Quick v1.3 benchmark — ASIA/Sachs/Child only for fast verification.
 */
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import {
  asiaGraph, sachsGraph, childGraph,
  generateLinearData, runBenchmark, formatBenchmarkTable,
  type BenchmarkResult,
} from '../src/benchmark.js';

const OUTPUT_DIR = join(import.meta.dirname, '..', 'benchmark-results');

const DATASETS = [
  { name: 'ASIA', fn: asiaGraph, n: 2000 },
  { name: 'Sachs', fn: sachsGraph, n: 2000 },
  { name: 'Child', fn: childGraph, n: 2000 },
];

const prevSHD: Record<string, Record<string, number>> = {
  ASIA: { GES: 4, BOSS: 2 },
  Sachs: { GES: 9, BOSS: 14 },
  Child: { GES: 22, BOSS: 8 },
};

mkdirSync(OUTPUT_DIR, { recursive: true });
const results: BenchmarkResult[] = [];

for (const ds of DATASETS) {
  console.log(`\n=== ${ds.name} (${ds.fn().nodes.length} nodes) ===`);
  const truth = ds.fn();
  const { data, nodeNames } = generateLinearData(truth, ds.n, 42);

  try {
    const r = runBenchmark(ds.name, truth, data, nodeNames);
    results.push(r);
    for (const a of r.algorithms) {
      const prev = prevSHD[ds.name]?.[a.algorithm];
      const delta = prev !== undefined
        ? (a.shd < prev ? `↓${prev - a.shd} BETTER` : a.shd > prev ? `↑${a.shd - prev}` : `SAME`)
        : 'NEW';
      console.log(`  ${a.algorithm.padEnd(8)} SHD=${String(a.shd).padStart(3)} F1=${a.f1.toFixed(3)} (was ${prev ?? '-'}) ${delta}`);
    }
  } catch (e) {
    console.log(`  Error: ${(e as Error).message}`);
  }
}

// Save
const report = formatBenchmarkTable(results);
writeFileSync(join(OUTPUT_DIR, 'benchmark-discovery.md'), report);
writeFileSync(join(OUTPUT_DIR, 'benchmark-discovery.json'), JSON.stringify(results, null, 2));
console.log(`\nReport saved to ${OUTPUT_DIR}/`);
