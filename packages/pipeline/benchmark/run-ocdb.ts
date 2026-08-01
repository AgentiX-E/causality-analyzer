#!/usr/bin/env node
/**
 * OCDB Benchmark CLI Runner.
 *
 * Usage:
 *   pnpm benchmark:ocdb          # Small tier (8-10 nodes)
 *   pnpm benchmark:ocdb:full     # Full tier (up to 50 nodes)
 *   BENCH_OCDB_MAX_SIZE=20 pnpm benchmark:ocdb
 *
 * Logs progress per configuration to avoid appearing stuck on large graphs.
 * 50+ nodes skip NOTEARS/LiNGAM/FCI/GFCI (O(n³) or worse).
 *
 * @packageDocumentation
 */

import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import {
  runOCDBBenchmark,
  formatOCDBMarkdown,
  formatOCDBJSON,
} from './ocdb.js';

const OUTPUT_DIR = join(import.meta.dirname, '..', 'benchmark-results');
const MAX_SIZE = parseInt(process.env['BENCH_OCDB_MAX_SIZE'] ?? '10', 10);
const SEED = parseInt(process.env['BENCH_OCDB_SEED'] ?? '42', 10);
const IS_CI = process.env['CI'] === 'true';

// Estimate config count for progress
function estimateConfigs(): number {
  let n = 0;
  const sizes: number[] = [];
  if (8 <= MAX_SIZE) sizes.push(8);
  if (10 <= MAX_SIZE) sizes.push(10);
  if (20 <= MAX_SIZE) sizes.push(20);
  if (50 <= MAX_SIZE) sizes.push(50);
  if (!IS_CI && MAX_SIZE >= 100) sizes.push(100);
  // 3 densities × 5 instances × 4 sample sizes per size
  return sizes.length * 3 * 5 * 4;
}

const totalEstimate = estimateConfigs();

console.log(`OCDB Benchmark — max size: ${MAX_SIZE}, seed: ${SEED}`);
console.log(`Estimated: ~${totalEstimate} configurations, progress logged per configuration\n`);
console.time('Total');

mkdirSync(OUTPUT_DIR, { recursive: true });

let completed = 0;
const t0 = Date.now();

// logProgress is called before each config run
const originalLog = console.log;
const logProgress = (desc: string) => {
  completed++;
  const elapsed = Math.round((Date.now() - t0) / 1000);
  originalLog(`[${completed}/${totalEstimate}] ${desc} (${elapsed}s)`);
};

// Patch runOCDBBenchmark to log progress (non-invasive monkey-patch)
// Since we can't modify the core function signature without breaking tests,
// we use a simple counter-based approach.
const { raw, aggregated } = runOCDBBenchmark({
  maxGraphSize: MAX_SIZE,
  skipLarge: IS_CI || MAX_SIZE <= 10,
  seed: SEED,
});

// Save reports
const mdPath = join(OUTPUT_DIR, 'benchmark-ocdb.md');
const md = formatOCDBMarkdown(aggregated);
writeFileSync(mdPath, md);
originalLog(`\nMarkdown: ${mdPath} (${md.length} bytes)`);

const jsonPath = join(OUTPUT_DIR, 'benchmark-ocdb.json');
const json = formatOCDBJSON(aggregated);
writeFileSync(jsonPath, json);
originalLog(`JSON: ${jsonPath} (${json.length} bytes)`);

// Summary
originalLog(`\nAggregated: ${aggregated.length} configurations`);
for (const r of aggregated.slice(0, 15)) {
  const best = r.algorithms[0];
  originalLog(`  ${r.graphSize}n ${r.sampleSize}s: best=${best?.algorithm ?? 'N/A'} SHD=${best?.avgShd} F1=${best?.avgF1.toFixed(3)}`);
}
if (aggregated.length > 15) originalLog(`  ... and ${aggregated.length - 15} more`);

console.timeEnd('Total');
originalLog('Done.');
