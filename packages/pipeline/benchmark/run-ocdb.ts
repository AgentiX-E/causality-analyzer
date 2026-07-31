#!/usr/bin/env node
/**
 * OCDB Benchmark CLI Runner.
 *
 * Usage:
 *   pnpm benchmark:ocdb          # Small tier (8-10 nodes)
 *   pnpm benchmark:ocdb:full     # Full tier (up to 50 nodes)
 *   BENCH_OCDB_MAX_SIZE=20 pnpm benchmark:ocdb
 *
 * Environment variables:
 *   BENCH_OCDB_MAX_SIZE  — max graph size (default: 10 for CI, 50 for full)
 *   BENCH_OCDB_SEED      — random seed (default: 42)
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

console.log(`OCDB Benchmark — max graph size: ${MAX_SIZE}, seed: ${SEED}`);
console.time('Total time');

mkdirSync(OUTPUT_DIR, { recursive: true });

const { raw, aggregated } = runOCDBBenchmark({
  maxGraphSize: MAX_SIZE,
  skipLarge: IS_CI || MAX_SIZE <= 10,
  seed: SEED,
});

// Save Markdown report
const mdPath = join(OUTPUT_DIR, 'benchmark-ocdb.md');
const md = formatOCDBMarkdown(aggregated);
writeFileSync(mdPath, md);
console.log(`\nMarkdown: ${mdPath} (${md.length} bytes)`);

// Save JSON report
const jsonPath = join(OUTPUT_DIR, 'benchmark-ocdb.json');
const json = formatOCDBJSON(aggregated);
writeFileSync(jsonPath, json);
console.log(`JSON: ${jsonPath} (${json.length} bytes)`);

// Summary
console.log(`\nAggregated results: ${aggregated.length} configurations`);
for (const r of aggregated) {
  const best = r.algorithms[0];
  console.log(
    `  ${r.graphSize}n @ ${r.sampleSize}s: ` +
    `best=${best?.algorithm ?? 'N/A'} (SHD=${best?.avgShd ?? 'N/A'}, F1=${best?.avgF1.toFixed(3) ?? 'N/A'})`,
  );
}

console.timeEnd('Total time');
console.log('\nDone.');
