#!/usr/bin/env node
/**
 * Export benchmark DAGs as CSV for cross-language comparison.
 *
 * Generates CSV data files for standard BNLearn networks (ASIA, Child,
 * Alarm, Mildew, Barley) that both CA and causal-learn can process.
 * Each dataset is saved as `benchmark-results/data/{network}_n{samples}.csv`
 * plus a `_truth.json` with the ground-truth adjacency for SHD computation.
 *
 * Usage: npx tsx benchmark/export-benchmark-data.ts
 *
 * @packageDocumentation
 */

import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import {
  asiaGraph,
  sachsGraph,
  childGraph,
  alarmGraph,
  barleyGraph,
  mildewGraph,
  generateLinearData,
} from '../src/benchmark.js';

const OUTPUT_DIR = join(import.meta.dirname, '..', 'benchmark-results', 'data');

interface BenchmarkDataset {
  name: string;
  fn: () => ReturnType<typeof asiaGraph>;
  nodes: number;
  edges: number;
}

const DATASETS: BenchmarkDataset[] = [
  { name: 'ASIA', fn: asiaGraph, nodes: 8, edges: 8 },
  { name: 'Sachs', fn: sachsGraph, nodes: 11, edges: 17 },
  { name: 'Child', fn: childGraph, nodes: 20, edges: 25 },
  { name: 'Alarm', fn: alarmGraph, nodes: 37, edges: 46 },
  { name: 'Barley', fn: barleyGraph, nodes: 48, edges: 84 },
  { name: 'Mildew', fn: mildewGraph, nodes: 35, edges: 46 },
];

const SAMPLE_SIZES = [500, 1000, 2000, 5000];
const SEED = 42;

mkdirSync(OUTPUT_DIR, { recursive: true });

interface TruthEntry {
  nodes: string[];
  edges: Array<[string, string]>;
}

console.log('Exporting benchmark datasets for cross-language comparison...\n');

const metadata: Record<string, unknown> = {};

for (const ds of DATASETS) {
  const truth = ds.fn();
  const truthData: TruthEntry = {
    nodes: [...truth.nodes],
    edges: truth.edges.map(e => [e.source, e.target] as [string, string]),
  };

  // Save ground truth once per dataset
  const truthPath = join(OUTPUT_DIR, `${ds.name}_truth.json`);
  writeFileSync(truthPath, JSON.stringify(truthData, null, 2));

  for (const n of SAMPLE_SIZES) {
    if (ds.nodes > 35 && n >= 2000) continue; // Skip heavy configs

    const { data, nodeNames } = generateLinearData(truth, n, SEED);

    // Write CSV: header row, then data rows
    const lines: string[] = [nodeNames.join(',')];
    for (const row of data) {
      lines.push(row.map(v => v.toFixed(6)).join(','));
    }
    const csvPath = join(OUTPUT_DIR, `${ds.name}_n${n}.csv`);
    writeFileSync(csvPath, lines.join('\n'));

    console.log(`  ${ds.name} n=${n}: ${data.length} rows × ${nodeNames.length} cols`);
  }

  metadata[ds.name] = { nodes: ds.nodes, trueEdges: ds.edges };
  console.log(`  ${ds.name}: truth saved (${truth.edges.length} edges)`);
}

// Save metadata
const metaPath = join(OUTPUT_DIR, 'benchmark-metadata.json');
writeFileSync(metaPath, JSON.stringify(metadata, null, 2));

console.log(`\nDone. ${Object.keys(metadata).length} datasets exported to ${OUTPUT_DIR}`);
