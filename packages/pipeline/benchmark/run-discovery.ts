/**
 * Benchmark CLI — discovery benchmarks using existing src/benchmark.ts.
 *
 * Run via: pnpm benchmark:discovery
 *          pnpm benchmark:estimation
 *          pnpm benchmark:all
 */
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import {
  asiaGraph, sachsGraph, alarmGraph, childGraph,
  barleyGraph, mildewGraph, win95ptsGraph, pathfinderGraph,
  generateLinearData, runBenchmark, formatBenchmarkTable,
  type BenchmarkResult,
} from '../src/benchmark.js';

const OUTPUT_DIR = join(import.meta.dirname, '..', 'benchmark-results');

const DISCOVERY_DATASETS = [
  { name: 'ASIA', fn: asiaGraph, nodes: 8, edges: 8, n: 2000 },
  { name: 'Sachs', fn: sachsGraph, nodes: 11, edges: 17, n: 2000 },
  { name: 'Child', fn: childGraph, nodes: 20, edges: 25, n: 2000 },
  { name: 'Mildew', fn: mildewGraph, nodes: 35, edges: 46, n: 2000 },
  { name: 'Alarm', fn: alarmGraph, nodes: 37, edges: 46, n: 2000 },
  { name: 'Barley', fn: barleyGraph, nodes: 48, edges: 84, n: 2000 },
  { name: 'Win95Pts', fn: win95ptsGraph, nodes: 76, edges: 112, n: 2000 },
  { name: 'Pathfinder', fn: pathfinderGraph, nodes: 109, edges: 195, n: 2000 },
];

function runDiscoveryBenchmarks() {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const results: BenchmarkResult[] = [];

  for (const ds of DISCOVERY_DATASETS) {
    console.log(`\n=== ${ds.name} (${ds.nodes} nodes, ${ds.edges} edges) ===`);
    const truth = ds.fn();
    const { data, nodeNames } = generateLinearData(truth, ds.n, 42);

    // Catch algorithm-level errors gracefully (known limitation on large graphs)
    try {
      const result = runBenchmark(ds.name, truth, data, nodeNames);
      for (const a of result.algorithms) {
        console.log(`  ${a.algorithm.padEnd(8)} SHD=${String(a.shd).padStart(3)} F1=${a.f1.toFixed(3)} TPR=${a.tpr.toFixed(3)} ${a.timeMs}ms`);
      }
      results.push(result);
    } catch (err) {
      console.log(`  Skipped: ${(err as Error).message}`);
    }
  }

  // Save report
  const report = formatBenchmarkTable(results);
  const reportPath = join(OUTPUT_DIR, 'benchmark-discovery.md');
  writeFileSync(reportPath, report);
  console.log(`\nReport: ${reportPath}`);

  const jsonPath = join(OUTPUT_DIR, 'benchmark-discovery.json');
  writeFileSync(jsonPath, JSON.stringify(results, null, 2));
  console.log(`JSON: ${jsonPath}`);
}

runDiscoveryBenchmarks();
