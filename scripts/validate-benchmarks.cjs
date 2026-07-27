#!/usr/bin/env node
/**
 * Benchmark SHD validator — checks cross-dataset results against baselines.
 * Reads .github/benchmark-baselines.json and compares with vitest output.
 * Exits 0 if all pass, exits 1 if any algorithm exceeds its maxSHD.
 *
 * Usage: node scripts/validate-benchmarks.cjs <vitest-output.log>
 */
const fs = require('fs');
const path = require('path');

const baselinePath = path.join(__dirname, '..', '.github', 'benchmark-baselines.json');
const baselines = JSON.parse(fs.readFileSync(baselinePath, 'utf8'));

const output = fs.readFileSync(process.argv[2], 'utf8');

// Parse vitest output: extract "Dataset (Nn, Ee, Ss):" blocks with algorithm stats.
// Dataset headers look like: "ASIA (8n, 8e, 2000s):"
const datasetRegex = /(\w+(?:-Bias)?)\s+\((\d+)n,\s*(\d+)e,\s*(\d+)s\):/g;
const algoRegex = /^\s+(\w+)\s+edges=(\d+)\/\d+\s+SHD=(\d+)\s+TPR=([\d.]+)\s+FPR=([\d.]+)/gm;

let match;
let violations = 0;
let totalAlgos = 0;

while ((match = datasetRegex.exec(output)) !== null) {
  const dataset = match[1];
  const blockStart = match.index;
  const datasetHeaderEnd = match.index + match[0].length;
  const nextHeader = output.slice(datasetHeaderEnd).search(/\w+(?:-Bias)?\s+\(\d+n/);
  const block = nextHeader > 0
    ? output.substring(blockStart, datasetHeaderEnd + nextHeader)
    : output.substring(blockStart);

  const thresholds = baselines.baselines[dataset]?.thresholds;
  if (!thresholds) {
    console.log(`[WARN] No baselines for dataset: ${dataset}`);
    continue;
  }

  // Reset regex via re-creation to avoid cross-block state leakage
  const lineAlgoRegex = /^\s+(\w+)\s+edges=(\d+)\/\d+\s+SHD=(\d+)\s+TPR=([\d.]+)\s+FPR=([\d.]+)/gm;
  let algoMatch;
  while ((algoMatch = lineAlgoRegex.exec(block)) !== null) {
    const algo = algoMatch[1];
    const shd = parseInt(algoMatch[3]);
    const maxSHD = thresholds[algo]?.maxSHD;

    totalAlgos++;
    if (maxSHD !== undefined) {
      if (shd <= maxSHD) {
        console.log(`[PASS] ${dataset}/${algo}: SHD=${shd} ≤ ${maxSHD}`);
      } else {
        console.log(`[FAIL] ${dataset}/${algo}: SHD=${shd} > ${maxSHD}`);
        violations++;
      }
    } else {
      console.log(`[SKIP] ${dataset}/${algo}: no baseline defined`);
    }
  }
}

console.log(`\nResults: ${totalAlgos - violations}/${totalAlgos} passed, ${violations} violations`);

if (violations > 0) {
  console.log('::error::Algorithms exceeding SHD baselines — precision regression detected!');
  process.exit(1);
} else {
  console.log('All algorithms within baseline thresholds.');
  process.exit(0);
}
