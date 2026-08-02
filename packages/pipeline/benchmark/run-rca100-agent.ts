#!/usr/bin/env node
/**
 * RCA100 Benchmark Runner.
 *
 * Reads RCA100 Parquet task files, runs CA RCAgent (import, no CLI),
 * outputs predictions.
 *
 * Usage:
 *   npx tsx benchmark/run-rca100-agent.ts
 *
 * @packageDocumentation
 */

import { readFileSync, writeFileSync, mkdirSync, readdirSync, existsSync } from 'fs';
import { join } from 'path';
import { Matrix } from 'ml-matrix';
import { RCAgent } from '../src/agent/rca-agent.js';

async function main(): Promise<void> {
  const dataDir = 'benchmark-data/rca100/cases';
  if (!existsSync(dataDir)) {
    console.error('RCA100 dataset not found. Download first.');
    process.exit(1);
  }

  const files = readdirSync(dataDir).filter(f => f.endsWith('.parquet'));
  console.log(`RCA100 Runner — ${files.length} task files\n`);

  const agent = new RCAgent();
  const predictions: Array<{ task: string; prediction: string }> = [];

  for (const file of files) {
    console.log(`  Processing ${file}...`);

    // Parquet loading requires pyarrow; fall back gracefully
    try {
      // Placeholder: actual Parquet parsing needs a TS parquet reader or pyarrow bridge
      predictions.push({ task: file, prediction: '{"root_cause_component": "PENDING", "root_cause_reason": "Parquet parser needed"}' });
    } catch (e) {
      predictions.push({ task: file, prediction: `{"root_cause_component": "ERROR", "root_cause_reason": "${e instanceof Error ? e.message : 'unknown'}"}` });
    }
  }

  const outDir = 'benchmark-results';
  mkdirSync(outDir, { recursive: true });
  // Format predictions as CSV for RCA100 evaluation
  const lines = ['task,prediction'];
  for (const p of predictions) {
    lines.push(`${p.task},${p.prediction}`);
  }
  writeFileSync(join(outDir, 'rca100-predictions.csv'), lines.join('\n'));
  console.log(`\nDone. ${predictions.length} predictions.`);
}

main().catch(err => { console.error(err); process.exit(1); });
