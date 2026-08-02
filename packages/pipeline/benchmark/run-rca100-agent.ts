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
  const dataDir = 'benchmark-data/rca100/csv';
  const parquetDir = 'benchmark-data/rca100/cases';

  // Prefer CSV (converted by CI via pyarrow), fall back to Parquet directory
  const exists = existsSync(dataDir);
  const lookupDir = exists ? dataDir : parquetDir;

  if (!existsSync(lookupDir)) {
    console.error('RCA100 dataset not found. Run benchmark-rca100 CI workflow.');
    process.exit(1);
  }

  const files = readdirSync(lookupDir).filter(f => f.endsWith('.csv'));
  console.log(`RCA100 Runner — ${files.length} task files\n`);

  const agent = new RCAgent();
  const predictions: Array<{ task: string; prediction: string }> = [];

  for (const file of files) {
    console.log(`  Processing ${file}...`);

    try {
      // Load CSV data into Matrix
      const csvPath = join(lookupDir, file);
      const content = readFileSync(csvPath, 'utf-8');
      const lines = content.trim().split('\n');
      if (lines.length < 2) {
        predictions.push({ task: file, prediction: '{"root_cause_component": "NO_DATA", "root_cause_reason": "CSV has no data rows"}' });
        continue;
      }

      const header = lines[0]!.split(',');
      const dataRows: number[][] = [];
      for (let i = 1; i < lines.length; i++) {
        const vals = lines[i]!.split(',').map(Number);
        if (vals.every(v => !isNaN(v))) dataRows.push(vals);
      }

      if (dataRows.length === 0) {
        predictions.push({ task: file, prediction: '{"root_cause_component": "NO_DATA", "root_cause_reason": "No numeric data rows"}' });
        continue;
      }

      // Build Matrix and detect anomalies
      const data = new Matrix(dataRows);
      const serviceNames = header.filter((h: string) => h !== 'timestamp' && h !== 'datetime' && h !== 'time');

      // Use column subset if full header doesn't match
      const colCount = data.columns;
      const names = serviceNames.length === colCount
        ? serviceNames
        : Array.from({ length: colCount }, (_, i) => `metric_${i}`);

      // Run diagnosis
      const diagnosis = agent.diagnose(data, names);

      // Generate prediction (no LLM if no API key — use top RCA rank)
      const topRC = diagnosis.ranking[0];
      predictions.push({
        task: file,
        prediction: JSON.stringify({
          root_cause_component: topRC?.component ?? 'UNKNOWN',
          root_cause_reason: topRC
            ? `Top RCA score=${topRC.score.toFixed(3)}, root=${topRC.isRoot}`
            : 'No ranking produced',
        }),
      });
    } catch (e) {
      predictions.push({
        task: file,
        prediction: JSON.stringify({
          root_cause_component: 'ERROR',
          root_cause_reason: (e as Error).message,
        }),
      });
    }
  }

  const outDir = 'benchmark-results';
  mkdirSync(outDir, { recursive: true });
  const lines = ['task,prediction'];
  for (const p of predictions) {
    lines.push(`${p.task},${p.prediction}`);
  }
  writeFileSync(join(outDir, 'rca100-predictions.csv'), lines.join('\n'));
  console.log(`\nDone. ${predictions.length} predictions written.`);
}

main().catch(err => { console.error(err); process.exit(1); });
