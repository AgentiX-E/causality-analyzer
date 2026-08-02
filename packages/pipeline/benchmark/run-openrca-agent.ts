#!/usr/bin/env node
/**
 * OpenRCA Benchmark Runner — CA RCAgent Integration.
 *
 * Reads OpenRCA telemetry CSV data, runs CA's RCAgent (PC causal discovery +
 * HeuristicPath RCA + DeepSeek LLM reasoning), and outputs predictions
 * in the format expected by OpenRCA's main/evaluate.py.
 *
 * Usage:
 *   npx tsx benchmark/run-openrca-agent.ts <system> <dataset-path>
 *
 * Example:
 *   npx tsx benchmark/run-openrca-agent.ts Telecom dataset/Telecom/
 *
 * Requires: DEEPSEEK_API_KEY environment variable (never in code)
 *
 * @packageDocumentation
 */

import { readFileSync, writeFileSync, mkdirSync, readdirSync, existsSync } from 'fs';
import { join } from 'path';
import { Matrix } from 'ml-matrix';
import { RCAgent, type RCAPrediction } from '../src/agent/rca-agent.js';

// ── Adapter: OpenRCA CSV → Matrix ────────────────────────────────────

interface OpenRCALoadResult {
  data: Matrix;
  serviceNames: string[];
  startTime: string;
  endTime: string;
  description: string;
}

/**
 * Load OpenRCA telemetry CSV for an incident window.
 *
 * OpenRCA data structure per date directory:
 *   telemetry/{DATE}/metric/*.csv — one CSV per service or one merged CSV
 *
 * We load the first available metric CSV in the date directory.
 */
function loadOpenRCAData(datasetPath: string, startTime: string, endTime: string): OpenRCALoadResult {
  // Find telemetry directories within the time window
  const telemetryDir = join(datasetPath, 'telemetry');
  const dateDirs = existsSync(telemetryDir)
    ? readdirSync(telemetryDir).filter(d => existsSync(join(telemetryDir, d, 'metric')))
    : [];

  if (dateDirs.length === 0) {
    throw new Error(`No telemetry found at ${telemetryDir}`);
  }

  // Use the first date directory (for now — OpenRCA may need specific date)
  const metricDir = join(telemetryDir, dateDirs[0]!, 'metric');
  const metricFiles = readdirSync(metricDir).filter(f => f.endsWith('.csv'));

  if (metricFiles.length === 0) {
    throw new Error(`No metric CSV files in ${metricDir}`);
  }

  // Load first metric file
  const csvPath = join(metricDir, metricFiles[0]!);
  return parseCSV(csvPath, startTime, endTime);
}

function parseCSV(csvPath: string, startTime: string, endTime: string): OpenRCALoadResult {
  const content = readFileSync(csvPath, 'utf-8');
  const lines = content.trim().split('\n');
  if (lines.length < 2) {
    return { data: new Matrix(0, 0), serviceNames: [], startTime, endTime, description: '' };
  }

  const header = lines[0]!.split(',').map(h => h.trim());
  const timestampIdx = header.findIndex(h => h.toLowerCase() === 'timestamp' || h.toLowerCase() === 'time');
  const serviceNames = header.filter((_, i) => i !== timestampIdx);

  const rows: number[][] = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i]!.split(',');
    const row: number[] = [];
    for (let j = 0; j < header.length; j++) {
      if (j === timestampIdx) continue;
      row.push(parseFloat(parts[j] ?? '0') || 0);
    }
    rows.push(row);
  }

  return {
    data: new Matrix(rows),
    serviceNames,
    startTime,
    endTime,
    description: `Failure window: ${startTime} to ${endTime}`,
  };
}

// ── Adapter: RCAgent prediction → OpenRCA CSV ───────────────────────

function formatOpenRCAPrediction(prediction: RCAPrediction, datetime?: string): string {
  const dt = prediction.datetime ?? datetime ?? 'unknown';
  return JSON.stringify({
    root_cause_occurrence_datetime: dt,
    root_cause_component: prediction.component,
    root_cause_reason: prediction.reason,
  });
}

// ── Main Runner ──────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.length < 2) {
    console.error('Usage: run-openrca-agent <system> <dataset-path>');
    console.error('Example: run-openrca-agent Telecom dataset/Telecom/');
    process.exit(1);
  }

  const system = args[0]!;
  const datasetPath = args[1]!;

  // Load queries (ground truth CSV — contains start/end times and descriptions)
  const queryPath = join(datasetPath, 'query.csv');
  if (!existsSync(queryPath)) {
    console.error(`query.csv not found at ${queryPath}. Download dataset first.`);
    process.exit(1);
  }

  const queryContent = readFileSync(queryPath, 'utf-8');
  const queryLines = queryContent.trim().split('\n');
  if (queryLines.length < 2) {
    console.error('query.csv is empty');
    process.exit(1);
  }

  console.log(`OpenRCA Runner — ${system} (${queryLines.length - 1} queries)\n`);

  const agent = new RCAgent();
  const predictions: Array<{ instruction: string; prediction: string }> = [];

  // Parse query CSV header
  const queryHeader = queryLines[0]!.split(',');
  const startIdx = queryHeader.findIndex(h => h.trim() === 'start_time');
  const endIdx = queryHeader.findIndex(h => h.trim() === 'end_time');
  const descIdx = queryHeader.findIndex(h => h.trim() === 'description');

  for (let i = 1; i < queryLines.length; i++) {
    const parts = queryLines[i]!.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/); // CSV with quotes
    const startTime = startIdx >= 0 ? (parts[startIdx] ?? '').replace(/"/g, '') : '';
    const endTime = endIdx >= 0 ? (parts[endIdx] ?? '').replace(/"/g, '') : '';
    const description = descIdx >= 0 ? (parts[descIdx] ?? '').replace(/"/g, '') : '';

    console.log(`  Query ${i}/${queryLines.length - 1}: ${description.slice(0, 60)}...`);

    try {
      const { data, serviceNames } = loadOpenRCAData(datasetPath, startTime, endTime);

      if (data.rows === 0 || serviceNames.length === 0) {
        predictions.push({
          instruction: description,
          prediction: formatOpenRCAPrediction({ component: 'UNKNOWN', reason: 'No telemetry data', rawLLMResponse: '{}' }),
        });
        continue;
      }

      const diagnosis = agent.diagnose(data, serviceNames);
      const prediction = await agent.reason(diagnosis, description);

      predictions.push({
        instruction: description,
        prediction: formatOpenRCAPrediction(prediction, startTime),
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'unknown';
      console.error(`  Failed query ${i}: ${msg}`);
      predictions.push({
        instruction: description,
        prediction: formatOpenRCAPrediction({ component: 'ERROR', reason: msg, rawLLMResponse: '{}' }),
      });
    }
  }

  // Save predictions
  const outDir = 'test/result';
  mkdirSync(outDir, { recursive: true });
  const outPath = join(outDir, `${system}/agent-ca-llm-deepseek-chat.csv`);
  mkdirSync(join(outDir, system), { recursive: true });

  const csvLines = ['instruction,prediction'];
  for (const p of predictions) {
    csvLines.push(`"${p.instruction.replace(/"/g, '""')}","${p.prediction.replace(/"/g, '""')}"`);
  }
  writeFileSync(outPath, csvLines.join('\n'));

  console.log(`\nDone. ${predictions.length} predictions saved to ${outPath}`);
  console.log('Evaluate with:');
  console.log(`  python -m main.evaluate -p ${outPath} -q ${queryPath} -r test/result/report.csv`);
}

main().catch(err => {
  console.error('Runner error:', err);
  process.exit(1);
});
