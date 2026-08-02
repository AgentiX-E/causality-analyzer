#!/usr/bin/env node
/**
 * AIOps2025 Benchmark Runner — CA RCAgent Integration.
 *
 * CCF International AIOps Challenge 2025, Track 1: Microservice RCA.
 * 561 teams validated. Evaluated by F1-score (component + reason).
 *
 * Reads AIOps2025 metric data (CSV), runs CA's RCAgent (PC causal discovery +
 * HeuristicPath RCA), and outputs predictions in AIOps2025 format.
 *
 * Usage:
 *   npx tsx benchmark/run-aiops2025-agent.ts <dataset-path>
 *
 * Data Structure (per case directory):
 *   {uuid}/
 *     input_time.json          ← [{uuid, start_time, end_time}]
 *     metric/
 *       selected_apm/          ← Pre-filtered APM metrics (CSV)
 *       selected_infra/        ← Pre-filtered infra metrics (CSV)
 *       all_metric/            ← Full metric set (CSV, fallback)
 *
 * Output Format:
 *   {"uuid": "...", "component": "...", "reason": "...", "reasoning_trace": [...]}
 *
 * @packageDocumentation
 */

import { readFileSync, writeFileSync, mkdirSync, readdirSync, existsSync } from 'fs';
import { join, basename } from 'path';
import { Matrix } from 'ml-matrix';
import { RCAgent, type RCADiagnosis } from '../src/agent/rca-agent.js';

// ── Types ────────────────────────────────────────────────────────────────

interface AIOpsCase {
  uuid: string;
  startTime: string;
  endTime: string;
  casePath: string;
}

interface AIOpsPrediction {
  uuid: string;
  component: string;
  reason: string;
  reasoning_trace: Array<{
    step: number;
    action: string;
    observation: string;
  }>;
}

// ── Data Loading ──────────────────────────────────────────────────────────

/**
 * Parse input_time.json to get case definitions.
 */
function loadCases(datasetPath: string): AIOpsCase[] {
  // Try single-case (one directory) or multi-case (many directories)
  const timePath = join(datasetPath, 'input_time.json');
  if (existsSync(timePath)) {
    const raw = readFileSync(timePath, 'utf-8');
    const entries = JSON.parse(raw) as Array<{ uuid: string; start_time?: string; end_time?: string }>;
    return entries.map(e => ({
      uuid: e.uuid,
      startTime: e.start_time ?? '',
      endTime: e.end_time ?? '',
      casePath: datasetPath,
    }));
  }

  // Multi-case: scan subdirectories
  const dirs = readdirSync(datasetPath, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => join(datasetPath, d.name));

  const cases: AIOpsCase[] = [];
  for (const dir of dirs) {
    const tp = join(dir, 'input_time.json');
    if (!existsSync(tp)) continue;
    const raw = readFileSync(tp, 'utf-8');
    const entries = JSON.parse(raw) as Array<{ uuid: string; start_time?: string; end_time?: string }>;
    for (const e of entries) {
      cases.push({
        uuid: e.uuid,
        startTime: e.start_time ?? '',
        endTime: e.end_time ?? '',
        casePath: dir,
      });
    }
  }
  return cases;
}

/**
 * Load metric data from a case directory. Prefers selected_apm metrics,
 * falls back to all_metric, then selected_infra.
 */
function loadMetrics(casePath: string): { data: Matrix; serviceNames: string[] } | null {
  const metricDirs = [
    join(casePath, 'metric', 'selected_apm'),
    join(casePath, 'metric', 'all_metric'),
    join(casePath, 'metric', 'selected_infra'),
  ];

  for (const metricDir of metricDirs) {
    if (!existsSync(metricDir)) continue;

    const files = readdirSync(metricDir).filter(f => f.endsWith('.csv'));
    if (files.length === 0) continue;

    // Load the first CSV file
    const csvPath = join(metricDir, files[0]!);
    try {
      return parseMetricCSV(csvPath);
    } catch {
      // Try next directory
      continue;
    }
  }

  return null;
}

/**
 * Parse a metric CSV file into Matrix + column names.
 * Expected format: timestamp,service_1,service_2,...,service_n
 */
function parseMetricCSV(csvPath: string): { data: Matrix; serviceNames: string[] } {
  const content = readFileSync(csvPath, 'utf-8');
  const lines = content.trim().split('\n');
  if (lines.length < 2) {
    throw new Error(`Empty CSV: ${csvPath}`);
  }

  const header = lines[0]!.split(',').map(h => h.trim());
  const tsIdx = header.findIndex(h =>
    h.toLowerCase() === 'timestamp' ||
    h.toLowerCase() === 'time' ||
    h.toLowerCase() === 'datetime'
  );
  const serviceNames = header.filter((_, i) => i !== tsIdx);

  const rows: number[][] = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i]!.split(',');
    const row: number[] = [];
    for (let j = 0; j < header.length; j++) {
      if (j === tsIdx) continue;
      const val = parseFloat(parts[j] ?? '');
      if (!isNaN(val)) row.push(val);
    }
    if (row.length === serviceNames.length) {
      rows.push(row);
    }
  }

  if (rows.length === 0) {
    throw new Error(`No valid data rows in ${csvPath}`);
  }

  return {
    data: new Matrix(rows),
    serviceNames,
  };
}

// ── Prediction ────────────────────────────────────────────────────────────

/**
 * Build reasoning trace from RCAgent diagnosis.
 */
function buildReasoningTrace(
  diagnosis: RCADiagnosis,
  serviceNames: string[],
): Array<{ step: number; action: string; observation: string }> {
  const trace: Array<{ step: number; action: string; observation: string }> = [];

  // Step 1: Anomaly detection
  trace.push({
    step: 1,
    action: 'AnomalyDetection(z-score)',
    observation: `Anomalous services: ${diagnosis.anomalousServices.length > 0 ? diagnosis.anomalousServices.join(', ') : 'None detected'}`,
  });

  // Step 2: Causal discovery
  trace.push({
    step: 2,
    action: 'CausalDiscovery(PC algorithm, α=0.05)',
    observation: `Discovered ${diagnosis.graph.edges.length} causal edges among ${diagnosis.graph.nodes.length} services`,
  });

  // Step 3: RCA ranking
  const top3 = diagnosis.ranking.slice(0, 3);
  trace.push({
    step: 3,
    action: 'RCARanking(HeuristicPathRCA)',
    observation: `Top candidates: ${top3.map(r => `${r.component}(score=${r.score.toFixed(3)})`).join(', ') || 'None'}`,
  });

  // Step 4: Root cause determination
  trace.push({
    step: 4,
    action: 'RootCauseDetermination',
    observation: `Selected ${diagnosis.ranking[0]?.component ?? 'UNKNOWN'} as most likely root cause`,
  });

  return trace;
}

/**
 * Generate prediction from diagnosis.
 */
function generatePrediction(
  case_: AIOpsCase,
  diagnosis: RCADiagnosis,
  serviceNames: string[],
): AIOpsPrediction {
  const topRC = diagnosis.ranking[0];

  const component = topRC?.component ?? 'UNKNOWN';
  const reason = topRC
    ? `RCA score=${topRC.score.toFixed(3)}, root=${topRC.isRoot}. ` +
      `Causal graph: ${diagnosis.graph.edges.length} edges. ` +
      `Anomalous services: ${diagnosis.anomalousServices.join(',') || 'none'}.`
    : 'No causal signal detected. Fallback: first anomalous service.';

  return {
    uuid: case_.uuid,
    component,
    reason,
    reasoning_trace: buildReasoningTrace(diagnosis, serviceNames),
  };
}

// ── Main Runner ───────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.length < 1) {
    console.error('Usage: run-aiops2025-agent <dataset-path>');
    console.error('Example: run-aiops2025-agent benchmark-data/aiops2025/cases/');
    process.exit(1);
  }

  const datasetPath = args[0]!;
  console.log(`AIOps2025 Runner — ${datasetPath}\n`);

  const cases = loadCases(datasetPath);
  console.log(`  Found ${cases.length} cases\n`);

  if (cases.length === 0) {
    console.error('No cases found. Check dataset path.');
    process.exit(1);
  }

  const agent = new RCAgent();
  const predictions: AIOpsPrediction[] = [];
  let success = 0;
  let noData = 0;
  let errors = 0;

  for (let i = 0; i < cases.length; i++) {
    const c = cases[i]!;
    console.log(`  [${i + 1}/${cases.length}] ${c.uuid}...`);

    try {
      const metrics = loadMetrics(c.casePath);
      if (!metrics) {
        console.log(`    ⚠ No metric data found`);
        predictions.push({
          uuid: c.uuid,
          component: 'UNKNOWN',
          reason: 'No metric data available',
          reasoning_trace: [],
        });
        noData++;
        continue;
      }

      const diagnosis = agent.diagnose(metrics.data, metrics.serviceNames);
      const prediction = generatePrediction(c, diagnosis, metrics.serviceNames);
      predictions.push(prediction);
      success++;

      if (prediction.component !== 'UNKNOWN') {
        console.log(`    → ${prediction.component} (${prediction.reason.slice(0, 80)}...)`);
      }
    } catch (e: unknown) {
      console.log(`    ✗ ${(e as Error).message}`);
      predictions.push({
        uuid: c.uuid,
        component: 'ERROR',
        reason: (e as Error).message,
        reasoning_trace: [],
      });
      errors++;
    }
  }

  // Save predictions
  const outDir = 'benchmark-results';
  mkdirSync(outDir, { recursive: true });

  const jsonPath = join(outDir, 'aiops2025-predictions.json');
  writeFileSync(jsonPath, JSON.stringify(predictions, null, 2));

  // Also save CSV for easy import
  const csvLines = ['uuid,component,reason'];
  for (const p of predictions) {
    csvLines.push(`"${p.uuid}","${p.component}","${p.reason.replace(/"/g, '""')}"`);
  }
  const csvPath = join(outDir, 'aiops2025-predictions.csv');
  writeFileSync(csvPath, csvLines.join('\n'));

  console.log(`\nDone.`);
  console.log(`  Success: ${success} | No data: ${noData} | Errors: ${errors}`);
  console.log(`  JSON: ${jsonPath}`);
  console.log(`  CSV:  ${csvPath}`);
}

main().catch(err => {
  console.error('Runner error:', err);
  process.exit(1);
});
