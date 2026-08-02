#!/usr/bin/env node
/**
 * RCAEval Real-Data Benchmark Runner.
 *
 * Evaluates RCAgent against the RCAEval real-world microservice dataset
 * (375 RE1 cases across 3 systems × 5 fault types × 5 repetitions).
 *
 * Data source: HuggingFace — phamquiluan/RCAEval (Parquet format)
 * Ground truth: cases.parquet
 * Python bridge: scripts/load-rcaeval.py for Parquet→JSON conversion
 *
 * Pipeline per case:
 *   1. Python loads metrics.parquet → aggregates per service → JSON
 *   2. TS parses JSON → Matrix + service names
 *   3. RCAgent.diagnose() on fault window
 *   4. Compare top rank vs ground truth from cases.parquet
 *
 * @packageDocumentation
 */

import { writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { execSync } from 'node:child_process';
import { Matrix } from 'ml-matrix';
import { RCAgent, type RCADiagnosis } from '../src/agent/rca-agent.js';

const OUTPUT_DIR = join(import.meta.dirname, '..', 'benchmark-results');

// ── Types ────────────────────────────────────────────────────────────────

interface RCACase {
  caseId: string;
  dataset: string;
  system: string;
  rootCauseService: string;
  fault: string;
  caseDir: string;
}

interface CaseResult {
  caseId: string;
  dataset: string;
  system: string;
  fault: string;
  groundTruth: string;
  predicted: string;
  topRank: number;
  top5: string[];
  correct: boolean;
  diagnosis: RCADiagnosis | null;
  error?: string;
  timeMs: number;
}

interface AggregateResult {
  method: string;
  dataset: string;
  total: number;
  top1Acc: number;
  top3Acc: number;
  top5Acc: number;
  avgAt5: number;
  mrr: number;
  byFault: Record<string, { total: number; top1: number; top5: number; avgAt5: number }>;
}

// ── Data Loading ──────────────────────────────────────────────────────────

function loadCaseIndex(datasetDir: string): RCACase[] {
  const scriptPath = join(import.meta.dirname, '..', '..', '..', 'scripts', 'load-rcaeval.py');
  const output = execSync(`python3 "${scriptPath}" case_index "${datasetDir}"`, {
    encoding: 'utf-8', maxBuffer: 50 * 1024 * 1024, timeout: 30000,
  });
  const records = JSON.parse(output) as Array<Record<string, unknown>>;
  return records.map(r => ({
    caseId: r['case'] as string,
    dataset: r['dataset'] as string,
    system: r['system'] as string,
    rootCauseService: r['root_cause_service'] as string,
    fault: r['fault'] as string,
    caseDir: join(datasetDir, r['case'] as string),
  }));
}

function loadServiceMetrics(
  caseDir: string,
): { data: Matrix; serviceNames: string[]; timesteps: number; faultIndex: number } | null {
  const metricPath = join(caseDir, 'metrics.parquet');
  if (!existsSync(metricPath)) return null;

  const scriptPath = join(import.meta.dirname, '..', '..', '..', 'scripts', 'load-rcaeval.py');

  try {
    const output = execSync(`python3 "${scriptPath}" case_metrics "${caseDir}"`, {
      encoding: 'utf-8', maxBuffer: 50 * 1024 * 1024, timeout: 60000,
    });
    const parsed = JSON.parse(output) as {
      error?: string; serviceNames: string[]; data: number[][]; nTimesteps: number; faultIndex: number;
    };
    if (parsed.error || parsed.serviceNames.length < 2 || parsed.nTimesteps === 0) return null;

    const rows: number[][] = [];
    for (let i = 0; i < parsed.nTimesteps; i++) {
      const row: number[] = [];
      for (let j = 0; j < parsed.serviceNames.length; j++) {
        row.push(parsed.data[j]?.[i] ?? 0);
      }
      rows.push(row);
    }

    return {
      data: new Matrix(rows),
      serviceNames: parsed.serviceNames,
      timesteps: parsed.nTimesteps,
      faultIndex: parsed.faultIndex,
    };
  } catch (e) {
    console.error(`  Failed: ${(e as Error).message}`);
    return null;
  }
}

// ── Evaluation ────────────────────────────────────────────────────────────

function evaluateCase(case_: RCACase): CaseResult {
  const t0 = performance.now();
  try {
    const metrics = loadServiceMetrics(case_.caseDir);
    if (!metrics) {
      return { caseId: case_.caseId, dataset: case_.dataset, system: case_.system, fault: case_.fault,
        groundTruth: case_.rootCauseService, predicted: 'NO_DATA', topRank: 999, top5: [], correct: false,
        diagnosis: null, error: 'No metrics', timeMs: performance.now() - t0 };
    }

    const faultData = metrics.faultIndex > 0 && metrics.faultIndex < metrics.timesteps
      ? new Matrix(metrics.data.to2DArray().slice(metrics.faultIndex))
      : metrics.data;

    if (faultData.rows < 5) {
      return { caseId: case_.caseId, dataset: case_.dataset, system: case_.system, fault: case_.fault,
        groundTruth: case_.rootCauseService, predicted: 'TOO_SHORT', topRank: 999, top5: [], correct: false,
        diagnosis: null, error: 'Fault window too short', timeMs: performance.now() - t0 };
    }

    const agent = new RCAgent({ multiFault: false });
    const diagnosis = agent.diagnose(faultData, metrics.serviceNames);
    const top5 = diagnosis.ranking.slice(0, 5).map(r => r.component);
    const gt = case_.rootCauseService.toLowerCase();
    const rank = top5.findIndex(
      s => s.toLowerCase() === gt || s.toLowerCase().includes(gt) || gt.includes(s.toLowerCase()),
    );
    return {
      caseId: case_.caseId, dataset: case_.dataset, system: case_.system, fault: case_.fault,
      groundTruth: case_.rootCauseService,
      predicted: diagnosis.ranking[0]?.component ?? 'UNKNOWN',
      topRank: rank >= 0 ? rank + 1 : top5.length + 1, top5,
      correct: rank === 0, diagnosis, timeMs: performance.now() - t0,
    };
  } catch (e) {
    return { caseId: case_.caseId, dataset: case_.dataset, system: case_.system, fault: case_.fault,
      groundTruth: case_.rootCauseService, predicted: 'ERROR', topRank: 999, top5: [], correct: false,
      diagnosis: null, error: (e as Error).message, timeMs: performance.now() - t0 };
  }
}

function aggregateResults(results: CaseResult[], method: string): AggregateResult[] {
  const byDataset = new Map<string, CaseResult[]>();
  for (const r of results) { const k = r.dataset; if (!byDataset.has(k)) byDataset.set(k, []); byDataset.get(k)!.push(r); }

  const agg: AggregateResult[] = [];
  for (const [dataset, cases] of byDataset) {
    const valid = cases.filter(c => !c.error);
    const T = cases.length;

    let avgAt5Sum = 0;
    for (const c of valid) for (let k = 1; k <= 5; k++) if (c.topRank === k) avgAt5Sum++;
    let mrrSum = 0;
    for (const c of valid) if (c.topRank <= 500) mrrSum += 1 / c.topRank;

    const byFault: Record<string, { total: number; top1: number; top5: number; avgAt5: number }> = {};
    for (const [f, fc] of groupBy(cases, c => c.fault)) {
      const ft = fc.filter(c => !c.error);
      let afSum = 0;
      for (const c of ft) for (let k = 1; k <= 5; k++) if (c.topRank === k) afSum++;
      byFault[f] = {
        total: fc.length, top1: fc.filter(c => c.topRank <= 1).length,
        top5: fc.filter(c => c.topRank <= 5).length,
        avgAt5: fc.length > 0 ? afSum / (fc.length * 5) : 0,
      };
    }

    agg.push({
      method, dataset, total: T,
      top1Acc: T > 0 ? valid.filter(c => c.topRank <= 1).length / T : 0,
      top3Acc: T > 0 ? valid.filter(c => c.topRank <= 3).length / T : 0,
      top5Acc: T > 0 ? valid.filter(c => c.topRank <= 5).length / T : 0,
      avgAt5: T > 0 ? avgAt5Sum / (T * 5) : 0,
      mrr: T > 0 ? mrrSum / T : 0,
      byFault,
    });
  }
  return agg;
}

function groupBy<T, K>(arr: T[], keyFn: (x: T) => K): Map<K, T[]> {
  const m = new Map<K, T[]>();
  for (const x of arr) { const k = keyFn(x); if (!m.has(k)) m.set(k, []); m.get(k)!.push(x); }
  return m;
}

// ── Reports ───────────────────────────────────────────────────────────────

function formatMarkdown(agg: AggregateResult[], cases: CaseResult[]): string {
  const lines: string[] = [
    '# RCAEval Real-Data Benchmark Results',
    '',
    '> Dataset: phamquiluan/RCAEval RE1 (375 cases, metric-only)',
    '> Method: RCAgent — PC causal discovery + HeuristicPathRCA',
    '',
    '## Overall Results',
    '',
    '| Dataset | Cases | Top-1 | Top-3 | Top-5 | Avg@5 | MRR |',
    '|---------|:---:|:---:|:---:|:---:|:---:|:---:|',
  ];
  for (const r of agg) {
    lines.push(`| ${r.dataset} | ${r.total} | ${pct(r.top1Acc)} | ${pct(r.top3Acc)} | ${pct(r.top5Acc)} | ${r.avgAt5.toFixed(3)} | ${r.mrr.toFixed(3)} |`);
  }

  // By fault type
  if (agg.length > 0) {
    const r = agg[0]!;
    lines.push('', `## By Fault Type — ${r.dataset}`, '',
      '| Fault | Cases | Top-1 | Top-5 | Avg@5 |',
      '|-------|:---:|:---:|:---:|:---:|');
    for (const [fault, f] of Object.entries(r.byFault).sort()) {
      lines.push(`| ${fault} | ${f.total} | ${pct(f.top1 / f.total)} | ${pct(f.top5 / f.total)} | ${f.avgAt5.toFixed(3)} |`);
    }
  }

  // Competitive comparison
  lines.push('', '## Competitive Comparison (Avg@5)',
    '| Method | Avg@5 | Source |',
    '|--------|:---:|--------|',
    '| **BARO** | **0.80** | RCAEval paper (WWW 2025) |',
    '| CIRCA | 0.46 | RCAEval paper |',
    '| RCD | 0.13 | RCAEval paper |',
    '| MicroCause | 0.20 | RCAEval paper |',
  );
  if (agg.length > 0) lines.push(`| **CA RCAgent** | **${agg[0]!.avgAt5.toFixed(3)}** | This benchmark |`);

  return lines.join('\n');
}

function formatJSON(agg: AggregateResult[], cases: CaseResult[]): string {
  return JSON.stringify({
    benchmark: 'RCAEval-Real', source: 'phamquiluan/RCAEval (HuggingFace)',
    timestamp: new Date().toISOString(),
    method: 'RCAgent — PC causal discovery + HeuristicPathRCA',
    summary: agg,
    cases: cases.map(c => ({ caseId: c.caseId, dataset: c.dataset, fault: c.fault, groundTruth: c.groundTruth, predicted: c.predicted, topRank: c.topRank, correct: c.correct, error: c.error })),
  }, null, 2);
}

function pct(val: number): string { return `${(val * 100).toFixed(1)}%`; }

// ── CLI ───────────────────────────────────────────────────────────────────

const isMainModule = process.argv[1]?.includes('run-rcaeval-real');

if (isMainModule) {
  const datasetDir = process.argv[2] ?? 'benchmark-data/rcaeval';
  if (!existsSync(datasetDir)) { console.error(`Dataset not found: ${datasetDir}`); process.exit(1); }

  console.log(`RCAEval Real-Data Benchmark — ${datasetDir}`);
  mkdirSync(OUTPUT_DIR, { recursive: true });

  console.log('Loading case index...');
  const cases = loadCaseIndex(datasetDir);
  console.log(`  ${cases.length} cases (${[...new Set(cases.map(c => c.dataset))].join(', ')})\n`);

  console.log('Running RCAgent on all cases...');
  console.time('RCAEval-Real');
  const results: CaseResult[] = [];
  let lastReport = 0;
  for (const c of cases) {
    const r = evaluateCase(c);
    results.push(r);
    if (results.length - lastReport >= 25 || results.length === cases.length) {
      const correct = results.filter(rr => rr.correct).length;
      console.log(`  [${results.length}/${cases.length}] Top-1: ${correct}/${results.length} (${(correct / results.length * 100).toFixed(1)}%)`);
      lastReport = results.length;
    }
  }
  console.timeEnd('RCAEval-Real');

  const agg = aggregateResults(results, 'RCAgent');
  for (const a of agg) {
    console.log(`  ${a.dataset}: Top-1=${pct(a.top1Acc)} Avg@5=${a.avgAt5.toFixed(3)} MRR=${a.mrr.toFixed(3)}`);
  }

  const mdPath = join(OUTPUT_DIR, 'benchmark-rcaeval-real.md');
  writeFileSync(mdPath, formatMarkdown(agg, results));
  const jsonPath = join(OUTPUT_DIR, 'benchmark-rcaeval-real.json');
  writeFileSync(jsonPath, formatJSON(agg, results));
  console.log(`\nReports: ${mdPath}, ${jsonPath}\nDone.`);
}
