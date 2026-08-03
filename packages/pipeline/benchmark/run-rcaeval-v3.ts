#!/usr/bin/env node
/**
 * RCAEval v3 Multi-modal Benchmark Runner.
 *
 * Full pipeline with trace topology + log error signals:
 *   1. Load metrics per service (Python bridge)
 *   2. Load traces → extract caller→callee topology (Python bridge)
 *   3. Load logs → extract error signals (Python bridge)
 *   4. diagnoseV3(): CUSUM + PC discovery + HeuristicPathRCA + MultiSourceRanker
 *   5. Compare against ground truth
 *
 * Evaluates on RE1 (metric-only), RE2 (metrics+traces+logs),
 * and RE3 (metrics+traces+logs) — 735 cases total.
 *
 * @packageDocumentation
 */

import { writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { execSync } from 'node:child_process';
import { Matrix } from 'ml-matrix';
import { RCAgent, type RCADiagnosis } from '../src/agent/rca-agent.js';

const SCRIPT_PATH = join(import.meta.dirname, '..', '..', '..', 'scripts', 'load-rcaeval.py');
const OUTPUT_DIR = join(import.meta.dirname, '..', 'benchmark-results');

// ── Types ────────────────────────────────────────────────────────────────

interface RCACase {
  caseId: string; dataset: string; suite: string; system: string;
  rootCauseService: string; fault: string; caseDir: string;
  hasLogs: boolean; hasTraces: boolean;
}

interface CaseResult {
  caseId: string; dataset: string; suite: string; system: string;
  fault: string; groundTruth: string; predicted: string;
  topRank: number; top5: string[]; correct: boolean;
  diagnosis: RCADiagnosis | null; error?: string; timeMs: number;
}

interface TraceEdge { source: string; target: string; callCount: number; }
interface LogError { service: string; errorCount: number; severity: string; errorTypes: string[]; }

// ── Data Loading ──────────────────────────────────────────────────────────

function loadCaseIndex(datasetDir: string, suiteFilter?: string): RCACase[] {
  const args = suiteFilter ? `"${datasetDir}" "${suiteFilter}"` : `"${datasetDir}"`;
  const output = execSync(`python3 "${SCRIPT_PATH}" case_index ${args}`, {
    encoding: 'utf-8', maxBuffer: 100 * 1024 * 1024, timeout: 30000,
  });
  const records = JSON.parse(output) as Array<Record<string, unknown>>;
  return records.map(r => ({
    caseId: r['case'] as string,
    dataset: r['dataset'] as string,
    suite: r['suite'] as string,
    system: r['system'] as string,
    rootCauseService: r['root_cause_service'] as string,
    fault: r['fault'] as string,
    caseDir: join(datasetDir, r['case'] as string),
    hasLogs: r['has_logs'] as boolean,
    hasTraces: r['has_traces'] as boolean,
  }));
}

function loadMetrics(caseDir: string): { data: Matrix; serviceNames: string[]; nTimesteps: number; faultIndex: number } | null {
  const output = execSync(`python3 "${SCRIPT_PATH}" case_metrics "${caseDir}"`, {
    encoding: 'utf-8', maxBuffer: 100 * 1024 * 1024, timeout: 60000,
  });
  const parsed = JSON.parse(output) as {
    error?: string; serviceNames: string[]; data: number[][]; nTimesteps: number; faultIndex: number;
  };
  if (parsed.error || parsed.serviceNames.length < 2 || parsed.nTimesteps === 0) return null;

  const rows: number[][] = [];
  for (let i = 0; i < parsed.nTimesteps; i++) {
    const row: number[] = [];
    for (let j = 0; j < parsed.serviceNames.length; j++) row.push(parsed.data[j]?.[i] ?? 0);
    rows.push(row);
  }
  return { data: new Matrix(rows), serviceNames: parsed.serviceNames, nTimesteps: parsed.nTimesteps, faultIndex: parsed.faultIndex };
}

function loadTraces(caseDir: string): TraceEdge[] {
  try {
    const output = execSync(`python3 "${SCRIPT_PATH}" case_traces "${caseDir}"`, {
      encoding: 'utf-8', maxBuffer: 50 * 1024 * 1024, timeout: 30000,
    });
    const parsed = JSON.parse(output) as { edges: TraceEdge[]; error?: string };
    return parsed.edges ?? [];
  } catch { return []; }
}

function loadLogs(caseDir: string): LogError[] {
  try {
    const output = execSync(`python3 "${SCRIPT_PATH}" case_logs "${caseDir}"`, {
      encoding: 'utf-8', maxBuffer: 50 * 1024 * 1024, timeout: 30000,
    });
    const parsed = JSON.parse(output) as { errors: LogError[]; error?: string };
    return parsed.errors ?? [];
  } catch { return []; }
}

// ── Evaluation ────────────────────────────────────────────────────────────

function evaluateCase(case_: RCACase): CaseResult {
  const t0 = performance.now();
  try {
    const metrics = loadMetrics(case_.caseDir);
    if (!metrics) {
      return result(case_, 'NO_DATA', 999, [], false, 'No metrics', t0);
    }

    const faultData = metrics.faultIndex > 0 && metrics.faultIndex < metrics.nTimesteps
      ? new Matrix(metrics.data.to2DArray().slice(metrics.faultIndex))
      : metrics.data;

    if (faultData.rows < 5) {
      return result(case_, 'TOO_SHORT', 999, [], false, 'Fault window too short', t0);
    }

    // ── Multi-modal pipeline ──
    const agent = new RCAgent();
    const diagnosis = agent.diagnoseV3(faultData, metrics.serviceNames);

    // ── Integrate trace topology (if available) ──
    if (case_.hasTraces) {
      const traceEdges = loadTraces(case_.caseDir);
      if (traceEdges.length > 0) {
        // Boost services that are root nodes in the trace topology (no incoming calls)
        const hasIncoming = new Set(traceEdges.map(e => e.target));
        const rootServices = traceEdges
          .filter(e => !hasIncoming.has(e.source))
          .map(e => e.source);
        const uniqueRoots = [...new Set(rootServices)];

        // Boost root service scores in the ranking
        const boostedRanking = diagnosis.ranking.map(r => ({
          ...r,
          score: uniqueRoots.some(root =>
            root.toLowerCase().includes(r.component.toLowerCase()) ||
            r.component.toLowerCase().includes(root.toLowerCase())
          ) ? r.score * 1.5 : r.score,
        }));
        diagnosis.ranking.length = 0;
        diagnosis.ranking.push(...boostedRanking.sort((a, b) => b.score - a.score));
      }
    }

    // ── Integrate log error signals (if available) ──
    if (case_.hasLogs) {
      const logErrors = loadLogs(case_.caseDir);
      if (logErrors.length > 0) {
        const criticalServices = new Set(
          logErrors.filter(e => e.severity === 'critical').map(e => e.service.toLowerCase())
        );
        // Boost services with critical log errors
        const boostedRanking = diagnosis.ranking.map(r => ({
          ...r,
          score: criticalServices.has(r.component.toLowerCase()) ? r.score * 1.3 : r.score,
        }));
        diagnosis.ranking.length = 0;
        diagnosis.ranking.push(...boostedRanking.sort((a, b) => b.score - a.score));
      }
    }

    const top5 = diagnosis.ranking.slice(0, 5).map(r => r.component);
    const gt = case_.rootCauseService.toLowerCase();
    const rank = top5.findIndex(
      s => s.toLowerCase() === gt || s.toLowerCase().includes(gt) || gt.includes(s.toLowerCase()),
    );

    return {
      caseId: case_.caseId, dataset: case_.dataset, suite: case_.suite, system: case_.system,
      fault: case_.fault, groundTruth: case_.rootCauseService,
      predicted: diagnosis.ranking[0]?.component ?? 'UNKNOWN',
      topRank: rank >= 0 ? rank + 1 : top5.length + 1, top5,
      correct: rank === 0, diagnosis, timeMs: performance.now() - t0,
    };
  } catch (e) {
    return result(case_, 'ERROR', 999, [], false, (e as Error).message, t0);
  }
}

function result(c: RCACase, pred: string, rank: number, top5: string[], correct: boolean, error: string, t0: number): CaseResult {
  return { caseId: c.caseId, dataset: c.dataset, suite: c.suite, system: c.system,
    fault: c.fault, groundTruth: c.rootCauseService, predicted: pred, topRank: rank,
    top5, correct, diagnosis: null, error, timeMs: performance.now() - t0 };
}

// ── Aggregation ───────────────────────────────────────────────────────────

function aggregate(results: CaseResult[]): string {
  const lines: string[] = [
    '# RCAEval v3 Multi-modal Benchmark Results',
    '',
    '> Method: RCAgent v3 — CUSUM + PC causal discovery + HeuristicPathRCA + MultiSourceRanker',
    '> Trace topology + log error signals integrated for RE2/RE3',
    '',
    '## Overall Results',
    '',
    '| Suite | Cases | Top-1 | Top-3 | Top-5 | Avg@5 | MRR |',
    '|-------|:---:|:---:|:---:|:---:|:---:|:---:|',
  ];

  for (const suite of ['RE1', 'RE2', 'RE3']) {
    const cases = results.filter(r => r.suite === suite);
    const valid = cases.filter(c => !c.error);
    const T = cases.length || 1;
    let avgAt5Sum = 0;
    for (const c of valid) for (let k = 1; k <= 5; k++) if (c.topRank === k) avgAt5Sum++;
    let mrrSum = 0;
    for (const c of valid) if (c.topRank <= 500) mrrSum += 1 / c.topRank;

    lines.push(`| ${suite} | ${cases.length} | ${pct(valid.filter(c => c.topRank <= 1).length / T)} | ${pct(valid.filter(c => c.topRank <= 3).length / T)} | ${pct(valid.filter(c => c.topRank <= 5).length / T)} | ${(T > 0 ? avgAt5Sum / (T * 5) : 0).toFixed(3)} | ${(T > 0 ? mrrSum / T : 0).toFixed(3)} |`);
  }

  // By system
  lines.push('', '## By System', '',
    '| System | Cases | Top-1 | Top-5 | Avg@5 |',
    '|--------|:---:|:---:|:---:|:---:|');
  for (const sys of ['ob', 'ss', 'tt']) {
    const cases = results.filter(r => r.system === sys);
    const valid = cases.filter(c => !c.error);
    const T = cases.length || 1;
    let avgAt5Sum = 0;
    for (const c of valid) for (let k = 1; k <= 5; k++) if (c.topRank === k) avgAt5Sum++;
    const sysName = sys === 'ob' ? 'Online Boutique (12 svc)' : sys === 'ss' ? 'Sock Shop (15 svc)' : 'Train Ticket (64 svc)';
    lines.push(`| ${sysName} | ${cases.length} | ${pct(valid.filter(c => c.topRank <= 1).length / T)} | ${pct(valid.filter(c => c.topRank <= 5).length / T)} | ${(T > 0 ? avgAt5Sum / (T * 5) : 0).toFixed(3)} |`);
  }

  // By fault type (RE2 — has most modalities)
  const re2Cases = results.filter(r => r.suite === 'RE2');
  if (re2Cases.length > 0) {
    lines.push('', '## By Fault Type (RE2)', '',
      '| Fault | Cases | Top-1 | Top-5 | Avg@5 |',
      '|-------|:---:|:---:|:---:|:---:|');
    const byFault = new Map<string, CaseResult[]>();
    for (const c of re2Cases) { if (!byFault.has(c.fault)) byFault.set(c.fault, []); byFault.get(c.fault)!.push(c); }
    for (const [fault, fc] of [...byFault].sort()) {
      const valid = fc.filter(c => !c.error);
      const T = fc.length || 1;
      let afSum = 0;
      for (const c of valid) for (let k = 1; k <= 5; k++) if (c.topRank === k) afSum++;
      lines.push(`| ${fault} | ${fc.length} | ${pct(valid.filter(c => c.topRank <= 1).length / T)} | ${pct(valid.filter(c => c.topRank <= 5).length / T)} | ${(T > 0 ? afSum / (T * 5) : 0).toFixed(3)} |`);
    }
  }

  // Competitive comparison
  lines.push('', '## Competitive Comparison (Avg@5)', '',
    '| Method | Avg@5 | Data | Source |',
    '|--------|:---:|------|--------|',
    '| **BARO** | **0.80** | Multi-modal (RE2) | FSE 2024 |',
    '| CIRCA | 0.46 | Multi-modal (RE2) | RCAEval paper |',
    '| MicroCause | 0.20 | Metric-only | RCAEval paper |',
    '| RCD | 0.13 | Metric-only | RCAEval paper |',
    `| **CA RCAgent v3** | **${computeOverallAvg5(results).toFixed(3)}** | Multi-modal | This benchmark |`,
  );

  return lines.join('\n');
}

function computeOverallAvg5(results: CaseResult[]): number {
  const valid = results.filter(c => !c.error);
  if (valid.length === 0) return 0;
  let sum = 0;
  for (const c of valid) for (let k = 1; k <= 5; k++) if (c.topRank === k) sum++;
  return sum / (valid.length * 5);
}

function pct(val: number): string { return `${(val * 100).toFixed(1)}%`; }

// ── CLI ───────────────────────────────────────────────────────────────────

const isMainModule = process.argv[1]?.includes('run-rcaeval-v3');

if (isMainModule) {
  const datasetDir = process.argv[2] ?? '/workspace/benchmark-data/rcaeval';
  const suiteFilter = process.argv[3] ?? 'RE1,RE2,RE3';
  if (!existsSync(datasetDir)) { console.error(`Dataset not found: ${datasetDir}`); process.exit(1); }

  console.log(`RCAEval v3 Multi-modal Benchmark — ${datasetDir}`);
  console.log(`Suites: ${suiteFilter}\n`);
  mkdirSync(OUTPUT_DIR, { recursive: true });

  const cases = loadCaseIndex(datasetDir, suiteFilter);
  console.log(`  ${cases.length} cases (${cases.filter(c => c.hasTraces).length} with traces, ${cases.filter(c => c.hasLogs).length} with logs)\n`);

  console.log('Running v3 pipeline...');
  console.time('RCAEval-v3');
  const results: CaseResult[] = [];
  let lastReport = 0;

  for (const c of cases) {
    const r = evaluateCase(c);
    results.push(r);
    if (results.length - lastReport >= 25 || results.length === cases.length) {
      const correct = results.filter(rr => rr.correct).length;
      const re2 = results.filter(rr => rr.suite === 'RE2');
      const re2Correct = re2.filter(rr => rr.correct).length;
      console.log(`  [${results.length}/${cases.length}] Overall: ${correct}/${results.length} (${(correct/results.length*100).toFixed(1)}%) | RE2: ${re2Correct}/${re2.length||1} (${(re2.length>0?(re2Correct/re2.length*100):0).toFixed(1)}%)`);
      lastReport = results.length;
    }
  }
  console.timeEnd('RCAEval-v3');

  const report = aggregate(results);
  const mdPath = join(OUTPUT_DIR, 'benchmark-rcaeval-v3.md');
  writeFileSync(mdPath, report);
  const jsonPath = join(OUTPUT_DIR, 'benchmark-rcaeval-v3.json');
  writeFileSync(jsonPath, JSON.stringify({ timestamp: new Date().toISOString(), results }, null, 2));
  console.log(`\nReports: ${mdPath}, ${jsonPath}\nDone.`);
}
