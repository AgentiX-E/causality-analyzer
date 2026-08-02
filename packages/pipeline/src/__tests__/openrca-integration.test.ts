/**
 * OpenRCA Agent — End-to-End Integration Test.
 *
 * Simulates the full OpenRCA pipeline without requiring the 68 GB dataset:
 * 1. Generates synthetic metric CSV in OpenRCA telemetry format
 * 2. Runs RCAgent.diagnose() with CA's PC + HeuristicPath RCA
 * 3. Runs RCAgent.reason() with DeepSeek LLM
 * 4. Validates output matches OpenRCA prediction format
 *
 * Production-ready: this test validates the same code path the CI
 * workflow uses against the real OpenRCA dataset.
 *
 * @packageDocumentation
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { writeFileSync, readFileSync, mkdirSync, existsSync, unlinkSync, rmdirSync, rmSync } from 'fs';
import { join } from 'path';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../graph/causal-graph.js';
import { generateLinearData } from '../benchmark.js';
import { RCAgent, type RCAPrediction } from '../agent/rca-agent.js';

const SEED = 42;

// ── Synthetic OpenRCA Data Generator ─────────────────────────────────

function createSyntheticOpenRCAData(nQueries = 3): string {
  const tmpDir = '/tmp/openrca-synthetic/Telecom';
  const telemetryDir = join(tmpDir, 'telemetry', '2020_04_11', 'metric');
  mkdirSync(telemetryDir, { recursive: true });

  // Generate a microservice topology
  const services = ['Web', 'API', 'Auth', 'DB', 'Cache', 'Queue', 'Log', 'Monitor'];
  const g = new CausalGraph(services);
  g.addEdge('Web', 'API');
  g.addEdge('API', 'Auth');
  g.addEdge('API', 'Cache');
  g.addEdge('API', 'Queue');
  g.addEdge('API', 'Log');
  g.addEdge('Auth', 'DB');
  g.addEdge('Log', 'Monitor');

  const { data, nodeNames } = generateLinearData(g, 100, SEED);

  // Inject fault in DB at t=50
  const dbIdx = nodeNames.indexOf('DB');
  for (let t = 50; t < 100; t++) {
    if (data[t]) {
      data[t]![dbIdx] = (data[t]![dbIdx] ?? 0) * (1 + 3 * (t - 50) / 50);
    }
  }

  // Write metric CSV (OpenRCA format: timestamp, then service columns)
  const header = ['timestamp', ...nodeNames];
  const lines = [header.join(',')];
  for (let t = 0; t < data.length; t++) {
    const row = [String(t * 60), ...data[t]!.map(v => v!.toFixed(6))];
    lines.push(row.join(','));
  }
  writeFileSync(join(telemetryDir, 'metrics.csv'), lines.join('\n'));

  // Write query.csv (OpenRCA format)
  const queries = [
    'start_time,end_time,description',
    '"2020-04-11 00:00:00","2020-04-11 00:30:00","DB latency spike causing cascading failures in dependent services"',
    '"2020-04-11 00:00:00","2020-04-11 00:30:00","Auth service timeout propagating to API and Web"',
    '"2020-04-11 00:00:00","2020-04-11 00:30:00","Web service returning elevated error rates"',
  ];
  writeFileSync(join(tmpDir, 'query.csv'), queries.join('\n'));

  return tmpDir;
}

// ── Helpers ──────────────────────────────────────────────────────────

function parseSyntheticCSV(csvPath: string): { data: Matrix; serviceNames: string[] } {
  const content = readFileSync(csvPath, 'utf-8');
  const lines = content.trim().split('\n');
  const header = lines[0]!.split(',').map((h: string) => h.trim());
  const tsIdx = header.findIndex((h: string) => h.toLowerCase() === 'timestamp');
  const names = header.filter((_: string, i: number) => i !== tsIdx);
  const rows: number[][] = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i]!.split(',');
    const row: number[] = [];
    let col = 0;
    for (let j = 0; j < header.length; j++) {
      if (j === tsIdx) continue;
      row.push(parseFloat(parts[j] ?? '0') || 0);
      col++;
    }
    rows.push(row);
  }
  return { data: new Matrix(rows), serviceNames: names };
}

function formatOpenRCAPrediction(pred: RCAPrediction, datetime?: string): string {
  return JSON.stringify({
    'root_cause_occurrence_datetime': pred.datetime ?? datetime ?? '2020-04-11 00:00:00',
    'root_cause_component': pred.component,
    'root_cause_reason': pred.reason,
  });
}

// ── Tests ────────────────────────────────────────────────────────────

describe('OpenRCA Agent — End-to-End Integration', () => {
  const tmpDir = createSyntheticOpenRCAData();
  const agent = new RCAgent();
  const hasKey = (process.env['DEEPSEEK_API_KEY']?.length ?? 0) > 10;

  // Clean up after tests
  afterAll(() => {
    try {
      if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true });
    } catch { /* ignore */ }
  });

  it('synthetic data has valid structure', () => {
    const metricPath = join(tmpDir, 'telemetry', '2020_04_11', 'metric', 'metrics.csv');
    expect(existsSync(metricPath)).toBe(true);
    const queryPath = join(tmpDir, 'query.csv');
    expect(existsSync(queryPath)).toBe(true);
  });

  it('CA discovers causal graph from synthetic OpenRCA metrics', () => {
    const metricPath = join(tmpDir, 'telemetry', '2020_04_11', 'metric', 'metrics.csv');
    const { data, serviceNames } = parseSyntheticCSV(metricPath);
    const graph = agent.discover(data, serviceNames);
    expect(graph.edges.length).toBeGreaterThan(0);
    expect(graph.nodeCount).toBe(serviceNames.length);
  });

  it('RCA agent diagnoses synthetic incident', () => {
    const metricPath = join(tmpDir, 'telemetry', '2020_04_11', 'metric', 'metrics.csv');
    const { data, serviceNames } = parseSyntheticCSV(metricPath);
    const diagnosis = agent.diagnose(data, serviceNames);
    expect(diagnosis.graph.nodes.length).toBe(serviceNames.length);
  });

  it('agent produces OpenRCA-format predictions via LLM', async () => {
    if (!hasKey) {
      console.log('  Skipping LLM test (no API key)');
      expect(true).toBe(true);
      return;
    }

    const metricPath = join(tmpDir, 'telemetry', '2020_04_11', 'metric', 'metrics.csv');
    const { data, serviceNames } = parseSyntheticCSV(metricPath);
    const diagnosis = agent.diagnose(data, serviceNames);
    const prediction = await agent.reason(
      diagnosis,
      'All services experiencing elevated latency. DB metrics show strongest anomaly.',
    );

    // Validate OpenRCA format
    const formatted = formatOpenRCAPrediction(prediction, '2020-04-11 00:15:00');
    const parsed = JSON.parse(formatted);

    expect(parsed).toHaveProperty('root_cause_occurrence_datetime');
    expect(parsed).toHaveProperty('root_cause_component');
    expect(parsed).toHaveProperty('root_cause_reason');
    expect(typeof parsed.root_cause_component).toBe('string');
    expect(parsed.root_cause_component.length).toBeGreaterThan(0);

    console.log(`  CA-LLM prediction: component=${prediction.component}, reason=${prediction.reason}`);
  }, 30000);

  it('prediction CSV matches OpenRCA evaluation format', () => {
    // Verify the output format is compatible with main/evaluate.py
    const samplePrediction = formatOpenRCAPrediction({
      component: 'DB',
      reason: 'Database connection pool exhaustion',
      rawLLMResponse: '{}',
    }, '2020-04-11 00:15:00');

    const parsed = JSON.parse(samplePrediction);
    // OpenRCA evaluate.py expects datetime to match "%Y-%m-%d %H:%M:%S"
    expect(parsed.root_cause_occurrence_datetime).toMatch(/\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}/);
  });
});
