#!/usr/bin/env node
/**
 * CA RCA CLI — Causal Discovery + Root Cause Analysis from CSV.
 *
 * Reads a metrics CSV file, runs PC causal discovery to build a
 * service dependency graph, then runs HeuristicPath RCA to rank
 * root cause candidates. Outputs JSON with graph edges and RCA
 * ranking to stdout.
 *
 * Usage:
 *   npx tsx benchmark/ca-rca-cli.ts metrics.csv
 *   npx tsx benchmark/ca-rca-cli.ts metrics.csv --anomalous Web,API,DB
 *
 * Output (stdout):
 *   {
 *     "graph": { "nodes": [...], "edges": [{"source","target"}] },
 *     "ranking": [{"component","score","isRoot"}],
 *     "anomalousServices": [...]
 *   }
 *
 * @packageDocumentation
 */

import { readFileSync } from 'fs';
import { Matrix } from 'ml-matrix';
import { pcAlgorithm } from '../src/graph/pc.js';
import { HeuristicPathRCA } from '../src/analyze/rca.js';
import type { CausalGraph } from '../src/graph/causal-graph.js';

// ── Parse CSV ────────────────────────────────────────────────────────

interface ParsedMetrics {
  serviceNames: string[];
  data: Matrix;
  rawRows: number[][];
}

function parseCSV(path: string): ParsedMetrics {
  const content = readFileSync(path, 'utf-8');
  const lines = content.trim().split('\n');
  if (lines.length < 2) throw new Error('CSV must have header + at least 1 data row');

  const header = lines[0]!.split(',').map(h => h.trim());
  const timestampIdx = header.findIndex(h => h.toLowerCase() === 'timestamp');
  const serviceNames = header.filter((_, i) => i !== timestampIdx);

  const rawRows: number[][] = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i]!.split(',');
    const row: number[] = [];
    for (let j = 0; j < header.length; j++) {
      if (j === timestampIdx) continue;
      row.push(parseFloat(parts[j] ?? '0') || 0);
    }
    rawRows.push(row);
  }

  const data = new Matrix(rawRows);
  return { serviceNames, data, rawRows };
}

// ── Anomaly Detection ────────────────────────────────────────────────

function detectAnomalies(
  serviceNames: string[],
  rawRows: number[][],
  tailFraction = 0.2,
): string[] {
  const n = rawRows.length;
  const tail = Math.max(1, Math.floor(n * tailFraction));
  const recent = rawRows.slice(-tail);

  const anomalous: string[] = [];
  for (let j = 0; j < serviceNames.length; j++) {
    const fullCol = rawRows.map(r => r[j] ?? 0);
    const recentCol = recent.map(r => r[j] ?? 0);
    const fullMean = fullCol.reduce((s, v) => s + v, 0) / fullCol.length;
    const fullStd = Math.sqrt(fullCol.reduce((s, v) => s + (v - fullMean) ** 2, 0) / fullCol.length) || 1;
    const recentMean = recentCol.reduce((s, v) => s + v, 0) / recentCol.length;
    const zScore = (recentMean - fullMean) / fullStd;
    if (Math.abs(zScore) > 0.3) anomalous.push(serviceNames[j]!);
  }
  return anomalous;
}

// ── Main CLI ──────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.error('Usage: ca-rca-cli <metrics.csv> [--anomalous Svc1,Svc2,...]');
    process.exit(1);
  }

  const csvPath = args[0]!;
  const anomalousFlagIdx = args.indexOf('--anomalous');
  let specifiedAnomalous: string[] | null = null;
  if (anomalousFlagIdx >= 0 && args[anomalousFlagIdx + 1]) {
    specifiedAnomalous = args[anomalousFlagIdx + 1]!.split(',').map(s => s.trim());
  }

  const { serviceNames, data, rawRows } = parseCSV(csvPath);

  // Step 1: PC causal discovery
  const pcResult = pcAlgorithm(data, serviceNames, { alpha: 0.05, stable: true });
  const discoveredGraph: CausalGraph = pcResult.graph;

  // Step 2: Anomaly detection
  const anomalousServices = specifiedAnomalous ?? detectAnomalies(serviceNames, rawRows);

  // Step 3: RCA ranking
  const rca = new HeuristicPathRCA();
  rca.train(discoveredGraph, new Set(anomalousServices), data);
  const rcaResult = rca.findRootCauses(anomalousServices);

  // Step 4: Build output
  const output = {
    graph: {
      nodes: [...discoveredGraph.nodes].map(n => ({ name: n })),
      edges: discoveredGraph.edges.map(e => ({
        source: e.source,
        target: e.target,
        weight: e.weight ?? 1.0,
      })),
    },
    ranking: rcaResult.rootCauses.map(rc => ({
      component: rc.name,
      score: rc.score,
      isRoot: discoveredGraph.parents(rc.name).length === 0,
    })),
    anomalousServices,
  };

  process.stdout.write(JSON.stringify(output, null, 2));
}

main().catch(err => {
  console.error('CA RCA CLI error:', err);
  process.exit(1);
});
