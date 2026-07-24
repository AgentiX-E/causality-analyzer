/**
 * Causal Discovery Benchmark Suite.
 *
 * Runs canonical causal discovery algorithms on standard DAGs and
 * produces SHD/TPR/FPR metrics for quantitative comparison.
 *
 * Design: deterministic seeds for reproducibility; standard
 * evaluation metrics from the causal discovery literature.
 *
 * @packageDocumentation
 */
import { CausalGraph } from './graph/causal-graph.js';
import { Matrix } from 'ml-matrix';
import { pcAlgorithm } from './graph/pc.js';
import { gesAlgorithm } from './graph/ges.js';
import { fciAlgorithm } from './graph/advanced-discovery.js';
import { directLiNGAM } from './graph/lingam.js';
import { notearsAlgorithm } from './graph/notears.js';
import { createRNG } from '@agentix-e/causality-analyzer-core';

// ── Types ────────────────────────────────────────────────────────────

export interface AlgorithmResult {
  algorithm: string;
  graph: string;
  shd: number;
  tpr: number;
  fpr: number;
  f1: number;
  nEdges: number;
  nCorrect: number;
  nMissing: number;
  nExtra: number;
  timeMs: number;
}

export interface BenchmarkResult {
  name: string;
  nodes: number;
  trueEdges: number;
  algorithms: AlgorithmResult[];
}

// ── Canonical Graphs ─────────────────────────────────────────────────

export function asiaGraph(): CausalGraph {
  const g = new CausalGraph(['Asia', 'Smoke', 'Tub', 'Lung', 'Bronc', 'Either', 'XRay', 'Dysp']);
  g.addEdge('Asia', 'Tub');
  g.addEdge('Smoke', 'Lung');
  g.addEdge('Smoke', 'Bronc');
  g.addEdge('Tub', 'Either');
  g.addEdge('Lung', 'Either');
  g.addEdge('Either', 'XRay');
  g.addEdge('Either', 'Dysp');
  g.addEdge('Bronc', 'Dysp');
  return g;
}

export function sachsGraph(): CausalGraph {
  const g = new CausalGraph(['PKC', 'PKA', 'Raf', 'Mek', 'Erk', 'Akt', 'P38', 'Jnk', 'Plcg', 'PIP2', 'PIP3']);
  g.addEdge('PKC', 'Raf');
  g.addEdge('PKC', 'P38');
  g.addEdge('PKC', 'Jnk');
  g.addEdge('PKC', 'Plcg');
  g.addEdge('PKA', 'Raf');
  g.addEdge('PKA', 'Mek');
  g.addEdge('PKA', 'Erk');
  g.addEdge('PKA', 'Akt');
  g.addEdge('PKA', 'P38');
  g.addEdge('PKA', 'Jnk');
  g.addEdge('Raf', 'Mek');
  g.addEdge('Mek', 'Erk');
  g.addEdge('Plcg', 'PIP2');
  g.addEdge('Plcg', 'PIP3');
  g.addEdge('PIP3', 'PIP2');
  g.addEdge('PIP2', 'PKC');
  g.addEdge('Akt', 'Erk');
  return g;
}

export function mBiasGraph(): CausalGraph {
  const g = new CausalGraph(['X', 'Y', 'C1', 'C2', 'M']);
  g.addEdge('C1', 'X');
  g.addEdge('C1', 'M');
  g.addEdge('C2', 'M');
  g.addEdge('C2', 'Y');
  return g;
}

export function butterflyGraph(): CausalGraph {
  const g = new CausalGraph(['X', 'Y', 'C', 'M']);
  g.addEdge('C', 'X');
  g.addEdge('C', 'Y');
  g.addEdge('X', 'M');
  g.addEdge('M', 'Y');
  return g;
}

export function randomDAG(nodes: number, density: number, seed: number): CausalGraph {
  const rng = createRNG(seed);
  const names = Array.from({ length: nodes }, (_, i) => `V${i}`);
  const g = new CausalGraph(names);

  for (let i = 0; i < nodes; i++) {
    for (let j = i + 1; j < nodes; j++) {
      if (rng() < density) g.addEdge(names[i]!, names[j]!);
    }
  }
  return g;
}

// ── Data Generation ──────────────────────────────────────────────────

export function generateLinearData(graph: CausalGraph, n: number, seed: number, noise: number = 0.1): { data: number[][]; nodeNames: string[] } {
  const rng = createRNG(seed);
  const nodes = [...graph.nodes];
  const order = graph.topologicalSort();
  const data: number[][] = Array.from({ length: n }, () => new Array(nodes.length).fill(0));

  for (let i = 0; i < n; i++) {
    for (const node of order) {
      const parents = graph.parents(node);
      let val = 0;
      for (const p of parents) {
        const pIdx = nodes.indexOf(p);
        val += 0.7 * data[i]![pIdx]!;
      }
      val += (rng() - 0.5) * noise * 2;
      data[i]![nodes.indexOf(node)] = val;
    }
  }
  return { data, nodeNames: nodes };
}

// ── Evaluation ───────────────────────────────────────────────────────

export function computeSHD(predicted: CausalGraph, truth: CausalGraph): {
  shd: number; tpr: number; fpr: number; f1: number; nCorrect: number; nMissing: number; nExtra: number;
} {
  const predEdges = new Set(predicted.edges.map(e => `${e.source}→${e.target}`));
  const trueEdges = new Set(truth.edges.map(e => `${e.source}→${e.target}`));

  let correct = 0;
  for (const e of predEdges) if (trueEdges.has(e)) correct++;

  const missing = Math.max(0, trueEdges.size - correct);
  const extra = Math.max(0, predEdges.size - correct);
  const shd = missing + extra;

  const tpr = trueEdges.size > 0 ? correct / trueEdges.size : 1;
  const fpr = predEdges.size > 0 ? extra / predEdges.size : 0;
  const precision = predEdges.size > 0 ? correct / predEdges.size : 1;
  const recall = tpr;
  const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;

  return { shd, tpr, fpr, f1, nCorrect: correct, nMissing: missing, nExtra: extra };
}

// ── Runner ───────────────────────────────────────────────────────────

export function runBenchmark(
  name: string,
  truth: CausalGraph,
  data: number[][],
  nodeNames: string[],
): BenchmarkResult {
  const algorithms: Array<{ name: string; fn: (d: Matrix, nodes: string[]) => CausalGraph }> = [
    { name: 'PC', fn: (d, n) => (pcAlgorithm as any)(d, n).graph },
    { name: 'GES', fn: (d, n) => gesAlgorithm(d as any, n) },
    { name: 'NOTEARS', fn: (_d, n) => notearsAlgorithm([...Array(_d.rows)].map((_, i) => [...Array(_d.columns)].map((_, j) => _d.get(i, j))), n, { lambda1: 0.1, maxOuterIter: 10, wThreshold: 0.2 }).graph },
    { name: 'LiNGAM', fn: (d, n) => (directLiNGAM as any)(d, n).graph },
    { name: 'FCI', fn: (d, n) => (fciAlgorithm as any)(d, n).graph },
  ];

  const results: AlgorithmResult[] = [];
  const matrix = new Matrix(data);

  for (const alg of algorithms) {
    const t0 = performance.now();
    const predicted = alg.fn(matrix, nodeNames);
    const timeMs = performance.now() - t0;

    const metrics = computeSHD(predicted, truth);
    results.push({
      algorithm: alg.name,
      graph: name,
      ...metrics,
      nEdges: predicted.edges.length,
      timeMs: Math.round(timeMs),
    });
  }

  return { name, nodes: truth.nodeCount, trueEdges: truth.edges.length, algorithms: results };
}

export function formatBenchmarkTable(results: BenchmarkResult[]): string {
  const lines: string[] = [];
  lines.push('# Causal Discovery Benchmark Results');
  lines.push('');
  lines.push('| Graph | Nodes | True Edges | Algorithm | SHD | TPR | FPR | F1 | Edges Found | Time (ms) |');
  lines.push('|-------|-------|------------|-----------|-----|-----|-----|----|-------------|-----------|');

  for (const r of results) {
    for (const a of r.algorithms) {
      lines.push(`| ${r.name} | ${r.nodes} | ${r.trueEdges} | ${a.algorithm} | ${a.shd} | ${a.tpr.toFixed(3)} | ${a.fpr.toFixed(3)} | ${a.f1.toFixed(3)} | ${a.nEdges} | ${a.timeMs} |`);
    }
  }

  return lines.join('\n');
}
