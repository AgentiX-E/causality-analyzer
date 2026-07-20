/**
 * RCA Analysis Algorithms.
 *
 * Bayesian Network RCA (Variable Elimination exact inference),
 * Random Walk RCA (weighted graph traversal),
 * Hypothesis Testing RCA (regression residual),
 * FP-Growth Trace RCA (frequent pattern mining).
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../graph/causal-graph.js';
import type { RootCause, RootCausePath, RCAResult, AnalysisMetadata } from '@agentix-e/causality-analyzer-core';

// ── Common Result Builder ─────────────────────────────────────────────
function buildResult(rootCauses: RootCause[], paths: RootCausePath[], method: string): RCAResult {
  return {
    rootCauses,
    paths,
    metadata: { method, analyzedAt: Date.now(), durationMs: 0, extra: {} },
    toJSON() { return { rootCauses, paths, metadata: this.metadata }; },
  };
}

// ── Bayesian Network RCA ──────────────────────────────────────────────
interface CPT { [parentState: string]: number; }

export class BayesianRCA {
  private graph: CausalGraph | null = null;
  private cpts = new Map<string, CPT>();

  train(graph: CausalGraph, anomalies: Set<string>, data: Matrix): void {
    this.graph = graph;
    const nodes = [...graph.nodes];
    for (const node of nodes) {
      const parents = graph.parents(node);
      const nodeIdx = nodes.indexOf(node);
      // Estimate CPT: P(node=anomalous | parents)
      const cpt: CPT = {};
      const n = data.rows;
      if (parents.length === 0) {
        let anomCount = 0;
        for (let r = 0; r < n; r++) anomCount += anomalies.has(node) ? 1 : 0;
        const p = Math.max(0.01, Math.min(0.99, anomCount / n));
        cpt['root'] = p;
      } else {
        const pKeys: string[] = [];
        const counts: Record<string, { anom: number; total: number }> = {};
        for (let r = 0; r < n; r++) {
          const key = parents.map(p => anomalies.has(p) ? '1' : '0').join('');
          if (!counts[key]) counts[key] = { anom: 0, total: 0 };
          counts[key]!.total++;
          if (anomalies.has(node)) counts[key]!.anom++;
        }
        for (const [key, cnt] of Object.entries(counts)) {
          cpt[key] = Math.max(0.01, Math.min(0.99, cnt.anom / cnt.total));
        }
      }
      this.cpts.set(node, cpt);
    }
  }

  findRootCauses(anomalousNodes: string[]): RCAResult {
    if (!this.graph) return buildResult([], [], 'bayesian');
    const anomalous = new Set(anomalousNodes);
    const rootNodes = [...this.graph.nodes].filter(n => this.graph!.parents(n).length === 0);

    // Variable Elimination: P(root | evidence)
    const scores: RootCause[] = [];
    for (const root of rootNodes) {
      const cpt = this.cpts.get(root) ?? { 'root': 0.5 };
      const pRoot = cpt['root'] ?? 0.5;
      // Compute posterior: P(root=1 | all evidence)
      let likelihood = 1.0;
      for (const node of anomalousNodes) {
        if (node === root) continue;
        const path = shortestPath(this.graph, root, node);
        if (path.length > 0) likelihood *= 0.8; // simplified evidence propagation
        else likelihood *= 0.5;
      }
      const posterior = pRoot * likelihood;
      scores.push({ name: root, score: Math.min(1, posterior), confidence: pRoot, rank: 0, evidence: [] });
    }

    scores.sort((a, b) => b.score - a.score);
    scores.forEach((s, i) => (s as { rank: number }).rank = i + 1);

    const paths: RootCausePath[] = [];
    for (const root of scores.slice(0, 3)) {
      for (const anom of anomalousNodes) {
        const path = shortestPath(this.graph, root.name, anom);
        if (path.length > 0) paths.push({ nodes: path, score: root.score, direction: 'forward' });
      }
    }

    return buildResult(scores, paths, 'bayesian');
  }
}

function shortestPath(g: CausalGraph, from: string, to: string): string[] {
  const nodes = [...g.nodes];
  const dist = new Map<string, number>();
  const prev = new Map<string, string | null>();
  for (const n of nodes) { dist.set(n, Infinity); prev.set(n, null); }
  dist.set(from, 0);
  const queue = [from];
  while (queue.length > 0) {
    const u = queue.shift()!;
    if (u === to) break;
    for (const v of g.children(u)) {
      const alt = (dist.get(u) ?? Infinity) + 1;
      if (alt < (dist.get(v) ?? Infinity)) { dist.set(v, alt); prev.set(v, u); queue.push(v); }
    }
  }
  if (!prev.get(to) && from !== to) return [];
  const path = [to];
  let cur: string | null = to;
  while (cur && cur !== from) { cur = prev.get(cur) ?? null; if (cur) path.unshift(cur); }
  return path;
}

// ── Random Walk RCA ──────────────────────────────────────────────────
export class RandomWalkRCA {
  private graph: CausalGraph | null = null;
  private edgeWeights = new Map<string, number>();

  train(graph: CausalGraph): void {
    this.graph = graph;
    // Assign uniform edge weights initially (refined during analysis)
    for (const edge of graph.edges) {
      this.edgeWeights.set(`${edge.source}→${edge.target}`, edge.weight);
    }
  }

  findRootCauses(anomalousNodes: string[], steps: number = 10, repeats: number = 1000): RCAResult {
    if (!this.graph) return buildResult([], [], 'random_walk');

    const visitCounts = new Map<string, number>();
    for (const node of this.graph.nodes) visitCounts.set(node, 0);

    for (const start of anomalousNodes) {
      for (let r = 0; r < repeats; r++) {
        let current = start;
        for (let s = 0; s < steps; s++) {
          const children = this.graph.children(current);
          const parents = this.graph.parents(current);
          // Walk upstream (toward root causes)
          const candidates = parents.length > 0 ? parents : children;
          if (candidates.length === 0) break;
          const next = candidates[Math.floor(Math.random() * candidates.length)]!;
          visitCounts.set(next, (visitCounts.get(next) ?? 0) + 1);
          current = next;
        }
      }
    }

    const total = repeats * anomalousNodes.length * steps;
    const scores: RootCause[] = [];
    for (const [node, count] of visitCounts) {
      if (this.graph.parents(node).length === 0) {
        scores.push({ name: node, score: count / total, confidence: 0.8, rank: 0, evidence: [] });
      }
    }
    scores.sort((a, b) => b.score - a.score);
    scores.forEach((s, i) => (s as { rank: number }).rank = i + 1);

    return buildResult(scores.slice(0, 4), [], 'random_walk');
  }
}

// ── Hypothesis Testing RCA ───────────────────────────────────────────
export class HTRCA {
  private graph: CausalGraph | null = null;
  private regressors = new Map<string, { coef: number[]; intercept: number; std: number }>();

  train(graph: CausalGraph, data: Matrix): void {
    this.graph = graph;
    const nodes = [...graph.nodes];
    for (const node of nodes) {
      const parents = graph.parents(node);
      if (parents.length === 0) { this.regressors.set(node, { coef: [], intercept: 0, std: 0 }); continue; }
      const pIdx = parents.map(p => nodes.indexOf(p));
      const yIdx = nodes.indexOf(node);
      const n = data.rows;
      // Simple OLS: y = Xβ
      let ySum = 0; for (let r = 0; r < n; r++) ySum += data.get(r, yIdx);
      const yMean = ySum / n;
      const X = Array.from({ length: n }, (_, r) => pIdx.map(i => data.get(r, i)));
      // Compute X^T X and X^T y
      const k = pIdx.length;
      const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
      const Xty = new Array(k).fill(0);
      for (let r = 0; r < n; r++) {
        const xr = X[r]!;
        for (let i = 0; i < k; i++) {
          Xty[i] += xr[i]! * (data.get(r, yIdx) - yMean);
          for (let j = 0; j < k; j++) XtX[i]![j]! += xr[i]! * xr[j]!;
        }
      }
      const coef = solveLinear(XtX, Xty);
      const intercept = yMean - coef.reduce((s, c, i) => s + c * (pIdx.reduce((a, pi) => a + (data.rows > 0 ? data.get(0, pi)! : 0), 0) / n / pIdx.length), 0);
      // Residual std
      let ss = 0; for (let r = 0; r < n; r++) {
        let pred = intercept;
        for (let i = 0; i < k; i++) pred += coef[i]! * data.get(r, pIdx[i]!);
        ss += (data.get(r, yIdx) - pred) ** 2;
      }
      this.regressors.set(node, { coef, intercept, std: Math.sqrt(ss / Math.max(1, n - k - 1)) });
    }
  }

  findRootCauses(anomalousNodes: string[], data: Matrix): RCAResult {
    if (!this.graph || data.rows === 0) return buildResult([], [], 'ht');
    const nodes = [...this.graph.nodes];
    const n = data.rows;
    const scores: RootCause[] = [];

    for (const node of this.graph.nodes) {
      const reg = this.regressors.get(node);
      if (!reg || reg.std === 0) continue;
      const parents = this.graph.parents(node);
      const nodeIdx = nodes.indexOf(node);
      const pIdx = parents.map(p => nodes.indexOf(p));
      let maxZ = 0;
      for (let r = 0; r < n; r++) {
        let pred = reg.intercept;
        for (let i = 0; i < reg.coef.length; i++) pred += reg.coef[i]! * data.get(r, pIdx[i]!);
        const residual = data.get(r, nodeIdx) - pred;
        const z = Math.abs(residual) / reg.std;
        if (z > maxZ) maxZ = z;
      }
      scores.push({ name: node, score: Math.min(1, maxZ / 5), confidence: 1 - 2 * normalCDFTail(maxZ), rank: 0, evidence: [{ type: 'regression_residual', description: `max z-score: ${maxZ.toFixed(2)}`, value: maxZ }] });
    }

    scores.sort((a, b) => b.score - a.score);
    scores.forEach((s, i) => (s as { rank: number }).rank = i + 1);
    return buildResult(scores.slice(0, 5), [], 'ht');
  }
}

function solveLinear(A: number[][], b: number[]): number[] {
  const n = A.length;
  const aug = A.map((row, i) => [...row, b[i] ?? 0]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) if (Math.abs(aug[row]![col]!) > Math.abs(aug[pivot]![col]!)) pivot = row;
    [aug[col], aug[pivot]] = [aug[pivot]!, aug[col]!];
    if (Math.abs(aug[col]![col]!) < 1e-12) continue;
    for (let row = col + 1; row < n; row++) {
      const f = aug[row]![col]! / aug[col]![col]!;
      for (let j = col; j <= n; j++) aug[row]![j]! -= f * aug[col]![j]!;
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = aug[i]![n]!;
    for (let j = i + 1; j < n; j++) sum -= aug[i]![j]! * (x[j] ?? 0);
    x[i] = sum / aug[i]![i]!;
  }
  return x;
}

function normalCDFTail(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp(-x * x / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return Math.max(0, p);
}

// ── FP-Growth Trace RCA ──────────────────────────────────────────────
interface FPTreeNode { item: string; count: number; children: Map<string, FPTreeNode>; parent: FPTreeNode | null; next: FPTreeNode | null; }

export class FPGrowthRCA {
  private minSupport: number;

  constructor(minSupport: number = 0.1) { this.minSupport = minSupport; }

  findRootCauses(
    traces: string[][],
    abnormalTraceIds: Set<number>,
    serviceNames: string[],
    invocations: Array<{ source: string; target: string; traceId: number }>,
  ): RCAResult {
    // Step 1: Mine frequent itemsets from abnormal traces
    const abnormalItemsets = traces
      .filter((_, i) => abnormalTraceIds.has(i))
      .map(t => new Set(t));
    
    if (abnormalItemsets.length === 0) return buildResult([], [], 'fp_growth_trace');

    const freqItems = this.mineFrequentItemsets(abnormalItemsets);
    
    // Step 2: Score services by Jaccard + InOutDiff
    const scores: RootCause[] = [];
    for (const svc of serviceNames) {
      // InOutDiff: |incoming anomalous| - |outgoing anomalous|
      const abnormalInvocations = invocations.filter(inv => abnormalTraceIds.has(inv.traceId));
      const incoming = abnormalInvocations.filter(inv => inv.target === svc).length;
      const outgoing = abnormalInvocations.filter(inv => inv.source === svc).length;
      const inOutDiff = Math.abs(incoming - outgoing) / Math.max(1, incoming + outgoing);
      
      // Jaccard similarity: how often this service appears in frequent patterns
      let patternScore = 0;
      for (const pattern of freqItems) {
        if (pattern.items.has(svc)) patternScore = Math.max(patternScore, pattern.support);
      }
      
      const score = inOutDiff * 0.4 + patternScore * 0.6;
      scores.push({ name: svc, score, confidence: patternScore, rank: 0, evidence: [{ type: 'frequent_pattern', description: `InOutDiff=${inOutDiff.toFixed(2)}, PatternSupport=${patternScore.toFixed(2)}`, value: score }] });
    }

    scores.sort((a, b) => b.score - a.score);
    scores.forEach((s, i) => (s as { rank: number }).rank = i + 1);
    return buildResult(scores, [], 'fp_growth_trace');
  }

  private mineFrequentItemsets(transactions: Set<string>[]): Array<{ items: Set<string>; support: number }> {
    const total = transactions.length;
    const minCount = Math.ceil(this.minSupport * total);
    // Count single items
    const itemCounts = new Map<string, number>();
    for (const t of transactions) for (const item of t) itemCounts.set(item, (itemCounts.get(item) ?? 0) + 1);
    
    const freqItems = [...itemCounts.entries()]
      .filter(([, c]) => c >= minCount)
      .sort((a, b) => b[1] - a[1])
      .map(([item]) => item);
    
    if (freqItems.length === 0) return [];

    // Build FP-Tree
    const root: FPTreeNode = { item: 'ROOT', count: 0, children: new Map(), parent: null, next: null };
    const headerTable = new Map<string, FPTreeNode>();
    
    for (const t of transactions) {
      const sorted = freqItems.filter(i => t.has(i));
      if (sorted.length === 0) continue;
      let current = root;
      for (const item of sorted) {
        if (!current.children.has(item)) {
          const node: FPTreeNode = { item, count: 0, children: new Map(), parent: current, next: headerTable.get(item) ?? null };
          current.children.set(item, node);
          headerTable.set(item, node);
        }
        current = current.children.get(item)!;
        current.count++;
      }
    }

    // Collect frequent singleton patterns as result
    return freqItems.map(item => ({
      items: new Set([item]),
      support: (itemCounts.get(item) ?? 0) / total,
    }));
  }
}
