import { CONSTANTS } from "../constants.js";
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
import { solveLinear, normalCDFTail } from '@agentix-e/causality-analyzer-core';
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

// ── Heuristic Path-based RCA ──────────────────────────────────────────

/**
 * Heuristic Path-based RCA.
 *
 * **IMPORTANT**: This is NOT a proper Bayesian network inference engine.
 * It uses a heuristic scoring formula:
 *   score = P(root) × 0.8^connected × 0.5^not_connected
 * where "connected" means a path exists from root to anomalous node.
 *
 * For rigorous Bayesian inference with variable elimination or belief
 * propagation, use a dedicated probabilistic graphical model library
 * or the StructuralCausalModel for counterfactual reasoning.
 */
interface CPT { [parentState: string]: number; }

export class HeuristicPathRCA {
  private graph: CausalGraph | null = null;
  private cpts = new Map<string, CPT>();

  /**
   * Train the heuristic path model.
   *
   * @param graph — causal DAG
   * @param anomalies — set of known anomalous node names (used to boost prior)
   * @param data — observational data matrix
   */
  train(graph: CausalGraph, anomalies: Set<string>, data: Matrix): void {
    this.graph = graph;
    const nodes = [...graph.nodes];
    for (const node of nodes) {
      const parents = graph.parents(node);
      const nodeIdx = nodes.indexOf(node);
      const n = data.rows;
      // Estimate per-row anomaly threshold from data
      let colSum = 0, colSq = 0;
      for (let r = 0; r < n; r++) { const v = data.get(r, nodeIdx); colSum += v; colSq += v * v; }
      const colMean = colSum / n;
      const colStd = Math.sqrt(Math.max(1e-10, colSq / n - colMean * colMean));
      const threshold = colMean + CONSTANTS.ANOMALY_THRESHOLD_SIGMA * colStd; // >2.5σ = anomalous
      const isAnomalous = (r: number) => data.get(r, nodeIdx) > threshold;

      // Estimate CPT: P(node=anomalous | parent configuration)
      const cpt: CPT = {};
      if (parents.length === 0) {
        let anomCount = 0;
        for (let r = 0; r < n; r++) if (isAnomalous(r)) anomCount++;
        // Boost prior for nodes listed in the anomalies set
        const basePrior = Math.max(0.01, Math.min(0.99, anomCount / n));
        cpt['root'] = anomalies.has(node) ? Math.min(0.99, basePrior * CONSTANTS.ANOMALY_PRIOR_BOOST) : basePrior;
      } else {
        const counts: Record<string, { anom: number; total: number }> = {};
        const pIndices = parents.map(p => nodes.indexOf(p));
        // Determine anomaly threshold for each parent
        const parentThresholds = pIndices.map(pi => {
          let s2 = 0, c2 = 0;
          for (let r = 0; r < n; r++) { const v = data.get(r, pi); s2 += v; c2 += v * v; }
          const m2 = s2 / n; const sd2 = Math.sqrt(Math.max(1e-10, c2 / n - m2 * m2)); return m2 + CONSTANTS.ANOMALY_THRESHOLD_SIGMA * sd2;
        });
        for (let r = 0; r < n; r++) {
          const key = parents.map((_, i) => data.get(r, pIndices[i]!) > parentThresholds[i]! ? '1' : '0').join('');
          if (!counts[key]) counts[key] = { anom: 0, total: 0 };
          counts[key]!.total++;
          if (isAnomalous(r)) counts[key]!.anom++;
        }
        for (const [key, cnt] of Object.entries(counts)) {
          cpt[key] = Math.max(0.01, Math.min(0.99, cnt.total > 0 ? cnt.anom / cnt.total : 0.5));
        }
      }
      this.cpts.set(node, cpt);
    }
  }

  findRootCauses(anomalousNodes: string[]): RCAResult {
    if (!this.graph) return buildResult([], [], 'heuristic_path');
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
        if (path.length > 0) likelihood *= CONSTANTS.PATH_LIKELIHOOD_CONNECTED; // simplified evidence propagation
        else likelihood *= CONSTANTS.PATH_LIKELIHOOD_DISCONNECTED;
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

    return buildResult(scores, paths, 'heuristic_path');
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
  private seed: number | null = null;

  constructor(seed?: number) {
    this.seed = seed ?? null;
  }

  /** Simple LCG for reproducible random numbers. */
  private nextRand(): number {
    if (this.seed == null) return Math.random();
    this.seed = (this.seed * 1664525 + 1013904223) % 0x100000000;
    return (this.seed >>> 0) / 0x100000000;
  }

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
          const next = candidates[Math.floor(this.nextRand() * candidates.length)]!;
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
      // Intercept = yMean - Σ(βi * xiMean) where xiMean = mean of parent column i
      const parentMeans = pIdx.map(pi => {
        let sum = 0; for (let r = 0; r < n; r++) sum += data.get(r, pi);
        return sum / n;
      });
      const intercept = yMean - coef.reduce((s, c, i) => s + c * (parentMeans[i] ?? 0), 0);
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

    // Recursive FP-Growth mining
    const result: Array<{ items: Set<string>; support: number }> = [];
    // Add singleton frequent items
    for (const item of freqItems) {
      result.push({ items: new Set([item]), support: (itemCounts.get(item) ?? 0) / total });
    }

    // Mine multi-item patterns (process items in reverse frequency order)
    const orderedItems = [...freqItems].reverse();
    for (const item of orderedItems) {
      // Build conditional pattern base for this item
      const condPatterns: Array<{ items: Set<string>; count: number }> = [];
      let node: FPTreeNode | null = headerTable.get(item) ?? null;
      while (node) {
        // Walk up to root collecting prefix path
        const path: string[] = [];
        let p = node.parent;
        while (p && p.item !== 'ROOT') { path.push(p.item); p = p.parent; }
        if (path.length > 0) {
          condPatterns.push({ items: new Set(path), count: node.count });
        }
        node = node.next;
      }
      // Recursively mine conditional FP-Tree
      this.mineFPTree(
        condPatterns,
        new Set([item]),
        (itemCounts.get(item) ?? 0) / total,
        minCount,
        total,
        result,
      );
    }

    return result;
  }

  /** Recursive FP-Growth: mine conditional FP-Tree for a given prefix. */
  private mineFPTree(
    patterns: Array<{ items: Set<string>; count: number }>,
    prefix: Set<string>,
    prefixSupport: number,
    minCount: number,
    total: number,
    result: Array<{ items: Set<string>; support: number }>,
  ): void {
    // Count items in conditional pattern base
    const counts = new Map<string, number>();
    for (const p of patterns) {
      for (const item of p.items) counts.set(item, (counts.get(item) ?? 0) + p.count);
    }
    const freqItems = [...counts.entries()]
      .filter(([, c]) => c >= minCount)
      .sort((a, b) => b[1] - a[1])
      .map(([item]) => item);

    if (freqItems.length === 0) return;

    // Process in reverse order for bottom-up recursive mining
    for (const item of [...freqItems].reverse()) {
      const newPrefix = new Set(prefix);
      newPrefix.add(item);
      result.push({ items: newPrefix, support: (counts.get(item) ?? 0) / total });

      // Build conditional pattern base for the extended prefix
      const nextPatterns: Array<{ items: Set<string>; count: number }> = [];
      for (const p of patterns) {
        if (p.items.has(item)) {
          const filtered = new Set(p.items);
          filtered.delete(item);
          if (filtered.size > 0) nextPatterns.push({ items: filtered, count: p.count });
        }
      }
      if (nextPatterns.length > 0) {
        this.mineFPTree(nextPatterns, newPrefix, (counts.get(item) ?? 0) / total, minCount, total, result);
      } else if (freqItems.length === 1) {
        // Single path: all combinations of freqItems
        for (let mask = 1; mask < (1 << freqItems.length); mask++) {
          const combo = new Set(prefix);
          for (let b = 0; b < freqItems.length; b++) {
            if (mask & (1 << b)) combo.add(freqItems[b]!);
          }
          if (combo.size > prefix.size) {
            // Find min support along path
            const minSupport = Math.min(...freqItems.map(f => counts.get(f) ?? 0));
            result.push({ items: combo, support: minSupport / total });
          }
        }
      }
    }
  }
}
