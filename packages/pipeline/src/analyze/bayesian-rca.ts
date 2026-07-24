/**
 * BayesianRCA — rigorous causal root cause analysis via Bayesian inference.
 *
 * REPLACES the heuristic HeuristicPathRCA with proper probabilistic
 * inference over a learned Bayesian network (CPTs estimated from data).
 *
 * Inference engines (selectable):
 * - variable_elimination (exact, single-query, default)
 * - junction_tree (exact, all-marginals via Hugin)
 * - loopy_bp (approximate, iterative message passing)
 * - likelihood_weighting (approximate, importance sampling)
 * - gibbs_sampling (approximate, MCMC)
 *
 * Design:
 * 1. Learn CPTs from data using Laplace-smoothed MLE
 * 2. Build factor graph from CPTs
 * 3. Compute P(root=anomalous | anomalous_nodes) via inference engine
 * 4. Rank roots by posterior probability
 *
 * Reference: Koller & Friedman (2009). PGM. MIT Press, Chapters 10-12.
 */
import type { RootCause, RootCausePath, RCAResult, AnalysisMetadata } from '@agentix-e/causality-analyzer-core';
import { CausalGraph } from '../graph/causal-graph.js';
import {
  estimateCPTs, cptToFactor, variableElimination, junctionTreeInference,
  loopyBeliefPropagation, likelihoodWeighting, gibbsSampling, bruteForceOracle,
} from '../infer/bayesian-network.js';
import type { CPT, Evidence, Factor, JunctionTreeResult } from '../infer/bayesian-network.js';

export type BayesianRCAEngine =
  | 'variable_elimination'
  | 'junction_tree'
  | 'loopy_bp'
  | 'likelihood_weighting'
  | 'gibbs_sampling'
  | 'brute_force';

export interface BayesianRCAOptions {
  engine?: BayesianRCAEngine;
  alpha?: number;          // Laplace smoothing (default 1)
  threshold?: number;      // anomaly discretization threshold (auto if omitted)
  nSamples?: number;       // for sampling engines (default 10000)
  seed?: number;           // for reproducibility
}

// ── Result Builder ────────────────────────────────────────────────────
function buildResult(rootCauses: RootCause[], paths: RootCausePath[], method: string): RCAResult {
  return {
    rootCauses, paths,
    metadata: { method, analyzedAt: Date.now(), durationMs: 0, extra: {} },
    toJSON() { return { rootCauses: this.rootCauses, paths: this.paths, metadata: this.metadata }; },
  };
}

// ── BayesianRCA ───────────────────────────────────────────────────────
export class BayesianRCA {
  private graph: CausalGraph | null = null;
  private cpts = new Map<string, CPT>();
  private parents = new Map<string, string[]>();
  private nodeNames: string[] = [];
  private nodeIndex = new Map<string, number>();
  private engine: BayesianRCAEngine;
  private alpha: number;
  private nSamples: number;
  private seed?: number;

  constructor(options: BayesianRCAOptions = {}) {
    this.engine = options.engine ?? 'variable_elimination';
    this.alpha = options.alpha ?? 1;
    this.nSamples = options.nSamples ?? 10000;
    this.seed = options.seed;
  }

  /**
   * Train: learn CPTs from data using maximum likelihood with Laplace smoothing.
   *
   * @param graph — causal DAG (correct topology is assumed — topology validation
   *               belongs to the causal discovery phase, not RCA)
   * @param data — observational data matrix (N×K, columns in graph.nodes order)
   */
  train(graph: CausalGraph, data: number[][]): void {
    this.graph = graph;
    this.nodeNames = [...graph.nodes];
    this.nodeIndex = new Map(this.nodeNames.map((n, i) => [n, i]));
    this.parents = new Map();
    for (const node of this.nodeNames) {
      this.parents.set(node, graph.parents(node));
    }

    this.cpts = estimateCPTs(
      data,
      this.nodeNames,
      { parents: (node: string) => this.parents.get(node) ?? [] },
      this.nodeIndex,
      { alpha: this.alpha },
    );
  }

  /**
   * Find root causes: infer P(root=anomalous | anomal_nodes) for each root.
   *
   * @param anomalousNodes — list of nodes observed to be anomalous
   * @returns ranked root causes with posterior probabilities
   */
  findRootCauses(anomalousNodes: string[]): RCAResult {
    if (!this.graph || this.cpts.size === 0) return buildResult([], [], 'bayesian');

    const evidence: Evidence = {};
    for (const node of anomalousNodes) {
      evidence[node] = 1; // anomalous
    }

    // Identify root nodes (no parents)
    const rootNodes = this.nodeNames.filter(n => this.parents.get(n)!.length === 0);
    if (rootNodes.length === 0) return buildResult([], [], 'bayesian');

    // Build factors from CPTs
    let factors: Factor[] = [];
    const domainSizes = new Map<string, number>();
    for (const node of this.nodeNames) domainSizes.set(node, 2);

    for (const node of this.nodeNames) {
      const cpt = this.cpts.get(node);
      if (!cpt) continue;
      factors.push(cptToFactor(node, this.parents.get(node) ?? [], cpt, domainSizes));
    }

    // Compute posterior for each root
    const scores: RootCause[] = [];
    for (const root of rootNodes) {
      const posterior = this._inferPosterior(factors, root, evidence);
      const pAnomalous = posterior.get(1) ?? 0;
      scores.push({
        name: root,
        score: Math.min(1, pAnomalous),
        confidence: pAnomalous,
        rank: 0,
        evidence: [{
          type: 'causal_effect',
          description: `P(${root}=anomalous | evidence) = ${(pAnomalous * 100).toFixed(1)}%`,
          value: pAnomalous,
        }],
      });
    }

    scores.sort((a, b) => b.score - a.score);
    scores.forEach((s, i) => (s as { rank: number }).rank = i + 1);

    // Build propagation paths
    const paths: RootCausePath[] = [];
    for (const root of scores.slice(0, 3)) {
      for (const anom of anomalousNodes) {
        const sp = shortestPath(this.graph, root.name, anom);
        if (sp.length > 0) {
          paths.push({ nodes: sp, score: root.confidence, direction: 'forward' });
        }
      }
    }

    return buildResult(scores, paths, `bayesian_${this.engine}`);
  }

  /** Dispatch to the selected inference engine */
  private _inferPosterior(
    factors: Factor[],
    query: string,
    evidence: Evidence,
  ): Map<number, number> {
    switch (this.engine) {
      case 'variable_elimination':
        return variableElimination(factors, query, evidence);

      case 'junction_tree': {
        const jt = junctionTreeInference(factors, evidence);
        return jt.posteriors.get(query) ?? new Map([[0, 0.5], [1, 0.5]]);
      }

      case 'loopy_bp': {
        const lbp = loopyBeliefPropagation(factors, evidence, { seed: this.seed });
        return lbp.posteriors.get(query) ?? new Map([[0, 0.5], [1, 0.5]]);
      }

      case 'likelihood_weighting':
        return likelihoodWeighting(
          this.cpts, this.nodeNames, this.parents,
          query, evidence, this.nSamples, this.seed,
        ).posterior;

      case 'gibbs_sampling':
        return gibbsSampling(
          this.cpts, this.nodeNames, this.parents,
          query, evidence, { iterations: this.nSamples, burnIn: 1000, seed: this.seed },
        ).posterior;

      case 'brute_force':
        return bruteForceOracle(this.cpts, this.nodeNames, this.parents, query, evidence);

      default:
        return variableElimination(factors, query, evidence);
    }
  }
}

// ── Shared shortest-path utility ───────────────────────────────────────
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
      if ((dist.get(v) ?? Infinity) > (dist.get(u) ?? Infinity) + 1) {
        dist.set(v, (dist.get(u) ?? 0) + 1);
        prev.set(v, u);
        queue.push(v);
      }
    }
  }
  if ((prev.get(to) ?? null) === null && from !== to) return [];
  const path = [to];
  let cur: string | null | undefined = prev.get(to);
  while (cur && cur !== from) { path.unshift(cur); cur = prev.get(cur); }
  if (cur === from) path.unshift(from);
  return path;
}
