/**
 * Stability Selection & StARS for causal discovery.
 *
 * Stability Selection (Meinshausen & Bühlmann 2010):
 *   Run the discovery algorithm on multiple bootstrap-resampled datasets.
 *   Edges that appear in ≥ threshold% of subsamples are kept.
 *   Eliminates false-positive edges from noisy data.
 *
 * StARS (Stability Approach to Regularization Selection, Liu et al. 2010):
 *   Run stability selection across a range of regularization parameters.
 *   Select the parameter that maximizes total edge stability while
 *   keeping variability below a target threshold.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import { createRNG } from '@agentix-e/causality-analyzer-core';

// ── Types ──────────────────────────────────────────────────────────

export interface StabilityResult {
  /** Stable edges (appeared in ≥ threshold% of subsamples) */
  stableGraph: CausalGraph;
  /** Per-edge stability scores [0, 1] */
  edgeStability: Map<string, number>;
  /** Number of subsamples used */
  nSubsamples: number;
  /** Fraction of original data per subsample */
  subsampleFraction: number;
}

export interface StARSResult {
  /** Best regularization parameter value */
  bestParam: number;
  /** Stability graph at best param */
  bestGraph: CausalGraph;
  /** D-stability value at each param */
  stabilityValues: Array<{ param: number; stability: number; nEdges: number }>;
}

// ── Stability Selection ────────────────────────────────────────────

/**
 * Stability Selection for causal discovery.
 *
 * Bootstrap-resamples the data, runs the discovery algorithm on each
 * subsample, and keeps only edges that appear consistently.
 *
 * @param data — original data matrix
 * @param nodeNames — variable names
 * @param discoverFn — discovery algorithm (must accept data + nodeNames → CausalGraph)
 * @param options — configuration
 */
export function stabilitySelection(
  data: Matrix,
  nodeNames: string[],
  discoverFn: (data: Matrix, nodeNames: string[]) => CausalGraph,
  options: {
    nSubsamples?: number;
    subsampleFraction?: number;
    edgeThreshold?: number;
    seed?: number;
  } = {},
): StabilityResult {
  const nSubsamples = options.nSubsamples ?? 50;
  const fraction = options.subsampleFraction ?? 0.8;
  const threshold = options.edgeThreshold ?? 0.6;
  const rng = createRNG(options.seed ?? null);

  const N = data.rows;
  const subsampleSize = Math.max(2, Math.floor(N * fraction));

  // Accumulate edge counts across subsamples
  const edgeCounts = new Map<string, number>();

  for (let s = 0; s < nSubsamples; s++) {
    // Bootstrap sample (with replacement)
    const indices: number[] = [];
    for (let i = 0; i < subsampleSize; i++) {
      indices.push(Math.floor(rng() * N));
    }

    const subData = new Matrix(subsampleSize, data.columns);
    for (let i = 0; i < subsampleSize; i++) {
      for (let c = 0; c < data.columns; c++) {
        subData.set(i, c, data.get(indices[i]!, c));
      }
    }

    try {
      const graph = discoverFn(subData, nodeNames);
      for (const edge of graph.edges) {
        const key = `${edge.source}→${edge.target}`;
        edgeCounts.set(key, (edgeCounts.get(key) ?? 0) + 1);
      }
    } catch {
      // Skip failed subsamples (e.g., singular matrix)
    }
  }

  // Filter edges above threshold
  const stableGraph = new CausalGraph([...nodeNames]);
  const edgeStability = new Map<string, number>();

  for (const [key, count] of edgeCounts) {
    const stability = count / nSubsamples;
    edgeStability.set(key, stability);
    if (stability >= threshold) {
      const [source, target] = key.split('→') as [string, string];
      stableGraph.addEdge(source, target);
    }
  }

  return {
    stableGraph,
    edgeStability,
    nSubsamples,
    subsampleFraction: fraction,
  };
}

// ── StARS ───────────────────────────────────────────────────────────

/**
 * StARS: Stability Approach to Regularization Selection.
 *
 * Searches a range of regularization parameter values, running
 * stability selection at each one. Returns the parameter that
 * maximizes edge stability while keeping instability low.
 *
 * @param data — original data
 * @param nodeNames — variable names
 * @param paramDiscoverFn — function mapping (param) → discovery fn
 * @param paramRange — parameter values to search
 * @param instabilityThreshold — max allowed instability (default 0.05)
 */
export function starsSelection(
  data: Matrix,
  nodeNames: string[],
  paramDiscoverFn: (param: number) => (data: Matrix, nodeNames: string[]) => CausalGraph,
  paramRange: number[],
  options: {
    nSubsamples?: number;
    subsampleFraction?: number;
    instabilityThreshold?: number;
    seed?: number;
  } = {},
): StARSResult {
  const nSubsamples = options.nSubsamples ?? 20;
  const fraction = options.subsampleFraction ?? 0.8;
  const targetInstability = options.instabilityThreshold ?? 0.05;
  const seed = options.seed ?? 42;

  const stabilityValues: StARSResult['stabilityValues'] = [];

  for (const param of paramRange) {
    const discoverFn = paramDiscoverFn(param);
    const result = stabilitySelection(data, nodeNames, discoverFn, {
      nSubsamples,
      subsampleFraction: fraction,
      edgeThreshold: 0,
      seed: seed + param * 100,
    });

    // Compute total edge instability: average variance of edge probabilities
    const edgeProbs = [...result.edgeStability.values()];
    let instability = 0;
    if (edgeProbs.length > 0) {
      // Instability = 2 × avg( p × (1-p) ) — from StARS definition
      instability = 2 * edgeProbs.reduce((s, p) => s + p * (1 - p), 0) / edgeProbs.length;
    }

    stabilityValues.push({
      param,
      stability: 1 - instability, // higher = more stable
      nEdges: edgeProbs.length,
    });
  }

  // Select best param: highest stability with nEdges > 0 and instability ≤ target
  let bestParam = paramRange[paramRange.length - 1]!;
  let bestScore = -Infinity;

  for (const v of stabilityValues) {
    const instability = 1 - v.stability;
    if (v.nEdges > 0 && instability <= targetInstability && v.stability > bestScore) {
      bestScore = v.stability;
      bestParam = v.param;
    }
  }

  // Run final stability selection at best param
  const bestDiscoverFn = paramDiscoverFn(bestParam);
  const bestResult = stabilitySelection(data, nodeNames, bestDiscoverFn, {
    nSubsamples,
    subsampleFraction: fraction,
    edgeThreshold: 0.6,
    seed,
  });

  return {
    bestParam,
    bestGraph: bestResult.stableGraph,
    stabilityValues,
  };
}
