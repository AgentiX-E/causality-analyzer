/**
 * GRaSP — Greedy Relaxations of Sparsest Permutation.
 *
 * Based on Lam, Andrews & Ramsey (AIStats 2022):
 * "GRaSP: Greedy Relaxations of the Sparsest Permutation for Causal Discovery"
 *
 * GRaSP uses permutation-based search to find the sparsest Markov
 * equivalence class. Unlike PC (constraint-based) or GES (score-based),
 * GRaSP directly optimizes sparsity via permutation scoring, making it
 * effective on high-dimensional data where other methods struggle.
 *
 * Algorithm:
 * 1. Start with a random topological ordering (permutation)
 * 2. Forward phase: greedily add edges within the permutation constraint
 *    that improve BIC score
 * 3. Backward phase: remove edges that degrade BIC
 * 4. Perturb the permutation and repeat
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { CausalGraph } from './causal-graph.js';

export interface GRaSPConfig {
  /** Number of restarts with different initial permutations */
  numStarts?: number;
  /** BIC penalty weight (default 1.0 for standard BIC) */
  bicLambda?: number;
  /** Maximum parent set size (-1 = unlimited) */
  maxParents?: number;
  /** Whether to use stable ordering */
  stable?: boolean;
}

/**
 * BIC score for a DAG given data.
 * BIC = -2 * log-likelihood + λ * k * ln(n) where k = number of parameters.
 * Lower BIC is better.
 */
function computeBICScore(
  data: Matrix,
  graph: CausalGraph,
  nodeNames: string[],
  lambda: number,
): number {
  const n = data.rows;
  if (n === 0) return 0;
  const nodeIndex = new Map(nodeNames.map((name, i) => [name, i]));

  let bic = 0;
  for (const node of nodeNames) {
    const parents = graph.parents(node);
    const yIdx = nodeIndex.get(node)!;
    const pIdxs = parents.map(p => nodeIndex.get(p)!);
    const k = pIdxs.length;

    if (k === 0) {
      // Root node: BIC = n * ln(variance) + ln(n)
      let sum = 0, sumSq = 0;
      for (let r = 0; r < n; r++) {
        const y = data.get(r, yIdx);
        sum += y;
        sumSq += y * y;
      }
      const variance = Math.max(1e-10, (sumSq - sum * sum / n) / n);
      bic += n * Math.log(variance) + lambda * Math.log(n);
    } else {
      // OLS regression: y = Xβ
      const XtX = Array.from({ length: k }, () => new Float64Array(k));
      const Xty = new Float64Array(k);
      let ySum = 0;

      for (let r = 0; r < n; r++) {
        const y = data.get(r, yIdx);
        ySum += y;
        for (let i = 0; i < k; i++) {
          const xi = data.get(r, pIdxs[i]!);
          Xty[i] += xi * y;
          for (let j = 0; j < k; j++) {
            XtX[i]![j] += xi * data.get(r, pIdxs[j]!);
          }
        }
      }

      // Solve OLS via Cholesky
      const coef = solveCovariant(XtX, Xty, k);
      const yMean = ySum / n;
      const intercept = yMean - pIdxs.reduce((s: number, _: number, i: number) => {
        let xSum = 0;
        for (let r = 0; r < n; r++) xSum += data.get(r, pIdxs[i]!);
        return s + coef[i]! * xSum / n;
      }, 0);

      let ss = 0;
      for (let r = 0; r < n; r++) {
        let pred = intercept;
        for (let i = 0; i < k; i++) pred += coef[i]! * data.get(r, pIdxs[i]!);
        ss += (data.get(r, yIdx) - pred) ** 2;
      }
      const variance = Math.max(1e-10, ss / n);
      bic += n * Math.log(variance) + lambda * (k + 1) * Math.log(n);
    }
  }
  return bic;
}

/** Solve Ax = b using Cholesky decomposition */
function solveCovariant(A: Float64Array[], b: Float64Array, k: number): number[] {
  // Build L (lower triangular) from A
  const L = new Float64Array(k * k);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i]![j]!;
      for (let p = 0; p < j; p++) sum -= L[i * k + p]! * L[j * k + p]!;
      if (i === j) {
        L[i * k + i] = Math.sqrt(Math.max(1e-12, sum));
      } else {
        L[i * k + j] = sum / L[j * k + j]!;
      }
    }
  }

  // Forward substitution: L y = b
  const y = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    let sum = b[i]!;
    for (let j = 0; j < i; j++) sum -= L[i * k + j]! * y[j]!;
    y[i] = sum / L[i * k + i]!;
  }

  // Back substitution: Lᵀ x = y
  const x = new Float64Array(k);
  for (let i = k - 1; i >= 0; i--) {
    let sum = y[i]!;
    for (let j = i + 1; j < k; j++) sum -= L[j * k + i]! * x[j]!;
    x[i] = sum / L[i * k + i]!;
  }

  return Array.from(x);
}

/**
 * GRaSP causal discovery algorithm.
 *
 * @param data — observation matrix (rows × columns)
 * @param nodeNames — variable names
 * @param config — algorithm configuration
 * @param domainKnowledge — optional domain constraints
 */
export function graspAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: Partial<GRaSPConfig> = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph } {
  const numStarts = config.numStarts ?? 3;
  const bicLambda = config.bicLambda ?? 2.0;
  const maxParents = config.maxParents ?? -1;
  const n = nodeNames.length;

  if (data.rows === 0 || n === 0) return { graph: new CausalGraph(nodeNames) };

  const maxP = maxParents === -1 ? n : Math.min(maxParents, n);

  let bestGraph: CausalGraph | null = null;
  let bestBIC = Infinity;

  // Phase 1: Permutation-based search
  for (let start = 0; start < numStarts; start++) {
    // Create a random permutation
    const perm = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [perm[i], perm[j]] = [perm[j]!, perm[i]!];
    }

    // Build an initial DAG respecting the permutation
    // (edges only from earlier to later nodes in permutation)
    const g = new CausalGraph(nodeNames);

    // Forward phase: greedily add edges in topological order
    for (let i = 1; i < n; i++) {
      const nodeI = nodeNames[perm[i]!]!;
      // Consider adding parents from earlier-position nodes
      const candidates: Array<{ parent: number; gain: number }> = [];

      for (let j = 0; j < i; j++) {
        const nodeJ = nodeNames[perm[j]!]!;
        // Check if adding this edge improves BIC
        const currentParents = g.parents(nodeI);
        if (currentParents.length >= maxP) break;

        g.addEdge(nodeJ, nodeI);
        const newBIC = computeBICScore(data, g, nodeNames, bicLambda);
        g.removeEdge(nodeJ, nodeI);

        const baseBIC = bestGraph
          ? computeBICScore(data, bestGraph, nodeNames, bicLambda)
          : computeBICScore(data, g, nodeNames, bicLambda);

        const gain = baseBIC - newBIC;
        if (gain > 0) candidates.push({ parent: j, gain });
      }

      // Add best parents
      candidates.sort((a, b) => b.gain - a.gain);
      for (const { parent } of candidates.slice(0, maxP)) {
        g.addEdge(nodeNames[perm[parent]!]!, nodeI);
      }
    }

    // Backward phase: remove edges that increase BIC
    let improved = true;
    while (improved) {
      improved = false;
      for (const node of nodeNames) {
        const parents = g.parents(node);
        for (const p of parents) {
          const currentBIC = computeBICScore(data, g, nodeNames, bicLambda);
          g.removeEdge(p, node);
          const newBIC = computeBICScore(data, g, nodeNames, bicLambda);
          if (newBIC >= currentBIC) {
            g.addEdge(p, node); // restore if not better
          } else {
            improved = true;
          }
        }
      }
    }

    const bic = computeBICScore(data, g, nodeNames, bicLambda);
    if (bic < bestBIC) {
      bestBIC = bic;
      bestGraph = g;
    }
  }

  const result = bestGraph ?? new CausalGraph(nodeNames);
  if (domainKnowledge) result.applyDomainKnowledge(domainKnowledge);

  return { graph: result };
}
