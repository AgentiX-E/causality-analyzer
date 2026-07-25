/**
 * BOSS — Best Order Score Search (NeurIPS 2023).
 *
 * State-of-the-art score-based causal discovery that searches the
 * space of variable permutations/orderings rather than the space of
 * graphs. This yields dramatically better scalability and precision
 * compared to GES and GRaSP.
 *
 * Core idea: a topological ordering of variables uniquely determines
 * a DAG under the BIC-score-optimal parent set for each variable
 * (restricted to predecessors in the ordering).
 *
 * Algorithm:
 *  1. Generate random permutation π
 *  2. For each variable v, select optimal parents Pa(v) ⊆ predecessors(π, v)
 *     using a Grow-Shrink Tree (GST) for cached BIC evaluation
 *  3. "Better mutation": for each variable, find the position in π
 *     that yields the best total BIC (swap positions)
 *  4. Repeat until convergence (no improvement)
 *  5. Multiple random restarts, keep best global BIC
 *  6. DAG → CPDAG (via pdag2dag)
 *
 * Reference: Lam et al. (NeurIPS 2023). "BOSS: Best Order Score Search."
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';
import { createRNG, solveLinear } from '@agentix-e/causality-analyzer-core';

// ── Config ──────────────────────────────────────────────────────────

export interface BOSSConfig {
  /** Number of random restarts (default: 10) */
  numStarts: number;
  /** Maximum iterations per restart (default: 50) */
  maxIter: number;
  /** Maximum number of parents per node (-1 = unlimited) */
  maxParents: number;
  /** Penalty discount for BIC (default: 1.0 = standard BIC) */
  penaltyDiscount: number;
  /** Random seed for reproducibility */
  seed?: number;
}

const DEFAULTS: BOSSConfig = {
  numStarts: 10,
  maxIter: 50,
  maxParents: -1,
  penaltyDiscount: 1.0,
};

// ── GST (Grow-Shrink Tree) ──────────────────────────────────────────

/**
 * Grow-Shrink Tree for efficient parent set scoring.
 *
 * Maintains a cache of BIC scores for candidate parent sets of a single
 * target variable. Uses grow-shrink heuristic to find the optimal set
 * without enumerating all 2^k subsets.
 */
class GrowShrinkTree {
  private cache = new Map<string, number>();
  private readonly dataColumns: number[];
  private readonly targetIdx: number;
  private readonly N: number;
  private readonly penaltyFactor: number;

  constructor(
    private data: Matrix,
    targetIdx: number,
    dataColumns: number[],
    penaltyDiscount: number,
  ) {
    this.targetIdx = targetIdx;
    this.N = data.rows;
    this.dataColumns = dataColumns;
    this.penaltyFactor = penaltyDiscount * Math.log(Math.max(2, this.N));
  }

  /** Score a parent set (with caching) */
  score(parents: number[]): number {
    const key = [...parents].sort((a, b) => a - b).join(',');
    if (this.cache.has(key)) return this.cache.get(key)!;

    const k = parents.length;
    // Extract target values
    const y: number[] = [];
    for (let r = 0; r < this.N; r++) y.push(this.data.get(r, this.targetIdx));

    if (k === 0) {
      const mean = y.reduce((s, v) => s + v, 0) / this.N;
      const rss = y.reduce((s, v) => s + (v - mean) ** 2, 0);
      const bic = -this.N * Math.log(Math.max(1e-10, rss / this.N)) - k * this.penaltyFactor;
      this.cache.set(key, bic);
      return bic;
    }

    // OLS regression: y ~ parents
    const X = Array.from({ length: this.N }, () => [1]);
    for (let r = 0; r < this.N; r++) {
      for (const p of parents) X[r].push(this.data.get(r, p));
    }

    const XtX = Array.from({ length: k + 1 }, () => new Float64Array(k + 1));
    const Xty = new Float64Array(k + 1);
    for (let r = 0; r < this.N; r++) {
      for (let i = 0; i <= k; i++) {
        Xty[i] += (X[r][i] ?? 0) * y[r];
        for (let j = 0; j <= k; j++) {
          XtX[i][j] += (X[r][i] ?? 0) * (X[r][j] ?? 0);
        }
      }
    }

    const A = XtX.map(row => Array.from(row));
    const b = Array.from(Xty);
    const beta = solveLinear(A, b);

    let rss = 0;
    for (let r = 0; r < this.N; r++) {
      let pred = 0;
      for (let i = 0; i <= k; i++) pred += (beta[i] ?? 0) * (X[r][i] ?? 0);
      rss += (y[r] - pred) ** 2;
    }

    const bic = -this.N * Math.log(Math.max(1e-10, rss / this.N)) - (k + 1) * this.penaltyFactor;
    this.cache.set(key, bic);
    return bic;
  }

  /**
   * Find optimal parent set from candidates using grow-shrink heuristic.
   *
   * Grow: greedily add candidates that improve BIC
   * Shrink: greedily remove candidates until BIC stops improving
   */
  findOptimal(candidates: number[], maxParents: number): number[] {
    let parents: number[] = [];
    let bestScore = this.score([]);

    // Grow phase
    const remaining = new Set(candidates);
    let changed = true;
    while (changed) {
      changed = false;
      if (maxParents >= 0 && parents.length >= maxParents) break;

      let bestCandidate = -1;
      let bestCandidateScore = bestScore;

      for (const c of remaining) {
        const candidateParents = [...parents, c];
        const s = this.score(candidateParents);
        if (s > bestCandidateScore) {
          bestCandidateScore = s;
          bestCandidate = c;
        }
      }

      if (bestCandidate >= 0 && bestCandidateScore > bestScore) {
        parents.push(bestCandidate);
        remaining.delete(bestCandidate);
        bestScore = bestCandidateScore;
        changed = true;
      }
    }

    // Shrink phase
    changed = true;
    while (changed) {
      changed = false;
      for (let i = 0; i < parents.length; i++) {
        const reduced = [...parents.slice(0, i), ...parents.slice(i + 1)];
        const s = this.score(reduced);
        if (s >= bestScore) {
          parents = reduced;
          bestScore = s;
          changed = true;
          break;
        }
      }
    }

    return parents;
  }

  /** Decay penalty factor (for BOSS inner loop) */
  scoreWithPenalty(parents: number[], penaltyMult: number): number {
    const baseKey = [...parents].sort((a, b) => a - b).join(',');
    const cacheKey = `${baseKey}|p${penaltyMult.toFixed(2)}`;
    if (this.cache.has(cacheKey)) return this.cache.get(cacheKey)!;

    const k = parents.length;
    const y: number[] = [];
    for (let r = 0; r < this.N; r++) y.push(this.data.get(r, this.targetIdx));

    if (k === 0) {
      const mean = y.reduce((s, v) => s + v, 0) / this.N;
      const rss = y.reduce((s, v) => s + (v - mean) ** 2, 0);
      const bic = -this.N * Math.log(Math.max(1e-10, rss / this.N));
      this.cache.set(cacheKey, bic);
      return bic;
    }

    const X = Array.from({ length: this.N }, () => [1]);
    for (let r = 0; r < this.N; r++) {
      for (const p of parents) X[r].push(this.data.get(r, p));
    }

    const kp = k + 1;
    const XtX = Array.from({ length: kp }, () => new Float64Array(kp));
    const Xty = new Float64Array(kp);
    for (let r = 0; r < this.N; r++) {
      for (let i = 0; i < kp; i++) {
        Xty[i] += (X[r][i] ?? 0) * y[r];
        for (let j = 0; j < kp; j++) {
          XtX[i][j] += (X[r][i] ?? 0) * (X[r][j] ?? 0);
        }
      }
    }

    const A = XtX.map(row => Array.from(row));
    const b = Array.from(Xty);
    const beta = solveLinear(A, b);

    let rss = 0;
    for (let r = 0; r < this.N; r++) {
      let pred = 0;
      for (let i = 0; i < kp; i++) pred += (beta[i] ?? 0) * (X[r][i] ?? 0);
      rss += (y[r] - pred) ** 2;
    }

    const bic = -this.N * Math.log(Math.max(1e-10, rss / this.N)) - penaltyMult * this.penaltyFactor * kp;
    this.cache.set(cacheKey, bic);
    return bic;
  }
}

// ── BOSS Core ───────────────────────────────────────────────────────

export function bossAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: Partial<BOSSConfig> = {},
  domainKnowledge?: DomainKnowledge,
): CausalGraph {
  const cfg = { ...DEFAULTS, ...config };
  const d = nodeNames.length;
  const N = data.rows;
  const rng = createRNG(cfg.seed ?? null);

  if (d === 0 || N < 2) return new CausalGraph([...nodeNames]);

  // Build GST trees for each variable (cached BIC scoring)
  const gstTrees: GrowShrinkTree[] = [];
  const allCols = Array.from({ length: d }, (_, i) => i);
  for (let i = 0; i < d; i++) {
    const candidates = allCols.filter(c => c !== i);
    gstTrees.push(new GrowShrinkTree(data, i, candidates, cfg.penaltyDiscount));
  }

  let bestGlobalBIC = -Infinity;
  let bestParents: number[][] = [];

  // Multiple random restarts
  for (let start = 0; start < cfg.numStarts; start++) {
    // Generate random permutation
    const perm = fisherYatesShuffle(allCols, rng);

    // Initial parent sets from GST trees
    const parents: number[][] = Array.from({ length: d }, () => [] as number[]);
    for (let pos = 0; pos < d; pos++) {
      const v = perm[pos];
      const predecessors = perm.slice(0, pos);
      if (predecessors.length > 0) {
        parents[v] = gstTrees[v].findOptimal(predecessors, cfg.maxParents);
      }
    }

    // Compute initial BIC
    let currentBIC = computeTotalBIC(gstTrees, parents, perm);
    let improved = true;
    let iter = 0;

    // Better mutation: try moving each variable to a better position
    while (improved && iter < cfg.maxIter) {
      improved = false;
      iter++;

      for (let v = 0; v < d; v++) {
        const currentPos = perm.indexOf(v);
        let bestPos = currentPos;
        let bestPosBIC = currentBIC;

        // Try inserting v at every other position
        for (let newPos = 0; newPos <= d; newPos++) {
          if (newPos === currentPos || newPos === currentPos + 1) continue;

          // Build candidate permutation with v at newPos
          const candidate: number[] = [];
          for (let i = 0; i < d; i++) {
            if (perm[i] !== v) candidate.push(perm[i]);
          }
          candidate.splice(newPos > currentPos ? newPos - 1 : newPos, 0, v);

          // Compute BIC for this candidate ordering
          const candBIC = computeBICForOrder(gstTrees, candidate, cfg.maxParents);
          if (candBIC > bestPosBIC) {
            bestPosBIC = candBIC;
            bestPos = newPos;
          }
        }

        if (bestPos !== currentPos) {
          // Apply the move
          const newPerm = [...perm.filter(x => x !== v)];
          newPerm.splice(bestPos > currentPos ? bestPos - 1 : bestPos, 0, v);

          // Copy back and recompute parents
          for (let i = 0; i < d; i++) perm[i] = newPerm[i]!;

          // Recompute parent sets for affected variables
          for (let pos = 0; pos < d; pos++) {
            const node = perm[pos];
            const preds = perm.slice(0, pos);
            parents[node] = gstTrees[node].findOptimal(preds, cfg.maxParents);
          }

          currentBIC = computeTotalBIC(gstTrees, parents, perm);
          improved = true;
        }
      }
    }

    if (currentBIC > bestGlobalBIC) {
      bestGlobalBIC = currentBIC;
      bestParents = parents.map(p => [...p]);
    }
  }

  // Build DAG from best ordering
  const g = new CausalGraph([...nodeNames]);
  for (let v = 0; v < d; v++) {
    for (const p of bestParents[v] ?? []) {
      g.addEdge(nodeNames[p], nodeNames[v]);
    }
  }

  // Convert to CPDAG
  const cpdag = g.pdag2dag();

  if (domainKnowledge) cpdag.applyDomainKnowledge(domainKnowledge);

  return cpdag;
}

// ── Helpers ─────────────────────────────────────────────────────────

function computeTotalBIC(
  trees: GrowShrinkTree[],
  parents: number[][],
  _perm: number[],
): number {
  let total = 0;
  for (let i = 0; i < parents.length; i++) {
    total += trees[i].score(parents[i] ?? []);
  }
  return total;
}

function computeBICForOrder(
  trees: GrowShrinkTree[],
  perm: number[],
  maxParents: number,
): number {
  let total = 0;
  for (let pos = 0; pos < perm.length; pos++) {
    const v = perm[pos];
    const preds = perm.slice(0, pos);
    const opt = trees[v].findOptimal(preds, maxParents);
    total += trees[v].score(opt);
  }
  return total;
}

function fisherYatesShuffle<T>(arr: T[], rng: () => number): T[] {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = result[i];
    result[i] = result[j];
    result[j] = tmp;
  }
  return result;
}
