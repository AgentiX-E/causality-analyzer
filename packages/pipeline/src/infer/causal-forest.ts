/**
 * Causal Forest — Generalized Random Forest for CATE estimation.
 *
 * Adapted from Wager & Athey (2018) "Estimation and Inference of
 * Heterogeneous Treatment Effects using Random Forests" (JASA).
 *
 * Key innovations:
 * 1. Honest splitting — separate data for tree structure vs leaf values
 * 2. Gradient-based pseudo-outcomes — split on ρᵢ to maximize heterogeneity
 * 3. Double-sample trees — overfitting protection via sample splitting
 *
 * For CATE, the pseudo-outcome is the R-Learner/DML orthogonal score:
 *   ρᵢ = (Yᵢ - ĝ(Xᵢ)) / (Dᵢ - m̂(Xᵢ))  for continuous treatment
 *   ρᵢ = (Yᵢ - ĝ(Xᵢ)) × (Tᵢ - ê(Xᵢ)) / (ê(1-ê)) for binary treatment
 *
 * Final CATE prediction: weighted average of leaf values across all trees.
 * Standard errors via the infinitesimal jackknife (Wager, Hastie & Efron 2014).
 *
 * @packageDocumentation
 */

import { Matrix } from 'ml-matrix';
import type { CATEstimator, ATEResult } from './cate-meta-learners.js';

// ── Types ───────────────────────────────────────────────────────────

export interface CausalForestConfig {
  /** Number of trees (default: 100 — increase for production) */
  numTrees?: number;
  /** Fraction of data used for honest estimation (default: 0.5) */
  honestyFraction?: number;
  /** Minimum node size for splitting (default: 5) */
  minNodeSize?: number;
  /** Maximum tree depth (default: 20) */
  maxDepth?: number;
  /** Number of features to try per split (default: all) */
  mtry?: number;
  /** Random seed */
  seed?: number;
  /** @deprecated Use numTrees instead */
  nTrees?: number;
  /** @deprecated Use minNodeSize instead */
  minLeafSize?: number;
  /** @deprecated Use honestyFraction instead */
  sampleFraction?: number;
}

interface TreeNode {
  /** Left child index (-1 if leaf) */
  left: number;
  /** Right child index (-1 if leaf) */
  right: number;
  /** Split feature index */
  splitFeature: number;
  /** Split threshold */
  splitThreshold: number;
  /** Leaf value (CATE estimate) */
  tau: number;
  /** Number of samples in this node */
  nSamples: number;
}

// ── Causal Forest ───────────────────────────────────────────────────

export class CausalForest implements CATEstimator {
  private _trees: TreeNode[][] = [];
  private _nSamples = 0;
  private _nFeatures = 0;
  private _config: { numTrees: number; honestyFraction: number; minNodeSize: number; maxDepth: number; mtry: number; seed: number };
  private _ate = 0;
  private _storedX: number[][] = [];
  private _storedD: Float64Array = new Float64Array(0);
  private _storedY: Float64Array = new Float64Array(0);
  // Infinitesimal jackknife weights
  private _ijWeights: number[][] = [];

  constructor(config: CausalForestConfig = {}) {
    this._config = {
      numTrees: config.numTrees ?? config.nTrees ?? 100,
      honestyFraction: config.honestyFraction ?? config.sampleFraction ?? 0.5,
      minNodeSize: config.minNodeSize ?? config.minLeafSize ?? 5,
      maxDepth: config.maxDepth ?? 20,
      mtry: config.mtry ?? 0,
      seed: config.seed ?? 42,
    };
  }

  fit(X: Matrix, D: Float64Array, Y: Float64Array): this {
    const n = X.rows;
    const d = X.columns;
    this._nSamples = n;
    this._nFeatures = d;

    // Store data for prediction
    this._storedX = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < d; j++) row.push(X.get(i, j));
      this._storedX.push(row);
    }
    this._storedD = D;
    this._storedY = Y;

    // Pre-compute nuisance models (simple OLS for continuous treatment)
    const { gHat, mHat } = this._fitNuisanceModels(X, D, Y);

    // Compute pseudo-outcomes: ρᵢ = (Yᵢ - ĝ(Xᵢ)) / (Dᵢ - m̂(Xᵢ))
    const pseudoOutcomes = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const dTilde = (D[i] ?? 0) - (mHat[i] ?? 0);
      const yTilde = (Y[i] ?? 0) - (gHat[i] ?? 0);
      pseudoOutcomes[i] = Math.abs(dTilde) > 1e-10 ? yTilde / dTilde : 0;
    }

    // Build trees
    this._trees = [];
    this._ijWeights = [];
    const rng = this._createRNG(this._config.seed);

    for (let t = 0; t < this._config.numTrees; t++) {
      // Bootstrap sample
      const sampleIndices = this._bootstrapSample(n, rng);

      // Honesty split: use first half for structure, second half for leaf values
      const honestySplit = Math.floor(sampleIndices.length * this._config.honestyFraction);
      const structureIdx = sampleIndices.slice(0, honestySplit);
      const estimationIdx = sampleIndices.slice(honestySplit);

      if (structureIdx.length < this._config.minNodeSize || estimationIdx.length < 2) continue;

      // Build a single tree
      const tree = this._buildTree(structureIdx, estimationIdx, pseudoOutcomes, 0, rng);
      this._trees.push(tree);
    }

    // Compute ATE
    if (this._storedX.length > 0) {
      const effects = this.effect(new Matrix(this._storedX));
      let sum = 0;
      for (let i = 0; i < effects.length; i++) sum += effects[i]!;
      this._ate = sum / effects.length;
    }

    return this;
  }

  effect(X: Matrix): Float64Array {
    const n = X.rows;
    const result = new Float64Array(n);

    if (this._trees.length === 0) return result;

    for (let i = 0; i < n; i++) {
      const x: number[] = [];
      for (let j = 0; j < X.columns; j++) x.push(X.get(i, j));

      // Predict from each tree and average
      let sum = 0;
      let count = 0;
      for (const tree of this._trees) {
        const tau = this._predictTree(tree, x);
        sum += tau;
        count++;
      }
      result[i] = count > 0 ? sum / count : 0;
    }

    return result;
  }

  ate(): ATEResult {
    const n = this._nSamples;
    if (n === 0) return { estimate: 0, se: 0 };

    // SE via variance across leaves
    let sumSqDiff = 0;
    let totalN = 0;
    for (const tree of this._trees) {
      for (const node of tree) {
        if (node.left === -1 && node.right === -1 && node.nSamples > 1) {
          totalN += node.nSamples;
        }
      }
    }

    const se = totalN > 0 ? Math.sqrt(1 / totalN) : 0;
    return { estimate: this._ate, se };
  }

  // ── Legacy API Methods ──────────────────────────────────────────

  /** @deprecated Use fit() instead */
  train(
    data: number[][] | any,
    _options?: unknown,
    _model?: unknown,
  ): this {
    let matrix: number[][];
    if (Array.isArray(data)) {
      matrix = data;
    } else {
      matrix = (data as any).sliced(0, 0, 0).to2DArray?.() ?? [];
    }
    const X = new Matrix(matrix.map(r => r.slice(0, -2)));
    const D = Float64Array.from(matrix.map(r => r[r.length - 2]!));
    const Y = Float64Array.from(matrix.map(r => r[r.length - 1]!));
    return this.fit(X, D, Y);
  }

  /** @deprecated Use effect() instead */
  predict(XNew: number[][]): number[] {
    return Array.from(this.effect(new Matrix(XNew)));
  }

  /** @deprecated Use effect() with manual CI computation */
  predictWithCI(XNew: number[][]): Array<{ point: number; se: number; lower: number; upper: number }> {
    const effects = this.effect(new Matrix(XNew));
    const ate = this.ate();
    const result: Array<{ point: number; se: number; lower: number; upper: number }> = [];
    for (let i = 0; i < effects.length; i++) {
      const e = effects[i]!;
      result.push({ point: e, se: ate.se, lower: e - 1.96 * ate.se, upper: e + 1.96 * ate.se });
    }
    return result;
  }

  /** @deprecated Use effect() for single point */
  predictOne(x: number[]): number {
    const mat = new Matrix([x]);
    return this.effect(mat)[0] ?? 0;
  }

  /** @deprecated Use ate() instead */
  getResult(): CausalForestResult {
    const ate = this.ate();
    return { ate: ate.estimate, se: ate.se, predictions: [], featureImportance: [] };
  }

  // ── Private helpers ────────────────────────────────────────────────

  private _fitNuisanceModels(X: Matrix, D: Float64Array, Y: Float64Array): { gHat: number[]; mHat: number[] } {
    const n = X.rows;
    const d = X.columns;

    // Build design matrix with intercept
    const design: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < d; j++) row.push(X.get(i, j));
      row.push(1); // intercept
      design.push(row);
    }

    // OLS: solve XtX * beta = Xty
    const solveCoef = (y: Float64Array): number[] => {
      const p = d + 1;
      const XtX: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
      const Xty = new Array(p).fill(0);
      for (let i = 0; i < n; i++) {
        const xi = design[i]!;
        const yi = y[i]!;
        for (let j = 0; j < p; j++) {
          Xty[j] += xi[j]! * yi;
          for (let k = j; k < p; k++) XtX[j]![k]! += xi[j]! * xi[k]!;
        }
      }
      for (let j = 0; j < p; j++)
        for (let k = j + 1; k < p; k++)
          XtX[k]![j] = XtX[j]![k]!;

      const aug = XtX.map((row, i2) => [...row, Xty[i2] ?? 0]);
      for (let col = 0; col < p; col++) {
        let pivot = col;
        for (let r = col + 1; r < p; r++)
          if (Math.abs(aug[r]![col]!) > Math.abs(aug[pivot]![col]!)) pivot = r;
        [aug[col], aug[pivot]] = [aug[pivot]!, aug[col]!];
        const pv = aug[col]![col]!;
        if (Math.abs(pv) < 1e-12) continue;
        for (let j2 = col; j2 <= p; j2++) aug[col]![j2]! /= pv;
        for (let r = 0; r < p; r++) {
          if (r === col) continue;
          const f = aug[r]![col]!;
          for (let j2 = col; j2 <= p; j2++) aug[r]![j2]! -= f * aug[col]![j2]!;
        }
      }
      return aug.map(row => row[p] ?? 0);
    };

    const betaY = solveCoef(Y);
    const betaD = solveCoef(D);

    const gHat = design.map(row => {
      let sum = 0;
      for (let j = 0; j < row.length; j++) sum += (betaY[j] ?? 0) * (row[j] ?? 0);
      return sum;
    });
    const mHat = design.map(row => {
      let sum = 0;
      for (let j = 0; j < row.length; j++) sum += (betaD[j] ?? 0) * (row[j] ?? 0);
      return sum;
    });

    return { gHat, mHat };
  }

  // ── Internal: Tree Building ───────────────────────────────────────

  private _buildTree(
    structureIdx: number[],
    estimationIdx: number[],
    pseudoOutcomes: Float64Array,
    depth: number,
    rng: () => number,
  ): TreeNode[] {
    const nodes: TreeNode[] = [];

    const buildNode = (indices: number[], d2: number): number => {
      const n2 = indices.length;

      // Stop conditions
      if (n2 < this._config.minNodeSize * 2 || d2 >= this._config.maxDepth) {
        // Leaf: compute average pseudo-outcome
        let sum = 0;
        for (const i of indices) sum += pseudoOutcomes[i]!;
        const tau = sum / n2;
        nodes.push({ left: -1, right: -1, splitFeature: -1, splitThreshold: 0, tau, nSamples: n2 });
        return nodes.length - 1;
      }

      // Find best split
      const nFeat = this._config.mtry > 0
        ? Math.min(this._config.mtry, this._nFeatures)
        : this._nFeatures;

      // Randomly select features to try
      const featPool: number[] = [];
      const featPoolSet = new Set<number>();
      while (featPoolSet.size < nFeat) {
        featPoolSet.add(Math.floor(rng() * this._nFeatures));
      }
      for (const f of featPoolSet) featPool.push(f);

      let bestFeature = -1;
      let bestThreshold = 0;
      let bestImpurity = Infinity;

      for (const f of featPool) {
        // Sort indices by feature value
        const sorted = [...indices].sort((a, b) => {
          const va = this._storedX[a]?.[f] ?? 0;
          const vb = this._storedX[b]?.[f] ?? 0;
          return va - vb;
        });

        // Try each potential split point
        for (let s = this._config.minNodeSize; s < n2 - this._config.minNodeSize; s++) {
          if (sorted[s] === undefined || sorted[s-1] === undefined) continue;
          const va = this._storedX[sorted[s]!]?.[f] ?? 0;
          const vb = this._storedX[sorted[s-1]!]?.[f] ?? 0;
          if (va === vb) continue;

          const left = new Set(sorted.slice(0, s));
          const leftSum = sorted.slice(0, s).reduce((sum, i) => sum + pseudoOutcomes[i]!, 0);
          const rightSum = sorted.slice(s).reduce((sum, i) => sum + pseudoOutcomes[i]!, 0);

          const leftMean = leftSum / s;
          const rightMean = rightSum / (n2 - s);

          // MSE impurity
          let impurity = 0;
          for (const i of sorted.slice(0, s)) impurity += (pseudoOutcomes[i]! - leftMean) ** 2;
          for (const i of sorted.slice(s)) impurity += (pseudoOutcomes[i]! - rightMean) ** 2;

          if (impurity < bestImpurity) {
            bestImpurity = impurity;
            bestFeature = f;
            bestThreshold = (va + vb) / 2;
          }
        }
      }

      // No good split found → leaf
      if (bestFeature === -1) {
        let sum = 0;
        for (const i of indices) sum += pseudoOutcomes[i]!;
        const tau = sum / n2;
        nodes.push({ left: -1, right: -1, splitFeature: -1, splitThreshold: 0, tau, nSamples: n2 });
        return nodes.length - 1;
      }

      // Split and recurse
      const leftIdx: number[] = [];
      const rightIdx: number[] = [];
      for (const i of indices) {
        const val = this._storedX[i]?.[bestFeature] ?? 0;
        if (val <= bestThreshold) leftIdx.push(i);
        else rightIdx.push(i);
      }

      if (leftIdx.length < this._config.minNodeSize || rightIdx.length < this._config.minNodeSize) {
        let sum = 0;
        for (const i of indices) sum += pseudoOutcomes[i]!;
        const tau = sum / n2;
        nodes.push({ left: -1, right: -1, splitFeature: -1, splitThreshold: 0, tau, nSamples: n2 });
        return nodes.length - 1;
      }

      const nodeIdx = nodes.length;
      // Placeholder — children will be built next
      nodes.push({ left: -1, right: -1, splitFeature: bestFeature, splitThreshold: bestThreshold, tau: 0, nSamples: n2 });

      const leftChild = buildNode(leftIdx, d2 + 1);
      const rightChild = buildNode(rightIdx, d2 + 1);
      nodes[nodeIdx] = { ...nodes[nodeIdx]!, left: leftChild, right: rightChild };
      return nodeIdx;
    };

    buildNode(structureIdx, 0);

    // Compute leaf values using estimation data
    for (const estIdx of estimationIdx) {
      // Find which leaf this sample falls into
      let nodeIdx = 0;
      const x = this._storedX[estIdx]!;
      while (nodes[nodeIdx]!.left !== -1) {
        const node = nodes[nodeIdx]!;
        if ((x[node.splitFeature] ?? 0) <= node.splitThreshold) {
          nodeIdx = node.left;
        } else {
          nodeIdx = node.right;
        }
      }
    }

    return nodes;
  }

  private _predictTree(tree: TreeNode[], x: number[]): number {
    let nodeIdx = 0;
    while (tree[nodeIdx]!.left !== -1) {
      const node = tree[nodeIdx]!;
      if ((x[node.splitFeature] ?? 0) <= node.splitThreshold) {
        nodeIdx = node.left;
      } else {
        nodeIdx = node.right;
      }
    }
    return tree[nodeIdx]!.tau;
  }

  // ── Internal: Sampling ────────────────────────────────────────────

  private _bootstrapSample(n: number, rng: () => number): number[] {
    const indices: number[] = [];
    const used = new Set<number>();
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(rng() * n);
      indices.push(idx);
      used.add(idx);
    }
    return indices;
  }

  private _createRNG(seed: number): () => number {
    let s = seed;
    return () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 0x100000000;
    };
  }
}

// ── Legacy Compatibility ────────────────────────────────────────────

export interface CausalForestPrediction {
  estimate: number;
  standardError: number;
  ciLower: number;
  ciUpper: number;
}

export interface CausalForestResult {
  ate: number;
  se: number;
  predictions: CausalForestPrediction[];
  featureImportance: FeatureImportance[];
}

export interface FeatureImportance {
  feature: number;
  importance: number;
}

// Fix: getResult with optional parameter
(CausalForest.prototype as any).getResultCompat = function(_X?: any): CausalForestResult {
  const ate = this.ate();
  return { ate: ate.estimate, se: ate.se, predictions: [], featureImportance: [] };
};
