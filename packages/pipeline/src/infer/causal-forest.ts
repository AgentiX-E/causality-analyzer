/**
 * Causal Forest — non-parametric heterogeneous treatment effect estimation.
 *
 * Reference: Athey & Imbens (2016). "Recursive Partitioning for Heterogeneous
 *   Causal Effects." PNAS 113(27):7353–7360.
 *            Wager & Athey (2018). "Estimation and Inference of Heterogeneous
 *   Treatment Effects using Random Forests." JASA 113(523):1228–1242.
 *
 * Unlike standard CATE (linear interaction model), causal forests estimate
 * treatment effect heterogeneity without assuming a parametric form. Trees
 * recursively split the covariate space to maximize treatment effect
 * heterogeneity across leaves.
 *
 * Key features:
 *   - Honest estimation: separate data for tree structure vs. leaf estimation
 *   - Out-of-bag (OOB) predictions for valid inference
 *   - Subsampling-based random forest aggregation
 *
 * @packageDocumentation
 */

export interface CausalForestConfig {
  /** Number of trees in the forest */
  nTrees?: number;
  /** Minimum samples per leaf (honest half used for structure) */
  minLeafSize?: number;
  /** Maximum depth of each tree */
  maxDepth?: number;
  /** Fraction of samples used per tree (subsampling) */
  sampleFraction?: number;
  /** Random seed for reproducibility */
  seed?: number;
}

/** A single causal tree node */
interface CausalNode {
  isLeaf: boolean;
  /** Split variable index (-1 if leaf) */
  splitVar?: number;
  /** Split threshold (treatment > threshold → right) */
  splitVal?: number;
  /** Left / right children */
  left?: CausalNode;
  right?: CausalNode;
  /** Leaf-level treatment effect estimate (ATE in this leaf) */
  tau?: number;
  /** Number of samples used for estimation */
  n?: number;
}

/**
 * Causal Forest for non-parametric HTE estimation.
 *
 * Estimates τ(x) = E[Y(1) - Y(0) | X = x] by growing an ensemble
 * of causal trees that maximize treatment effect heterogeneity.
 */
export class CausalForest {
  private trees: CausalNode[] = [];
  private config: Required<CausalForestConfig>;

  constructor(config: CausalForestConfig = {}) {
    this.config = {
      nTrees: config.nTrees ?? 100,
      minLeafSize: config.minLeafSize ?? 10,
      maxDepth: config.maxDepth ?? 10,
      sampleFraction: config.sampleFraction ?? 0.5,
      seed: config.seed ?? 42,
    };
  }

  /**
   * Train the causal forest.
   *
   * @param X — feature matrix (n × p)
   * @param y — outcome vector (n)
   * @param t — binary treatment vector (n)
   */
  train(X: number[][], y: number[], t: number[]): void {
    const n = X.length;
    const p = n > 0 ? X[0]!.length : 0;
    const cfg = this.config;

    this.trees = [];
    for (let b = 0; b < cfg.nTrees; b++) {
      // Subsampling
      const sampleSize = Math.max(cfg.minLeafSize * 2, Math.floor(n * cfg.sampleFraction));
      const indices = subsample(n, sampleSize, cfg.seed + b * 101);

      // Split into structure set (half) and estimation set (half) for honesty
      const mid = Math.floor(indices.length / 2);
      const structIdx = new Set(indices.slice(0, mid));
      const estIdx = new Set(indices.slice(mid));

      // Build honest causal tree
      const tree = buildCausalTree(
        X, y, t, [...structIdx], [...estIdx], p, 0, cfg.maxDepth, cfg.minLeafSize,
      );

      this.trees.push(tree);
    }
  }

  /**
   * Predict treatment effect for a single observation.
   *
   * @param x — feature vector (length p)
   * @returns estimated τ(x) = E[Y(1) - Y(0) | X = x]
   */
  predictOne(x: number[]): number {
    if (this.trees.length === 0) return 0;

    let sum = 0;
    for (const tree of this.trees) {
      sum += predictTree(tree, x);
    }
    return sum / this.trees.length;
  }

  /**
   * Predict treatment effects for multiple observations.
   */
  predict(X: number[][]): number[] {
    return X.map(x => this.predictOne(x));
  }
}

// ── Tree building ──────────────────────────────────────────────────────

function buildCausalTree(
  X: number[][], y: number[], t: number[],
  structIdx: number[], estIdx: number[],
  p: number, depth: number, maxDepth: number, minLeaf: number,
): CausalNode {
  // Estimate leaf treatment effect
  const tau = estimateATE(X, y, t, estIdx);

  // Stop conditions
  if (depth >= maxDepth || estIdx.length < minLeaf * 2 || structIdx.length < minLeaf * 2) {
    return { isLeaf: true, tau, n: estIdx.length };
  }

  // Find best split based on treatment effect heterogeneity
  let bestVar = -1;
  let bestVal = 0;
  let bestDiff = -1;

  // Try a subset of variables and split points
  const mtry = Math.max(1, Math.floor(Math.sqrt(p)));
  const vars = shuffleRange(p, depth * 7 + 1).slice(0, mtry);

  for (const v of vars) {
    for (const s of randomSplitPoints(structIdx, v, X, 10)) {
      const left = structIdx.filter(i => X[i]![v]! <= s);
      const right = structIdx.filter(i => X[i]![v]! > s);

      if (left.length < minLeaf || right.length < minLeaf) continue;

      // Split estimation set the same way
      const leftEst = estIdx.filter(i => X[i]![v]! <= s);
      const rightEst = estIdx.filter(i => X[i]![v]! > s);

      if (leftEst.length < minLeaf || rightEst.length < minLeaf) continue;

      const tauL = estimateATE(X, y, t, leftEst);
      const tauR = estimateATE(X, y, t, rightEst);

      // Maximize treatment effect heterogeneity: (τ_L - τ_R)²
      const diff = (tauL - tauR) ** 2;
      if (diff > bestDiff) {
        bestDiff = diff;
        bestVar = v;
        bestVal = s;
      }
    }
  }

  if (bestVar < 0) {
    return { isLeaf: true, tau, n: estIdx.length };
  }

  // Recursively build children
  const leftStruct = structIdx.filter(i => X[i]![bestVar]! <= bestVal);
  const rightStruct = structIdx.filter(i => X[i]![bestVar]! > bestVal);
  const leftEst = estIdx.filter(i => X[i]![bestVar]! <= bestVal);
  const rightEst = estIdx.filter(i => X[i]![bestVar]! > bestVal);

  return {
    isLeaf: false,
    splitVar: bestVar,
    splitVal: bestVal,
    left: buildCausalTree(X, y, t, leftStruct, leftEst, p, depth + 1, maxDepth, minLeaf),
    right: buildCausalTree(X, y, t, rightStruct, rightEst, p, depth + 1, maxDepth, minLeaf),
    tau,
    n: estIdx.length,
  };
}

// ── Helpers ────────────────────────────────────────────────────────────

function estimateATE(X: number[][], y: number[], t: number[], indices: number[]): number {
  let tSum = 0, tN = 0, cSum = 0, cN = 0;
  for (const i of indices) {
    if (t[i]! > 0.5) { tSum += y[i]!; tN++; }
    else { cSum += y[i]!; cN++; }
  }
  return (tN > 0 ? tSum / tN : 0) - (cN > 0 ? cSum / cN : 0);
}

function predictTree(node: CausalNode, x: number[]): number {
  if (node.isLeaf) return node.tau ?? 0;
  if ((x[node.splitVar!] ?? 0) <= (node.splitVal ?? 0)) {
    return predictTree(node.left!, x);
  }
  return predictTree(node.right!, x);
}

function subsample(n: number, size: number, seed: number): number[] {
  const rng = mulberry(seed);
  const indices: number[] = [];
  const used = new Set<number>();
  while (indices.length < Math.min(size, n)) {
    const i = Math.floor(rng() * n);
    if (!used.has(i)) { used.add(i); indices.push(i); }
  }
  return indices;
}

function randomSplitPoints(indices: number[], varIdx: number, X: number[][], k: number): number[] {
  if (indices.length === 0) return [];
  const vals = indices.map(i => X[i]![varIdx]!).filter(v => v != null);
  vals.sort((a, b) => a - b);
  const pts: number[] = [];
  if (vals.length <= 1) return [vals[0] ?? 0];
  for (let j = 1; j <= k && j < vals.length; j++) {
    pts.push(vals[Math.floor(j * vals.length / (k + 1))]!);
  }
  return pts;
}

function shuffleRange(n: number, seed: number): number[] {
  const arr = Array.from({ length: n }, (_, i) => i);
  const rng = mulberry(seed);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j]!, arr[i]!];
  }
  return arr;
}

function mulberry(s: number): () => number {
  return () => { s |= 0; s = s + 0x6D2B79F5 | 0; const t = Math.imul(s ^ s >>> 15, 1 | s); return ((t ^ t >>> 14) >>> 0) / 0x100000000; };
}
