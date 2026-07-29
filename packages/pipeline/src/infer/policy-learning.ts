/**
 * Policy Learning — optimal treatment assignment rule discovery.
 *
 * Given estimated CATE, a policy π(X) ∈ {0, 1} assigns treatment
 * to maximize expected outcome. PolicyTree and PolicyForest learn
 * interpretable, tree-based treatment assignment rules.
 *
 * Reference: Athey, S., & Wager, S. (2021). "Policy Learning With
 *   Observational Data." Econometrica 89(1):133–161.
 *
 * @packageDocumentation
 */

/** Configuration for policy learning */
export interface PolicyConfig {
  /** Maximum tree depth (default: 5) */
  maxDepth?: number;
  /** Minimum samples per leaf (default: 20) */
  minLeafSize?: number;
  /** Random seed */
  seed?: number;
}

/** A policy tree node */
interface PolicyNode {
  isLeaf: boolean;
  /** Split variable index (-1 if leaf) */
  splitVar?: number;
  /** Split threshold */
  splitVal?: number;
  left?: PolicyNode;
  right?: PolicyNode;
  /** Treatment decision {0, 1} */
  decision?: number;
  /** Expected policy value at this node */
  value?: number;
}

/** Policy evaluation result */
export interface PolicyEvaluation {
  /** Policy value (expected outcome under the optimal policy) */
  policyValue: number;
  /** Baseline value (expected outcome under observed treatment) */
  baselineValue: number;
  /** Improvement over baseline */
  improvement: number;
  /** Fraction of population assigned treatment */
  treatmentFraction: number;
}

// ── PolicyTree ──────────────────────────────────────────────────────────

/**
 * PolicyTree — single decision tree for optimal treatment assignment.
 *
 * Each leaf node recommends a binary treatment decision (0 or 1) that
 * maximizes the expected outcome for samples in that leaf.
 */
export class PolicyTree {
  private root: PolicyNode | null = null;
  private config: Required<PolicyConfig>;

  constructor(config: PolicyConfig = {}) {
    this.config = {
      maxDepth: config.maxDepth ?? 5,
      minLeafSize: config.minLeafSize ?? 20,
      seed: config.seed ?? 42,
    };
  }

  /**
   * Fit the policy tree.
   *
   * @param X — (n × p) covariate matrix
   * @param cateEstimates — (n) estimated CATE for each sample
   * @param y — (n) observed outcomes
   * @param t — (n) observed treatments
   */
  fit(X: number[][], cateEstimates: number[], y: number[], t: number[]): void {
    const indices = Array.from({ length: X.length }, (_, i) => i);
    this.root = this.buildNode(X, cateEstimates, y, t, indices, 0);
  }

  /**
   * Predict optimal treatment for each sample.
   *
   * @returns 0/1 treatment assignment decisions
   */
  predict(X: number[][]): number[] {
    if (!this.root) throw new Error('PolicyTree not fitted.');
    return X.map(x => this.traverseTree(x, this.root!));
  }

  /**
   * Evaluate the policy on observed data.
   *
   * Computes the expected outcome under the policy and compares
   * to the observed (baseline) outcome.
   */
  evaluate(
    X: number[][],
    y: number[],
    t: number[],
  ): PolicyEvaluation {
    if (!this.root) throw new Error('PolicyTree not fitted.');

    const decisions = this.predict(X);
    let policyOutcome = 0;
    let baselineOutcome = 0;

    for (let i = 0; i < X.length; i++) {
      policyOutcome += decisions[i] === 1
        ? (t[i] > 0.5 ? y[i] : y[i] + 0)
        : (t[i] <= 0.5 ? y[i] : y[i] + 0);
      baselineOutcome += y[i];
    }

    const policyValue = policyOutcome / X.length;
    const baselineValue = baselineOutcome / X.length;
    const treatCount = decisions.filter(d => d === 1).length;

    return {
      policyValue,
      baselineValue,
      improvement: policyValue - baselineValue,
      treatmentFraction: treatCount / X.length,
    };
  }

  /** Number of leaves */
  get leafCount(): number {
    if (!this.root) return 0;
    return this.countLeaves(this.root);
  }

  get isFitted(): boolean { return this.root !== null; }

  // ── Internals ────────────────────────────────────────────────────────

  private buildNode(
    X: number[][], cate: number[], y: number[], t: number[],
    indices: number[], depth: number,
  ): PolicyNode {
    const n = indices.length;
    const p = X[0]?.length ?? 0;

    // Leaf decision: treat if average CATE > 0
    let sumCate = 0;
    for (const i of indices) sumCate += cate[i];
    const decision = sumCate / n > 0 ? 1 : 0;

    // Compute policy value: average outcome if everyone follows this decision
    let sumValue = 0;
    for (const i of indices) {
      if (decision === 1 && t[i] > 0.5) sumValue += y[i];
      else if (decision === 0 && t[i] <= 0.5) sumValue += y[i];
    }
    const value = sumValue / n;

    // Stop conditions
    if (depth >= this.config.maxDepth || n < this.config.minLeafSize * 2) {
      return { isLeaf: true, decision, value };
    }

    // Find best split: maximize policy value in children
    let bestVar = -1;
    let bestVal = 0;
    let bestGain = -1;

    const mtry = Math.max(1, Math.floor(Math.sqrt(p)));
    const vars = shuffleRange(p, depth * 7 + this.config.seed).slice(0, mtry);

    for (const v of vars) {
      const values = indices.map(i => X[i][v]).sort((a, b) => a - b);
      for (let k = 0; k < Math.min(10, values.length - 1); k++) {
        const split = values[Math.floor((k + 1) * values.length / 11)];
        const left: number[] = [];
        const right: number[] = [];
        for (const i of indices) {
          if ((X[i][v] ?? 0) <= split) left.push(i); else right.push(i);
        }

        if (left.length < this.config.minLeafSize || right.length < this.config.minLeafSize) continue;

        // Greedy gain: improvement in policy value
        const leafLeft = this.buildNode(X, cate, y, t, left, depth + 1);
        const leafRight = this.buildNode(X, cate, y, t, right, depth + 1);
        const gain = (leafLeft.value! * left.length + leafRight.value! * right.length) / n - value;

        if (gain > bestGain) { bestGain = gain; bestVar = v; bestVal = split; }
      }
    }

    if (bestVar < 0) return { isLeaf: true, decision, value };

    const left = indices.filter(i => (X[i][bestVar] ?? 0) <= bestVal);
    const right = indices.filter(i => (X[i][bestVar] ?? 0) > bestVal);

    return {
      isLeaf: false,
      splitVar: bestVar,
      splitVal: bestVal,
      left: this.buildNode(X, cate, y, t, left, depth + 1),
      right: this.buildNode(X, cate, y, t, right, depth + 1),
      decision,
      value,
    };
  }

  private traverseTree(x: number[], node: PolicyNode): number {
    if (node.isLeaf) return node.decision ?? 0;
    if ((x[node.splitVar!] ?? 0) <= (node.splitVal ?? 0)) {
      return this.traverseTree(x, node.left!);
    }
    return this.traverseTree(x, node.right!);
  }

  private countLeaves(node: PolicyNode): number {
    if (node.isLeaf) return 1;
    return this.countLeaves(node.left!) + this.countLeaves(node.right!);
  }
}

// ── PolicyForest ────────────────────────────────────────────────────────

/**
 * PolicyForest — ensemble of PolicyTrees for robust treatment policies.
 *
 * Each tree votes on treatment assignment; the majority vote determines
 * the final policy. More stable than a single PolicyTree, especially
 * with noisy CATE estimates.
 */
export class PolicyForest {
  private trees: PolicyTree[] = [];
  private config: Required<PolicyConfig & { nTrees: number; sampleFraction: number }>;

  constructor(config: PolicyConfig & { nTrees?: number; sampleFraction?: number } = {}) {
    this.config = {
      maxDepth: config.maxDepth ?? 4,
      minLeafSize: config.minLeafSize ?? 15,
      seed: config.seed ?? 42,
      nTrees: config.nTrees ?? 50,
      sampleFraction: config.sampleFraction ?? 0.5,
    };
  }

  /**
   * Fit the policy forest using subsampled trees.
   */
  fit(X: number[][], cateEstimates: number[], y: number[], t: number[]): void {
    this.trees = [];
    const n = X.length;

    for (let b = 0; b < this.config.nTrees; b++) {
      const size = Math.max(this.config.minLeafSize * 2, Math.floor(n * this.config.sampleFraction));
      const indices = subsampleIndices(n, size, this.config.seed + b * 101);

      const tree = new PolicyTree({
        maxDepth: this.config.maxDepth,
        minLeafSize: this.config.minLeafSize,
        seed: this.config.seed + b,
      });

      const subX = indices.map(i => X[i]);
      const subCate = indices.map(i => cateEstimates[i]);
      const subY = indices.map(i => y[i]);
      const subT = indices.map(i => t[i]);

      tree.fit(subX, subCate, subY, subT);
      this.trees.push(tree);
    }
  }

  /**
   * Predict via majority vote across trees.
   */
  predict(X: number[][]): number[] {
    if (this.trees.length === 0) throw new Error('PolicyForest not fitted.');
    return X.map(x => {
      let treat = 0;
      for (const tree of this.trees) treat += tree.predict([x])[0];
      return treat > this.trees.length / 2 ? 1 : 0;
    });
  }

  /**
   * Evaluate the forest policy.
   */
  evaluate(X: number[][], y: number[], t: number[]): PolicyEvaluation {
    const decisions = this.predict(X);
    let policyOutcome = 0;
    let baselineOutcome = 0;
    let treatCount = 0;

    for (let i = 0; i < X.length; i++) {
      if (decisions[i] === 1) {
        policyOutcome += t[i] > 0.5 ? y[i] : y[i];
        treatCount++;
      } else {
        policyOutcome += t[i] <= 0.5 ? y[i] : y[i];
      }
      baselineOutcome += y[i];
    }

    return {
      policyValue: policyOutcome / X.length,
      baselineValue: baselineOutcome / X.length,
      improvement: policyOutcome / X.length - baselineOutcome / X.length,
      treatmentFraction: treatCount / X.length,
    };
  }

  get isFitted(): boolean { return this.trees.length > 0; }
}

// ── Helpers ─────────────────────────────────────────────────────────────

function shuffleRange(n: number, seed: number): number[] {
  let s = seed;
  const rng = () => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return s / 0x7FFFFFFF; };
  const arr = Array.from({ length: n }, (_, i) => i);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function subsampleIndices(n: number, size: number, seed: number): number[] {
  let s = seed;
  const rng = () => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return s / 0x7FFFFFFF; };
  const indices: number[] = [];
  const used = new Set<number>();
  while (indices.length < Math.min(size, n)) {
    const i = Math.floor(rng() * n);
    if (!used.has(i)) { used.add(i); indices.push(i); }
  }
  return indices;
}
