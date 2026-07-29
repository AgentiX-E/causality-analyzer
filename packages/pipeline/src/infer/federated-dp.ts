/**
 * Federated Causal Learning with Differential Privacy.
 *
 * Enables causal effect estimation across multiple data silos without
 * exposing raw individual-level data. Each node computes local
 * statistics, applies differential privacy noise, and transmits only
 * privatized aggregates to the coordinator.
 *
 * Privacy mechanisms:
 *   - Laplace mechanism: ε-differential privacy for scalar statistics
 *   - Gaussian mechanism: (ε, δ)-differential privacy for vector statistics
 *
 * Reference: Dwork, C., & Roth, A. (2014). "The Algorithmic Foundations
 *   of Differential Privacy." Foundations and Trends in TCS 9(3–4).
 *
 * @packageDocumentation
 */

/** Configuration for federated causal learning */
export interface FederatedLearningConfig {
  /** Privacy budget ε (smaller = stronger privacy) */
  epsilon: number;
  /** Privacy mechanism */
  mechanism: 'laplace' | 'gaussian';
  /** Delta parameter for (ε,δ)-DP (Gaussian mechanism only) */
  delta?: number;
  /** Sensitivity of the statistic (max change from one sample) */
  sensitivity: number;
  /** Minimum node count required for aggregation */
  minNodes: number;
  /** Random seed for reproducibility */
  seed?: number;
}

/** Per-node federated statistic with noise */
export interface FederatedNodeResult {
  nodeId: string;
  /** Raw statistic (never leaves the node) */
  rawValue: number;
  /** Privatized statistic transmitted to coordinator */
  privatizedValue: number;
  /** Noise added */
  noiseAdded: number;
  /** Sample count for weighted aggregation */
  sampleCount: number;
}

/** Aggregated federated result */
export interface FederatedAggregation {
  /** Weighted mean of privatized values */
  aggregateValue: number;
  /** Standard error of the aggregate */
  standardError: number;
  /** Effective privacy budget consumed */
  effectiveEpsilon: number;
  /** Nodes contributing */
  nodeCount: number;
  /** Total samples across all nodes */
  totalSamples: number;
}

// ── Differential Privacy Primitives ─────────────────────────────────────

/**
 * Apply Laplace noise for ε-differential privacy.
 *
 * Laplace(b) has PDF: f(x) = (1/2b)·exp(-|x|/b)
 * where b = sensitivity / epsilon.
 *
 * @param value — raw statistic
 * @param sensitivity — max change in statistic from one sample
 * @param epsilon — privacy budget
 * @param seed — reproducibility seed
 * @returns value + Laplace(0, sensitivity / epsilon)
 */
export function laplaceMechanism(
  value: number,
  sensitivity: number,
  epsilon: number,
  seed?: number,
): { privatizedValue: number; noiseAdded: number } {
  let s = seed ?? Date.now();
  const rng = () => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return s / 0x7FFFFFFF; };

  const scale = sensitivity / epsilon;
  // Inverse CDF sampling: F⁻¹(u) = b·sign(u-0.5)·ln(1-2|u-0.5|)
  const u = rng() - 0.5;
  const noise = -scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));

  return { privatizedValue: value + noise, noiseAdded: noise };
}

/**
 * Apply Gaussian noise for (ε, δ)-differential privacy.
 *
 * Gaussian(0, σ²) where σ = (sensitivity / ε) · √(2·ln(1.25/δ)).
 *
 * @returns value + Gaussian(0, sigma²)
 */
export function gaussianMechanism(
  value: number,
  sensitivity: number,
  epsilon: number,
  delta: number = 1e-5,
  seed?: number,
): { privatizedValue: number; noiseAdded: number } {
  let s = seed ?? Date.now();
  const rng = () => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return s / 0x7FFFFFFF; };

  const sigma = (sensitivity / epsilon) * Math.sqrt(2 * Math.log(1.25 / delta));
  // Box-Muller transform
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  const noise = sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);

  return { privatizedValue: value + noise, noiseAdded: noise };
}

/**
 * Compute the sensitivity of an ATE statistic.
 *
 * For binary treatment and bounded outcome [0, Y_max], the sensitivity
 * of the mean difference is Y_max / min(n_treated, n_control).
 *
 * @param yMax — max possible outcome value
 * @param nTreated — number of treated samples
 * @param nControl — number of control samples
 * @returns sensitivity of the ATE estimate
 */
export function computeATESensitivity(
  yMax: number,
  nTreated: number,
  nControl: number,
): number {
  const minN = Math.min(nTreated, nControl);
  return minN > 0 ? yMax / minN : yMax;
}

// ── Federated DML with DP ───────────────────────────────────────────────

/**
 * Run federated Double Machine Learning with differential privacy.
 *
 * Each node:
 *   1. Runs local DML to compute orthogonalized ATE
 *   2. Applies DP noise to the local ATE
 *   3. Transmits only the privatized value + sample count
 *
 * Coordinator aggregates using inverse-variance weighting.
 *
 * @param nodes — per-node data and statistics
 * @param config — privacy and federation configuration
 * @returns aggregated result with privacy accounting
 */
export function federatedDMLWithDP(
  nodes: Array<{
    nodeId: string;
    localATE: number;
    localSE: number;
    nTreated: number;
    nControl: number;
    outcomeMax: number;
  }>,
  config: FederatedLearningConfig,
): {
  nodes: FederatedNodeResult[];
  aggregation: FederatedAggregation;
} {
  if (nodes.length < config.minNodes) {
    throw new Error(
      `Need at least ${config.minNodes} nodes, got ${nodes.length}`,
    );
  }

  const nodeResults: FederatedNodeResult[] = [];

  for (const node of nodes) {
    const sensitivity = computeATESensitivity(
      node.outcomeMax,
      node.nTreated,
      node.nControl,
    );

    const privatized = config.mechanism === 'gaussian'
      ? gaussianMechanism(node.localATE, sensitivity, config.epsilon, config.delta, config.seed)
      : laplaceMechanism(node.localATE, sensitivity, config.epsilon, config.seed);

    nodeResults.push({
      nodeId: node.nodeId,
      rawValue: node.localATE,
      privatizedValue: privatized.privatizedValue,
      noiseAdded: privatized.noiseAdded,
      sampleCount: node.nTreated + node.nControl,
    });
  }

  // Weighted aggregation: weight ∝ 1/variance, variance ∝ 1/sampleCount
  let weightedSum = 0;
  let totalWeight = 0;
  let totalSamples = 0;

  for (const nr of nodeResults) {
    const weight = nr.sampleCount; // more samples = more weight
    weightedSum += weight * nr.privatizedValue;
    totalWeight += weight;
    totalSamples += nr.sampleCount;
  }

  const aggregateValue = totalWeight > 0 ? weightedSum / totalWeight : 0;
  const standardError = totalWeight > 0
    ? Math.sqrt(nodeResults.reduce((s, nr) => s + nr.sampleCount * (nr.privatizedValue - aggregateValue) ** 2, 0) / totalWeight) / Math.sqrt(nodeResults.length)
    : 0;

  return {
    nodes: nodeResults,
    aggregation: {
      aggregateValue,
      standardError,
      effectiveEpsilon: config.epsilon,
      nodeCount: nodes.length,
      totalSamples,
    },
  };
}

/**
 * Secure aggregation for federated statistics.
 *
 * In a real deployment, secure aggregation would use multi-party
 * computation (MPC) or trusted execution environments (TEE) to
 * ensure the coordinator cannot inspect individual node values,
 * only the final aggregate.
 *
 * This implementation provides the aggregation API; the MPC/TEE
 * layer is a deployment concern.
 *
 * @param privatizedValues — DP-noised values from each node
 * @param weights — node weights (typically sample counts)
 * @returns weighted aggregate
 */
export function secureAggregate(
  privatizedValues: number[],
  weights: number[],
): number {
  let weightedSum = 0;
  let totalWeight = 0;
  for (let i = 0; i < privatizedValues.length; i++) {
    weightedSum += weights[i] * privatizedValues[i];
    totalWeight += weights[i];
  }
  return totalWeight > 0 ? weightedSum / totalWeight : 0;
}

/**
 * Privacy accounting: compute total privacy cost.
 *
 * For sequential composition with k mechanisms, each with privacy
 * budget εᵢ, the total privacy budget is Σεᵢ.
 *
 * @param epsilons — privacy budgets consumed
 * @returns total privacy budget consumed
 */
export function totalPrivacyCost(epsilons: number[]): number {
  return epsilons.reduce((a, b) => a + b, 0);
}
