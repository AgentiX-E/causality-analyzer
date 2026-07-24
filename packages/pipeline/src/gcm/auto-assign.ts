/**
 * GCM Auto Mechanism Assignment.
 *
 * Automatically selects the most appropriate causal mechanism (ANM, PNL,
 * or discrete) and ML model for each node in a Structural Causal Model,
 * based on data characteristics.
 *
 * This mirrors DoWhy's `gcm.auto.assign_causal_mechanisms()` but is
 * tailored for TypeScript with explicit selection criteria.
 *
 * Selection criteria:
 *   - Continuous, high skew (> 2)    → PostNonlinearModel
 *   - Continuous, low skew (≤ 2)     → AdditiveNoiseModel
 *   - Discrete (< 10 unique values)  → DiscreteAdditiveNoiseModel
 *   - Small samples (n < 30)         → fallback to simpler model
 *
 * @packageDocumentation
 */
import type { CausalGraph } from '../graph/causal-graph.js';
import type { StructuralCausalModel } from '../gcm/structural-causal-model.js';

// ── Types ───────────────────────────────────────────────────────────

export type MechanismType = 'anm' | 'pnl' | 'discrete' | 'constant';

export interface MechanismAssignment {
  /** Node name */
  node: string;
  /** Assigned mechanism type */
  mechanism: MechanismType;
  /** Confidence in the assignment [0, 1] */
  confidence: number;
  /** Reason for the assignment */
  reason: string;
  /** Parent names for the mechanism */
  parents: string[];
}

export interface AutoAssignResult {
  /** Per-node assignments */
  assignments: MechanismAssignment[];
  /** Whether any nodes needed fallback assignment */
  hasFallbacks: boolean;
  /** Summary statistics */
  summary: {
    anmCount: number;
    pnlCount: number;
    discreteCount: number;
    constantCount: number;
    totalNodes: number;
  };
}

// ── Auto Assignment ──────────────────────────────────────────────────

/**
 * Automatically assign causal mechanisms based on data characteristics.
 *
 * @param graph — CausalGraph defining parent-child relationships
 * @param data — data matrix (n rows × d columns)
 * @param nodeIndex — mapping from node name to column index
 * @param options — configuration
 */
export function autoAssignMechanisms(
  graph: CausalGraph,
  data: number[][],
  nodeIndex: Map<string, number>,
  options: {
    /** Minimum unique values to consider continuous (default: 10) */
    discreteThreshold?: number;
    /** Skewness threshold for PNL vs ANM (default: 2.0) */
    skewThreshold?: number;
    /** Minimum sample size for full analysis (default: 30) */
    minSamples?: number;
  } = {},
): AutoAssignResult {
  const discreteThreshold = options.discreteThreshold ?? 10;
  const skewThreshold = options.skewThreshold ?? 2.0;
  const minSamples = options.minSamples ?? 30;
  const n = data.length;

  const assignments: MechanismAssignment[] = [];

  for (const node of graph.nodes) {
    const colIdx = nodeIndex.get(node);
    if (colIdx === undefined) {
      assignments.push({
        node,
        mechanism: 'constant',
        confidence: 0.5,
        reason: 'No data column — assuming constant mechanism',
        parents: graph.parents(node),
      });
      continue;
    }

    const parents = graph.parents(node);

    // Extract node values
    const values = data.map(r => r[colIdx] ?? 0);

    // Check for constant values
    const uniqueValues = new Set(values);
    const nUnique = uniqueValues.size;

    // Case 1: Constant / near-constant
    if (nUnique <= 1) {
      assignments.push({
        node,
        mechanism: 'constant',
        confidence: 1.0,
        reason: `Only ${nUnique} unique value(s) — constant mechanism`,
        parents,
      });
      continue;
    }

    // Case 2: Discrete (few unique values)
    if (nUnique < discreteThreshold) {
      assignments.push({
        node,
        mechanism: 'discrete',
        confidence: 1 - (nUnique / discreteThreshold) * 0.2,
        reason: `Discrete variable — ${nUnique} unique values (threshold: ${discreteThreshold})`,
        parents,
      });
      continue;
    }

    // Case 3: Continuous — choose ANM or PNL based on skewness
    const mean = values.reduce((s, v) => s + v, 0) / n;
    let m2 = 0, m3 = 0;
    for (const v of values) {
      const d = v - mean;
      m2 += d * d;
      m3 += d * d * d;
    }
    const variance = m2 / n;
    const std = Math.sqrt(Math.max(1e-10, variance));
    const skewness = m3 / (n * std * std * std + 1e-10);

    // Small sample fallback
    if (n < minSamples) {
      assignments.push({
        node,
        mechanism: 'anm',
        confidence: 0.6,
        reason: `Small sample (n=${n} < ${minSamples}) — fallback to ANM`,
        parents,
      });
      continue;
    }

    if (Math.abs(skewness) > skewThreshold) {
      assignments.push({
        node,
        mechanism: 'pnl',
        confidence: Math.min(0.95, Math.abs(skewness) / (skewThreshold * 2)),
        reason: `High skewness (${skewness.toFixed(2)} > ${skewThreshold}) — PNL mechanism`,
        parents,
      });
    } else {
      assignments.push({
        node,
        mechanism: 'anm',
        confidence: Math.min(0.95, 1 - Math.abs(skewness) / skewThreshold),
        reason: `Low skewness (${skewness.toFixed(2)} ≤ ${skewThreshold}) — ANM mechanism`,
        parents,
      });
    }
  }

  const hasFallbacks = assignments.some(a => a.mechanism === 'constant' || a.confidence < 0.7);

  return {
    assignments,
    hasFallbacks,
    summary: {
      anmCount: assignments.filter(a => a.mechanism === 'anm').length,
      pnlCount: assignments.filter(a => a.mechanism === 'pnl').length,
      discreteCount: assignments.filter(a => a.mechanism === 'discrete').length,
      constantCount: assignments.filter(a => a.mechanism === 'constant').length,
      totalNodes: graph.nodes.length,
    },
  };
}
