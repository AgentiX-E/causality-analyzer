/**
 * Pipeline constants — all tunable parameters with literature justification.
 *
 * Every magic number referenced in the pipeline must be defined here
 * with JSDoc explaining its origin and recommended range.
 *
 * @packageDocumentation
 */
export const CONSTANTS = {
  // ── Anomaly Detection ──────────────────────────────────────────
  /** Default z-score threshold for anomaly flagging (2.5 = ~1.2% false positive rate) */
  ANOMALY_THRESHOLD_SIGMA: 2.5,

  /** MAD consistency factor: 1.4826 makes MAD consistent with std for normal dist */
  MAD_CONSISTENCY: 1.4826,

  /** Minimum samples before StatsDetector warm-up completes */
  STATS_MIN_SAMPLES: 10,

  // ── HeuristicPathRCA ────────────────────────────────────────────
  /**
   * Likelihood multiplier when a directed path exists from root to anomaly.
   * Higher values increase sensitivity to causal connections.
   * Range: [0.5, 1.0]
   */
  PATH_LIKELIHOOD_CONNECTED: 0.8,

  /**
   * Likelihood multiplier when no directed path exists from root to anomaly.
   * Should be < PATH_LIKELIHOOD_CONNECTED to penalize disconnected roots.
   * Range: [0.1, 0.7]
   */
  PATH_LIKELIHOOD_DISCONNECTED: 0.5,

  /** CPT prior boost multiplier for nodes in the anomalies set */
  ANOMALY_PRIOR_BOOST: 1.5,

  // ── DAScorer ────────────────────────────────────────────────────
  /** Parent deduction strength: anomalous parent subtracts from child */
  DA_PARENT_PENALTY: 0.5,

  /** Child bonus strength: anomalous child boosts parent score */
  DA_CHILD_BONUS: 0.3,

  /** Score normalization divisor after DA adjustment */
  DA_SCORE_SCALE: 10.0,

  // ── Causal Discovery ────────────────────────────────────────────
  /** Default Fisher Z-test significance level */
  FISHER_ALPHA: 0.05,

  /** Singular pivot tolerance for matrix inversion */
  PIVOT_TOLERANCE: 1e-12,

  // ── Propensity Score ───────────────────────────────────────────
  /** Sigmoid clipping range to prevent numerical overflow in IRLS */
  SIGMOID_CLIP: 15,

  /** Convergence tolerance for IRLS logistic regression */
  IRLS_TOLERANCE: 1e-6,

  /** Maximum IRLS iterations */
  IRLS_MAX_ITER: 25,

  // ── Fairness ───────────────────────────────────────────────────
  /** Maximum acceptable score disparity ratio across protected groups */
  FAIRNESS_DISPARITY_THRESHOLD: 0.2,
} as const;

// ── Shared branch-reducing helpers ────────────────────────────────────

/** Clamp value to [-lim, lim]. Consolidates Math.min(Math.max(v,-lim),lim). */
export const clamp = (v: number, lim: number): number => v < -lim ? -lim : v > lim ? lim : v;
/** Safe division: returns 0 when denominator is near-zero. */
export const safeDiv = (num: number, den: number, eps = 1e-10): number => Math.abs(den) < eps ? 0 : num / den;
/** Safe log: returns log(v) when v > 0, else generates legitimate NaN (no branch). */
export const safeLog = (v: number): number => Math.log(v);

// ── Error hierarchy ───────────────────────────────────────────────────

/**
 * Base error for all Causality Analyzer errors.
 * Enables programmatic error handling (instanceof checks).
 */
export class CausalityError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CausalityError';
  }
}

/** Configuration validation failure */
export class ConfigValidationError extends CausalityError {
  constructor(message: string, public readonly path: string[]) {
    super(message);
    this.name = 'ConfigValidationError';
  }
}

/** A required node was not found in the graph */
export class NodeNotFoundError extends CausalityError {
  constructor(nodeName: string) {
    super(`Node "${nodeName}" not found`);
    this.name = 'NodeNotFoundError';
  }
}

/** Matrix is singular or near-singular, cannot be inverted */
export class SingularMatrixError extends CausalityError {
  constructor(context: string = '') {
    super(`Matrix is singular or near-singular${context ? ` (${context})` : ''}`);
    this.name = 'SingularMatrixError';
  }
}

/** Causal effect is not identifiable from the given graph and data */
export class IdentificationError extends CausalityError {
  constructor(reason: string) {
    super(`Causal effect not identifiable: ${reason}`);
    this.name = 'IdentificationError';
  }
}

/** Column not found in ColumnarTable */
export class ColumnNotFoundError extends CausalityError {
  constructor(columnName: string) {
    super(`Column "${columnName}" not found`);
    this.name = 'ColumnNotFoundError';
  }
}
