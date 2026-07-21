/**
 * Core type definitions for Causality Analyzer.
 *
 * All fundamental data structures used across the library are defined here.
 * These types form the universal data contract — every package in the
 * causality-analyzer ecosystem depends on them.
 *
 * @packageDocumentation
 */

// ── Graph Types ────────────────────────────────────────────────────────

/** A weighted, directed edge in a causal graph */
export interface CausalEdge {
  /** Source node name */
  readonly source: string;
  /** Target node name */
  readonly target: string;
  /** Causal strength [0, 1]. 0 = no causal effect, 1 = deterministic cause */
  readonly weight: number;
  /** Whether the edge direction is confirmed (false = CPDAG undirected) */
  readonly directed: boolean;
}

/** A directed acyclic graph representing causal relationships */
export interface CausalGraph {
  /** Ordered list of node names */
  readonly nodes: ReadonlyArray<string>;
  /** Directed edges */
  readonly edges: ReadonlyArray<CausalEdge>;
  /** Computed adjacency matrix (row-major, n×n Float64Array) */
  readonly adjacency?: Float64Array;
}

/** Metadata attached to a causal graph */
export interface GraphMetadata {
  /** Unique graph identifier */
  readonly id: string;
  /** Algorithm used to discover the graph (PC, GES, NOTEARS, etc.) */
  readonly method: string;
  /** Unix timestamp (ms) when the graph was computed */
  readonly computedAt: number;
  /** Parameters used in discovery (alpha, maxDegree, etc.) */
  readonly parameters: Record<string, unknown>;
  /** Estimated confidence in the overall graph structure [0, 1] */
  readonly confidence: number;
}

/** A named and timestamped version of a causal graph */
export interface GraphVersion {
  /** Graph identifier */
  readonly graphId: string;
  /** Monotonically increasing version number */
  readonly version: number;
  /** Unix timestamp (ms) of this version */
  readonly timestamp: number;
  /** Description of what changed from the previous version */
  readonly changeDescription?: string;
}

// ── RCA Result Types ───────────────────────────────────────────────────

/** Evidence item explaining why a node is considered a root cause */
export interface Evidence {
  /** Type of evidence */
  readonly type: 'regression_residual' | 'parent_anomaly' | 'descendant_score' | 'frequent_pattern' | 'causal_effect';
  /** Human-readable description */
  readonly description: string;
  /** Quantitative evidence value (higher = stronger evidence) */
  readonly value: number;
}

/** A single root cause with its score and supporting evidence */
export interface RootCause {
  /** Metric or service name */
  readonly name: string;
  /** Root cause score [0, 1]. Higher = more likely to be the root cause */
  readonly score: number;
  /** Statistical confidence [0, 1] */
  readonly confidence: number;
  /** Rank position (1 = most likely root cause) */
  readonly rank: number;
  /** Supporting evidence items */
  readonly evidence: ReadonlyArray<Evidence>;
}

/** A causal propagation path from a root cause to an anomalous node */
export interface RootCausePath {
  /** Ordered list of nodes from root cause to anomaly */
  readonly nodes: ReadonlyArray<string>;
  /** Path score [0, 1] */
  readonly score: number;
  /** Propagation direction relative to causal graph */
  readonly direction: 'forward' | 'backward';
}

/** Analysis metadata */
export interface AnalysisMetadata {
  /** Method used for RCA */
  readonly method: string;
  /** Unix timestamp (ms) of analysis */
  readonly analyzedAt: number;
  /** Duration of analysis in ms */
  readonly durationMs: number;
  /** Additional method-specific metadata */
  readonly extra: Record<string, unknown>;
}

/** Complete root cause analysis result */
export interface RCAResult {
  /** Root causes sorted by score (descending) */
  readonly rootCauses: ReadonlyArray<RootCause>;
  /** Propagation paths from root causes to anomalies */
  readonly paths: ReadonlyArray<RootCausePath>;
  /** Analysis metadata */
  readonly metadata: AnalysisMetadata;

  /** Serialize to plain JSON object */
  toJSON(): Record<string, unknown>;
}

// ── Detection Types ─────────────────────────────────────────────────────

/** Result from an anomaly detector */
export interface DetectionResult {
  /** Whether any anomaly was detected */
  readonly isAnomalous: boolean;
  /** Per-metric anomaly labels: 1 = anomalous, 0 = normal */
  readonly labels: Float64Array;
  /** Per-metric anomaly scores. Higher = more anomalous */
  readonly scores: Float64Array;
  /** Unix timestamp (ms) of detection */
  readonly timestamp: number;
  /** Detector-specific metadata */
  readonly metadata: Record<string, unknown>;
}

// ── Causal Inference Types ─────────────────────────────────────────────

/** An identified causal estimand */
export interface IdentifiedEstimand {
  /** Type of estimand */
  readonly estimandType: 'nonparametric_ate' | 'nonparametric_nde' | 'nonparametric_nie';
  /** Treatment variable names */
  readonly treatmentVariables: ReadonlyArray<string>;
  /** Outcome variable names */
  readonly outcomeVariables: ReadonlyArray<string>;
  /** Backdoor adjustment sets (keyed by method name) */
  readonly backdoorVariables: Record<string, ReadonlyArray<string>>;
  /** Instrumental variables (if IV method identified) */
  readonly instrumentalVariables: ReadonlyArray<string>;
  /** Frontdoor variables (if frontdoor method identified) */
  readonly frontdoorVariables: ReadonlyArray<string>;
}

/** A causal effect estimate */
export interface CausalEstimate {
  /** Estimated effect value */
  readonly value: number;
  /** The estimand that was estimated */
  readonly targetEstimand: IdentifiedEstimand;
  /** Name of the estimation method */
  readonly methodName: string;
}

// ── Domain Knowledge ────────────────────────────────────────────────────

/** Domain knowledge constraints for causal graph discovery */
export interface DomainKnowledge {
  /** Forbidden edges: (from, to) pairs that must NOT appear */
  readonly forbids?: ReadonlyArray<readonly [string, string]>;
  /** Required edges: (from, to) pairs that MUST appear */
  readonly requires?: ReadonlyArray<readonly [string, string]>;
  /** Nodes that must be root nodes (no incoming edges) */
  readonly rootNodes?: ReadonlyArray<string>;
  /** Nodes that must be leaf nodes (no outgoing edges) */
  readonly leafNodes?: ReadonlyArray<string>;
  /** Temporal lag constraints */
  readonly temporal?: ReadonlyArray<{
    readonly cause: string;
    readonly effect: string;
    readonly minLag: number;
    readonly maxLag: number;
  }>;
}

// ── Pipeline Types ──────────────────────────────────────────────────────

/** Pipeline stages in order of execution */
export enum PipelineStage {
  INGEST = 'ingest',
  DETECT = 'detect',
  GRAPH = 'graph',
  ANALYZE = 'analyze',
  INFER = 'infer',
  VALIDATE = 'validate',
}

// ── Metric Types ────────────────────────────────────────────────────────

/** A query for reading metric data from storage */
export interface MetricQuery {
  /** Start timestamp (inclusive, ms since epoch) */
  readonly start: number;
  /** End timestamp (exclusive, ms since epoch) */
  readonly end: number;
  /** Metric column names to retrieve (empty = all) */
  readonly metrics?: ReadonlyArray<string>;
}

/** A query for retrieving historical RCA results */
export interface ResultQuery {
  /** Start timestamp (inclusive, ms since epoch) */
  readonly start?: number;
  /** End timestamp (exclusive, ms since epoch) */
  readonly end?: number;
  /** Filter by root cause name */
  readonly rootCause?: string;
  /** Maximum number of results */
  readonly limit?: number;
}

// ── Conditional Probability Table ───────────────────────────────────────

/** A conditional probability table entry */
export interface ConditionalProbabilityTable {
  /** Node name */
  readonly node: string;
  /** Parent node names (in order) */
  readonly parents: ReadonlyArray<string>;
  /** Map of parent_state_key → probability */
  readonly entries: Record<string, number>;
}

// ── Regression Model Parameters ─────────────────────────────────────────

/** Parameters for a linear regression model used in HT analysis */
export interface RegressionParams {
  /** Coefficient array (one per parent + intercept) */
  readonly coefficients: ReadonlyArray<number>;
  /** Intercept term */
  readonly intercept: number;
  /** Standard deviation of residuals on training data */
  readonly residualStdDev: number;
}

// ── Visualization Data Types ───────────────────────────────────────────

export interface GraphVizNode {
  id: string; label: string;
  type: 'root_cause' | 'anomaly' | 'intermediate' | 'healthy';
  score: number; isAnomalous: boolean;
}
export interface GraphVisualizationData { nodes: GraphVizNode[]; edges: Array<{ source: string; target: string; weight: number; directed: boolean }>; }

export interface TimeSeriesDataPoint { ts: number; value: number; q10?: number; q90?: number; }
export interface AnomalyRegion { start: number; end: number; severity: 'critical' | 'warning' | 'info'; rootCause?: string; }
export interface TimeSeriesChartData { series: Array<{ name: string; data: TimeSeriesDataPoint[] }>; anomalyRegions: AnomalyRegion[]; }

export interface RankingEntry { rank: number; name: string; score: number; confidence: number; evidence: ReadonlyArray<Evidence>; }
export interface PropagationPath { root: string; path: string[]; score: number; }
export interface RCARankingData { rootCauses: RankingEntry[]; propagationPaths: PropagationPath[]; }
