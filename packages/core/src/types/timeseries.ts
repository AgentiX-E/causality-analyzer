/**
 * Time-series causal discovery types for PCMCI+ and related algorithms.
 *
 * Extends the core type system with temporal semantics: lagged edges,
 * contemporaneous edges, CPDAG orientation marks, and PCMCI+-specific
 * configuration and result types.
 *
 * Reference: Runge, J. (2020). "Discovering contemporaneous and lagged
 *   causal relations in autocorrelated nonlinear time series datasets."
 *   Proceedings of the 36th UAI Conference.
 *
 * @packageDocumentation
 */

// ── Edge Orientation Marks ──────────────────────────────────────────────

/**
 * Edge endpoint mark following the Tetrad / PAG convention.
 *
 * - `'tail'`  : standard directed edge tail  (—)
 * - `'arrow'` : directed edge arrowhead      (>)
 * - `'circle'`: partially directed (unknown)  (o)
 *
 * A fully directed edge A → B uses sourceMark='tail', targetMark='arrow'.
 * An undirected edge A — B uses sourceMark='tail', targetMark='tail'.
 * A partially oriented edge A o→ B uses sourceMark='circle', targetMark='arrow'.
 */
export type EdgeMark = 'tail' | 'arrow' | 'circle';

// ── Time-Series Graph Types ─────────────────────────────────────────────

/**
 * A directed or partially directed edge in a time-series causal graph.
 *
 * Unlike {@link CausalEdge}, this type carries a `lag` field:
 * - lag = 0: contemporaneous edge (same time step, may be partially oriented)
 * - lag > 0: lagged edge (past → present, always fully directed)
 *
 * Negative lags are never produced by the algorithm (temporal ordering is
 * explicit via the lag field).
 */
export interface TimeSeriesEdge {
  /** Source variable name (root name without lag suffix) */
  readonly source: string;
  /** Target variable name (root name without lag suffix) */
  readonly target: string;
  /**
   * Time lag:
   *   - 0 = contemporaneous (X_i[t] causes X_j[t])
   *   - >0 = X_i[t - lag] causes X_j[t]
   */
  readonly lag: number;
  /**
   * Causal strength in [-1, 1].
   * Sign indicates direction of effect; magnitude indicates normalized
   * effect size. Computed from the partial correlation or conditional
   * mutual information of the final CI test.
   */
  readonly strength: number;
  /** p-value from the conditional independence test */
  readonly pValue: number;
  /** Source endpoint mark (typically 'tail' for lagged, variable for contemporaneous) */
  readonly sourceMark: EdgeMark;
  /** Target endpoint mark (typically 'arrow' for lagged, variable for contemporaneous) */
  readonly targetMark: EdgeMark;
  /**
   * Discovery phase:
   *   - `'pc1'`: discovered in Phase 1 (PC₁ skeleton)
   *   - `'mci'`: survived Phase 2 (MCI+ conditioning)
   */
  readonly phase: 'pc1' | 'mci';
}

/**
 * Complete time-series causal graph with both lagged and contemporaneous
 * edges.
 *
 * The graph is always a CPDAG for the contemporaneous layer (edges may be
 * partially oriented). Lagged edges are always fully directed (arrow target)
 * because the past cannot depend on the future.
 */
export interface TimeSeriesGraph {
  /** Ordered list of variable names */
  readonly nodes: ReadonlyArray<string>;
  /** All edges (lagged + contemporaneous), each with orientation marks */
  readonly edges: ReadonlyArray<TimeSeriesEdge>;
  /** Maximum lag considered by the discovery algorithm */
  readonly tauMax: number;
  /** Number of time steps in the input data */
  readonly timeSteps: number;
  /**
   * Whether the contemporaneous layer is a CPDAG.
   * Always true for PCMCI+ output; may be false for fully oriented output
   * from other time-series algorithms.
   */
  readonly isCPDAG: boolean;
}

// ── PCMCI+ Configuration & Result ───────────────────────────────────────

/**
 * Valid conditional independence test backends for PCMCI+.
 *
 * - `'parcorr'`: Partial correlation (Fisher Z transform). Fast, O(n).
 *   Best for linear-Gaussian data. Default backend.
 * - `'cmiknn'`: Conditional mutual information via k-NN KSG estimator.
 *   Detects nonlinear dependencies. O(n²).
 * - `'gsquared'`: G-squared log-likelihood ratio test.
 *   Uses regression residuals for continuous data; permutation-based
 *   p-values for non-Gaussian regimes.
 */
export type CIBackend = 'parcorr' | 'cmiknn' | 'gsquared';

/**
 * Full configuration for the PCMCI+ algorithm.
 *
 * All fields have reasonable defaults; pass a Partial<> to pcmciPlusAlgorithm().
 */
export interface PCMCIPlusConfig {
  /**
   * Significance level for conditional independence tests.
   * @default 0.05
   */
  readonly alpha: number;
  /**
   * Maximum time lag to consider.
   * @default min(5, Math.floor(T / 20)) — auto-computed from data length
   */
  readonly tauMax: number;
  /**
   * Maximum number of conditioning variables in PC₁ phase.
   * @default 5
   */
  readonly maxCondVars: number;
  /**
   * Which CI test backend to use.
   * @default 'parcorr'
   */
  readonly ciBackend: CIBackend;
  /**
   * Number of nearest neighbors for the CMIknn backend.
   * Only meaningful when ciBackend === 'cmiknn'.
   * @default 5
   */
  readonly knnK?: number;
  /**
   * Number of permutations for building the null distribution in
   * CMIknn and Gsquared backends.
   * @default 200 (CMIknn), 500 (Gsquared)
   */
  readonly nPermutations?: number;
}

/**
 * Structured summary of a PCMCI+ execution result.
 */
export interface PCMCIPlusEdgeSummary {
  /** Total number of edges in the discovered graph */
  readonly totalEdges: number;
  /** Number of edges with lag > 0 */
  readonly laggedEdges: number;
  /** Number of edges with lag = 0 */
  readonly contemporaneousEdges: number;
  /** Number of fully directed contemporaneous edges */
  readonly directedEdges: number;
  /** Number of partially oriented contemporaneous edges (CPDAG) */
  readonly partiallyDirectedEdges: number;
}

/**
 * The result of running pcmciPlusAlgorithm().
 */
export interface PCMCIPlusResult {
  /** The discovered time-series causal graph (CPDAG for contemporaneous layer) */
  readonly graph: TimeSeriesGraph;
  /**
   * Per-variable parent sets from Phase 1 (PC₁).
   * Key = target variable name.
   * Value = list of significant parent edges before MCI+ pruning.
   */
  readonly parents: Readonly<Map<string, ReadonlyArray<TimeSeriesEdge>>>;
  /** Counts by edge category */
  readonly summary: Readonly<PCMCIPlusEdgeSummary>;
  /** The configuration used (with defaults filled in) */
  readonly config: PCMCIPlusConfig;
  /** Wall-clock time from start to end of algorithm execution (ms) */
  readonly runtimeMs: number;
}

// ── Observer / Callback ─────────────────────────────────────────────────

/**
 * Callback invoked for every conditional independence test during PCMCI+
 * execution. Useful for debugging, logging, or real-time visualization.
 */
export type CITestObserver = (
  /** Source variable name */
  source: string,
  /** Target variable name */
  target: string,
  /** Time lag (0 = contemporaneous, >0 = lagged) */
  lag: number,
  /** Names of conditioning variables */
  condSet: ReadonlyArray<string>,
  /** Computed p-value */
  pValue: number,
  /** Raw test statistic (Fisher Z, CMI, or G²) */
  testStatistic: number,
) => void;

// ── CI Test Result ──────────────────────────────────────────────────────

/**
 * Standardized result of a conditional independence test.
 * All CI backends (ParCorr, CMIknn, Gsquared) return this type.
 */
export interface CITestResult {
  /** Two-sided p-value of the test */
  readonly pValue: number;
  /** Raw test statistic (partial correlation |ρ|, CMI value, or G²) */
  readonly testStatistic: number;
}

// ── CPDAG Orientation ───────────────────────────────────────────────────

/**
 * Input for the orientCPDAG() function.
 * Represents the contemporaneous adjacency structure with separation sets.
 */
export interface CPDAGInput {
  /**
   * List of adjacent pairs (undirected edges) in the contemporaneous skeleton.
   * Each entry is a [nodeA, nodeB] pair.
   */
  readonly adjacencies: ReadonlyArray<readonly [string, string]>;
  /**
   * Separation sets from the skeleton phase.
   * Key format: "i|j" (lexicographically ordered pair).
   * Value: set of variable names that d-separate i from j.
   */
  readonly sepSets: Readonly<Map<string, ReadonlySet<string>>>;
  /** All node names in the contemporaneous layer */
  readonly nodes: ReadonlyArray<string>;
}
