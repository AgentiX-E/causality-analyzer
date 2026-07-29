/**
 * Distributed computing types for I8 — stateless Worker architecture
 * with vector clock versioning and Fisher's method CI merge.
 *
 * All persistent state lives in external Raft clusters (SQL + Graph).
 * Causality Analyzer Workers are pure computation units with zero
 * local durable state. This enables arbitrary horizontal scaling with
 * no data migration on node addition.
 *
 * @packageDocumentation
 */

// ── Vector Clock ────────────────────────────────────────────────────────

/**
 * Vector clock: Map<workerId, monotonically-increasing sequence number>.
 *
 * Used for tracking causal history of distributed graph versions.
 * A version A "happened-before" version B if every worker's sequence
 * in A is <= that in B, and at least one is strictly <.
 *
 * Reference: Lamport, L. (1978). "Time, Clocks, and the Ordering of
 *   Events in a Distributed System." CACM 21(7):558–565.
 *            Mattern, F. (1989). "Virtual Time and Global States of
 *   Distributed Systems."
 */
export type VectorClock = Readonly<Record<string, number>>;

/** Ordering relationship between two vector clocks */
export type ClockOrder = 'before' | 'after' | 'concurrent' | 'equal';

/**
 * Compare two vector clocks to determine their causal relationship.
 *
 * @returns 'before' if a happened-before b, 'after' if b happened-before a,
 *          'concurrent' if neither dominates, 'equal' if identical.
 */
export function compareClocks(a: VectorClock, b: VectorClock): ClockOrder {
  const allKeys = new Set([...Object.keys(a), ...Object.keys(b)]);
  let aGte = true;
  let bGte = true;

  for (const wid of allKeys) {
    const aSeq = a[wid] ?? 0;
    const bSeq = b[wid] ?? 0;
    if (aSeq > bSeq) bGte = false;
    if (bSeq > aSeq) aGte = false;
  }

  if (aGte && bGte) return 'equal';
  if (aGte) return 'after';    // a dominates b → a is after b
  if (bGte) return 'before';   // b dominates a → a is before b
  return 'concurrent';
}

/**
 * Merge two vector clocks by taking the pairwise maximum.
 * Used when reconciling graph versions from different workers.
 *
 * @returns merged clock with max sequence per worker
 */
export function mergeClocks(a: VectorClock, b: VectorClock): VectorClock {
  const merged: Record<string, number> = { ...a };
  for (const [wid, seq] of Object.entries(b)) {
    merged[wid] = Math.max(merged[wid] ?? 0, seq);
  }
  return merged;
}

/**
 * Increment a worker's entry in a vector clock.
 * Used when a worker publishes a new graph version.
 */
export function incrementClock(clock: VectorClock, workerId: string): VectorClock {
  const next: Record<string, number> = { ...clock };
  next[workerId] = (next[workerId] ?? 0) + 1;
  return next;
}

// ── Distributed CI Task ──────────────────────────────────────────────────

/**
 * A single conditional independence test dispatched to a Worker.
 *
 * The Coordinator partitions the full CI test matrix (all variable
 * pairs × all condition set sizes) into individual tasks. Each task
 * specifies exactly which (source, target, lag, condSet) to test.
 */
export interface DistributedCITask {
  /** Unique task identifier */
  readonly taskId: string;
  /** Source variable index */
  readonly source: number;
  /** Target variable index */
  readonly target: number;
  /** Time lag (0 = contemporaneous, >0 = lagged; for PCMCI+) */
  readonly lag: number;
  /** Indices of conditioning variables */
  readonly condSet: ReadonlyArray<number>;
  /** Partition key for SQL shard routing (if applicable) */
  readonly partitionKey?: string;
  /** Significance level */
  readonly alpha: number;
  /** CI test backend to use */
  readonly ciBackend: 'parcorr' | 'cmiknn' | 'gsquared';
}

/**
 * Result of a single CI test from a Worker.
 */
export interface DistributedCIResult {
  /** Matching task identifier */
  readonly taskId: string;
  /** Worker that performed the test */
  readonly workerId: string;
  /** Source variable index */
  readonly source: number;
  /** Target variable index */
  readonly target: number;
  /** Time lag */
  readonly lag: number;
  /** Conditioning set used */
  readonly condSet: ReadonlyArray<number>;
  /** Computed p-value */
  readonly pValue: number;
  /** Raw test statistic */
  readonly testStatistic: number;
  /** Sample size for the test (for weighted merging) */
  readonly sampleSize: number;
  /** Wall-clock time for this test (ms) */
  readonly runtimeMs: number;
}

/**
 * A batch of CI tasks dispatched together to a Worker.
 */
export interface DistributedCITaskBatch {
  /** Batch identifier */
  readonly batchId: string;
  /** Tasks in this batch */
  readonly tasks: ReadonlyArray<DistributedCITask>;
  /** Target worker */
  readonly workerId: string;
  /** Columns needed from SQL cluster (for pushdown optimization) */
  readonly requiredColumns: ReadonlyArray<number>;
}

// ── Distributed Graph Version ───────────────────────────────────────────

/**
 * Metadata for a distributed graph version.
 * Uses vector clock instead of monotonic integer version.
 */
export interface DistributedGraphVersion {
  /** Unique graph identifier */
  readonly graphId: string;
  /** Vector clock at this version */
  readonly vectorClock: VectorClock;
  /** Workers that contributed edges to this version */
  readonly contributors: ReadonlyArray<string>;
  /** Discovery method */
  readonly method: string;
  /** Unix timestamp (ms) */
  readonly computedAt: number;
  /** Algorithm parameters */
  readonly parameters: Record<string, unknown>;
}

/**
 * Distributed CI test batch result from a single Worker.
 */
export interface DistributedCIBatchResult {
  /** Batch identifier */
  readonly batchId: string;
  /** Worker that executed the batch */
  readonly workerId: string;
  /** Per-task results */
  readonly results: ReadonlyArray<DistributedCIResult>;
  /** Worker's current vector clock after batch */
  readonly vectorClock: VectorClock;
  /** Total wall-clock time for the batch (ms) */
  readonly batchRuntimeMs: number;
}

// ── Cluster Configuration ───────────────────────────────────────────────

/** SQL cluster mode: redundancy or sharding */
export type ClusterMode = 'redundancy' | 'sharding';

/** SQL cluster configuration */
export interface SQLClusterConfig {
  /** Deployment mode */
  readonly mode: ClusterMode;
  /** Node URIs (pg-wire compatible) */
  readonly nodes: ReadonlyArray<string>;
  /** Shard key when mode = 'sharding' */
  readonly shardKey?: 'partition_key' | 'hash';
  /** Raft consistency level */
  readonly consistencyLevel: 'strong' | 'eventual';
  /** Read preference */
  readonly readPreference: 'leader' | 'follower' | 'any';
}

/** Graph cluster configuration */
export interface GraphClusterConfig {
  /** Deployment mode */
  readonly mode: ClusterMode;
  /** Node URIs (cypher compatible) */
  readonly nodes: ReadonlyArray<string>;
  /** Shard key when mode = 'sharding' */
  readonly shardKey?: 'graphId';
  /** Raft consistency level */
  readonly consistencyLevel: 'strong';
  /** Read preference */
  readonly readPreference: 'leader' | 'follower' | 'any';
}

// ── Distributed Discovery Configuration ─────────────────────────────────

/**
 * Full configuration for distributed causal discovery.
 * SQL and Graph clusters are independently configurable.
 */
export interface DistributedDiscoveryConfig {
  /** SQL cluster configuration */
  readonly sql: SQLClusterConfig;
  /** Graph cluster configuration */
  readonly graph: GraphClusterConfig;
  /** Worker configuration */
  readonly workers: {
    /** Number of stateless workers to use */
    readonly count: number;
    /** Task distribution strategy */
    readonly taskStrategy: 'round-robin' | 'least-loaded' | 'partition-aware';
    /** CI test backend for all workers */
    readonly ciBackend: 'parcorr' | 'cmiknn' | 'gsquared';
  };
  /** Coordinator configuration */
  readonly coordinator: {
    /** Merge strategy for combining CI test results */
    readonly mergeStrategy: 'fisher-method' | 'weighted-mean' | 'majority-vote';
    /** Consensus threshold: fraction of workers that must agree */
    readonly consensusThreshold: number;
    /** Conflict resolution for contradictory edge directions */
    readonly conflictResolution: 'fisher-method' | 'most-recent' | 'manual';
  };
  /** Significance level */
  readonly alpha: number;
  /** Maximum conditioning set size */
  readonly maxCondVars: number;
}

// ── Federated Learning Types ────────────────────────────────────────────

/**
 * Federated learning configuration with differential privacy.
 */
export interface FederatedConfig {
  /** Privacy budget (ε in differential privacy) */
  readonly epsilon: number;
  /** Privacy mechanism */
  readonly privacyMechanism: 'laplace' | 'gaussian';
  /** Sensitivity of the statistic being privatized */
  readonly sensitivity: number;
  /** Minimum number of nodes required for aggregation */
  readonly minNodes: number;
  /** Whether to use secure aggregation (multi-party computation) */
  readonly secureAggregation: boolean;
}

/**
 * Federated statistic with noise for privacy.
 */
export interface FederatedStatistic {
  /** Node identifier */
  readonly nodeId: string;
  /** Raw statistic value */
  readonly value: number;
  /** Noise scale parameter (sensitivity / epsilon) */
  readonly noiseScale: number;
  /** Privatized statistic (value + noise) */
  readonly privatizedValue: number;
  /** Sample count for weighted merging */
  readonly sampleCount: number;
}
