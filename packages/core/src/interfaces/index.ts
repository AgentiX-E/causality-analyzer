/**
 * Storage interface contracts for Causality Analyzer.
 *
 * These interfaces define the persistence abstraction layer.
 * Implementations (embed, remote) live in separate packages
 * and are injected via DI at runtime.
 *
 * @packageDocumentation
 */

import type { ColumnarTable, TableSchema } from '../table/index.js';
import type {
  ConditionalProbabilityTable,
  CausalEdge,
  CausalGraph,
  DetectionResult,
  GraphMetadata,
  GraphVersion,
  MetricQuery,
  RCAResult,
  RegressionParams,
  ResultQuery,
} from '../types/index.js';

// ── Relational Store ────────────────────────────────────────────────────

/**
 * Relational storage interface.
 *
 * Responsible for: time-series metrics, model parameters (CPT, regression),
 * RCA results, and transaction management (SAVEPOINT).
 *
 * Typical backends:
 * - Embedded: better-sqlite3 (in-process, WAL mode)
 * - Distributed: PostgreSQL or any PG-wire compatible database
 */
export interface IRelationalStore {
  // ── Time-series data ───────────────────────────────────────────────
  /** Read metric data within a time window */
  readMetrics<S extends TableSchema>(query: MetricQuery): Promise<ColumnarTable<S>>;

  /** Persist anomaly detection results */
  writeDetections(detections: DetectionResult[]): Promise<void>;

  // ── Model parameters ───────────────────────────────────────────────
  /** Save a conditional probability table for a Bayesian network node */
  saveCPT(graphId: string, node: string, cpt: ConditionalProbabilityTable): Promise<void>;

  /** Load a CPT for a specific graph/node combination */
  loadCPT(graphId: string, node: string): Promise<ConditionalProbabilityTable | null>;

  /** Save regression model parameters for HT-based RCA */
  saveRegressionModel(graphId: string, node: string, model: RegressionParams): Promise<void>;

  /** Load regression model parameters */
  loadRegressionModel(graphId: string, node: string): Promise<RegressionParams | null>;

  // ── RCA results ────────────────────────────────────────────────────
  /** Persist an RCA analysis result */
  saveRCAResult(caseId: string, result: RCAResult): Promise<void>;

  /** Query historical RCA results by time range and optional filters */
  queryHistoricalResults(query: ResultQuery): Promise<RCAResult[]>;

  // ── Transaction management ─────────────────────────────────────────
  /** Begin a new transaction (maps to SAVEPOINT in SQL databases) */
  beginTransaction(sessionId: string): Promise<void>;

  /** Commit the current transaction */
  commitTransaction(sessionId: string): Promise<void>;

  /** Roll back to a previously set checkpoint */
  rollbackToCheckpoint(sessionId: string, checkpoint: string): Promise<void>;

  /** Set a named checkpoint within the current transaction */
  setCheckpoint(sessionId: string, name: string): Promise<void>;

  /** Close the store and release all resources */
  close(): void;
}

// ── Graph Store ─────────────────────────────────────────────────────────

/**
 * Graph storage interface.
 *
 * Responsible for: causal graph CRUD, graph versioning, graph similarity search.
 *
 * Typical backends:
 * - Embedded: overgraph (Rust/napi-rs, LSM-tree, temporal edges)
 * - Distributed: Neo4j or any Cypher/Bolt-compatible graph database
 */
export interface IGraphStore {
  /** Persist a causal graph with metadata. Returns the assigned graph ID. */
  saveGraph(graph: CausalGraph, metadata: GraphMetadata): Promise<string>;

  /** Load the latest version of a causal graph by ID */
  loadGraph(graphId: string): Promise<CausalGraph | null>;

  /** Load a specific version of a causal graph */
  loadGraphVersion(graphId: string, version: number): Promise<CausalGraph | null>;

  /** List all versions of a causal graph */
  listGraphVersions(graphId: string): Promise<GraphVersion[]>;

  /**
   * Find causal graphs structurally similar to the given graph.
   * Used for detecting structural drift over time.
   */
  findSimilarGraphs(graph: CausalGraph, limit: number): Promise<CausalGraph[]>;

  /** Close the store and release all resources */
  close(): void;
}
