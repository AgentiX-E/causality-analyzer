/**
 * Storage interface contracts for Causality Analyzer.
 *
 * These interfaces define the persistence abstraction layer.
 * Implementations (embed, remote) live in separate packages
 * and are injected via DI at runtime.
 *
 * All async methods accept an optional AbortSignal for cancellation.
 *
 * Interface Segregation Principle:
 * - IMetricStore — time-series metrics + anomaly detections
 * - IModelStore  — CPT + regression model CRUD
 * - IResultStore — RCA result persistence
 * - ITransactionStore — SAVEPOINT-based transaction management
 * - IRelationalStore — composition of all four + lifecycle
 *
 * @packageDocumentation
 */

import type { ColumnarTable, TableSchema } from '../table/index.js';
import type {
  ConditionalProbabilityTable,
  CausalGraph,
  DetectionResult,
  GraphMetadata,
  GraphVersion,
  MetricQuery,
  RCAResult,
  RegressionParams,
  ResultQuery,
} from '../types/index.js';

// ── Focused Interfaces (ISP) ────────────────────────────────────────────

/**
 * Metric storage — time-series data and anomaly detection results.
 *
 * Typical backends: better-sqlite3 (embed), PostgreSQL (remote).
 */
export interface IMetricStore {
  /** Read metric data within a time window */
  readMetrics<S extends TableSchema>(query: MetricQuery): Promise<ColumnarTable<S>>;

  /** Persist anomaly detection results */
  writeDetections(detections: DetectionResult[], signal?: AbortSignal): Promise<void>;
}

/**
 * Model storage — CPT and regression model parameters.
 *
 * Typical backends: better-sqlite3 (embed), PostgreSQL (remote).
 */
export interface IModelStore {
  /** Save a conditional probability table for a Bayesian network node */
  saveCPT(graphId: string, node: string, cpt: ConditionalProbabilityTable, signal?: AbortSignal): Promise<void>;

  /** Load a CPT for a specific graph/node combination */
  loadCPT(graphId: string, node: string, signal?: AbortSignal): Promise<ConditionalProbabilityTable | null>;

  /** Save regression model parameters for HT-based RCA */
  saveRegressionModel(graphId: string, node: string, model: RegressionParams, signal?: AbortSignal): Promise<void>;

  /** Load regression model parameters */
  loadRegressionModel(graphId: string, node: string, signal?: AbortSignal): Promise<RegressionParams | null>;
}

/**
 * Result storage — RCA analysis result persistence and querying.
 *
 * Typical backends: better-sqlite3 (embed), PostgreSQL (remote).
 */
export interface IResultStore {
  /** Persist an RCA analysis result */
  saveRCAResult(caseId: string, result: RCAResult, signal?: AbortSignal): Promise<void>;

  /** Query historical RCA results by time range and optional filters */
  queryHistoricalResults(query: ResultQuery): Promise<RCAResult[]>;
}

/**
 * Transaction management — SAVEPOINT-based nested transaction support.
 */
export interface ITransactionStore {
  /** Begin a new transaction (maps to SAVEPOINT in SQL databases) */
  beginTransaction(sessionId: string, signal?: AbortSignal): Promise<void>;

  /** Commit the current transaction */
  commitTransaction(sessionId: string, signal?: AbortSignal): Promise<void>;

  /** Roll back to a previously set checkpoint */
  rollbackToCheckpoint(sessionId: string, checkpoint: string, signal?: AbortSignal): Promise<void>;

  /** Set a named checkpoint within the current transaction */
  setCheckpoint(sessionId: string, name: string, signal?: AbortSignal): Promise<void>;
}

// ── Composite Interface ─────────────────────────────────────────────────

/**
 * Relational storage — composition of metric, model, result, and transaction stores.
 *
 * Consumers that only need metrics should depend on IMetricStore,
 * not IRelationalStore, to follow the Interface Segregation Principle.
 * IRelationalStore exists for convenience when a full relational store is needed.
 */
export interface IRelationalStore extends IMetricStore, IModelStore, IResultStore, ITransactionStore {
  /** Close the store and release all resources */
  close(): void;
  /** Health check: true if store is accessible */
  healthCheck?(): Promise<boolean> | boolean;
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
  saveGraph(graph: CausalGraph, metadata: GraphMetadata, signal?: AbortSignal): Promise<string>;
  loadGraph(graphId: string, signal?: AbortSignal): Promise<CausalGraph | null>;
  loadGraphVersion(graphId: string, version: number, signal?: AbortSignal): Promise<CausalGraph | null>;
  listGraphVersions(graphId: string, signal?: AbortSignal): Promise<GraphVersion[]>;
  findSimilarGraphs(graph: CausalGraph, limit: number, signal?: AbortSignal): Promise<CausalGraph[]>;
  close(): void;
}
