/**
 * Storage interface contracts for Causality Analyzer.
 *
 * These interfaces define the persistence abstraction layer.
 * Implementations (embed, remote) live in separate packages
 * and are injected via DI at runtime.
 *
 * All async methods accept an optional AbortSignal for cancellation.
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
  readMetrics<S extends TableSchema>(query: MetricQuery): Promise<ColumnarTable<S>>;
  writeDetections(detections: DetectionResult[], signal?: AbortSignal): Promise<void>;
  saveCPT(graphId: string, node: string, cpt: ConditionalProbabilityTable, signal?: AbortSignal): Promise<void>;
  loadCPT(graphId: string, node: string, signal?: AbortSignal): Promise<ConditionalProbabilityTable | null>;
  saveRegressionModel(graphId: string, node: string, model: RegressionParams, signal?: AbortSignal): Promise<void>;
  loadRegressionModel(graphId: string, node: string, signal?: AbortSignal): Promise<RegressionParams | null>;
  saveRCAResult(caseId: string, result: RCAResult, signal?: AbortSignal): Promise<void>;
  queryHistoricalResults(query: ResultQuery): Promise<RCAResult[]>;
  beginTransaction(sessionId: string, signal?: AbortSignal): Promise<void>;
  commitTransaction(sessionId: string, signal?: AbortSignal): Promise<void>;
  rollbackToCheckpoint(sessionId: string, checkpoint: string, signal?: AbortSignal): Promise<void>;
  setCheckpoint(sessionId: string, name: string, signal?: AbortSignal): Promise<void>;
  close(): void;
}

// ── Graph Store ─────────────────────────────────────────────────────────

export interface IGraphStore {
  saveGraph(graph: CausalGraph, metadata: GraphMetadata, signal?: AbortSignal): Promise<string>;
  loadGraph(graphId: string, signal?: AbortSignal): Promise<CausalGraph | null>;
  loadGraphVersion(graphId: string, version: number, signal?: AbortSignal): Promise<CausalGraph | null>;
  listGraphVersions(graphId: string, signal?: AbortSignal): Promise<GraphVersion[]>;
  findSimilarGraphs(graph: CausalGraph, limit: number, signal?: AbortSignal): Promise<CausalGraph[]>;
  close(): void;
}
