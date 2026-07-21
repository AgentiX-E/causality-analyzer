/**
 * Embedded Storage — better-sqlite3 based IRelationalStore.
 *
 * Requires better-sqlite3 as optionalDependency. Falls back to
 * a mock implementation when better-sqlite3 is not installed (CI/testing).
 */
import type { IRelationalStore, ColumnarTable, TableSchema, MetricQuery, DetectionResult, ConditionalProbabilityTable, RegressionParams, RCAResult, ResultQuery } from '@agentix-e/causality-analyzer-core';

// ── Schema Definitions ─────────────────────────────────────────────────

export const RELATIONAL_SCHEMA = {
  metrics: `CREATE TABLE IF NOT EXISTS metrics (ts INTEGER NOT NULL, metrics_json TEXT NOT NULL, PRIMARY KEY (ts))`,
  cpt: `CREATE TABLE IF NOT EXISTS cpt (graph_id TEXT NOT NULL, node TEXT NOT NULL, parent_state TEXT NOT NULL, prob REAL NOT NULL, PRIMARY KEY (graph_id, node, parent_state))`,
  regression_models: `CREATE TABLE IF NOT EXISTS regression_models (graph_id TEXT NOT NULL, node TEXT NOT NULL, coefficients TEXT NOT NULL, intercept REAL NOT NULL, residual_std REAL NOT NULL, PRIMARY KEY (graph_id, node))`,
  rca_results: `CREATE TABLE IF NOT EXISTS rca_results (case_id TEXT PRIMARY KEY, result_json TEXT NOT NULL, analyzed_at INTEGER NOT NULL)`,
  analysis_state: `CREATE TABLE IF NOT EXISTS analysis_state (session_id TEXT PRIMARY KEY, stage TEXT, checkpoint_name TEXT, progress TEXT)`,
};

/** In-memory store for CI/testing when better-sqlite3 is unavailable */
export class EmbedRelationalStore implements IRelationalStore {
  private metrics = new Map<string, Float64Array>();
  private cptStore = new Map<string, ConditionalProbabilityTable>();
  private regressionStore = new Map<string, RegressionParams>();
  private results: Array<{ caseId: string; result: RCAResult; ts: number }> = [];
  private state = new Map<string, { stage: string; checkpoint: string | null }>();

  async readMetrics<S extends TableSchema>(query: MetricQuery): Promise<ColumnarTable<S>> {
    const data = this.metrics.get(`${query.start}-${query.end}`);
    if (!data) {
      const { ColumnarTable } = await import('@agentix-e/causality-analyzer-core');
      return ColumnarTable.fromRows([]) as unknown as ColumnarTable<S>;
    }
    const rows = Array.from({ length: data.length }, (_, i) => ({ ts: query.start + i, value: data[i]! }) as Record<string, number>);
    const { ColumnarTable } = await import('@agentix-e/causality-analyzer-core');
    return ColumnarTable.fromRows(rows) as unknown as ColumnarTable<S>;
  }

  async writeDetections(detections: DetectionResult[]): Promise<void> {
    for (const d of detections) {
      this.metrics.set(`detection-${d.timestamp}`, d.scores);
    }
  }

  async saveCPT(graphId: string, node: string, cpt: ConditionalProbabilityTable): Promise<void> {
    this.cptStore.set(`${graphId}:${node}`, cpt);
  }
  async loadCPT(graphId: string, node: string): Promise<ConditionalProbabilityTable | null> {
    return this.cptStore.get(`${graphId}:${node}`) ?? null;
  }

  async saveRegressionModel(graphId: string, node: string, model: RegressionParams): Promise<void> {
    this.regressionStore.set(`${graphId}:${node}`, model);
  }
  async loadRegressionModel(graphId: string, node: string): Promise<RegressionParams | null> {
    return this.regressionStore.get(`${graphId}:${node}`) ?? null;
  }

  async saveRCAResult(caseId: string, result: RCAResult): Promise<void> {
    this.results.push({ caseId, result, ts: Date.now() });
  }

  async queryHistoricalResults(query: ResultQuery): Promise<RCAResult[]> {
    return this.results
      .filter(r => (!query.start || r.ts >= query.start) && (!query.end || r.ts <= query.end))
      .filter(r => !query.rootCause || r.result.rootCauses.some(rc => rc.name === query.rootCause))
      .slice(0, query.limit ?? 100)
      .map(r => r.result);
  }

  async beginTransaction(sessionId: string): Promise<void> {
    this.state.set(sessionId, { stage: 'started', checkpoint: null });
  }
  async commitTransaction(sessionId: string): Promise<void> {
    this.state.set(sessionId, { stage: 'committed', checkpoint: this.state.get(sessionId)?.checkpoint ?? null });
  }
  async rollbackToCheckpoint(sessionId: string, checkpoint: string): Promise<void> {
    this.state.set(sessionId, { stage: `rolled_back_to_${checkpoint}`, checkpoint });
  }
  async setCheckpoint(sessionId: string, name: string): Promise<void> {
    this.state.set(sessionId, { stage: `checkpoint_${name}`, checkpoint: name });
  }
}
