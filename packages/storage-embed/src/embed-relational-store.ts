/**
 * EmbedRelationalStore — better-sqlite3 backed IRelationalStore.
 *
 * Uses real SQLite via the better-sqlite3 native addon.
 * :memory: mode for CI/testing. Persistent WAL for production.
 * All operations are synchronous internally, wrapped in Promise
 * for IRelationalStore contract compliance.
 */
import Database from 'better-sqlite3';
import type {
  IRelationalStore, MetricQuery, DetectionResult,
  ConditionalProbabilityTable, RegressionParams, RCAResult, ResultQuery,
  ColumnarTable, TableSchema,
} from '@agentix-e/causality-analyzer-core';

// ── Schema ──────────────────────────────────────────────────────────────

export const SQL = {
  metrics:      `CREATE TABLE IF NOT EXISTS metrics (ts INTEGER NOT NULL, value REAL NOT NULL, metric_name TEXT NOT NULL, metadata_json TEXT, PRIMARY KEY (ts, metric_name))`,
  cpt:          `CREATE TABLE IF NOT EXISTS cpt (graph_id TEXT NOT NULL, node TEXT NOT NULL, parent_state TEXT NOT NULL, prob REAL NOT NULL, PRIMARY KEY (graph_id, node, parent_state))`,
  regression:   `CREATE TABLE IF NOT EXISTS regression_models (graph_id TEXT NOT NULL, node TEXT NOT NULL, coefficients TEXT NOT NULL, intercept REAL NOT NULL, residual_std REAL NOT NULL, PRIMARY KEY (graph_id, node))`,
  rca_results:  `CREATE TABLE IF NOT EXISTS rca_results (case_id TEXT PRIMARY KEY, result_json TEXT NOT NULL, analyzed_at INTEGER NOT NULL, root_cause TEXT)`,
  analysis:     `CREATE TABLE IF NOT EXISTS analysis_state (session_id TEXT NOT NULL, stage TEXT NOT NULL, checkpoint_name TEXT, progress TEXT, PRIMARY KEY (session_id, checkpoint_name))`,
};

export interface EmbedStoreOptions {
  /** Path to SQLite database file. Omit for :memory: */
  dbPath?: string;
  /** Enable WAL journal mode (default: true) */
  wal?: boolean;
}

// ── Store Implementation ────────────────────────────────────────────────

export class EmbedRelationalStore implements IRelationalStore {
  private db: any;
  private init: ReturnType<typeof setInterval> | null = null;

  constructor(options: EmbedStoreOptions = {}) {
    this.db = new Database(options.dbPath ?? ':memory:');
    this.db.pragma(options.wal !== false ? 'journal_mode = WAL' : 'journal_mode = DELETE');
    this.db.pragma('synchronous = NORMAL');
    for (const ddl of Object.values(SQL)) this.db.exec(ddl);

    // Prepared statements (compiled once at construction)
    this._stmt = {
      insertMetric:   this.db.prepare('INSERT OR REPLACE INTO metrics (ts, value, metric_name, metadata_json) VALUES (?, ?, ?, ?)'),
      insertCPT:      this.db.prepare('INSERT OR REPLACE INTO cpt (graph_id, node, parent_state, prob) VALUES (?, ?, ?, ?)'),
      loadCPT:        this.db.prepare('SELECT parent_state, prob FROM cpt WHERE graph_id = ? AND node = ? ORDER BY parent_state'),
      insertReg:      this.db.prepare('INSERT OR REPLACE INTO regression_models (graph_id, node, coefficients, intercept, residual_std) VALUES (?, ?, ?, ?, ?)'),
      loadReg:        this.db.prepare('SELECT coefficients, intercept, residual_std FROM regression_models WHERE graph_id = ? AND node = ?'),
      insertRCA:      this.db.prepare('INSERT OR REPLACE INTO rca_results (case_id, result_json, analyzed_at, root_cause) VALUES (?, ?, ?, ?)'),
      queryRCA:       this.db.prepare('SELECT result_json FROM rca_results WHERE (? IS NULL OR analyzed_at >= ?) AND (? IS NULL OR analyzed_at <= ?) AND (? IS NULL OR root_cause = ?) ORDER BY analyzed_at DESC LIMIT ?'),
      upsertState:    this.db.prepare('INSERT OR REPLACE INTO analysis_state (session_id, stage, checkpoint_name, progress) VALUES (?, ?, ?, ?)'),
      readMetrics:    this.db.prepare('SELECT ts, value FROM metrics WHERE ts >= ? AND ts <= ? ORDER BY ts'),
    };
  }

  private _stmt!: Record<string, any>;

  // ── IRelationalStore ──────────────────────────────────────────────

  async readMetrics<S extends TableSchema>(query: MetricQuery): Promise<ColumnarTable<S>> {
    const rows = this._stmt.readMetrics.all(query.start, query.end) as Array<{ ts: number; value: number }>;
    const col = new Float64Array(rows.length);
    for (let i = 0; i < rows.length; i++) { col[i] = rows[i]!.value; }
    const { ColumnarTable } = await import('@agentix-e/causality-analyzer-core');
    return ColumnarTable.fromColumnar({ ts: new Float64Array(rows.map(r => r.ts)), value: col }) as unknown as ColumnarTable<S>;
  }

  async writeDetections(detections: DetectionResult[]): Promise<void> {
    const insert = this._stmt.insertMetric;
    const tx = this.db.transaction((items: DetectionResult[]) => {
      for (const d of items) {
        for (let i = 0; i < d.scores.length; i++) {
          insert.run(d.timestamp, d.scores[i]!, `metric_${i}`, JSON.stringify(d.metadata));
        }
      }
    });
    tx(detections);
  }

  async saveCPT(graphId: string, node: string, cpt: ConditionalProbabilityTable): Promise<void> {
    const insert = this._stmt.insertCPT;
    const tx = this.db.transaction(() => {
      for (const [state, prob] of Object.entries(cpt.entries)) {
        insert.run(graphId, node, state, prob);
      }
    });
    tx();
  }

  async loadCPT(graphId: string, node: string): Promise<ConditionalProbabilityTable | null> {
    const rows = this._stmt.loadCPT.all(graphId, node) as Array<{ parent_state: string; prob: number }>;
    if (rows.length === 0) return null;
    const entries: Record<string, number> = {};
    for (const r of rows) entries[r.parent_state] = r.prob;
    return { node, parents: [], entries };
  }

  async saveRegressionModel(graphId: string, node: string, model: RegressionParams): Promise<void> {
    this._stmt.insertReg.run(graphId, node, JSON.stringify(model.coefficients), model.intercept, model.residualStdDev);
  }

  async loadRegressionModel(graphId: string, node: string): Promise<RegressionParams | null> {
    const row = this._stmt.loadReg.get(graphId, node) as { coefficients: string; intercept: number; residual_std: number } | undefined;
    if (!row) return null;
    return { coefficients: JSON.parse(row.coefficients), intercept: row.intercept, residualStdDev: row.residual_std };
  }

  async saveRCAResult(caseId: string, result: RCAResult): Promise<void> {
    this._stmt.insertRCA.run(caseId, JSON.stringify(result.toJSON()), Date.now(), result.rootCauses[0]?.name ?? null);
  }

  async queryHistoricalResults(query: ResultQuery): Promise<RCAResult[]> {
    const rows = this._stmt.queryRCA.all(
      query.start ?? null, query.start ?? 0,
      query.end ?? null, query.end ?? Number.MAX_SAFE_INTEGER,
      query.rootCause ?? null, query.rootCause ?? null,
      query.limit ?? 100,
    ) as Array<{ result_json: string }>;
    return rows.map(r => JSON.parse(r.result_json) as RCAResult);
  }

  async beginTransaction(sessionId: string): Promise<void> {
    this.db.exec(`SAVEPOINT "${sessionId.replace(/"/g, '""')}"`);
    this._stmt.upsertState.run(sessionId, 'started', null, null);
  }

  async commitTransaction(sessionId: string): Promise<void> {
    this.db.exec(`RELEASE SAVEPOINT "${sessionId.replace(/"/g, '""')}"`);
    this._stmt.upsertState.run(sessionId, 'committed', null, null);
  }

  async rollbackToCheckpoint(sessionId: string, checkpoint: string): Promise<void> {
    this.db.exec(`ROLLBACK TO SAVEPOINT "${checkpoint.replace(/"/g, '""')}"`);
    this._stmt.upsertState.run(sessionId, `rolled_back`, checkpoint, null);
  }

  async setCheckpoint(sessionId: string, name: string): Promise<void> {
    this.db.exec(`SAVEPOINT "${name.replace(/"/g, '""')}"`);
    this._stmt.upsertState.run(sessionId, 'checkpoint', name, null);
  }

  /** Close the database connection */
  close(): void { this.db.close(); }
}
