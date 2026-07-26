/**
 * Embedded Relational Store — Node.js 22+ built-in SQLite (node:sqlite).
 *
 * Uses Node.js 22's native `node:sqlite` DatabaseSync for zero-dependency,
 * zero-compilation embedded SQL storage. No better-sqlite3 required.
 *
 * The synchronous DatabaseSync API maps naturally to our async store interface
 * since SQLite is inherently single-writer and fast enough for embedded use.
 *
 * @packageDocumentation
 */
import { DatabaseSync } from 'node:sqlite';
import type { StatementSync } from 'node:sqlite';
import type {
  IRelationalStore, MetricQuery, DetectionResult,
  ConditionalProbabilityTable, RegressionParams, RCAResult, ResultQuery,
  ColumnarTable, TableSchema,
} from '@agentix-e/causality-analyzer-core';

// ── Row types ───────────────────────────────────────────────────────

interface MetricRow { ts: number; metric_name: string; value: number; }
interface CptRow { parent_state: string; prob: number; }
interface RegressionRow { coefficients: string; intercept: number; residual_std: number; }
interface RcaResultRow { result_json: string; }

// ── DDL ─────────────────────────────────────────────────────────────

const DDL: Record<string, string> = {
  metrics:     "CREATE TABLE IF NOT EXISTS metrics (ts INTEGER NOT NULL, value REAL NOT NULL, metric_name TEXT NOT NULL, PRIMARY KEY (ts, metric_name))",
  cpt:         "CREATE TABLE IF NOT EXISTS cpt (graph_id TEXT NOT NULL, node TEXT NOT NULL, parent_state TEXT NOT NULL, prob REAL NOT NULL, PRIMARY KEY (graph_id, node, parent_state))",
  regression:  "CREATE TABLE IF NOT EXISTS regression_models (graph_id TEXT NOT NULL, node TEXT NOT NULL, coefficients TEXT NOT NULL, intercept REAL NOT NULL, residual_std REAL NOT NULL, PRIMARY KEY (graph_id, node))",
  rca_results: "CREATE TABLE IF NOT EXISTS rca_results (case_id TEXT PRIMARY KEY, result_json TEXT NOT NULL, analyzed_at INTEGER NOT NULL, root_cause TEXT)",
  analysis:    "CREATE TABLE IF NOT EXISTS analysis_state (session_id TEXT NOT NULL, stage TEXT NOT NULL, checkpoint_name TEXT, progress TEXT, PRIMARY KEY (session_id, checkpoint_name))",
};

const SAFEPOINT_ESC = /"/g;
const DOUBLE_QUOTE = '""';

export interface EmbedStoreOptions {
  dbPath?: string;
}

function checkAborted(signal?: AbortSignal): void {
  signal?.throwIfAborted();
}

export class EmbedRelationalStore implements IRelationalStore {
  private db: DatabaseSync;

  // Prepared statements
  private mInsert: StatementSync;
  private mRead: StatementSync;
  private cSave: StatementSync;
  private cLoad: StatementSync;
  private rSave: StatementSync;
  private rLoad: StatementSync;
  private aSave: StatementSync;
  private aQuery: StatementSync;
  private sUpsert: StatementSync;

  constructor(opts: EmbedStoreOptions = {}) {
    const dbPath = opts.dbPath || "./causality-analyzer.db";
    this.db = new DatabaseSync(dbPath);
    this.db.exec(dbPath === ":memory:" ? "PRAGMA journal_mode = MEMORY" : "PRAGMA journal_mode = WAL");
    this.db.exec("PRAGMA synchronous = NORMAL");
    for (const ddl of Object.values(DDL)) this.db.exec(ddl);

    this.mInsert = this.db.prepare("INSERT OR REPLACE INTO metrics VALUES (?, ?, ?)");
    this.mRead   = this.db.prepare("SELECT ts, metric_name, value FROM metrics WHERE ts >= ? AND ts <= ? ORDER BY ts");
    this.cSave   = this.db.prepare("INSERT OR REPLACE INTO cpt VALUES (?, ?, ?, ?)");
    this.cLoad   = this.db.prepare("SELECT parent_state, prob FROM cpt WHERE graph_id = ? AND node = ? ORDER BY parent_state");
    this.rSave   = this.db.prepare("INSERT OR REPLACE INTO regression_models VALUES (?, ?, ?, ?, ?)");
    this.rLoad   = this.db.prepare("SELECT coefficients, intercept, residual_std FROM regression_models WHERE graph_id = ? AND node = ?");
    this.aSave   = this.db.prepare("INSERT OR REPLACE INTO rca_results VALUES (?, ?, ?, ?)");
    this.aQuery  = this.db.prepare("SELECT result_json FROM rca_results WHERE (? IS NULL OR analyzed_at >= ?) AND (? IS NULL OR analyzed_at <= ?) AND (? IS NULL OR root_cause = ?) ORDER BY analyzed_at DESC LIMIT ?");
    this.sUpsert = this.db.prepare("INSERT OR REPLACE INTO analysis_state VALUES (?, ?, ?, ?)");
  }

  private esc(s: string): string { return s.replace(SAFEPOINT_ESC, DOUBLE_QUOTE); }

  async readMetrics<S extends TableSchema>(query: MetricQuery): Promise<ColumnarTable<S>> {
    checkAborted(query.signal);
    const rows = this.mRead.all(query.start, query.end) as unknown as MetricRow[];
    const metricFilter = query.metrics ? new Set(query.metrics) : null;
    const filtered = metricFilter ? rows.filter(r => metricFilter.has(r.metric_name)) : rows;
    const { ColumnarTable } = await import("@agentix-e/causality-analyzer-core");
    return ColumnarTable.fromColumnar({
      ts: new Float64Array(filtered.map(r => r.ts)),
      value: new Float64Array(filtered.map(r => r.value)),
    }) as ColumnarTable<S>;
  }

  async writeDetections(d: DetectionResult[], signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    for (const x of d) {
      for (let i = 0; i < x.scores.length; i++) {
        this.mInsert.run(x.timestamp, x.scores[i]!, 'm' + i);
      }
    }
  }

  async saveCPT(gid: string, node: string, cpt: ConditionalProbabilityTable, signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    for (const [s, p] of Object.entries(cpt.entries)) {
      this.cSave.run(gid, node, s, p);
    }
  }

  async loadCPT(gid: string, node: string, signal?: AbortSignal): Promise<ConditionalProbabilityTable | null> {
    checkAborted(signal);
    const rows = this.cLoad.all(gid, node) as unknown as CptRow[];
    if (!rows.length) return null;
    const e: Record<string, number> = {};
    for (const r of rows) e[r.parent_state] = r.prob;
    return { node, parents: [], entries: e };
  }

  async saveRegressionModel(gid: string, node: string, m: RegressionParams, signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    this.rSave.run(gid, node, JSON.stringify(m.coefficients), m.intercept, m.residualStdDev);
  }

  async loadRegressionModel(gid: string, node: string, signal?: AbortSignal): Promise<RegressionParams | null> {
    checkAborted(signal);
    const r = this.rLoad.get(gid, node) as unknown as RegressionRow | undefined;
    if (!r) return null;
    return {
      coefficients: JSON.parse(r.coefficients) as number[],
      intercept: r.intercept,
      residualStdDev: r.residual_std,
    };
  }

  async saveRCAResult(cid: string, r: RCAResult, signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    this.aSave.run(cid, JSON.stringify(r.toJSON()), Date.now(), r.rootCauses[0]?.name ?? null);
  }

  async queryHistoricalResults(q: ResultQuery): Promise<RCAResult[]> {
    const rows = this.aQuery.all(
      q.start ?? null, q.start ?? 0,
      q.end ?? null, q.end ?? Number.MAX_SAFE_INTEGER,
      q.rootCause ?? null, q.rootCause ?? null,
      q.limit ?? 100,
    ) as unknown as RcaResultRow[];
    return rows.map(r => JSON.parse(r.result_json) as RCAResult);
  }

  async beginTransaction(sid: string, signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    this.db.exec('SAVEPOINT "' + this.esc(sid) + '"');
    this.sUpsert.run(sid, 'started', null, null);
  }

  async commitTransaction(sid: string, signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    this.db.exec('RELEASE SAVEPOINT "' + this.esc(sid) + '"');
  }

  async rollbackToCheckpoint(sid: string, cp: string, signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    this.db.exec('ROLLBACK TO SAVEPOINT "' + this.esc(cp) + '"');
  }

  async setCheckpoint(sid: string, name: string, signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    this.db.exec('SAVEPOINT "' + this.esc(name) + '"');
    this.sUpsert.run(sid, 'checkpoint', name, null);
  }

  private _closed = false;

  close(): void {
    if (this._closed) return;
    this._closed = true;
    this.db.close();
  }

  healthCheck(): boolean {
    if (this._closed) return false;
    try {
      this.db.prepare('SELECT 1').get();
      return true;
    } catch {
      return false;
    }
  }
}
