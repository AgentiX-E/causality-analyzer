/**
 * Browser Relational Store — IRelationalStore backed by SqlitePort (WASM SQLite).
 *
 * Implements the full IRelationalStore interface using any SqlitePort
 * (WorkerSqlitePort in production, DirectSqlitePort in tests).
 *
 * Identical table schema to storage-embed (node:sqlite) for cross-package
 * compatibility — a database created by one can be read by the other.
 *
 * @packageDocumentation
 */
import type {
  IRelationalStore, MetricQuery, DetectionResult,
  ConditionalProbabilityTable, RegressionParams, RCAResult, ResultQuery,
  TableSchema,
} from '@agentix-e/causality-analyzer-core';
import type { SqlitePort, SqliteRow } from './sqlite-port.js';

const DDL = [
  "CREATE TABLE IF NOT EXISTS metrics (ts INTEGER NOT NULL, value REAL NOT NULL, metric_name TEXT NOT NULL, PRIMARY KEY (ts, metric_name))",
  "CREATE TABLE IF NOT EXISTS cpt (graph_id TEXT NOT NULL, node TEXT NOT NULL, parent_state TEXT NOT NULL, prob REAL NOT NULL, PRIMARY KEY (graph_id, node, parent_state))",
  "CREATE TABLE IF NOT EXISTS regression_models (graph_id TEXT NOT NULL, node TEXT NOT NULL, coefficients TEXT NOT NULL, intercept REAL NOT NULL, residual_std REAL NOT NULL, PRIMARY KEY (graph_id, node))",
  "CREATE TABLE IF NOT EXISTS rca_results (case_id TEXT PRIMARY KEY, result_json TEXT NOT NULL, analyzed_at INTEGER NOT NULL, root_cause TEXT)",
  "CREATE TABLE IF NOT EXISTS analysis_state (session_id TEXT NOT NULL, stage TEXT NOT NULL, checkpoint_name TEXT, progress TEXT, PRIMARY KEY (session_id, checkpoint_name))",
];

interface MetricRow extends SqliteRow { ts: number; metric_name: string; value: number; }
interface CptRow extends SqliteRow { parent_state: string; prob: number; }
interface RegressionRow extends SqliteRow { coefficients: string; intercept: number; residual_std: number; }
interface RcaResultRow extends SqliteRow { result_json: string; }

function checkAborted(signal?: AbortSignal): void {
  signal?.throwIfAborted();
}

export class WasmRelationalStore implements IRelationalStore {
  private port: SqlitePort;

  constructor(port: SqlitePort) {
    this.port = port;
    for (const d of DDL) this.port.exec(d);
  }

  private esc(s: string): string { return s.replace(/"/g, '""'); }

  async readMetrics<S extends TableSchema>(query: MetricQuery): Promise<unknown> {
    checkAborted(query.signal);
    const rows = await this.port.all(
      "SELECT ts, metric_name, value FROM metrics WHERE ts >= ? AND ts <= ? ORDER BY ts",
      [query.start, query.end],
    ) as unknown as MetricRow[];
    const metricFilter = query.metrics ? new Set(query.metrics) : null;
    const filtered = metricFilter ? rows.filter(r => metricFilter.has(r.metric_name)) : rows;
    return {
      ts: new Float64Array(filtered.map(r => r.ts)),
      value: new Float64Array(filtered.map(r => r.value)),
    };
  }

  async writeDetections(d: DetectionResult[], signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    for (const x of d) {
      for (let i = 0; i < x.scores.length; i++) {
        await this.port.run(
          "INSERT OR REPLACE INTO metrics VALUES (?, ?, ?)",
          [x.timestamp, x.scores[i]!, 'm' + i],
        );
      }
    }
  }

  async saveCPT(gid: string, node: string, cpt: ConditionalProbabilityTable, signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    for (const [s, p] of Object.entries(cpt.entries)) {
      await this.port.run("INSERT OR REPLACE INTO cpt VALUES (?, ?, ?, ?)", [gid, node, s, p]);
    }
  }

  async loadCPT(gid: string, node: string, signal?: AbortSignal): Promise<ConditionalProbabilityTable | null> {
    checkAborted(signal);
    const rows = await this.port.all(
      "SELECT parent_state, prob FROM cpt WHERE graph_id = ? AND node = ? ORDER BY parent_state",
      [gid, node],
    ) as unknown as CptRow[];
    if (!rows.length) return null;
    const e: Record<string, number> = {};
    for (const r of rows) e[r.parent_state] = r.prob;
    return { node, parents: [], entries: e };
  }

  async saveRegressionModel(gid: string, node: string, m: RegressionParams, signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    await this.port.run(
      "INSERT OR REPLACE INTO regression_models VALUES (?, ?, ?, ?, ?)",
      [gid, node, JSON.stringify(m.coefficients), m.intercept, m.residualStdDev],
    );
  }

  async loadRegressionModel(gid: string, node: string, signal?: AbortSignal): Promise<RegressionParams | null> {
    checkAborted(signal);
    const r = await this.port.get(
      "SELECT coefficients, intercept, residual_std FROM regression_models WHERE graph_id = ? AND node = ?",
      [gid, node],
    ) as unknown as RegressionRow | undefined;
    if (!r) return null;
    return {
      coefficients: JSON.parse(r.coefficients) as number[],
      intercept: r.intercept,
      residualStdDev: r.residual_std,
    };
  }

  async saveRCAResult(cid: string, r: RCAResult, signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    await this.port.run(
      "INSERT OR REPLACE INTO rca_results VALUES (?, ?, ?, ?)",
      [cid, JSON.stringify(r.toJSON()), Date.now(), r.rootCauses[0]?.name ?? null],
    );
  }

  async queryHistoricalResults(q: ResultQuery): Promise<RCAResult[]> {
    const rows = await this.port.all(
      `SELECT result_json FROM rca_results
       WHERE (? IS NULL OR analyzed_at >= ?)
         AND (? IS NULL OR analyzed_at <= ?)
         AND (? IS NULL OR root_cause = ?)
       ORDER BY analyzed_at DESC LIMIT ?`,
      [q.start ?? null, q.start ?? 0, q.end ?? null, q.end ?? Number.MAX_SAFE_INTEGER,
       q.rootCause ?? null, q.rootCause ?? null, q.limit ?? 100],
    ) as unknown as RcaResultRow[];
    return rows.map(r => JSON.parse(r.result_json) as RCAResult);
  }

  async beginTransaction(sid: string, signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    await this.port.exec('SAVEPOINT "' + this.esc(sid) + '"');
    await this.port.run(
      "INSERT OR REPLACE INTO analysis_state VALUES (?, ?, ?, ?)",
      [sid, 'started', null, null],
    );
  }

  async commitTransaction(sid: string, signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    await this.port.exec('RELEASE SAVEPOINT "' + this.esc(sid) + '"');
  }

  async rollbackToCheckpoint(_sid: string, cp: string, signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    await this.port.exec('ROLLBACK TO SAVEPOINT "' + this.esc(cp) + '"');
  }

  async setCheckpoint(sid: string, name: string, signal?: AbortSignal): Promise<void> {
    checkAborted(signal);
    await this.port.exec('SAVEPOINT "' + this.esc(name) + '"');
    await this.port.run(
      "INSERT OR REPLACE INTO analysis_state VALUES (?, ?, ?, ?)",
      [sid, 'checkpoint', name, null],
    );
  }

  close(): void { this.port.close(); }
}
