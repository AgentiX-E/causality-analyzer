import { Client as PgClient, Pool as PgPool } from 'pg';
import type { IRelationalStore, MetricQuery, DetectionResult, ConditionalProbabilityTable, RegressionParams, RCAResult, ResultQuery, ColumnarTable, TableSchema } from '@agentix-e/causality-analyzer-core';
import type { MtlsConfig } from './types.js';

// ── PgClientLike — minimal pg.Client interface for instance DI ─────────

/**
 * Minimal interface satisfied by `pg.Client` and pg-mem adapters.
 *
 * When `config.client` is provided, the Store uses it directly
 * (no new() call). When absent, the Store creates its own `pg.Client`
 * from `connectionString` / `mtls` / `ssl` config.
 */
export interface PgClientLike {
  connect(): Promise<void>;
  query(sql: string, params?: unknown[]): Promise<{ rows: unknown[] }>;
  end(): Promise<void>;
}

// ── Row types ────────────────────────────────────────────────────────

interface MetricRow { ts: number; value: number; }
interface CptRow { parent_state: string; prob: number; }
interface RegressionRow { coefficients: string; intercept: number; residual_std: number; }
interface RcaResultRow { result_json: string; }

// ── DDL ──────────────────────────────────────────────────────────────

const DDL = [
  "CREATE TABLE IF NOT EXISTS metrics (ts BIGINT NOT NULL, value REAL NOT NULL, metric_name TEXT NOT NULL, PRIMARY KEY (ts, metric_name))",
  "CREATE TABLE IF NOT EXISTS cpt (graph_id TEXT NOT NULL, node TEXT NOT NULL, parent_state TEXT NOT NULL, prob REAL NOT NULL, PRIMARY KEY (graph_id, node, parent_state))",
  "CREATE TABLE IF NOT EXISTS regression_models (graph_id TEXT NOT NULL, node TEXT NOT NULL, coefficients JSONB NOT NULL, intercept REAL NOT NULL, residual_std REAL NOT NULL, PRIMARY KEY (graph_id, node))",
  "CREATE TABLE IF NOT EXISTS rca_results (case_id TEXT PRIMARY KEY, result_json JSONB NOT NULL, analyzed_at BIGINT NOT NULL, root_cause TEXT)",
  "CREATE TABLE IF NOT EXISTS analysis_state (session_id TEXT NOT NULL, stage TEXT NOT NULL, checkpoint_name TEXT, progress JSONB, PRIMARY KEY (session_id, checkpoint_name))",
];

// ── Config ───────────────────────────────────────────────────────────

export interface RemoteRelationalConfig {
  connectionString?: string;
  mtls?: MtlsConfig;
  ssl?: boolean | Record<string, unknown>;
  client?: PgClientLike;
  poolSize?: number;
}

// ── Retry ────────────────────────────────────────────────────────────

/** Error shape for connection-related errors. */
interface ConnectionError {
  code?: string;
  message?: string;
}

function isConnectionError(e: unknown): boolean {
  if (e == null || typeof e !== 'object') return false;
  const err = e as ConnectionError;
  if (err.code === 'ECONNREFUSED' || err.code === 'ETIMEDOUT') return true;
  if (typeof err.message === 'string' && /connection/.test(err.message)) return true;
  return false;
}

async function withRetry<T>(fn: () => Promise<T>, maxRetries = 3, baseMs = 100): Promise<T> {
  let last: unknown;
  for (let a = 0; a <= maxRetries; a++) {
    try { return await fn(); } catch (e: unknown) {
      last = e;
      if (a === maxRetries) break;
      if (!isConnectionError(e)) throw e;
      await new Promise(r => setTimeout(r, baseMs * Math.pow(2, a)));
    }
  }
  throw last;
}

// ── Config builder (pure, testable without PG) ───────────────────────

export function buildPgClientOpts(config: RemoteRelationalConfig): Record<string, unknown> {
  const opts: Record<string, unknown> = config.connectionString
    ? { connectionString: config.connectionString }
    : {};

  if (config.mtls) {
    const sslObj: Record<string, unknown> = { rejectUnauthorized: true };
    if (config.mtls.ca) sslObj.ca = config.mtls.ca;
    sslObj.cert = config.mtls.cert;
    sslObj.key = config.mtls.key;
    if (config.mtls.passphrase) sslObj.passphrase = config.mtls.passphrase;
    opts.ssl = config.ssl && typeof config.ssl === 'object'
      ? { ...sslObj, ...config.ssl }
      : sslObj;
  } else if (config.ssl !== undefined) {
    opts.ssl = config.ssl;
  }

  return opts;
}

// ── Store ────────────────────────────────────────────────────────────

export class RemoteRelationalStore implements IRelationalStore {
  private client: PgClientLike;
  private _ready: Promise<void>;

  constructor(config: RemoteRelationalConfig = {}) {
    if (config.client) {
      this.client = config.client;
    } else if ((config.poolSize ?? 1) > 1) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
      this.client = new PgPool({
        ...buildPgClientOpts(config),
        max: config.poolSize ?? 4,
      }) as unknown as PgClientLike;
    } else {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
      this.client = new PgClient(buildPgClientOpts(config)) as unknown as PgClientLike;
    }
    this._ready = withRetry(() => this._init());
  }

  private async _init(): Promise<void> {
    await this.client.connect();
    for (const d of DDL) await this.client.query(d);
  }

  private async _w() { await this._ready; }
  private q(s: string, p?: unknown[]) { return this.client.query(s, p); }
  private e(s: string) { return this.client.query(s); }
  private esc(s: string) { return s.replace(/'/g, "''"); }

  async readMetrics<S extends TableSchema>(q: MetricQuery): Promise<ColumnarTable<S>> {
    await this._w();
    const r = await this.q('SELECT ts, value FROM metrics WHERE ts >= $1 AND ts <= $2 ORDER BY ts', [q.start, q.end]);
    const x = r.rows as MetricRow[];
    const { ColumnarTable } = await import('@agentix-e/causality-analyzer-core');
    return ColumnarTable.fromColumnar({
      ts: new Float64Array(x.map(v => v.ts)),
      value: new Float64Array(x.map(v => v.value)),
    }) as ColumnarTable<S>;
  }

  async writeDetections(d: DetectionResult[], signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    await this._w();
    for (const x of d) {
      for (let i = 0; i < x.scores.length; i++) {
        await this.q(
          'INSERT INTO metrics VALUES ($1,$2,$3) ON CONFLICT (ts, metric_name) DO UPDATE SET value = EXCLUDED.value',
          [x.timestamp, x.scores[i]!, 'm' + i],
        );
      }
    }
  }

  async saveCPT(gid: string, n: string, cpt: ConditionalProbabilityTable, signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    await this._w();
    for (const [s, p] of Object.entries(cpt.entries)) {
      await this.q(
        'INSERT INTO cpt VALUES ($1,$2,$3,$4) ON CONFLICT (graph_id, node, parent_state) DO UPDATE SET prob = EXCLUDED.prob',
        [gid, n, s, p],
      );
    }
  }

  async loadCPT(gid: string, n: string, signal?: AbortSignal): Promise<ConditionalProbabilityTable | null> {
    await this._w();
    signal?.throwIfAborted();
    const r = await this.q(
      'SELECT parent_state, prob FROM cpt WHERE graph_id = $1 AND node = $2 ORDER BY parent_state',
      [gid, n],
    );
    const x = r.rows as CptRow[];
    if (!x.length) return null;
    const e: Record<string, number> = {};
    for (const v of x) e[v.parent_state] = v.prob;
    return { node: n, parents: [], entries: e };
  }

  async saveRegressionModel(gid: string, n: string, m: RegressionParams, signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    await this._w();
    await this.q(
      'INSERT INTO regression_models VALUES ($1,$2,$3,$4,$5) ON CONFLICT (graph_id, node) DO UPDATE SET coefficients = EXCLUDED.coefficients, intercept = EXCLUDED.intercept, residual_std = EXCLUDED.residual_std',
      [gid, n, JSON.stringify(m.coefficients), m.intercept, m.residualStdDev],
    );
  }

  async loadRegressionModel(gid: string, n: string, signal?: AbortSignal): Promise<RegressionParams | null> {
    await this._w();
    signal?.throwIfAborted();
    const r = await this.q(
      'SELECT coefficients, intercept, residual_std FROM regression_models WHERE graph_id = $1 AND node = $2',
      [gid, n],
    );
    const x = r.rows as RegressionRow[];
    if (x.length === 0) return null;
    const first = x[0]!;
    return {
      coefficients: JSON.parse(first.coefficients) as number[],
      intercept: first.intercept,
      residualStdDev: first.residual_std,
    };
  }

  async saveRCAResult(cid: string, r: RCAResult, signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    await this._w();
    await this.q(
      'INSERT INTO rca_results VALUES ($1,$2,$3,$4) ON CONFLICT (case_id) DO UPDATE SET result_json = EXCLUDED.result_json',
      [cid, JSON.stringify(r.toJSON()), Date.now(), r.rootCauses[0]?.name ?? null],
    );
  }

  async queryHistoricalResults(q: ResultQuery): Promise<RCAResult[]> {
    await this._w();
    let s = 'SELECT result_json FROM rca_results WHERE 1=1';
    const p: unknown[] = [];
    let i = 1;
    if (q.start != null) { s += ' AND analyzed_at>=$' + String(i++); p.push(q.start); }
    if (q.end != null) { s += ' AND analyzed_at<=$' + String(i++); p.push(q.end); }
    if (q.rootCause) { s += ' AND root_cause=$' + String(i++); p.push(q.rootCause); }
    s += ' ORDER BY analyzed_at DESC LIMIT $' + String(i++);
    p.push(q.limit ?? 100);
    return ((await this.q(s, p)).rows as RcaResultRow[]).map(x => JSON.parse(x.result_json) as RCAResult);
  }

  async beginTransaction(sid: string, signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    await this._w();
    await this.e('BEGIN');
    await this.e('SAVEPOINT sp_' + this.esc(sid));
    await this.q(
      'INSERT INTO analysis_state VALUES ($1,$2,$3,$4) ON CONFLICT (session_id, checkpoint_name) DO UPDATE SET stage = EXCLUDED.stage',
      [sid, 'started', null, null],
    );
  }

  async commitTransaction(sid: string, signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    await this._w();
    await this.e('RELEASE SAVEPOINT sp_' + this.esc(sid));
    await this.e('COMMIT');
  }

  async rollbackToCheckpoint(_sid: string, cp: string, signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    await this._w();
    await this.e('ROLLBACK TO SAVEPOINT sp_' + this.esc(cp));
  }

  async setCheckpoint(sid: string, name: string, signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    await this._w();
    await this.e('SAVEPOINT sp_' + this.esc(name));
    await this.q(
      'INSERT INTO analysis_state VALUES ($1,$2,$3,$4) ON CONFLICT (session_id, checkpoint_name) DO UPDATE SET stage = EXCLUDED.stage',
      [sid, 'checkpoint', name, null],
    );
  }

  async close(): Promise<void> { await this.client.end(); }

  async healthCheck(): Promise<boolean> {
    try { await this.q('SELECT 1'); return true; } catch { return false; }
  }

  async gracefulShutdown(ms = 5000): Promise<void> {
    await Promise.race([this.close(), new Promise(r => setTimeout(r, ms))]);
  }
}
