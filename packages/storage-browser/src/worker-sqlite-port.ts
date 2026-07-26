/**
 * WorkerSqlitePort — SqlitePort backed by a Web Worker running WASM SQLite + OPFS.
 *
 * Sends SQL execution messages to the worker and awaits responses.
 * Implements the same SqlitePort interface as DirectSqlitePort,
 * making IRelationalStore/IGraphStore implementations work identically
 * in browser and test environments.
 *
 * @packageDocumentation
 */
import type { SqlitePort, SqliteRow } from './sqlite-port.js';

interface WorkerResponse {
  id: number;
  result?: unknown;
  error?: string;
}

export class WorkerSqlitePort implements SqlitePort {
  private worker: Worker;
  private _nextId = 0;
  private _pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();

  constructor(workerScriptUrl: string | URL) {
    this.worker = new Worker(workerScriptUrl, { type: 'module' });
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const { id, result, error } = event.data;
      const pending = this._pending.get(id);
      if (pending) {
        this._pending.delete(id);
        if (error) pending.reject(new Error(error));
        else pending.resolve(result);
      }
    };
    this.worker.onerror = (e: ErrorEvent) => {
      // Reject all pending on worker crash
      for (const [, p] of this._pending) {
        p.reject(new Error(e.message || 'Worker error'));
      }
      this._pending.clear();
    };
  }

  private _send(type: string, sql: string, params?: (number | string | null)[]): Promise<unknown> {
    const id = ++this._nextId;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type, sql, params });
    });
  }

  async run(sql: string, params: (number | string | null)[] = []): Promise<void> {
    await this._send('run', sql, params);
  }

  async all(sql: string, params: (number | string | null)[] = []): Promise<SqliteRow[]> {
    return (await this._send('all', sql, params)) as SqliteRow[];
  }

  async get(sql: string, params: (number | string | null)[] = []): Promise<SqliteRow | undefined> {
    return (await this._send('get', sql, params)) as SqliteRow | undefined;
  }

  async exec(sql: string): Promise<void> {
    await this._send('exec', sql);
  }

  close(): void {
    this.worker.postMessage({ id: ++this._nextId, type: 'close' });
    this.worker.terminate();
    this._pending.clear();
  }
}
