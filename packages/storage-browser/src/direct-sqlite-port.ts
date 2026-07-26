/**
 * DirectSqlitePort — node:sqlite-based SqlitePort for vitest testing.
 *
 * Wraps Node.js 22+ built-in SQLite DatabaseSync with async interface
 * matching SqlitePort. DatabaseSync is inherently synchronous but wrapping
 * in Promise ensures interface compatibility with WorkerSqlitePort.
 *
 * @packageDocumentation
 */
import { DatabaseSync } from 'node:sqlite';
import type { SqlitePort, SqliteRow } from './sqlite-port.js';

export class DirectSqlitePort implements SqlitePort {
  private db: DatabaseSync;
  private _closed = false;

  constructor(path: string = ':memory:') {
    this.db = new DatabaseSync(path);
  }

  async run(sql: string, params: (number | string | null)[] = []): Promise<void> {
    const stmt = this.db.prepare(sql);
    stmt.run(...params);
  }

  async all(sql: string, params: (number | string | null)[] = []): Promise<SqliteRow[]> {
    const stmt = this.db.prepare(sql);
    return stmt.all(...params) as SqliteRow[];
  }

  async get(sql: string, params: (number | string | null)[] = []): Promise<SqliteRow | undefined> {
    const stmt = this.db.prepare(sql);
    return stmt.get(...params) as SqliteRow | undefined;
  }

  async exec(sql: string): Promise<void> {
    this.db.exec(sql);
  }

  close(): void {
    if (this._closed) return;
    this._closed = true;
    this.db.close();
  }
}
