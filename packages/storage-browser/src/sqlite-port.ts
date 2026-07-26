/**
 * SqlitePort — abstract SQLite execution interface.
 *
 * Browser: WorkerSqlitePort (Web Worker + WASM + OPFS)
 * Test:    DirectSqlitePort (node:sqlite in-memory)
 *
 * This abstraction enables testing IRelationalStore/IGraphStore
 * implementations with vitest (happy-dom / node environment) without
 * requiring a real browser or OPFS.
 *
 * @packageDocumentation
 */

export interface SqliteRow { [column: string]: number | string | null; }

export interface SqlitePort {
  /** Execute a write statement (INSERT/UPDATE/DELETE/CREATE) */
  run(sql: string, params?: (number | string | null)[]): Promise<void>;

  /** Execute a read statement and return all rows */
  all(sql: string, params?: (number | string | null)[]): Promise<SqliteRow[]>;

  /** Execute a read statement and return the first row */
  get(sql: string, params?: (number | string | null)[]): Promise<SqliteRow | undefined>;

  /** Execute raw SQL without returning rows */
  exec(sql: string): Promise<void>;

  /** Close the port and release resources */
  close(): void;
}
