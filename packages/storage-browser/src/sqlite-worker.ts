/**
 * SQLite Web Worker — OPFS + WASM SQLite via wa-sqlite.
 *
 * Runs inside a Web Worker. Initializes wa-sqlite with AccessHandlePoolVFS
 * (synchronous OPFS via createSyncAccessHandle) and responds to SQL
 * execution messages from the main thread.
 *
 * Message protocol:
 *   Main → Worker: { type: 'run'|'all'|'get'|'exec', id, sql, params }
 *   Worker → Main: { id, result?, error? }
 *
 * Migrated from @sqlite.org/sqlite-wasm to wa-sqlite (I55):
 *   Same SQLite engine · OPFS persistence · standard npm · MIT license
 *
 * @packageDocumentation
 */
import SQLiteESMFactory from 'wa-sqlite/dist/wa-sqlite.mjs';
import * as SQLite from 'wa-sqlite';
import { AccessHandlePoolVFS } from 'wa-sqlite/src/examples/AccessHandlePoolVFS.js';

// ── Worker message types ────────────────────────────────────────────

interface WorkerRequest {
  id: number;
  type: 'run' | 'all' | 'get' | 'exec' | 'close';
  sql?: string;
  params?: (number | string | null)[];
}

interface WorkerResponse {
  id: number;
  result?: unknown;
  error?: string;
}

// ── SQLite state ────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let sqlite3: any = null;
let db: number | null = null;

async function initSqlite(): Promise<void> {
  const module = await SQLiteESMFactory();
  sqlite3 = SQLite.Factory(module);

  const vfs = new AccessHandlePoolVFS('causality-analyzer');
  sqlite3.vfs_register(vfs, true);

  db = await sqlite3.open_v2('causality-analyzer.sqlite3');
}

// ── SQL execution ───────────────────────────────────────────────────

async function handleMessage(req: WorkerRequest): Promise<WorkerResponse> {
  const { id, type, sql, params } = req;

  try {
    if (!db) await initSqlite();
    if (!sqlite3 || !db) throw new Error('SQLite not initialized');

    const bind = params ?? [];

    switch (type) {
      case 'run': {
        await sqlite3.exec(db, sql!, { bind });
        return { id, result: undefined };
      }
      case 'all': {
        const rows: unknown[] = [];
        await sqlite3.exec(db, sql!, {
          bind,
          rowMode: 'object',
          callback: (row: unknown) => { rows.push(row); },
        });
        return { id, result: rows };
      }
      case 'get': {
        let first: unknown = undefined;
        await sqlite3.exec(db, sql!, {
          bind,
          rowMode: 'object',
          callback: (row: unknown) => { if (first === undefined) first = row; },
        });
        return { id, result: first };
      }
      case 'exec': {
        await sqlite3.exec(db, sql!);
        return { id, result: undefined };
      }
      case 'close': {
        if (db) {
          await sqlite3.close(db);
          db = null;
          sqlite3 = null;
        }
        return { id, result: undefined };
      }
    }
  } catch (e) {
    return { id, error: e instanceof Error ? e.message : 'Unknown error' };
  }
}

// ── Worker event listener ───────────────────────────────────────────

self.onmessage = async (event: MessageEvent<WorkerRequest>): Promise<void> => {
  const response = await handleMessage(event.data);
  self.postMessage(response);
};
