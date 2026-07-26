/**
 * SQLite Web Worker — OPFS + WASM SQLite.
 *
 * This file runs inside a Web Worker (DedicatedWorkerGlobalScope).
 * It initializes @sqlite.org/sqlite-wasm with OPFS persistence and
 * responds to SQL execution messages from the main thread.
 *
 * Message protocol:
 *   Main → Worker: { type: 'run'|'all'|'get'|'exec', id, sql, params }
 *   Worker → Main: { id, result?, error? }
 *
 * OPFS requires synchronous I/O via createSyncAccessHandle,
 * which is ONLY available in Web Workers (not main thread).
 *
 * @packageDocumentation
 */

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

// ── OPFS + WASM SQLite initialization ──────────────────────────────

let db: unknown = null;

async function initSqlite(): Promise<void> {
  // Dynamic import of @sqlite.org/sqlite-wasm inside Worker
  // The module provides sqlite3InitModule() for OPFS-based persistence
  const sqlite3 = await import('@sqlite.org/sqlite-wasm');

  // Initialize with OPFS access
  const sqlite = await (sqlite3 as { default?: { oo1: { OpfsDb: new (path: string, mode: string) => unknown } } }).default?.oo1;
  if (sqlite) {
    db = new sqlite.OpfsDb('/causality-analyzer.sqlite3', 'ct');
  }
}

async function handleMessage(req: WorkerRequest): Promise<WorkerResponse> {
  const { id, type, sql, params } = req;

  try {
    if (!db) await initSqlite();

    switch (type) {
      case 'run': {
        (db as { exec(sql: string, opts: { bind: (number | string | null)[] }): void }).exec(sql!, { bind: params ?? [] });
        return { id, result: undefined };
      }
      case 'all': {
        const rows: unknown[] = [];
        (db as { exec(sql: string, opts: { bind: (number | string | null)[]; rowMode: string; callback: (row: unknown) => void }): void })
          .exec(sql!, {
            bind: params ?? [],
            rowMode: 'object',
            callback: (row: unknown) => { rows.push(row); },
          });
        return { id, result: rows };
      }
      case 'get': {
        let first: unknown = undefined;
        (db as { exec(sql: string, opts: { bind: (number | string | null)[]; rowMode: string; callback: (row: unknown) => void }): void })
          .exec(sql!, {
            bind: params ?? [],
            rowMode: 'object',
            callback: (row: unknown) => { if (first === undefined) first = row; },
          });
        return { id, result: first };
      }
      case 'exec': {
        (db as { exec(sql: string): void }).exec(sql!);
        return { id, result: undefined };
      }
      case 'close': {
        if (db) {
          (db as { close(): void }).close();
          db = null;
        }
        return { id, result: undefined };
      }
    }
  } catch (e) {
    const message = e instanceof Error ? e.message : 'Unknown error';
    return { id, error: message };
  }
}

// ── Worker event listener ───────────────────────────────────────────

self.onmessage = async (event: MessageEvent<WorkerRequest>): Promise<void> => {
  const response = await handleMessage(event.data);
  self.postMessage(response);
};
