import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { RELATIONAL_SCHEMA } from '../embed-relational-store.js';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);

// Synchronous guard — vitest needs describe() to be registered at module load time
let Database: any = null;
try { Database = require('better-sqlite3'); } catch { /* CI without native addon */ }

const _describe = Database ? describe : describe.skip;

_describe('Real SQLite (persistent WAL)', () => {
  const fs = require('fs');
  const os = require('os');
  const path = require('path');
  let db: any, dbPath: string;

  beforeEach(() => {
    dbPath = path.join(os.tmpdir(), 'ca-sqlite-' + Date.now() + '-' + Math.random().toString(36).slice(2) + '.db');
    db = new Database(dbPath);
    db.pragma('journal_mode = WAL');
    for (const ddl of Object.values(RELATIONAL_SCHEMA)) db.exec(ddl);
  });
  afterEach(() => { db?.close(); try { fs.unlinkSync(dbPath); } catch {} });

  it('creates 5 tables on disk', () => {
    expect(db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all()).toHaveLength(5);
    expect(fs.statSync(dbPath).size).toBeGreaterThan(0);
  });

  it('SAVEPOINT rollback + release', () => {
    db.exec('SAVEPOINT s1');
    db.prepare('INSERT INTO cpt VALUES (?,?,?,?)').run('g1','A','0',0.3);
    db.exec('ROLLBACK TO SAVEPOINT s1');
    expect(db.prepare('SELECT * FROM cpt WHERE graph_id=?').all('g1')).toHaveLength(0);
    db.exec('SAVEPOINT s2');
    db.prepare('INSERT INTO cpt VALUES (?,?,?,?)').run('g2','B','1',0.7);
    db.exec('RELEASE SAVEPOINT s2');
    expect(db.prepare('SELECT * FROM cpt WHERE graph_id=? AND node=?').get('g2','B').prob).toBe(0.7);
  });
});
