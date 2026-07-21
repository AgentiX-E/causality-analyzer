import { RELATIONAL_SCHEMA } from '../embed-relational-store.js';

(async () => {
  let DB: any = null;
  try { const m = await import('better-sqlite3'); DB = m.default; } catch { return; }

  const { describe, it, expect, beforeEach, afterEach } = await import('vitest');
  const fs = await import('fs'); const os = await import('os'); const path = await import('path');

  describe('Real SQLite (persistent WAL)', () => {
    let db: any, dbPath: string;
    beforeEach(() => {
      dbPath = path.join(os.tmpdir(), 'ca-sqlite-' + Date.now() + '-' + Math.random().toString(36).slice(2) + '.db');
      db = new DB(dbPath);
      db.pragma('journal_mode = WAL');
      for (const ddl of Object.values(RELATIONAL_SCHEMA)) db.exec(ddl);
    });
    afterEach(() => { db?.close(); try { fs.unlinkSync(dbPath); } catch {} });

    it('creates tables on disk', () => {
      expect(db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all()).toHaveLength(5);
      expect(fs.statSync(dbPath).size).toBeGreaterThan(0);
    });

    it('SAVEPOINT rollback + release with WAL', () => {
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
})();
