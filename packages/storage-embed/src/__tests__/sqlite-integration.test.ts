import { RELATIONAL_SCHEMA } from '../embed-relational-store.js';

// Guard: skip when native addon unavailable (e.g. CI without prebuilts)
let hasSQLite = false; try { require('better-sqlite3'); hasSQLite = true; } catch {}

if (hasSQLite) {
  const { describe, it, expect, beforeEach, afterEach } = await import('vitest');
  const Database = require('better-sqlite3');

  describe('Real SQLite integration', () => {
    const fs = await import('fs');
    const os = await import('os');
    const path = await import('path');
    
    let db: any;
    let dbPath: string;

    // Use temp file for persistent WAL testing, clean up after
    beforeEach(() => {
      dbPath = path.join(os.tmpdir(), `ca-sqlite-test-${Date.now()}-${Math.random().toString(36).slice(2)}.db`);
      db = new Database(dbPath);
      db.pragma('journal_mode = WAL');
      db.pragma('synchronous = NORMAL');
      for (const ddl of Object.values(RELATIONAL_SCHEMA)) db.exec(ddl);
    });
    afterEach(() => {
      db?.close();
      try { fs.unlinkSync(dbPath); fs.unlinkSync(dbPath + '-wal'); fs.unlinkSync(dbPath + '-shm'); } catch {}
    });

    it('creates all 5 tables', () => {
      const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").all();
      expect(tables.map((t: any) => t.name)).toEqual(expect.arrayContaining(['metrics','cpt','regression_models','rca_results','analysis_state']));
    });

    it('SAVEPOINT rollback + release with persistent WAL', () => {
      db.exec("SAVEPOINT sp1");
      db.prepare('INSERT INTO cpt VALUES (?,?,?,?)').run('g1','X','0',0.3);
      db.exec("ROLLBACK TO SAVEPOINT sp1");
      expect(db.prepare('SELECT * FROM cpt WHERE graph_id=?').all('g1')).toHaveLength(0);

      db.exec("SAVEPOINT sp2");
      db.prepare('INSERT INTO cpt VALUES (?,?,?,?)').run('g2','Y','1',0.7);
      db.exec("RELEASE SAVEPOINT sp2");
      expect(db.prepare('SELECT * FROM cpt WHERE graph_id=?').get('g2','Y').prob).toBe(0.7);
    });

    it('persists data to disk (verify file exists before cleanup)', () => {
      db.prepare('INSERT INTO rca_results VALUES (?,?,?)').run('case-persist','{"ok":true}',Date.now());
      expect(fs.existsSync(dbPath)).toBe(true);
      const stat = fs.statSync(dbPath);
      expect(stat.size).toBeGreaterThan(0);
    });
  });
}
