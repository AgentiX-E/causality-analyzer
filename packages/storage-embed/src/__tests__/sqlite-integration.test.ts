/**
 * Real SQLite Integration Tests — better-sqlite3 backed EmbedRelationalStore.
 *
 * These tests use actual SQLite via the native better-sqlite3 addon.
 * The :memory: database provides real SQL, transactions, and SAVEPOINT.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { RELATIONAL_SCHEMA } from '../embed-relational-store.js';

describe('EmbedRelationalStore with real SQLite', () => {
  let db: Database.Database;

  beforeEach(() => {
    db = new Database(':memory:');
    db.pragma('journal_mode = WAL');
    db.pragma('foreign_keys = ON');
    for (const ddl of Object.values(RELATIONAL_SCHEMA)) {
      db.exec(ddl);
    }
  });

  afterEach(() => { db.close(); });

  it('creates all 5 tables with correct schemas', () => {
    const tables = db.prepare(
      "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).all() as { name: string }[];
    expect(tables.map(t => t.name)).toEqual(
      expect.arrayContaining(['metrics', 'cpt', 'regression_models', 'rca_results', 'analysis_state'])
    );
  });

  it('CPT: atomic insert + query with composite key', () => {
    const insert = db.prepare(
      'INSERT OR REPLACE INTO cpt (graph_id, node, parent_state, prob) VALUES (?, ?, ?, ?)'
    );
    insert.run('g1', 'Latency', '00', 0.05);
    insert.run('g1', 'Latency', '10', 0.40);
    insert.run('g1', 'Latency', '01', 0.30);
    insert.run('g1', 'Latency', '11', 0.85);

    const rows = db.prepare(
      'SELECT prob FROM cpt WHERE graph_id=? AND node=? ORDER BY parent_state'
    ).all('g1', 'Latency') as { prob: number }[];

    expect(rows).toHaveLength(4);
    expect(rows[0]!.prob).toBe(0.05);
    expect(rows[3]!.prob).toBe(0.85);
  });

  it('SAVEPOINT: rollback discards changes', () => {
    db.exec("SAVEPOINT sp1");
    db.prepare('INSERT INTO cpt (graph_id, node, parent_state, prob) VALUES (?,?,?,?)')
      .run('g2', 'X', '0', 0.3);
    db.exec("ROLLBACK TO SAVEPOINT sp1");

    const rows = db.prepare('SELECT * FROM cpt WHERE graph_id=?').all('g2');
    expect(rows).toHaveLength(0);
  });

  it('SAVEPOINT: release preserves changes', () => {
    db.exec("SAVEPOINT sp2");
    db.prepare('INSERT INTO cpt (graph_id, node, parent_state, prob) VALUES (?,?,?,?)')
      .run('g3', 'Y', '1', 0.7);
    db.exec("RELEASE SAVEPOINT sp2");

    const row = db.prepare('SELECT prob FROM cpt WHERE graph_id=? AND node=?').get('g3', 'Y') as { prob: number };
    expect(row.prob).toBe(0.7);
  });

  it('SAVEPOINT: nested checkpoints work correctly', () => {
    db.exec("SAVEPOINT outer");
    db.prepare('INSERT INTO rca_results (case_id, result_json, analyzed_at) VALUES (?,?,?)')
      .run('c1', '{"x":1}', 100);
    
    db.exec("SAVEPOINT inner");
    db.prepare('INSERT INTO rca_results (case_id, result_json, analyzed_at) VALUES (?,?,?)')
      .run('c2', '{"x":2}', 200);
    db.exec("ROLLBACK TO SAVEPOINT inner");
    // inner rollback: c2 gone, c1 remains
    db.exec("RELEASE SAVEPOINT outer");

    const rows = db.prepare('SELECT case_id FROM rca_results ORDER BY case_id').all() as { case_id: string }[];
    expect(rows).toHaveLength(1);
    expect(rows[0]!.case_id).toBe('c1');
  });

  it('regression_models: store and retrieve coefficients array', () => {
    db.prepare(
      'INSERT OR REPLACE INTO regression_models (graph_id, node, coefficients, intercept, residual_std) VALUES (?,?,?,?,?)'
    ).run('g1', 'CPU', JSON.stringify([1.5, 2.0, -0.5]), 0.25, 0.12);

    const row = db.prepare(
      'SELECT * FROM regression_models WHERE graph_id=? AND node=?'
    ).get('g1', 'CPU') as { coefficients: string; intercept: number; residual_std: number };

    expect(JSON.parse(row.coefficients)).toEqual([1.5, 2.0, -0.5]);
    expect(row.intercept).toBe(0.25);
    expect(row.residual_std).toBe(0.12);
  });

  it('rca_results: time-range query with ORDER BY', () => {
    db.prepare(
      'INSERT INTO rca_results (case_id, result_json, analyzed_at) VALUES (?,?,?)'
    ).run('early', '{}', 1000);
    db.prepare(
      'INSERT INTO rca_results (case_id, result_json, analyzed_at) VALUES (?,?,?)'
    ).run('mid', '{}', 2000);
    db.prepare(
      'INSERT INTO rca_results (case_id, result_json, analyzed_at) VALUES (?,?,?)'
    ).run('late', '{}', 3000);

    const rows = db.prepare(
      'SELECT case_id FROM rca_results WHERE analyzed_at >= ? AND analyzed_at <= ? ORDER BY analyzed_at'
    ).all(1500, 2500) as { case_id: string }[];

    expect(rows).toHaveLength(1);
    expect(rows[0]!.case_id).toBe('mid');
  });

  it('analysis_state: upsert pipeline progress', () => {
    db.prepare(
      'INSERT OR REPLACE INTO analysis_state (session_id, stage, checkpoint_name, progress) VALUES (?,?,?,?)'
    ).run('s1', 'detect', 'before_graph', '{"step":1}');

    const row = db.prepare(
      'SELECT stage, checkpoint_name FROM analysis_state WHERE session_id=?'
    ).get('s1') as { stage: string; checkpoint_name: string };
    expect(row.stage).toBe('detect');
    expect(row.checkpoint_name).toBe('before_graph');
  });
});
