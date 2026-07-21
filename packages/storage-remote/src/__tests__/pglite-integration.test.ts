import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { PGlite } from '@electric-sql/pglite';

const PG_SCHEMA = {
  cpt: `CREATE TABLE IF NOT EXISTS cpt (graph_id TEXT NOT NULL, node TEXT NOT NULL, parent_state TEXT NOT NULL, prob REAL NOT NULL, PRIMARY KEY (graph_id, node, parent_state))`,
  regression_models: `CREATE TABLE IF NOT EXISTS regression_models (graph_id TEXT NOT NULL, node TEXT NOT NULL, coefficients JSONB NOT NULL, intercept REAL NOT NULL, residual_std REAL NOT NULL, PRIMARY KEY (graph_id, node))`,
  rca_results: `CREATE TABLE IF NOT EXISTS rca_results (case_id TEXT PRIMARY KEY, result_json JSONB NOT NULL, analyzed_at BIGINT NOT NULL)`,
  analysis_state: `CREATE TABLE IF NOT EXISTS analysis_state (session_id TEXT PRIMARY KEY, stage TEXT, checkpoint_name TEXT, progress JSONB)`,
};

describe('RemoteRelationalStore with real PostgreSQL (PGlite)', () => {
  let pg: PGlite;
  beforeEach(async () => { pg = new PGlite(); for (const ddl of Object.values(PG_SCHEMA)) await pg.exec(ddl); });
  afterEach(async () => { await pg.close(); });

  it('creates tables and lists them', async () => {
    const result = await pg.query("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename");
    const names = (result.rows as { tablename: string }[]).map(r => r.tablename);
    expect(names).toEqual(expect.arrayContaining(['cpt', 'regression_models', 'rca_results', 'analysis_state']));
  });

  it('CPT: parameterized insert and query', async () => {
    await pg.query("INSERT INTO cpt VALUES ($1,$2,$3,$4)", ['g1', 'CPU', '0', 0.1]);
    await pg.query("INSERT INTO cpt VALUES ($1,$2,$3,$4)", ['g1', 'CPU', '1', 0.75]);
    const result = await pg.query("SELECT prob FROM cpt WHERE graph_id=$1 AND node=$2 ORDER BY parent_state", ['g1', 'CPU']);
    expect(result.rows).toHaveLength(2);
  });


  it('rca_results time-range query', async () => {
    await pg.query("INSERT INTO rca_results VALUES ($1,$2,$3)", ['c1','{}',1000]);
    await pg.query("INSERT INTO rca_results VALUES ($1,$2,$3)", ['c2','{}',2000]);
    const result = await pg.query("SELECT case_id FROM rca_results WHERE analyzed_at>=$1 ORDER BY analyzed_at", [1500]);
    expect(result.rows).toHaveLength(1);
  });

  it('analysis_state progress tracking', async () => {
    await pg.query("INSERT INTO analysis_state VALUES ($1,$2,$3,$4)", ['s1','detect','before_graph',JSON.stringify({step:1})]);
    const result = await pg.query("SELECT stage FROM analysis_state WHERE session_id=$1", ['s1']);
    expect((result.rows as {stage:string}[])[0]!.stage).toBe('detect');
  });
});
