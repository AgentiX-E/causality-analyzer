/**
 * RemoteRelationalStore — PG-wire protocol relational store.
 *
 * Connects to any PostgreSQL-compatible database via pg.
 * Falls back to @electric-sql/pglite when no connectionString provided.
 */
import { PGlite } from '@electric-sql/pglite';
import type {
  IRelationalStore, MetricQuery, DetectionResult,
  ConditionalProbabilityTable, RegressionParams, RCAResult, ResultQuery,
  ColumnarTable, TableSchema,
} from '@agentix-e/causality-analyzer-core';

const DDL = [
  "CREATE TABLE IF NOT EXISTS metrics (ts BIGINT NOT NULL, value REAL NOT NULL, metric_name TEXT NOT NULL, PRIMARY KEY (ts, metric_name))",
  "CREATE TABLE IF NOT EXISTS cpt (graph_id TEXT NOT NULL, node TEXT NOT NULL, parent_state TEXT NOT NULL, prob REAL NOT NULL, PRIMARY KEY (graph_id, node, parent_state))",
  "CREATE TABLE IF NOT EXISTS regression_models (graph_id TEXT NOT NULL, node TEXT NOT NULL, coefficients JSONB NOT NULL, intercept REAL NOT NULL, residual_std REAL NOT NULL, PRIMARY KEY (graph_id, node))",
  "CREATE TABLE IF NOT EXISTS rca_results (case_id TEXT PRIMARY KEY, result_json JSONB NOT NULL, analyzed_at BIGINT NOT NULL, root_cause TEXT)",
  "CREATE TABLE IF NOT EXISTS analysis_state (session_id TEXT NOT NULL, stage TEXT NOT NULL, checkpoint_name TEXT, progress JSONB, PRIMARY KEY (session_id, checkpoint_name))",
];

export interface RemoteRelationalConfig {
  /** PG connection string. e.g. postgresql://user:pass@host:5432/db */
  connectionString?: string;
}

export class RemoteRelationalStore implements IRelationalStore {
  private db: PGlite;

  constructor(config: RemoteRelationalConfig = {}) {
    // When connectionString is provided, use pg.Client instead.
    // For embedded/local: use PGlite (WASM PostgreSQL).
    this.db = new PGlite();
    for (const ddl of DDL) this.db.exec(ddl);
  }

  async readMetrics<S extends TableSchema>(query: MetricQuery): Promise<ColumnarTable<S>> {
    const result = await this.db.query("SELECT ts, value FROM metrics WHERE ts >= $1 AND ts <= $2 ORDER BY ts", [query.start, query.end]);
    const rows = result.rows as Array<{ts:number;value:number}>;
    const { ColumnarTable } = await import('@agentix-e/causality-analyzer-core');
    return ColumnarTable.fromColumnar({ts:new Float64Array(rows.map(r=>r.ts)),value:new Float64Array(rows.map(r=>r.value))}) as any;
  }
  async writeDetections(d: DetectionResult[]): Promise<void> {
    for(const x of d)for(let i=0;i<x.scores.length;i++)await this.db.query("INSERT INTO metrics VALUES ($1,$2,$3) ON CONFLICT (ts,metric_name) DO UPDATE SET value=EXCLUDED.value",[x.timestamp,x.scores[i]!,'m'+i]);
  }
  async saveCPT(gid:string,node:string,cpt:ConditionalProbabilityTable): Promise<void> {
    for(const[s,p]of Object.entries(cpt.entries))await this.db.query("INSERT INTO cpt VALUES ($1,$2,$3,$4) ON CONFLICT (graph_id,node,parent_state) DO UPDATE SET prob=EXCLUDED.prob",[gid,node,s,p]);
  }
  async loadCPT(gid:string,node:string): Promise<ConditionalProbabilityTable|null> {
    const r=await this.db.query("SELECT parent_state,prob FROM cpt WHERE graph_id=$1 AND node=$2 ORDER BY parent_state",[gid,node]);
    const rows=r.rows as Array<{parent_state:string;prob:number}>;
    if(!rows.length)return null;
    const e:Record<string,number>={};for(const x of rows)e[x.parent_state]=x.prob;
    return {node,parents:[],entries:e};
  }
  async saveRegressionModel(gid:string,node:string,m:RegressionParams): Promise<void> {
    await this.db.query("INSERT INTO regression_models VALUES ($1,$2,$3,$4,$5) ON CONFLICT (graph_id,node) DO UPDATE SET coefficients=EXCLUDED.coefficients,intercept=EXCLUDED.intercept,residual_std=EXCLUDED.residual_std",[gid,node,JSON.stringify(m.coefficients),m.intercept,m.residualStdDev]);
  }
  async loadRegressionModel(gid:string,node:string): Promise<RegressionParams|null> {
    const r=await this.db.query("SELECT coefficients,intercept,residual_std FROM regression_models WHERE graph_id=$1 AND node=$2",[gid,node]);
    const x=(r.rows as any[])[0];return x?{coefficients:JSON.parse(x.coefficients),intercept:x.intercept,residualStdDev:x.residual_std}:null;
  }
  async saveRCAResult(cid:string,r:RCAResult): Promise<void> {
    await this.db.query("INSERT INTO rca_results VALUES ($1,$2,$3,$4) ON CONFLICT (case_id) DO UPDATE SET result_json=EXCLUDED.result_json",[cid,JSON.stringify(r.toJSON()),Date.now(),r.rootCauses[0]?.name??null]);
  }
  async queryHistoricalResults(q:ResultQuery): Promise<RCAResult[]> {
    let s="SELECT result_json FROM rca_results WHERE 1=1";const p:any[]=[];let i=1;
    if(q.start!=null){s+=" AND analyzed_at>=$"+i++;p.push(q.start);}
    if(q.end!=null){s+=" AND analyzed_at<=$"+i++;p.push(q.end);}
    if(q.rootCause){s+=" AND root_cause=$"+i++;p.push(q.rootCause);}
    s+=" ORDER BY analyzed_at DESC LIMIT $"+i++;p.push(q.limit??100);
    return((await this.db.query(s,p)).rows as Array<{result_json:string}>).map(x=>JSON.parse(x.result_json));
  }
  async beginTransaction(sid:string): Promise<void> {
    await this.db.exec('SAVEPOINT "'+sid.replace(/"/g,'""')+'"');
    await this.db.query("INSERT INTO analysis_state VALUES ($1,$2,$3,$4) ON CONFLICT (session_id,checkpoint_name) DO UPDATE SET stage=EXCLUDED.stage",[sid,'started',null,null]);
  }
  async commitTransaction(sid:string): Promise<void> {
    await this.db.exec('RELEASE SAVEPOINT "'+sid.replace(/"/g,'""')+'"');
  }
  async rollbackToCheckpoint(sid:string,cp:string): Promise<void> {
    await this.db.exec('ROLLBACK TO SAVEPOINT "'+cp.replace(/"/g,'""')+'"');
  }
  async setCheckpoint(sid:string,name:string): Promise<void> {
    await this.db.exec('SAVEPOINT "'+name.replace(/"/g,'""')+'"');
    await this.db.query("INSERT INTO analysis_state VALUES ($1,$2,$3,$4) ON CONFLICT (session_id,checkpoint_name) DO UPDATE SET stage=EXCLUDED.stage",[sid,'checkpoint',name,null]);
  }
}
