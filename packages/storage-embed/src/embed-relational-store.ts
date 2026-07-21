import Database from 'better-sqlite3';
import type {
  IRelationalStore, MetricQuery, DetectionResult,
  ConditionalProbabilityTable, RegressionParams, RCAResult, ResultQuery,
  ColumnarTable, TableSchema,
} from '@agentix-e/causality-analyzer-core';

const DDL: Record<string,string> = {
  metrics:     "CREATE TABLE IF NOT EXISTS metrics (ts INTEGER NOT NULL, value REAL NOT NULL, metric_name TEXT NOT NULL, PRIMARY KEY (ts, metric_name))",
  cpt:         "CREATE TABLE IF NOT EXISTS cpt (graph_id TEXT NOT NULL, node TEXT NOT NULL, parent_state TEXT NOT NULL, prob REAL NOT NULL, PRIMARY KEY (graph_id, node, parent_state))",
  regression:  "CREATE TABLE IF NOT EXISTS regression_models (graph_id TEXT NOT NULL, node TEXT NOT NULL, coefficients TEXT NOT NULL, intercept REAL NOT NULL, residual_std REAL NOT NULL, PRIMARY KEY (graph_id, node))",
  rca_results: "CREATE TABLE IF NOT EXISTS rca_results (case_id TEXT PRIMARY KEY, result_json TEXT NOT NULL, analyzed_at INTEGER NOT NULL, root_cause TEXT)",
  analysis:    "CREATE TABLE IF NOT EXISTS analysis_state (session_id TEXT NOT NULL, stage TEXT NOT NULL, checkpoint_name TEXT, progress TEXT, PRIMARY KEY (session_id, checkpoint_name))",
};

export interface EmbedStoreOptions {
  dbPath?: string;
}

export class EmbedRelationalStore implements IRelationalStore {
  private db: any;
  private q: Record<string,any> = {};

  constructor(opts: EmbedStoreOptions = {}) {
    const path = opts.dbPath || "./causality-analyzer.db";
    this.db = new Database(path);
    this.db.pragma(path === ":memory:" ? "journal_mode = MEMORY" : "journal_mode = WAL");
    this.db.pragma("synchronous = NORMAL");
    for (const ddl of Object.values(DDL)) this.db.exec(ddl);

    this.q.mInsert = this.db.prepare("INSERT OR REPLACE INTO metrics VALUES (?, ?, ?)");
    this.q.mRead   = this.db.prepare("SELECT ts, value FROM metrics WHERE ts >= ? AND ts <= ? ORDER BY ts");
    this.q.cSave   = this.db.prepare("INSERT OR REPLACE INTO cpt VALUES (?, ?, ?, ?)");
    this.q.cLoad   = this.db.prepare("SELECT parent_state, prob FROM cpt WHERE graph_id = ? AND node = ? ORDER BY parent_state");
    this.q.rSave   = this.db.prepare("INSERT OR REPLACE INTO regression_models VALUES (?, ?, ?, ?, ?)");
    this.q.rLoad   = this.db.prepare("SELECT coefficients, intercept, residual_std FROM regression_models WHERE graph_id = ? AND node = ?");
    this.q.aSave   = this.db.prepare("INSERT OR REPLACE INTO rca_results VALUES (?, ?, ?, ?)");
    this.q.aQuery  = this.db.prepare("SELECT result_json FROM rca_results WHERE (? IS NULL OR analyzed_at >= ?) AND (? IS NULL OR analyzed_at <= ?) AND (? IS NULL OR root_cause = ?) ORDER BY analyzed_at DESC LIMIT ?");
    this.q.sUpsert = this.db.prepare("INSERT OR REPLACE INTO analysis_state VALUES (?, ?, ?, ?)");
  }

  private esc(s: string): string { return s.replace(/"/g, '""'); }

  async readMetrics<S extends TableSchema>(query: MetricQuery): Promise<ColumnarTable<S>> {
    const rows = this.q.mRead.all(query.start, query.end) as Array<{ts:number;value:number}>;
    const { ColumnarTable } = await import("@agentix-e/causality-analyzer-core");
    return ColumnarTable.fromColumnar({
      ts: new Float64Array(rows.map(r=>r.ts)),
      value: new Float64Array(rows.map(r=>r.value)),
    }) as any;
  }
  async writeDetections(d: DetectionResult[]): Promise<void> {
    this.db.transaction(()=>{for(const x of d)for(let i=0;i<x.scores.length;i++)this.q.mInsert.run(x.timestamp,x.scores[i]!,'m'+i);})();
  }
  async saveCPT(gid: string, node: string, cpt: ConditionalProbabilityTable): Promise<void> {
    this.db.transaction(()=>{for(const[s,p]of Object.entries(cpt.entries))this.q.cSave.run(gid,node,s,p);})();
  }
  async loadCPT(gid: string, node: string): Promise<ConditionalProbabilityTable|null> {
    const rows=this.q.cLoad.all(gid,node) as Array<{parent_state:string;prob:number}>;
    if(!rows.length)return null;
    const e:Record<string,number>={};for(const r of rows)e[r.parent_state]=r.prob;
    return {node,parents:[],entries:e};
  }
  async saveRegressionModel(gid: string, node: string, m: RegressionParams): Promise<void> {
    this.q.rSave.run(gid,node,JSON.stringify(m.coefficients),m.intercept,m.residualStdDev);
  }
  async loadRegressionModel(gid: string, node: string): Promise<RegressionParams|null> {
    const r=this.q.rLoad.get(gid,node) as any;
    return r?{coefficients:JSON.parse(r.coefficients),intercept:r.intercept,residualStdDev:r.residual_std}:null;
  }
  async saveRCAResult(cid: string, r: RCAResult): Promise<void> {
    this.q.aSave.run(cid,JSON.stringify(r.toJSON()),Date.now(),r.rootCauses[0]?.name??null);
  }
  async queryHistoricalResults(q: ResultQuery): Promise<RCAResult[]> {
    const rows=this.q.aQuery.all(q.start??null,q.start??0,q.end??null,q.end??Number.MAX_SAFE_INTEGER,q.rootCause??null,q.rootCause??null,q.limit??100) as Array<{result_json:string}>;
    return rows.map(r=>JSON.parse(r.result_json));
  }
  async beginTransaction(sid: string): Promise<void> {
    this.db.exec('SAVEPOINT "'+this.esc(sid)+'"'); this.q.sUpsert.run(sid,'started',null,null);
  }
  async commitTransaction(sid: string): Promise<void> {
    this.db.exec('RELEASE SAVEPOINT "'+this.esc(sid)+'"');
  }
  async rollbackToCheckpoint(sid: string, cp: string): Promise<void> {
    this.db.exec('ROLLBACK TO SAVEPOINT "'+this.esc(cp)+'"');
  }
  async setCheckpoint(sid: string, name: string): Promise<void> {
    this.db.exec('SAVEPOINT "'+this.esc(name)+'"'); this.q.sUpsert.run(sid,'checkpoint',name,null);
  }
  close(): void { this.db.close(); }
}
