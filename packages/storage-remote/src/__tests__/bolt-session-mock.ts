/**
 * TEST-ONLY: Minimal Bolt protocol mock for RemoteGraphStore testing.
 *
 * Implements the neo4j-driver-lite Session/Transaction/Result/Record
 * API surface minimally for testing. Supports MERGE, MATCH, UNWIND,
 * RETURN, ORDER BY, SET, max(), and DISTINCT.
 *
 * NEVER imported by production code. Lives in __tests__/.
 */
import type { DriverLike, SessionLike, TxLike, ResultLike, RecordLike } from '../remote-graph-store.js';

// ── In-memory graph database ──────────────────────────────────────────

interface MockNode {
  graphId: string;
  name: string;
  version: number;
}

interface MockRelationship {
  from: string;  // "graphId::name::version"
  to: string;    // "graphId::name::version"
  weight: number;
  directed: boolean;
}

interface MockGraphMeta {
  id: string;
  method: string;
  confidence: number;
  computedAt: number;
}

class BoltDatabase {
  nodes: MockNode[] = [];
  relationships: MockRelationship[] = [];
  graphMetas: MockGraphMeta[] = [];

  reset(): void {
    this.nodes = [];
    this.relationships = [];
    this.graphMetas = [];
  }
}

// ── Minimal Cypher executor ───────────────────────────────────────────

interface CypherParam {
  gid?: string;
  id?: string;
  name?: string;
  ver?: number;
  version?: number;
  method?: string;
  confidence?: number;
  computedAt?: number;
  src?: string;
  tgt?: string;
  w?: number;
  d?: boolean;
  limit?: number;
  nodes?: Array<{ gid: string; name: string; ver: number }>;
  edges?: Array<{ gid: string; ver: number; src: string; tgt: string; w: number; d: boolean }>;
}

function nodeKey(n: { graphId: string; name: string; version: number }): string {
  return `${n.graphId}::${n.name}::${n.version}`;
}

/** Execute a minimal Cypher subset and return RecordLike[]. */
function executeCypher(db: BoltDatabase, query: string, params: Record<string, unknown> = {}): RecordLike[] {
  const q = query.trim();

  // ── MERGE (g:Graph { ... }) SET ... ──────────────────────────────────
  if (q.includes('MERGE (g:Graph')) {
    const id = (params.id ?? params['id']) as string;
    const existing = db.graphMetas.find(m => m.id === id);
    if (existing) {
      if (params.method !== undefined) existing.method = params.method as string;
      if (params.confidence !== undefined) existing.confidence = params.confidence as number;
      if (params.computedAt !== undefined) existing.computedAt = params.computedAt as number;
    } else {
      db.graphMetas.push({
        id,
        method: (params.method as string) ?? '',
        confidence: (params.confidence as number) ?? 0,
        computedAt: (params.computedAt as number) ?? Date.now(),
      });
    }
    return [];
  }

  // ── UNWIND $nodes AS n MERGE (:Node { ... }) ────────────────────────
  if (q.includes('UNWIND $nodes')) {
    const nodesArr = (params.nodes ?? params['nodes']) as Array<{ gid: string; name: string; ver: number }> | undefined;
    if (nodesArr) {
      for (const n of nodesArr) {
        const exists = db.nodes.some(ex =>
          ex.graphId === n.gid && ex.name === n.name && ex.version === n.ver);
        if (!exists) {
          db.nodes.push({ graphId: n.gid, name: n.name, version: n.ver });
        }
      }
    }
    return [];
  }

  // ── UNWIND $edges AS e MATCH ... MERGE ... SET ... ────────────────────
  if (q.includes('UNWIND $edges')) {
    const edgesArr = (params.edges ?? params['edges']) as Array<{
      gid: string; ver: number; src: string; tgt: string; w: number; d: boolean;
    }> | undefined;
    if (edgesArr) {
      for (const e of edgesArr) {
        const fromKey = `${e.gid}::${e.src}::${e.ver}`;
        const toKey = `${e.gid}::${e.tgt}::${e.ver}`;
        const aExists = db.nodes.some(n => nodeKey(n) === fromKey);
        const bExists = db.nodes.some(n => nodeKey(n) === toKey);
        if (!aExists || !bExists) continue;
        const exists = db.relationships.some(r => r.from === fromKey && r.to === toKey);
        if (exists) {
          const rel = db.relationships.find(r => r.from === fromKey && r.to === toKey)!;
          rel.weight = e.w;
          rel.directed = e.d;
        } else {
          db.relationships.push({ from: fromKey, to: toKey, weight: e.w, directed: e.d });
        }
      }
    }
    return [];
  }

  // ── MATCH (n:Node { graphId, version }) RETURN n.name ───────────────
  if (q.includes('RETURN n.name') && q.includes('MATCH (n:Node')) {
    const gid = (params.id ?? params['id']) as string;
    const ver = (params.ver ?? params['ver'] ?? params.version ?? params['version']) as number | undefined;

    let results = db.nodes.filter(n => n.graphId === gid);
    if (ver !== undefined) results = results.filter(n => n.version === ver);

    if (q.includes('ORDER BY n.name')) {
      results.sort((a, b) => a.name.localeCompare(b.name));
    }

    return results.map(n => new MockRecord({ 'n.name': n.name }));
  }

  // ── MATCH (n:Node { graphId }) RETURN max(n.version) ─────────────────
  if (q.includes('max(n.version)') || q.includes('max(n.version) as latestVer')) {
    const gid = (params.id ?? params['id']) as string;
    const versions = db.nodes.filter(n => n.graphId === gid).map(n => n.version);
    const maxVer = versions.length > 0 ? Math.max(...versions) : null;
    return [new MockRecord({ latestVer: maxVer })];
  }

  // ── MATCH (a:Node)-[r:DEPENDS_ON]->(b:Node) RETURN a.name, b.name, r.weight, r.directed ──
  if (q.includes('r.weight') && q.includes('r.directed')) {
    const gid = (params.id ?? params['id']) as string;
    const ver = (params.ver ?? params['ver'] ?? params.version ?? params['version']) as number | undefined;

    const nodeMap = new Map<string, string>(); // key → name
    for (const n of db.nodes) {
      if (n.graphId === gid && (ver === undefined || n.version === ver)) {
        nodeMap.set(nodeKey(n), n.name);
      }
    }

    return db.relationships
      .filter(r => nodeMap.has(r.from) && nodeMap.has(r.to))
      .map(r => new MockRecord({
        source: nodeMap.get(r.from),
        target: nodeMap.get(r.to),
        weight: r.weight,
        directed: r.directed,
      }));
  }

  // ── MATCH (g:Graph) RETURN g.id ─────────────────────────────────────
  if (q.includes('MATCH (g:Graph)') && q.includes('RETURN g.id')) {
    let metas = [...db.graphMetas];
    if (q.includes('LIMIT $limit')) {
      metas = metas.slice(0, (params.limit as number) ?? 100);
    }
    return metas.map(m => new MockRecord({ id: m.id }));
  }

  // ── MATCH (g:Graph { id }) ... RETURN DISTINCT n.version, g.computedAt, g.id ──
  if (q.includes('DISTINCT n.version')) {
    const gid = (params.id ?? params['id']) as string;
    const meta = db.graphMetas.find(m => m.id === gid);
    const versions = [...new Set(db.nodes.filter(n => n.graphId === gid).map(n => n.version))].sort((a, b) => a - b);
    return versions.map(v => new MockRecord({
      graphId: gid,
      version: v,
      computedAt: meta?.computedAt ?? Date.now(),
    }));
  }

  // ── MATCH ... RETURN a.name as source, b.name as target, r.weight as weight, r.directed as directed ──
  // (used in loadGraph parallel query)
  if (q.includes('a.name as source') && q.includes('b.name as target')) {
    const gid = (params.id ?? params['id']) as string;
    const ver = (params.ver ?? params['ver'] ?? params.version ?? params['version']) as number | undefined;

    const nodeMap = new Map<string, string>();
    for (const n of db.nodes) {
      if (n.graphId === gid && (ver === undefined || n.version === ver)) {
        nodeMap.set(nodeKey(n), n.name);
      }
    }

    return db.relationships
      .filter(r => nodeMap.has(r.from) && nodeMap.has(r.to))
      .map(r => new MockRecord({
        source: nodeMap.get(r.from),
        target: nodeMap.get(r.to),
        weight: r.weight,
        directed: r.directed,
      }));
  }

  return [];
}

// ── Record mock ────────────────────────────────────────────────────────

class MockRecord implements RecordLike {
  private data: Record<string, unknown>;
  constructor(data: Record<string, unknown>) { this.data = data; }
  get(key: string): unknown { return this.data[key]; }
}

// ── Session / Transaction mocks ────────────────────────────────────────

class MockTx implements TxLike {
  constructor(private db: BoltDatabase) {}
  async run(query: string, params?: Record<string, unknown>): Promise<ResultLike> {
    return { records: executeCypher(this.db, query, params) };
  }
}

class MockSession implements SessionLike {
  private db: BoltDatabase;
  constructor(db: BoltDatabase) { this.db = db; }
  async run(query: string, params?: Record<string, unknown>): Promise<ResultLike> {
    return { records: executeCypher(this.db, query, params) };
  }
  async executeWrite<T>(txWork: (tx: TxLike) => Promise<T>): Promise<T> {
    return txWork(new MockTx(this.db));
  }
  async close(): Promise<void> { /* no-op */ }
}

// ── Driver mock ────────────────────────────────────────────────────────

export class BoltDriverMock implements DriverLike {
  private db = new BoltDatabase();

  /** Access the underlying database for test assertions. */
  get database(): BoltDatabase { return this.db; }

  /** Reset all data between tests. */
  reset(): void { this.db.reset(); }

  session(_config?: { defaultAccessMode?: string; database?: string }): SessionLike {
    return new MockSession(this.db);
  }

  async close(): Promise<void> {
    this.reset();
  }
}

// ── Convenience: throw-if-retryable error helper ───────────────────────

export function makeRetryableError(code: string): Error {
  const err = new Error(`Mock ${code}`);
  (err as any).code = code;
  return err;
}

export function makeNonRetryableError(): Error {
  const err = new Error('Mock permanent error');
  (err as any).code = 'Neo.ClientError.Statement.SyntaxError';
  return err;
}
