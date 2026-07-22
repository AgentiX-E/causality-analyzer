/**
 * RemoteGraphStore — enterprise-grade Bolt protocol graph store.
 *
 * Connects to Neo4j or any Bolt-compatible graph database via neo4j-driver-lite.
 * Requires a Bolt URI at construction — no in-process fallback.
 *
 * Features:
 * - Full auth: basic, bearer, kerberos, custom, none
 * - TLS: TRUST_SYSTEM_CA_SIGNED_CERTIFICATES by default, mTLS via clientCertificate
 * - Connection lifecycle: maxLifetime, acquisitionTimeout, liveness check
 * - UNWIND batched writes (single-transaction saveGraph)
 * - Exponential backoff retry using Neo4jError.isRetryable
 * - Structured logging via DI
 * - _Driver DI for test injection (BoltSessionMock)
 */
import type { IGraphStore, CausalGraph, GraphMetadata, GraphVersion } from '@agentix-e/causality-analyzer-core';
import type { MtlsConfig, TrustStrategy } from './types.js';
import { createRequire } from 'module';
import { writeFileSync, mkdtempSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
const _require = createRequire(import.meta.url);

// ── Types ──────────────────────────────────────────────────────────────

/** Auth discriminant. Covers the full Neo4j auth matrix. */
export type RemoteGraphAuth =
  | { type: 'basic'; user: string; password: string; realm?: string }
  | { type: 'bearer'; token: string }
  | { type: 'kerberos'; ticket: string }
  | { type: 'custom'; principal: string; credentials: string; realm: string; scheme: string; parameters?: Record<string, unknown> }
  | { type: 'none' };

/** Structured logger interface. Defaults to console. */
export interface GraphLogger {
  info(msg: string, meta?: Record<string, unknown>): void;
  warn(msg: string, meta?: Record<string, unknown>): void;
  error(msg: string, meta?: Record<string, unknown>): void;
  debug(msg: string, meta?: Record<string, unknown>): void;
}

/** Minimal neo4j.Driver-like interface for DI. */
export interface DriverLike {
  session(config?: { defaultAccessMode?: string; database?: string }): SessionLike;
  close(): Promise<void>;
}

/** Minimal neo4j.Session-like interface for DI. */
export interface SessionLike {
  run(query: string, params?: Record<string, unknown>): Promise<ResultLike>;
  executeWrite<T>(txWork: (tx: TxLike) => Promise<T>): Promise<T>;
  close(): Promise<void>;
}

/** Minimal neo4j.ManagedTransaction-like for executeWrite callback. */
export interface TxLike {
  run(query: string, params?: Record<string, unknown>): Promise<ResultLike>;
}

/** Minimal neo4j.Result-like interface. */
export interface ResultLike {
  records: RecordLike[];
}

/** Minimal neo4j.Record-like interface. */
export interface RecordLike {
  get(key: string): unknown;
}

/** Full RemoteGraphStore constructor config. */
export interface RemoteGraphConfig {
  /** REQUIRED: Bolt URI, e.g. neo4j+s://host:7687. Throws if absent. */
  uri: string;
  /** Auth credentials (default: basic neo4j/password). */
  auth?: RemoteGraphAuth;
  /** Connection pool size (default: 4). */
  maxPoolSize?: number;
  /** Max transaction retry time in ms (default: 15000). */
  maxTransactionRetryTime?: number;
  /** Retry attempts on transient failures (default: 3). */
  maxRetries?: number;
  /** Retry backoff base in ms (default: 100). */
  retryBaseMs?: number;
  /** TLS trust strategy (default: TRUST_SYSTEM_CA_SIGNED_CERTIFICATES). */
  trustStrategy?: TrustStrategy;
  /** PEM-encoded CA certificates for TRUST_CUSTOM_CA_SIGNED_CERTIFICATES. */
  trustedCertificates?: string[];
  /**
   * File-path-based mTLS client certificate (neo4j-native format).
   * Prefer `mtls` (PEM-string-based) unless you specifically need file paths.
   */
  clientCertificate?: { certfile: string; keyfile: string; password?: string };
  /**
   * PEM-string-based mTLS configuration — canonical format shared with PG-wire.
   * When provided, PEM certs are written to temp files for the Bolt driver.
   * Mutually exclusive with `clientCertificate`.
   */
  mtls?: MtlsConfig;
  /** Max connection lifetime in ms (default: 3_600_000 = 1 hour). */
  maxConnectionLifetime?: number;
  /** Connection acquisition timeout in ms (default: 60_000). */
  connectionAcquisitionTimeout?: number;
  /** Connection liveness check timeout in ms (default: 30_000). */
  connectionLivenessCheckTimeout?: number;
  /** Structured logger (default: console-based). */
  logger?: GraphLogger;
  /** Test DI: custom Driver constructor. Underscore = internal/DI only. */
  _Driver?: new (url: string, auth: unknown, config: Record<string, unknown>) => DriverLike;
}

// ── Helpers ────────────────────────────────────────────────────────────

const consoleLogger: GraphLogger = {
  info: (msg, meta) => console.info(`[RemoteGraphStore] ${msg}`, meta ?? ''),
  warn: (msg, meta) => console.warn(`[RemoteGraphStore] ${msg}`, meta ?? ''),
  error: (msg, meta) => console.error(`[RemoteGraphStore] ${msg}`, meta ?? ''),
  debug: () => { /* no-op by default */ },
};

function jaccardSimilarity(a: ReadonlyArray<string>, b: ReadonlyArray<string>): number {
  const sa = new Set(a), sb = new Set(b);
  const intersection = [...sa].filter(x => sb.has(x)).length;
  const union = new Set([...sa, ...sb]).size;
  return union === 0 ? 0 : intersection / union;
}

// ── RemoteGraphStore ───────────────────────────────────────────────────

export class RemoteGraphStore implements IGraphStore {
  private driver: DriverLike;
  private config: RemoteGraphConfig;
  private log: GraphLogger;
  private versionCounter = new Map<string, number>();
  private mtlsTempDir: string | null = null;

  constructor(config: RemoteGraphConfig) {
    if (!config.uri) throw new TypeError('RemoteGraphStore: "uri" is required — no in-process fallback exists. Provide a Bolt URI (e.g. neo4j+s://host:7687).');

    this.config = config;
    this.log = config.logger ?? consoleLogger;

    // Resolve Driver constructor: _Driver (test DI) or real neo4j-driver-lite
    const DriverCtor = config._Driver ?? _require('neo4j-driver-lite').driver;

    // Build auth token
    const auth = this.buildAuth(config.auth);

    // Build driver config
    const driverConfig: Record<string, unknown> = {
      maxConnectionPoolSize: config.maxPoolSize ?? 4,
      maxTransactionRetryTime: config.maxTransactionRetryTime ?? 15000,
      maxConnectionLifetime: config.maxConnectionLifetime ?? 3_600_000,
      connectionAcquisitionTimeout: config.connectionAcquisitionTimeout ?? 60_000,
      connectionLivenessCheckTimeout: config.connectionLivenessCheckTimeout ?? 30_000,
      trustStrategy: config.trustStrategy ?? 'TRUST_SYSTEM_CA_SIGNED_CERTIFICATES',
    };

    // CA certs
    if (config.trustedCertificates) driverConfig.trustedCertificates = config.trustedCertificates;

    // mTLS: prefer PEM-based mtls, fallback to file-path-based clientCertificate
    if (config.mtls) {
      if (config.clientCertificate) {
        throw new TypeError('RemoteGraphStore: "mtls" and "clientCertificate" are mutually exclusive — use one.');
      }
      driverConfig.clientCertificate = this.setupMtlsTempFiles(config.mtls);
    } else if (config.clientCertificate) {
      driverConfig.clientCertificate = config.clientCertificate;
    }

    this.driver = new DriverCtor(config.uri, auth, driverConfig);
    this.log.info('connected', {
      uri: config.uri,
      poolSize: driverConfig.maxConnectionPoolSize,
      mtls: !!config.mtls,
    });
  }

  /**
   * Write PEM cert/key strings to temp files for neo4j-driver-lite
   * (which requires file paths, not inline PEM). Temp dir is cleaned
   * up in close().
   */
  private setupMtlsTempFiles(mtls: MtlsConfig): { certfile: string; keyfile: string; password?: string } {
    this.mtlsTempDir = mkdtempSync(join(tmpdir(), 'ca-bolt-mtls-'));
    const certfile = join(this.mtlsTempDir, 'client.crt');
    const keyfile = join(this.mtlsTempDir, 'client.key');
    writeFileSync(certfile, mtls.cert, { mode: 0o600 });
    writeFileSync(keyfile, mtls.key, { mode: 0o600 });
    this.log.debug('mtls temp files written', { certfile, keyfile });
    return { certfile, keyfile, password: mtls.passphrase };
  }

  /** Build the appropriate neo4j AuthToken from config. */
  private buildAuth(authConfig?: RemoteGraphAuth): unknown {
    const neo4j = _require('neo4j-driver-lite');
    if (!authConfig) return neo4j.auth.basic('neo4j', 'password');

    switch (authConfig.type) {
      case 'basic':
        return neo4j.auth.basic(authConfig.user, authConfig.password, authConfig.realm);
      case 'bearer':
        return neo4j.auth.bearer(authConfig.token);
      case 'kerberos':
        return neo4j.auth.kerberos(authConfig.ticket);
      case 'custom':
        return neo4j.auth.custom(
          authConfig.principal,
          authConfig.credentials,
          authConfig.realm,
          authConfig.scheme,
          authConfig.parameters,
        );
      case 'none':
        return neo4j.auth.none();
    }
  }

  // ── Retry logic ─────────────────────────────────────────────────────

  private async retry<T>(fn: () => Promise<T>, op: string): Promise<T> {
    const { Neo4jError } = _require('neo4j-driver-lite');
    const max = this.config.maxRetries ?? 3;
    const base = this.config.retryBaseMs ?? 100;
    let attempt = 0;

    const go = async (): Promise<T> => {
      try { return await fn(); }
      catch (e: unknown) {
        attempt++;
        const retryable =
          (e instanceof Error && (e as any).code && Neo4jError.isRetryable(e)) ||
          ((e as any)?.code === 'ECONNREFUSED') ||
          ((e as any)?.code === 'ECONNRESET') ||
          ((e as any)?.code === 'ETIMEDOUT');

        if (attempt < max && retryable) {
          const delay = base * Math.pow(2, attempt);
          this.log.warn('retrying', { op, attempt, delayMs: delay });
          await new Promise(r => setTimeout(r, delay));
          return go();
        }
        this.log.error('operation failed', { op, attempt, max });
        throw e;
      }
    };
    return go();
  }

  // ── IGraphStore implementation ──────────────────────────────────────

  async saveGraph(graph: CausalGraph, metadata: GraphMetadata): Promise<string> {
    return this.retry(async () => {
      const id = metadata.id;
      const ver = (this.versionCounter.get(id) ?? 0) + 1;
      this.versionCounter.set(id, ver);

      const s = this.driver.session({ defaultAccessMode: 'WRITE' });
      try {
        // Single write transaction: upsert metadata, batch-insert nodes, batch-insert edges
        await s.executeWrite(async (tx: TxLike) => {
          // 1. Upsert graph metadata
          await tx.run(
            'MERGE (g:Graph { id: $id }) SET g.method = $method, g.confidence = $confidence, g.computedAt = $computedAt',
            { id, method: metadata.method, confidence: metadata.confidence, computedAt: metadata.computedAt },
          );

          // 2. Batch-create nodes via UNWIND
          if (graph.nodes.length > 0) {
            const nodesParam = graph.nodes.map(name => ({ gid: id, name, ver }));
            await tx.run(
              'UNWIND $nodes AS n MERGE (:Node { graphId: n.gid, name: n.name, version: n.ver })',
              { nodes: nodesParam },
            );
          }

          // 3. Batch-create edges via UNWIND
          if (graph.edges.length > 0) {
            const edgesParam = graph.edges.map(e => ({
              gid: id, ver,
              src: e.source, tgt: e.target,
              w: e.weight, d: e.directed,
            }));
            await tx.run(
              `UNWIND $edges AS e
               MATCH (a:Node { graphId: e.gid, name: e.src, version: e.ver })
               MATCH (b:Node { graphId: e.gid, name: e.tgt, version: e.ver })
               MERGE (a)-[r:DEPENDS_ON]->(b)
               SET r.weight = e.w, r.directed = e.d`,
              { edges: edgesParam },
            );
          }
        });
        this.log.debug('graph saved', { graphId: id, version: ver, nodes: graph.nodes.length, edges: graph.edges.length });
        return id;
      } finally { await s.close(); }
    }, 'saveGraph');
  }

  async loadGraph(graphId: string): Promise<CausalGraph | null> {
    return this.retry(async () => {
      const s = this.driver.session({ defaultAccessMode: 'READ' });
      try {
        // Find latest version
        const verRec = await s.run(
          'MATCH (n:Node { graphId: $id }) RETURN max(n.version) as latestVer',
          { id: graphId },
        );
        const latestVer = verRec.records[0]?.get('latestVer') as number | null;
        if (latestVer == null) return null;

        // Sequential: neo4j-driver v6 does not allow concurrent session.run() calls
        const nl = await s.run(
          'MATCH (n:Node { graphId: $id, version: $ver }) RETURN n.name ORDER BY n.name', { id: graphId, ver: latestVer },
        );
        const el = await s.run(
          `MATCH (a:Node { graphId: $id, version: $ver })-[r:DEPENDS_ON]->(b:Node { graphId: $id, version: $ver })
           RETURN a.name as source, b.name as target, r.weight as weight, r.directed as directed`,
          { id: graphId, ver: latestVer },
        );

        const nodes = nl.records.map((r: RecordLike) => r.get('n.name') as string);
        if (nodes.length === 0) return null;

        const edges = el.records.map((r: RecordLike) => ({
          source: r.get('source') as string,
          target: r.get('target') as string,
          weight: r.get('weight') as number,
          directed: r.get('directed') as boolean,
        }));

        return { nodes, edges };
      } finally { await s.close(); }
    }, 'loadGraph');
  }

  async loadGraphVersion(graphId: string, ver: number): Promise<CausalGraph | null> {
    return this.retry(async () => {
      const s = this.driver.session({ defaultAccessMode: 'READ' });
      try {
        // Sequential: neo4j-driver v6 does not allow concurrent s.run() on same session
        const nl = await s.run(
          'MATCH (n:Node { graphId: $id, version: $ver }) RETURN n.name ORDER BY n.name', { id: graphId, ver },
        );
        const el = await s.run(
          `MATCH (a:Node { graphId: $id, version: $ver })-[r:DEPENDS_ON]->(b:Node { graphId: $id, version: $ver })
           RETURN a.name as source, b.name as target, r.weight as weight, r.directed as directed`,
          { id: graphId, ver },
        );

        const nodes = nl.records.map((r: RecordLike) => r.get('n.name') as string);
        if (nodes.length === 0) return null;

        const edges = el.records.map((r: RecordLike) => ({
          source: r.get('source') as string,
          target: r.get('target') as string,
          weight: r.get('weight') as number,
          directed: r.get('directed') as boolean,
        }));

        return { nodes, edges };
      } finally { await s.close(); }
    }, 'loadGraphVersion');
  }

  async listGraphVersions(graphId: string): Promise<GraphVersion[]> {
    return this.retry(async () => {
      const s = this.driver.session({ defaultAccessMode: 'READ' });
      try {
        // Get distinct versions and their creation timestamps from Graph metadata
        const r = await s.run(
          `MATCH (g:Graph { id: $id })
           OPTIONAL MATCH (n:Node { graphId: $id })
           RETURN DISTINCT n.version as version, g.computedAt as computedAt, g.id as graphId
           ORDER BY n.version`,
          { id: graphId },
        );
        return r.records.map((rec: RecordLike) => ({
          graphId: rec.get('graphId') as string,
          version: rec.get('version') as number,
          timestamp: (rec.get('computedAt') as number) ?? Date.now(),
        }));
      } finally { await s.close(); }
    }, 'listGraphVersions');
  }

  async findSimilarGraphs(target: CausalGraph, limit: number): Promise<CausalGraph[]> {
    return this.retry(async () => {
      const s = this.driver.session({ defaultAccessMode: 'READ' });
      try {
        // Get all graph IDs
        const idRec = await s.run('MATCH (g:Graph) RETURN g.id as id');
        const allIds = idRec.records.map((r: RecordLike) => r.get('id') as string);

        // Load each, compute Jaccard similarity, sort descending
        const scored: Array<{ graph: CausalGraph; score: number }> = [];
        for (const gid of allIds) {
          const g = await this.loadGraph(gid);
          if (g) {
            const score = jaccardSimilarity(target.nodes, g.nodes);
            scored.push({ graph: g, score });
          }
        }
        scored.sort((a, b) => b.score - a.score);
        return scored.slice(0, limit).map(s => s.graph);
      } finally { await s.close(); }
    }, 'findSimilarGraphs');
  }

  async close(): Promise<void> {
    this.log.info('closing driver');
    await this.driver.close();
    // Clean up mTLS temp files
    if (this.mtlsTempDir) {
      try {
        const { rmSync } = await import('fs');
        rmSync(this.mtlsTempDir, { recursive: true, force: true });
        this.log.debug('mtls temp dir removed', { dir: this.mtlsTempDir });
      } catch { /* best-effort cleanup */ }
    }
  }
}
