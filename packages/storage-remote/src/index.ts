/**
 * Remote Storage — PG-wire relational + Bolt protocol graph clients.
 *
 * RemoteRelationalStore: PostgreSQL via pg.Client, testable via pg-mem.
 * RemoteGraphStore: Neo4j via neo4j-driver-lite, testable via _Driver DI.
 *
 * Both stores share the MtlsConfig type for unified PEM-based mTLS.
 */
export { RemoteRelationalStore } from './remote-relational-store.js';
export { RemoteGraphStore } from './remote-graph-store.js';
export type { RemoteGraphConfig, RemoteGraphAuth, TrustStrategy, MtlsConfig } from './remote-graph-store.js';
