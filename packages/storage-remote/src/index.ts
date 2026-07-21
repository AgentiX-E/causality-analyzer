/**
 * Remote Storage — PG-wire relational + Bolt protocol graph clients.
 *
 * RemoteRelationalStore: PostgreSQL via pg.Client, testable via pg-mem.
 * RemoteGraphStore: Neo4j via neo4j-driver-lite, testable via _Driver DI.
 *
 * Both stores share types from ./types.js (MtlsConfig, TrustStrategy).
 */
export { RemoteRelationalStore } from './remote-relational-store.js';
export { RemoteGraphStore } from './remote-graph-store.js';
export type { RemoteGraphConfig, RemoteGraphAuth } from './remote-graph-store.js';
export type { MtlsConfig, TrustStrategy } from './types.js';
