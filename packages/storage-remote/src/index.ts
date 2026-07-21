/**
 * Remote Storage — PG-wire relational + Bolt protocol graph clients.
 *
 * RemoteRelationalStore: PostgreSQL via pg.Client, testable via client DI.
 * RemoteGraphStore: Neo4j via neo4j-driver-lite, testable via _Driver DI.
 *
 * Shared types: MtlsConfig, TrustStrategy (./types.js)
 * Shared interface: PgClientLike (./remote-relational-store.js)
 */
export { RemoteRelationalStore } from './remote-relational-store.js';
export type { PgClientLike } from './remote-relational-store.js';
export { RemoteGraphStore } from './remote-graph-store.js';
export type { RemoteGraphConfig, RemoteGraphAuth } from './remote-graph-store.js';
export type { MtlsConfig, TrustStrategy } from './types.js';
