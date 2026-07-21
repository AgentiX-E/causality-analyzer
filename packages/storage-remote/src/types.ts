/**
 * Shared types for @agentix-e/causality-analyzer-storage-remote.
 *
 * Protocol-agnostic types used by both RemoteGraphStore (Bolt) and
 * RemoteRelationalStore (PG-wire). No backend-specific types here —
 * those live in their respective implementation files.
 *
 * @packageDocumentation
 */

/**
 * Canonical mTLS configuration — PEM-string-based, backend-agnostic.
 *
 * Shared by both RemoteGraphStore (Bolt) and RemoteRelationalStore (PG-wire).
 * Each backend adapts to its native format:
 *   - Bolt: CA → trustedCertificates, cert+key → temp files → ClientCertificate
 *   - PG:  CA → ssl.ca, cert+key → ssl.cert + ssl.key (native PEM support)
 */
export interface MtlsConfig {
  /** PEM-encoded CA certificate(s). Omit to use system trust store. */
  ca?: string | string[];
  /** PEM-encoded client certificate (for mTLS). */
  cert: string;
  /** PEM-encoded client private key (for mTLS). */
  key: string;
  /** Private key passphrase. */
  passphrase?: string;
}

/**
 * Typed TLS trust strategy.
 *
 * Mirrors neo4j-driver Config.TrustStrategy but also applies conceptually
 * to PG-wire (pg.Client ssl.rejectUnauthorized + ca).
 */
export type TrustStrategy =
  | 'TRUST_ALL_CERTIFICATES'
  | 'TRUST_CUSTOM_CA_SIGNED_CERTIFICATES'
  | 'TRUST_SYSTEM_CA_SIGNED_CERTIFICATES';
