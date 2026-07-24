# @agentix-e/causality-analyzer-storage-remote

> Enterprise-grade remote storage — PostgreSQL for relational data, Neo4j for causal graphs, with full mTLS support.

[![npm](https://img.shields.io/badge/version-1.0.0-blue)](https://www.npmjs.com/package/@agentix-e/causality-analyzer-storage-remote)

## Overview

`@agentix-e/causality-analyzer-storage-remote` provides remote storage backends for production deployments. Both relational (PostgreSQL via `pg`) and graph (Neo4j via `neo4j-driver-lite`) stores support mutual TLS authentication, connection lifecycle management, and instance injection for testing.

### Architecture

```
RemoteRelationalStore          RemoteGraphStore
        │                            │
        │ PgClientLike               │ DriverLike
        │ (instance DI)              │ (_Driver DI)
        │                            │
   ┌────┴────┐                  ┌────┴────┐
   │ pg.Client│                  │neo4j    │
   │ (prod)   │                  │driver   │
   └────┬────┘                  └────┬────┘
        │                            │
   ┌────┴────┐                  ┌────┴────┐
   │ pg-mem  │                  │BoltMock │
   │ (test)  │                  │(test)   │
   └────┬────┘                  └────┬────┘
        │                            │
   PostgreSQL                   Neo4j 5+
   (PG-wire)                    (Bolt)
```

## Installation

```bash
npm install @agentix-e/causality-analyzer-storage-remote
```

pg (PostgreSQL) is an optional dependency — install separately if needed:
```bash
npm install pg
```

neo4j-driver-lite is a required dependency (no in-process fallback).

## Quick Start

### RemoteRelationalStore (PostgreSQL)

```typescript
import { RemoteRelationalStore } from '@agentix-e/causality-analyzer-storage-remote';

// Production with connection string
const store = new RemoteRelationalStore({
  connectionString: 'postgresql://user:pass@db.example.com:5432/mydb',
  mtls: {
    cert: fs.readFileSync('/etc/certs/client.crt', 'utf8'),
    key:  fs.readFileSync('/etc/certs/client.key', 'utf8'),
  },
});

// Testing with pg-mem (no real PostgreSQL!)
import { newDb } from 'pg-mem';
const { Client } = newDb().adapters.createPg();
const testStore = new RemoteRelationalStore({ client: new Client() });

// Usage
await store.saveCPT('g1', 'CPU', { node: 'CPU', parents: [], entries: { '0': 0.3 } });
const cpt = await store.loadCPT('g1', 'CPU');

await store.close();
```

### RemoteGraphStore (Neo4j)

```typescript
import { RemoteGraphStore } from '@agentix-e/causality-analyzer-storage-remote';

// Production with mTLS
const store = new RemoteGraphStore({
  uri: 'neo4j+s://db.example.com:7687',
  auth: { type: 'basic', user: 'neo4j', password: 'password' },
  mtls: {
    cert: fs.readFileSync('/etc/certs/client.crt', 'utf8'),
    key:  fs.readFileSync('/etc/certs/client.key', 'utf8'),
  },
  trustStrategy: 'TRUST_CUSTOM_CA_SIGNED_CERTIFICATES',
  maxPoolSize: 8,
});

// Save a graph (single-transaction UNWIND batch)
const id = await store.saveGraph(
  { nodes: ['A', 'B', 'C'], edges: [{ source: 'A', target: 'B', weight: 1, directed: true }] },
  { id: 'g1', method: 'pc', computedAt: Date.now(), parameters: {}, confidence: 0.9 },
);

// Load with versioning
const latest = await store.loadGraph(id);
const v1 = await store.loadGraphVersion(id, 1);

await store.close();
```

## Authentication

RemoteGraphStore supports the full Neo4j auth matrix:

```typescript
// Basic
{ type: 'basic', user: 'neo4j', password: 'secret' }

// Bearer token (SSO/OIDC)
{ type: 'bearer', token: 'eyJhbGci...' }

// Kerberos (Enterprise)
{ type: 'kerberos', ticket: 'base64encoded' }

// Custom (LDAP, etc.)
{ type: 'custom', principal: 'user', credentials: 'pw', realm: 'LDAP', scheme: 'basic' }

// No auth
{ type: 'none' }
```

## mTLS Configuration

Both stores share the same `MtlsConfig` type:

```typescript
interface MtlsConfig {
  ca?: string | string[];   // CA certificate (PEM)
  cert: string;             // Client certificate (PEM)
  key: string;              // Client private key (PEM)
  passphrase?: string;      // Key passphrase
}
```

**Bolt path:** PEM → temp files → `neo4j.ClientCertificate(certfile, keyfile)` (auto-cleaned on `close()`)

**PG path:** PEM → `ssl.cert`, `ssl.key` (pg.Client natively supports inline PEM)

## API Reference

### `RemoteRelationalStore` implements `IRelationalStore`

Full CRUD for metrics, CPTs, regression models, RCA results, and transaction management. Supports `client` instance injection for testing.

### `RemoteGraphStore` implements `IGraphStore`

Graph CRUD with UNWIND batched writes, versioned storage, and Jaccard similarity search. Requires Bolt URI — no in-process fallback.

### Shared Types

| Type | Description |
|------|-------------|
| `MtlsConfig` | Canonical PEM-based mTLS configuration |
| `TrustStrategy` | `TRUST_ALL_CERTIFICATES` / `TRUST_CUSTOM_CA_SIGNED_CERTIFICATES` / `TRUST_SYSTEM_CA_SIGNED_CERTIFICATES` |
| `RemoteGraphAuth` | Discriminated union for all auth types |
| `PgClientLike` | Minimal pg.Client interface for instance DI |

## License

MIT
