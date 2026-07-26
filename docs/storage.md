# Data Storage

## Choosing a Backend

| Criteria | Embedded (Node) | Browser | Remote |
|----------|:---:|:---:|:---:|
| **Package** | storage-embed | storage-browser | storage-remote |
| **Setup** | Zero config | Worker + OPFS | PostgreSQL + Neo4j |
| **Persistence** | File (node:sqlite) | OPFS (survives reload) | Server-managed |
| **Offline** | ✅ | ✅ | ❌ |
| **Browser** | ❌ | ✅ | ❌ |
| **mTLS** | N/A | N/A | ✅ |
| **Best For** | CLI, server, CI | PWA, SPA, desktop | Enterprise, multi-node |

All three implement `IRelationalStore` + `IGraphStore`. Pipeline code identical across backends.

---

## Embedded Storage

### EmbedRelationalStore (SQLite)

Zero-configuration relational storage using `better-sqlite3`. Supports file-based persistence, `:memory:` mode, and transaction management with savepoints.

```typescript
import { EmbedRelationalStore } from '@agentix-e/causality-analyzer-storage-embed';

// In-memory (fast, ephemeral)
const store = new EmbedRelationalStore({ dbPath: ':memory:' });

// Persistent file
const store2 = new EmbedRelationalStore({ dbPath: './causality.db' });

// Transaction with checkpoints
await store.beginTransaction('session1');
await store.saveCPT('graph1', 'CPU', cpt);
await store.setCheckpoint('session1', 'before_update');
// ... more operations ...
await store.rollbackToCheckpoint('session1', 'before_update'); // undo
await store.commitTransaction('session1');
```

### EmbedGraphStore (OverGraph)

LSM-tree based graph storage. Each graph version is stored under a dedicated label for O(1) lookup.

```typescript
import { EmbedGraphStore } from '@agentix-e/causality-analyzer-storage-embed';

const store = new EmbedGraphStore({ dbPath: './graphs' });
const id = await store.saveGraph(graph, metadata);
const latest = await store.loadGraph(id);
const v2 = await store.loadGraphVersion(id, 2);
```

---

## Browser Storage (WASM SQLite + OPFS)

### Architecture

```
Main Thread              Web Worker
WasmRelationalStore ───▶ sqlite-worker.ts
WasmGraphStore      ◀─── OPFS + WASM SQLite
```

The `SqlitePort` abstraction (`DirectSqlitePort` for vitest, `WorkerSqlitePort` for browser) makes the same `WasmRelationalStore`/`WasmGraphStore` code run identically in both environments.

### WasmRelationalStore

Identical SQL schema to storage-embed. Supports all `IRelationalStore` operations including SAVEPOINT/ROLLBACK.

### WasmGraphStore

Stores causal graphs in `graph_nodes` + `graph_edges` SQLite tables. Supports versioning and causal fingerprint-based similarity search.

### Browser Support

| Browser | Min Version |
|---------|------------|
| Chrome | 102+ |
| Edge | 102+ |
| Firefox | 111+ |
| Safari | 15.2+ |

Requires HTTPS or localhost (OPFS = secure context only).

---

## Remote Storage

### RemoteRelationalStore (PostgreSQL)

Connection-pooled PostgreSQL client with configurable SSL/mTLS. Supports instance injection for testing with `pg-mem`.

**Production:**

```typescript
import { RemoteRelationalStore } from '@agentix-e/causality-analyzer-storage-remote';

const store = new RemoteRelationalStore({
  connectionString: 'postgresql://user:pass@host:5432/db',
  mtls: {
    cert: fs.readFileSync('/etc/ssl/client.crt', 'utf8'),
    key:  fs.readFileSync('/etc/ssl/client.key', 'utf8'),
    ca:   fs.readFileSync('/etc/ssl/ca.crt', 'utf8'),
  },
});
```

**Testing (no real PostgreSQL):**

```typescript
import { newDb } from 'pg-mem';

const { Client } = newDb().adapters.createPg();
const store = new RemoteRelationalStore({ client: new Client() });
```

### RemoteGraphStore (Neo4j)

Bolt protocol graph store with connection lifecycle management, exponential backoff retry, and UNWIND batched writes (single-transaction saveGraph).

**Production with mTLS:**

```typescript
import { RemoteGraphStore } from '@agentix-e/causality-analyzer-storage-remote';

const store = new RemoteGraphStore({
  uri: 'neo4j+s://db.example.com:7687',
  auth: { type: 'basic', user: 'neo4j', password: 'secret' },
  mtls: { cert: pemCert, key: pemKey },
  maxPoolSize: 8,
  maxConnectionLifetime: 3_600_000,
});
```

**Authentication matrix:**

| Type | Use Case | Config |
|------|----------|--------|
| `basic` | Username/password | `{ type: 'basic', user, password }` |
| `bearer` | SSO/OIDC token | `{ type: 'bearer', token }` |
| `kerberos` | Enterprise Kerberos | `{ type: 'kerberos', ticket }` |
| `custom` | LDAP, custom | `{ type: 'custom', principal, credentials, realm, scheme }` |
| `none` | No auth | `{ type: 'none' }` |

**Testing (no real Neo4j):**

```typescript
import { BoltDriverMock } from '@agentix-e/causality-analyzer-storage-remote/__tests__/bolt-session-mock';

const driver = new BoltDriverMock();
const store = new RemoteGraphStore({
  uri: 'bolt://localhost:7687',
  _Driver: class { constructor() {} session(cfg) { return driver.session(cfg); } close() {} },
});
```

### Interface-Based Backend Switching

Both stores implement standard interfaces (`IRelationalStore`, `IGraphStore`), enabling zero-code-change backend switching:

```typescript
function analyze(store: IRelationalStore, graphStore: IGraphStore) {
  // Works with any backend
}

// Development
analyze(new EmbedRelationalStore({ dbPath: ':memory:' }), new EmbedGraphStore());

// Production
analyze(
  new RemoteRelationalStore({ connectionString: '...', mtls: {...} }),
  new RemoteGraphStore({ uri: 'neo4j+s://...', mtls: {...} }),
);
```

[← Back to User Guide](../user-guide.md)
