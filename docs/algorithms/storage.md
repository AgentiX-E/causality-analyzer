# Data Storage

## Choosing a Backend

| Criteria | Embedded | Remote |
|----------|----------|--------|
| **Setup** | Zero config | PostgreSQL + Neo4j servers |
| **Persistence** | File-based (SQLite, OverGraph) | Server-managed |
| **Performance** | Single-process, in-memory possible | Connection-pooled, UNWIND batched |
| **Replication/Backup** | Manual file copy | Native PostgreSQL + Neo4j |
| **Testing** | Direct instantiation | Instance injection (pg-mem, BoltMock) |
| **mTLS** | N/A | Full support (Bolt + PG-wire) |
| **Best For** | Development, single-node, CI | Production, multi-node, enterprise |

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
