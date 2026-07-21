# ADR-003: Unified mTLS via PEM-based MtlsConfig

## Status
Accepted (2026-07)

## Context
Both Bolt (neo4j-driver-lite) and PG-wire (pg.Client) backends need mTLS.
Neo4j requires file paths for client certificates; pg.Client accepts inline
PEM strings. We needed a unified API that works for both without coupling.

## Decision
Define `MtlsConfig` as the canonical PEM-string-based type:

```typescript
interface MtlsConfig {
  ca?: string | string[];
  cert: string;
  key: string;
  passphrase?: string;
}
```

Bolt adapter writes PEM to temp files → `ClientCertificate(certfile, keyfile)`.
PG adapter passes PEM directly → `ssl.cert`, `ssl.key`.

Shared type lives in `packages/storage-remote/src/types.ts` —
both store implementations import from this independent module.

## Consequences

- ✅ Single public API for mTLS across all remote backends
- ✅ PEM strings are the most portable format (env vars, vault, config files)
- ✅ Bolt temp files auto-cleaned in `close()`
- ⚠️ Bolt path requires filesystem access (temp dir) — sandboxed environments
  must allow `fs.writeFileSync`
