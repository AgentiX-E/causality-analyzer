# ADR-002: Instance Injection over Constructor Injection

## Status
Accepted (2026-07)

## Context
Storage backends need to be testable without real database servers.
We explored constructor injection (`_Client?: new (...a: any[]) => any`)
vs instance injection (`client?: PgClientLike`).

Constructor injection requires the test to construct a mock class that
matches the real constructor signature — fragile and magic-string dependent.

## Decision
Use **instance injection** via config objects:

```typescript
// Production
new RemoteRelationalStore({ connectionString: 'postgresql://...' })

// Test
new RemoteRelationalStore({ client: pgMemAdapter })
```

The Store creates its own client when no instance is provided, or uses
the pre-configured instance directly. This is the "doorman" pattern:
either the caller provides a client, or the Store creates one.

## Consequences

- ✅ Clean public API — no underscore-prefixed magic fields
- ✅ Tests provide real adapter instances (pg-mem Client, BoltSessionMock)
- ✅ Production config builder (`buildPgClientOpts`) is a pure testable function
- ⚠️ `new pg.Client()` line is inherently integration-only (1 uncovered line)
- ✅ Counterpart: `_Driver` for RemoteGraphStore (neo4j has no pg-mem equivalent)

## Alternatives Considered

- **Constructor injection** (`_Client: MockClass`): Rejected — underscore convention
  signals "internal" but tests must touch it.
- **Docker-only testing**: Rejected — too slow for local dev iteration.
- **Monkey-patching `pg` module**: Rejected — fragile, breaks parallel test runs.
