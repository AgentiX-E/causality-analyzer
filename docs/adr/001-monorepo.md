# ADR-001: Monorepo with pnpm Workspaces

## Status
Accepted (2026-06)

## Context
Causality Analyzer spans causal discovery, RCA, anomaly detection, storage,
and visualization. Each domain has distinct dependencies and consumers.
A single-package architecture would create tight coupling and giant dependency
trees for consumers who only need the core types.

## Decision
Use `pnpm` workspaces monorepo with 5 packages:

```
packages/
├── core/          # Zero runtime deps — types + ColumnarTable + math
├── pipeline/      # Causal algorithms — depends on core
├── storage-embed/ # SQLite + OverGraph — depends on core
├── storage-remote/# PostgreSQL + Neo4j — depends on core
└── visual/        # Lit Web Components — depends on core + pipeline
```

## Consequences

- ✅ Consumer can install only needed packages (e.g., `core` for type-only usage)
- ✅ Each package has independent `tsconfig`, `vitest.config`, coverage thresholds
- ✅ Cross-package refactoring tracked by pnpm workspace protocol
- ⚠️ Core rebuild required after interface changes before other packages see them
- ⚠️ CI must run `build` before `test` for dependent packages
