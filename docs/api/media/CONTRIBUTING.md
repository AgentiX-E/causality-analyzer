# Contributing to Causality Analyzer

## Development Setup

```bash
git clone https://github.com/AgentiX-E/causality-analyzer.git
cd causality-analyzer
pnpm install
pnpm run --filter @agentix-e/causality-analyzer-core build
```

## Monorepo Structure

```
packages/
├── core/          # Types, interfaces, ColumnarTable, PluginRegistry, math
├── pipeline/      # Detection, causal discovery, RCA, inference, GCM, visualization
├── storage-embed/ # SQLite (better-sqlite3) + OverGraph embedded stores
├── storage-remote/# PostgreSQL (pg) + Neo4j (neo4j-driver-lite) remote stores
└── visual/        # Lit 3 Web Components for causal graph + timeseries visualization
```

## Quality Gates (CI enforced)

| Gate | Command | Description |
|------|---------|-------------|
| Lint | `pnpm -r lint` | ESLint with `@typescript-eslint/recommended` |
| Typecheck | `pnpm -r typecheck` | `tsc --noEmit` on all 5 packages |
| Test | `pnpm -r test` | Vitest with v8 coverage (95%+ threshold) |
| Browser | `pnpm run --filter ...visual browser-test` | Playwright Chromium |
| Neo4j | `pnpm run --filter ...storage-remote test:neo4j` | Docker Neo4j 5 with mTLS |

## Development Workflow

1. **Create a branch** from `main`
2. **Write tests first** — verify they fail
3. **Implement** the feature or fix
4. **Run quality gates** locally:
   ```bash
   pnpm run --filter @agentix-e/causality-analyzer-core build
   pnpm -r lint
   pnpm -r typecheck
   pnpm -r test
   ```
5. **Push** and open a PR — CI runs all gates + browser + Neo4j tests

## Coverage Requirements

Every package must maintain ≥ 95% coverage on statements, branches, functions, and lines.

| Package | Statements | Branches | Functions | Lines |
|---------|-----------|----------|-----------|-------|
| core | 95% | 95% | 95% | 95% |
| pipeline | 95% | 90% | 95% | 95% |
| storage-embed | 95% | 95% | 95% | 95% |
| storage-remote | 85% | 80% | 55% | 85% |
| visual | 48% | 70% | 70% | 48% |

(Visual thresholds are lower because Lit decorators are framework infrastructure.)

## Architecture Principles

1. **core is pure contracts** — interfaces, types, and foundational data structures. No heavy implementations.
2. **DI over inheritance** — stores are injected via instance config, not constructor classes.
3. **No in-process fallbacks** — remote stores fail fast, embedded stores are explicitly chosen.
4. **Test with real backends** — pg-mem for PostgreSQL, overgraph for graph, BoltSessionMock + real Neo4j in CI.
5. **Type safety first** — `any` is a warning, `as` casts must be justified.

## Adding a New Causal Discovery Algorithm

1. Implement in `packages/pipeline/src/graph/`
2. Export via `graph/index.ts` and `pipeline/src/index.ts`
3. Test: `i{N}-*.test.ts` in `__tests__/` with synthetic DAGs
4. Document with JSDoc — all public API must have `@param` and `@returns`

## Commit Convention

```
I{N}: <short description>

- bullet points of changes
- tests: N added, M modified
```

## Questions?

Open an issue on [GitHub](https://github.com/AgentiX-E/causality-analyzer/issues).
