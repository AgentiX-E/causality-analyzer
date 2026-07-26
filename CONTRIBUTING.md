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
├── core/            # Types, interfaces, ColumnarTable, PluginRegistry, math
├── pipeline/        # Detection, causal discovery, RCA, inference, GCM, HTTP API
├── storage-embed/   # node:sqlite + OverGraph embedded stores
├── storage-browser/ # WASM SQLite + OPFS browser stores
├── storage-remote/  # PostgreSQL (pg) + Neo4j remote stores
└── visual/          # Lit 3 Web Components for causal graph + timeseries
```

## Quality Gates (CI enforced)

| Gate | Command | Description |
|------|---------|-------------|
| Lint | `pnpm -r lint` | ESLint — no-unsafe-* rules enforced as errors |
| Typecheck | `pnpm -r typecheck` | `tsc --noEmit` on all 6 packages |
| Test | `pnpm -r test` | Vitest with v8 coverage |
| Browser | Playwright | Visual + storage-browser E2E (Chromium) |
| Neo4j | `test:neo4j` | Docker Neo4j integration tests |

## Development Workflow

1. **Create a branch** from `master`
2. **Write tests first** — verify they fail
3. **Implement** the feature or fix
4. **Run quality gates** locally:
   ```bash
   pnpm run --filter @agentix-e/causality-analyzer-core build
   pnpm -r lint
   pnpm -r test
   ```
5. **Push** and open a PR — CI runs all gates

## Coverage Requirements (actual, CI enforced)

| Package | Stmts | Branches | Funcs | Lines |
|---------|-------|----------|-------|-------|
| core | 95% | 83% | 95% | 95% |
| pipeline | 92% | 77% | 92% | 94% |
| storage-embed | 95% | 95% | 95% | 95% |
| storage-browser | 90% | 80% | 90% | 90% |
| storage-remote | 77% | 58% | 55% | 77% |
| visual | 55% | 52% | 42% | 50% |

Storage-remote thresholds reflect pg-mem test coverage (Neo4j tests run in CI via Docker).
Visual thresholds reflect Lit decorator infrastructure (Canvas2DRenderer at 98% individually).

## Architecture Principles

1. **core is pure contracts** — interfaces, types, and foundational data structures.
2. **DI over inheritance** — stores are injected via instance config.
3. **No in-process fallbacks** — remote stores fail fast, embedded stores are explicitly chosen.
4. **Test with real backends** — pg-mem for PostgreSQL, overgraph for graph, BoltMock for Neo4j.
5. **Type safety first** — `no-unsafe-*` rules enforced as errors.

## Known Limitations

### NOTEARS Algorithm
NOTEARS converged to zero edges on ASIA benchmarks with certain seeds. The gradient-based optimization is sensitive to initialization and hyperparameters.

## Commit Convention

```
I{N}: <short description>

- bullet points of changes
- tests: N added, M modified
```

## Questions?

Open an issue on [GitHub](https://github.com/AgentiX-E/causality-analyzer/issues).
