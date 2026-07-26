# @agentix-e/causality-analyzer-storage-browser

> Browser-native storage — WASM SQLite + OPFS persistence, zero native dependencies.

[![npm](https://img.shields.io/npm/v/@agentix-e/causality-analyzer-storage-browser)](https://npmjs.com/package/@agentix-e/causality-analyzer-storage-browser)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

`@agentix-e/causality-analyzer-storage-browser` provides persistent relational and graph storage that runs entirely in the browser. Powered by the official SQLite WASM build with Origin Private File System (OPFS) for durability, it enables full causality-analyzer pipelines to run offline in a browser tab — no server required.

**Key features:**
- **Zero native dependencies** — pure WASM, no node-gyp, no native addons
- **OPFS persistence** — data survives page reloads (Chrome 102+, Firefox 111+, Safari 15.2+)
- **Full IRelationalStore + IGraphStore** — drop-in replacement for storage-embed/storage-remote
- **SqlitePort abstraction** — swap between WorkerSqlitePort (browser) and DirectSqlitePort (testing)
- **Shared WASM instance** — one SQLite database serves both relational and graph storage

### Architecture

```
Main Thread                     Web Worker
┌─────────────────┐            ┌──────────────────────────────┐
│ WasmGraphStore   │──msg──▶   │  sqlite-worker.ts            │
│ WasmRelationalS. │◀──msg──   │  OPFS + WASM SQLite          │
│ WorkerSqlitePort │            │  ┌───────────────────────┐  │
└─────────────────┘            │  │ /causality-analyzer.db │  │
                               │  │  ├── metrics           │  │
                               │  │  ├── cpt               │  │
                               │  │  ├── regression_models │  │
                               │  │  ├── rca_results       │  │
                               │  │  ├── analysis_state    │  │
                               │  │  ├── graph_nodes       │  │
                               │  │  └── graph_edges       │  │
                               │  └───────────────────────┘  │
                               └──────────────────────────────┘
```

## Installation

```bash
npm install @agentix-e/causality-analyzer-storage-browser
npm install @agentix-e/causality-analyzer-core  # peer dependency
```

## Quick Start

### Browser (Production)

```typescript
import { WorkerSqlitePort, WasmRelationalStore, WasmGraphStore } from '@agentix-e/causality-analyzer-storage-browser';

// Create a port backed by a Web Worker running OPFS SQLite
const port = new WorkerSqlitePort(
  new URL('./node_modules/@agentix-e/causality-analyzer-storage-browser/dist/src/sqlite-worker.js', import.meta.url)
);

// Create stores sharing the same SQLite instance
const relStore = new WasmRelationalStore(port);
const graphStore = new WasmGraphStore(port);

// Use them exactly like any other IRelationalStore / IGraphStore
await graphStore.saveGraph(
  { nodes: ['CPU', 'Memory', 'Latency'], edges: [{ source: 'Memory', target: 'CPU', weight: 1, directed: true }] },
  { id: 'g1', method: 'PC', computedAt: Date.now(), parameters: {}, confidence: 0.9 }
);

const graph = await graphStore.loadGraph('g1');
// Data persists across page reloads via OPFS!
```

### Testing (vitest)

```typescript
import { DirectSqlitePort, WasmRelationalStore, WasmGraphStore } from '@agentix-e/causality-analyzer-storage-browser';

// DirectSqlitePort uses node:sqlite — zero setup, fast tests
const port = new DirectSqlitePort(':memory:');
const store = new WasmRelationalStore(port);

await store.saveCPT('g1', 'X', { node: 'X', parents: [], entries: { '0': 0.3 } });
const loaded = await store.loadCPT('g1', 'X');
// loaded.entries['0'] === 0.3
```

## API Reference

### SqlitePort Interface

| Method | Description |
|--------|-------------|
| `run(sql, params?)` | Execute INSERT/UPDATE/DELETE |
| `all(sql, params?)` | Execute SELECT, return all rows |
| `get(sql, params?)` | Execute SELECT, return first row |
| `exec(sql)` | Execute raw SQL (DDL, PRAGMA) |
| `close()` | Release resources |

### Implementations

| Class | Backend | Use Case |
|-------|---------|----------|
| `WorkerSqlitePort` | Web Worker + OPFS | Browser production |
| `DirectSqlitePort` | node:sqlite | vitest testing |

### Stores

| Class | Interface | Description |
|-------|-----------|-------------|
| `WasmRelationalStore` | `IRelationalStore` | Metrics, CPT, regression, RCA results |
| `WasmGraphStore` | `IGraphStore` | Causal graphs with versioning |

### Table Schema (identical to storage-embed)

```
metrics             (ts, value, metric_name)
cpt                 (graph_id, node, parent_state, prob)
regression_models   (graph_id, node, coefficients, intercept, residual_std)
rca_results         (case_id, result_json, analyzed_at, root_cause)
analysis_state      (session_id, stage, checkpoint_name, progress)
graph_nodes         (graph_id, name, version)
graph_edges         (graph_id, source, target, weight, directed, version)
```

## Browser Support

| Browser | Version | OPFS | Status |
|---------|---------|------|--------|
| Chrome | 102+ | ✅ | Supported |
| Edge | 102+ | ✅ | Supported |
| Firefox | 111+ | ✅ | Supported |
| Safari | 15.2+ | ✅ | Supported |

**Requirements:** HTTPS or localhost (OPFS requires secure context). COOP/COEP headers recommended.

## Testing

```bash
# Unit tests (vitest + DirectSqlitePort)
pnpm test

# E2E tests (Playwright + real browser)
pnpm browser-test
```

## License

MIT
