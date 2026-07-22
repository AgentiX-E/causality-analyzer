# @agentix-e/causality-analyzer-storage-embed

> Embedded storage for relational metrics and causal graphs — zero-configuration persistence.

[![npm](https://img.shields.io/badge/version-0.1.0-blue)](https://www.npmjs.com/package/@agentix-e/causality-analyzer-storage-embed)

## Overview

`@agentix-e/causality-analyzer-storage-embed` provides file-based persistence for both relational data (metrics, detections, CPTs, regression models, RCA results) and causal graphs. No external database server required — everything runs in-process.

### Backends

| Store | Backend | Format |
|-------|---------|--------|
| `EmbedRelationalStore` | [better-sqlite3](https://github.com/WiseLibs/better-sqlite3) | SQLite file or `:memory:` |
| `EmbedGraphStore` | OverGraph | LSM-tree file directory |

## Installation

```bash
npm install @agentix-e/causality-analyzer-storage-embed
```

## Quick Start

### Relational Store

```typescript
import { EmbedRelationalStore } from '@agentix-e/causality-analyzer-storage-embed';

// In-memory (fast, ephemeral)
const store = new EmbedRelationalStore({ dbPath: ':memory:' });

// Persistent (survives restarts)
const store2 = new EmbedRelationalStore({ dbPath: './causality.db' });

// Save and load CPTs
await store.saveCPT('graph1', 'CPU', {
  node: 'CPU', parents: ['Memory'],
  entries: { '0': 0.3, '1': 0.75 },
});
const cpt = await store.loadCPT('graph1', 'CPU');

// Save RCA results
await store.saveRCAResult('case-001', rcaResult);
const history = await store.queryHistoricalResults({ rootCause: 'Memory' });

// Transaction support
await store.beginTransaction('session1');
await store.saveCPT('g1', 'X', cpt);
await store.setCheckpoint('session1', 'checkpoint1');
await store.rollbackToCheckpoint('session1', 'checkpoint1');
await store.commitTransaction('session1');

// Clean up
store.close();
```

### Graph Store

```typescript
import { EmbedGraphStore } from '@agentix-e/causality-analyzer-storage-embed';

const graphStore = new EmbedGraphStore({ dbPath: './graph-data' });

// Save a causal graph
const id = await graphStore.saveGraph(
  { nodes: ['A', 'B', 'C'], edges: [{ source: 'A', target: 'B', weight: 1, directed: true }] },
  { id: 'graph-001', method: 'pc', computedAt: Date.now(), parameters: {}, confidence: 0.9 },
);

// Load latest version
const graph = await graphStore.loadGraph(id);

// Load specific version
const v1 = await graphStore.loadGraphVersion(id, 1);

// List all versions
const versions = await graphStore.listGraphVersions(id);

graphStore.close();
```

## API Reference

### `EmbedRelationalStore` implements `IRelationalStore`

| Method | Description |
|--------|-------------|
| `saveCPT(graphId, node, cpt)` | Store conditional probability table |
| `loadCPT(graphId, node)` | Retrieve CPT |
| `saveRegressionModel(graphId, node, model)` | Store regression coefficients |
| `loadRegressionModel(graphId, node)` | Retrieve model |
| `saveRCAResult(caseId, result)` | Store RCA result with timestamp |
| `queryHistoricalResults(query)` | Filtered query with pagination |
| `readMetrics(query)` | Read metric data with metric_name filter |
| `writeDetections(detections)` | Batch write anomaly detection results |
| `beginTransaction(sessionId)` | Start savepoint |
| `commitTransaction(sessionId)` | Commit savepoint |
| `rollbackToCheckpoint(sessionId, checkpoint)` | Rollback to named checkpoint |
| `setCheckpoint(sessionId, name)` | Create named checkpoint |
| `close()` | Close database connection |

### `EmbedGraphStore` implements `IGraphStore`

| Method | Description |
|--------|-------------|
| `saveGraph(graph, metadata)` | Save causal graph (versioned) |
| `loadGraph(graphId)` | Load latest version |
| `loadGraphVersion(graphId, ver)` | Load specific version |
| `listGraphVersions(graphId)` | List all versions with timestamps |
| `findSimilarGraphs(graph, limit)` | Graph similarity search |
| `close()` | Close graph database |

## License

MIT
