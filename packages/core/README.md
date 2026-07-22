# @agentix-e/causality-analyzer-core

> Foundation layer — types, interfaces, universal data primitives, and shared math.

[![npm](https://img.shields.io/badge/version-0.1.0-blue)](https://www.npmjs.com/package/@agentix-e/causality-analyzer-core)

## Overview

`@agentix-e/causality-analyzer-core` defines the contracts that every other package in the causality-analyzer ecosystem depends on. It contains the `ColumnarTable` data structure (the only implementation class — by explicit design exception, as it serves the same foundational role as `Array` in the runtime), plus shared mathematical utilities extracted from the pipeline.

**This package does not import any external runtime dependencies** — only `zod` for schema validation.

### What's Inside

```
src/
├── types/        # CausalEdge, CausalGraph, RCAResult, MetricQuery, etc.
├── interfaces/   # IRelationalStore, IGraphStore (storage contracts)
├── table/        # ColumnarTable — zero-copy columnar data structure
├── math.ts       # solveLinear, normalTail, erf, colMean, createRNG
├── registry/     # PluginRegistry (detectors, graphs, analyzers)
├── config/       # BaseConfig with Zod validation (abstract getSchema)
└── di/           # CausalityAnalyzerConfig (dependency injection)
```

## Installation

```bash
npm install @agentix-e/causality-analyzer-core
```

## Quick Start

### ColumnarTable

```typescript
import { ColumnarTable } from '@agentix-e/causality-analyzer-core';

const table = ColumnarTable.fromRows([
  { ts: 1000, cpu: 0.5, mem: 0.8 },
  { ts: 2000, cpu: 0.9, mem: 0.85 },
]);

// Zero-copy access
const cpuColumn = table.column('cpu');    // Float64Array
const row0 = table.row(0);               // { ts: 1000, cpu: 0.5, mem: 0.8 }
const sliced = table.slice(0, 1);        // ColumnarTable
```

### Shared Math

```typescript
import { solveLinear, normalTail, createRNG } from '@agentix-e/causality-analyzer-core';

// Gaussian elimination with partial pivoting
const x = solveLinear([[2, 1], [1, 3]], [5, 6]);
// x ≈ [1.8, 1.4]

// Upper-tail normal probability (Abramowitz & Stegun 7.1.26)
const p = normalTail(1.96);  // ≈ 0.025

// Seeded PRNG for reproducibility
const rng = createRNG(42);
rng();  // deterministic, reproducible
```

### Plugin Registry

```typescript
import { PluginRegistry, PluginCategory } from '@agentix-e/causality-analyzer-core';

PluginRegistry.register(PluginCategory.DETECTOR, 'StatsDetector', StatsDetector);
const names = PluginRegistry.listDetectors();  // ['StatsDetector']
```

## API Reference

📚 Full TypeDoc API: `pnpm docs` from the monorepo root.

### Key Exports

**Types:** `CausalGraph`, `CausalEdge`, `RootCause`, `RCAResult`, `MetricQuery`, `DetectionResult`, `Estimand`, `CausalEstimate`, `DomainKnowledge`, `ConditionalProbabilityTable`, `RegressionParams`

**Table:** `ColumnarTable`, `TableSchema`, `ColumnNames`, `DataRow`

**Math:** `solveLinear`, `normalTail`, `normalCDF`, `normalCDFTail`, `erf`, `colMean`, `createRNG`

**Registry:** `PluginRegistry`, `PluginCategory`, `RegisterDetector`, `RegisterGraph`, `RegisterAnalyzer`

**Config:** `BaseConfig`, `ValidationResult`

**Interfaces:** `IRelationalStore`, `IGraphStore`

## License

MIT
