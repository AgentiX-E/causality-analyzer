# Causality Analyzer — High-Level Architecture Design

> **Author:** Lambertyan
> **Version:** 1.1.0
> **Date:** 2026-07-29
> **License:** MIT

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Layered Architecture](#2-layered-architecture)
3. [Package Dependency Graph](#3-package-dependency-graph)
4. [Core Package — Types and Contracts](#4-core-package--types-and-contracts)
5. [Pipeline Package — Algorithm Engine](#5-pipeline-package--algorithm-engine)
6. [Storage Architecture](#6-storage-architecture)
7. [Visual Architecture](#7-visual-architecture)
8. [HTTP API and CLI Architecture](#8-http-api-and-cli-architecture)
9. [Design Patterns](#9-design-patterns)
10. [Security Architecture](#10-security-architecture)
11. [Observability Architecture](#11-observability-architecture)
12. [Data Flow Diagrams](#12-data-flow-diagrams)
13. [Extensibility](#13-extensibility)

---

## 1. Design Philosophy

Causality Analyzer is built on five architectural axioms. Every design decision, every abstraction boundary, and every dependency choice traces back to these principles.

### 1.1 Contract-First Development

All cross-package communication occurs through **TypeScript interfaces** defined in the `@agentix-e/causality-analyzer-core` package. Implementation classes reside exclusively in their owning packages and are never imported across package boundaries.

```
┌────────────────────────────────────────────┐
│  core defines: IGraphStore                  │
│                                              │
│  storage-embed implements: EmbedGraphStore   │
│  storage-remote implements: RemoteGraphStore │
│  storage-browser implements: WasmGraphStore  │
│                                              │
│  pipeline consumes: IGraphStore (interface)  │
└────────────────────────────────────────────┘
```

This enables:
- **Swappable storage backends** without changing pipeline code
- **Test doubles** at every interface boundary
- **Zero package coupling** — any package can be replaced independently
- **Clear API contracts** — interfaces serve as living documentation

### 1.2 Dependency Injection

All variable behavior — storage, logging, telemetry, configuration — is injected through a DI container (`CausalityAnalyzerConfig`) rather than imported directly. The pipeline has **zero hard dependency** on any concrete storage, logger, or telemetry implementation.

```typescript
// Consumers inject their chosen implementations
const config: CausalityAnalyzerConfig = {
  relationalStore: new EmbedRelationalStore(dbPath),
  graphStore: new EmbedGraphStore(graphPath),
  logger: new ConsoleLogger('debug'),
  // telemetry defaults to no-op
};
```

### 1.3 Progressive Complexity

The minimal runtime requires only `core + pipeline`. Every other package is optional:

```
Minimum: core + pipeline
  + storage-embed     → persistent results
  + storage-browser   → browser analytics
  + storage-remote    → enterprise deployment
  + visual            → Web Component visualization
  + server.ts         → HTTP API
```

Users pay only for what they use — in bundle size, in dependencies, and in complexity.

### 1.4 Zero External Service Dependency

In embedded mode, the entire analysis pipeline runs **in-process** with no external services:

- **Relational Storage**: `node:sqlite` DatabaseSync (Node.js 22+ built-in, zero npm deps)
- **Graph Storage**: OverGraph (native addon via napi-rs, in-process)
- **No PostgreSQL, no Neo4j, no Docker** required for basic operation

Remote storage tiers (PostgreSQL + Neo4j) are available for enterprise deployments but are never the default path.

### 1.5 Platform Agnosticism

The same API works identically across:
- **Node.js**: Direct execution with optional native addons
- **Browser**: WASM SQLite + Web Workers for non-blocking computation
- **Edge**: Core + pipeline (no file system, no native addons)

```typescript
// Same code, different runtime:
import { pcAlgorithm } from '@agentix-e/causality-analyzer-pipeline';
const { graph } = pcAlgorithm(data, varNames);
// Works in Node.js, browser, and edge without modification.
```

---

## 2. Layered Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 4: Visualization                                           │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  visual (Web Components, framework-agnostic)                │  │
│  │  <ca-causal-graph>  <ca-time-series>  <ca-root-cause-rank> │  │
│  │  Canvas2DRenderer  GraphRenderer (pluggable interface)     │  │
│  └────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  Layer 3: Application Logic                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  pipeline (Algorithm Engine)                                │  │
│  │  ┌──────────┬──────────┬──────────┬──────────┬──────────┐ │  │
│  │  │ Discovery │Inference │   RCA    │   GCM    │  Detect  │ │  │
│  │  │  28+ alg  │ do-Calc  │CIRCA+4   │ SCM+CF   │  5+ det  │ │  │
│  │  ├──────────┼──────────┼──────────┼──────────┼──────────┤ │  │
│  │  │Server(HTTPS)│CLI(stdio)│Benchmark  │Streaming│WorkerPool│ │  │
│  │  └──────────┴──────────┴──────────┴──────────┴──────────┘ │  │
│  └────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2: Contracts and Primitives                                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  core (Types + Interfaces + Math + Registry)                │  │
│  │  types | interfaces | registry | table | math | optimize   │  │
│  │  errors | telemetry | config | logger | graph-similarity   │  │
│  └────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  Layer 1: Storage and Persistence                                 │
│  ┌──────────────────┬──────────────────┬───────────────────┐    │
│  │  storage-embed   │  storage-browser │  storage-remote    │    │
│  │  node:sqlite     │  wa-sqlite WASM  │  PostgreSQL        │    │
│  │  + OverGraph     │  + OPFS          │  + Neo4j (Bolt)    │    │
│  └──────────────────┴──────────────────┴───────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

**Key property**: Dependencies flow **downward only**. Layer 4 can import from Layers 3 and 2. Layer 3 can import from Layer 2. Layer 2 imports nothing from Layers 1/3/4. Layer 1 imports only from Layer 2.

---

## 3. Package Dependency Graph

```
visual
  ├── core (types, interfaces)
  └── lit, uplot (external)

pipeline
  ├── core (types, interfaces, math)
  └── ml-matrix, simple-statistics, @kanaries/ml (external)

storage-embed
  ├── core (interfaces)
  └── node:sqlite, overgraph (external, native)

storage-browser
  ├── core (interfaces)
  └── wa-sqlite (external, WASM)

storage-remote
  ├── core (interfaces)
  └── pg (optional), neo4j-driver-lite, pg-mem, @electric-sql/pglite (external)

core
  └── zod (external, schema validation only)
```

**Rationale for each dependency:**

| Dependency | Why Not Internal? |
|-----------|-------------------|
| `zod` in core | Runtime schema validation is essential for `BaseConfig.validate()`. Zod is the most mature, tree-shakeable TypeScript schema library. Could be replaced with custom validators, but Zod provides composability, error messages, and inference that would require hundreds of lines to replicate. |
| `ml-matrix` in pipeline | Full-featured matrix library with BLAS-like operations, SVD, eigenvalue decomposition. Writing a competitive matrix library is a multi-year project outside our scope. |
| `simple-statistics` in pipeline | Zero-dependency statistical functions (quantile, MAD, linear regression). Could be replaced with `@stdlib/stats` modules but `simple-statistics` provides a consistent API surface. |
| `@kanaries/ml` in pipeline | LassoRegression for LiNGAM adjacency estimation. No other npm package provides scikit-learn-compatible Lasso with cross-validation. |
| `node:sqlite` in storage-embed | Built into Node.js 22+. Zero npm dependencies for embedded relational storage. |
| `overgraph` in storage-embed | Rust/napi-rs LSM-tree graph database. No pure-JS embedded graph database exists that matches its performance. Writing one would be a separate project. |
| `lit` in visual | 6KB gzipped, standards-based Web Components. The most lightweight and widely-adopted Web Component library. |
| `uplot` in visual | 35KB gzipped, GPU-accelerated Canvas time series chart. 10-100x faster than D3/SVG alternatives for streaming data. |

---

## 4. Core Package — Types and Contracts

The core package is the **constitutional document** of the Causality Analyzer ecosystem. It defines what every other package can and cannot do.

### 4.1 Content Boundaries

The core package contains **exactly one implementation class** (`ColumnarTable`), by explicit design exception. The design rule is:

> Core defines **interfaces**. Packages implement them. Do not put real implementations in core.

The exception for `ColumnarTable` exists because it serves the same foundational role as `Array` in the JavaScript runtime — it is the universal data primitive that all algorithms operate on. Having it in core avoids circular dependency issues.

### 4.2 Type System Design

The type system is organized into domain-specific modules:

```
core/src/types/index.ts
├── CausalGraph, CausalEdge       — DAG structure with adjacency matrix
├── GraphMetadata, GraphVersion   — Graph provenance and versioning
├── DetectionResult               — Anomaly detection output format
├── Evidence, RootCause, RCAResult — RCA result types
├── IdentifiedEstimand            — Causal estimand (backdoor/frontdoor/IV sets)
├── CausalEstimate                — Effect estimate with standard errors
├── DomainKnowledge               — Expert constraints (forbidden/required edges)
├── ConditionalProbabilityTable   — CPT for Bayesian networks
├── RegressionParams              — OLS model parameters for HTRCA
├── PipelineStage                 — Enum: INGEST → DETECT → GRAPH → ANALYZE → INFER → VALIDATE
└── Visualization types           — Data contracts for graph/time-series/ranking display
```

**Design decisions:**
- All types are **interfaces**, not classes — enables structural typing and easy test construction
- `Float64Array` is used for adjacency matrices — cache-friendly, zero boxing overhead
- `toJSON()` methods on result types enable serialization without a serialization library
- All numeric fields use `number` (TypeScript's 64-bit float) — sufficient precision for statistical computation

### 4.3 Interface Segregation

Storage interfaces follow the **Interface Segregation Principle**:

```typescript
// Fine-grained interfaces
interface IMetricStore {
  insertMetrics(...): Promise<void>;
  queryMetrics(query: MetricQuery): Promise<...>;
}

interface IModelStore {
  saveCPT(...): Promise<void>;
  getCPT(...): Promise<ConditionalProbabilityTable>;
}

interface IResultStore {
  saveResult(result: RCAResult): Promise<void>;
  queryResults(query: ResultQuery): Promise<RCAResult[]>;
}

interface ITransactionStore {
  beginTransaction(): Promise<void>;
  commit(): Promise<void>;
  rollback(): Promise<void>;
}

// Composition interface
interface IRelationalStore extends
  IMetricStore, IModelStore, IResultStore, ITransactionStore {}

// Separate interface for graph storage
interface IGraphStore {
  saveGraph(graph: CausalGraph, metadata: GraphMetadata): Promise<GraphVersion>;
  loadGraph(id: string, version?: number): Promise<CausalGraph>;
  // ...
}
```

This segregation means:
- Unit tests can mock only the interface needed (e.g., `IMetricStore` for anomaly detection tests)
- Storage backends can implement subsets of functionality (e.g., read-only backends)
- New interfaces can be added without breaking existing implementations

### 4.4 Plugin Registry

The plugin registry is a **static singleton** with class-level decorators for auto-discovery:

```typescript
@RegisterDetector('spectral_residual')
export class SpectralResidualDetector implements IDetector { ... }

@RegisterGraph('pc')
export class PCDiscoverer implements IGraphDiscoverer { ... }

// Consumers query the registry
const detectors = PluginRegistry.listDetectors();
const hasPC = PluginRegistry.hasGraph('pc');
```

**Design rationale:**
- **Decorator pattern** — zero-config registration, file import triggers registration
- **Static singleton** — avoids passing registry instances through every constructor
- **PluginCategory enum** — DETECTOR, GRAPH, ANALYZER categories enable namespacing
- **Unregister/clear** methods support testing and hot-reload scenarios

### 4.5 ColumnarTable — Universal Data Primitive

`ColumnarTable<S>` is the single implementation class permitted in core. It provides:

```typescript
const table = ColumnarTable.fromRows([
  { x: 1.0, y: 2.0, label: 'A' },
  { x: 3.0, y: 4.0, label: 'B' },
]);

// Type-safe column access
const xCol: Float64Array = table.column('x');

// Immutable transformations return new instances
const normalized = table.normalize('x', 'zscore');

// Zero-copy views
const window = table.slice(0, 100);
```

**Key design properties:**
- **Generic schema** `S extends TableSchema` maps column names to types at compile time
- **Immutable operations** — all transformations return new instances (no shared mutable state)
- **Zero-copy views** — `slice()` returns views sharing the underlying Float64Array buffers
- **Columnar layout** — data stored as `Map<string, Float64Array>`, cache-friendly for vectorized operations

---

## 5. Pipeline Package — Algorithm Engine

The pipeline package is the computational heart of Causality Analyzer. It implements all algorithms across five domains and provides HTTP API, CLI, and runtime infrastructure.

### 5.1 Module Organization

```
pipeline/src/
├── graph/           — 28+ causal discovery algorithms
│   ├── causal-graph.ts   — DAG data structure (maintained separately from core type)
│   ├── pc.ts, pc-max.ts  — Constraint-based discovery
│   ├── ges.ts, boss.ts, grasp.ts  — Score-based discovery
│   ├── notears.ts, dagma.ts, golem.ts  — Continuous optimization
│   ├── lingam.ts, ica-lingam.ts, var-lingam.ts, fask.ts  — Functional models
│   ├── fci/advanced-discovery.ts  — FCI + Grow-Shrink + Targeted
│   ├── gfci.ts, rfci.ts, ccd.ts  — PAG-based methods
│   ├── kci.ts, rcd.ts, cdnod.ts, mvpc.ts  — Specialized
│   ├── tsicd.ts, tsfci.ts, pcmci.ts  — Time series
│   ├── images.ts, gin.ts, timino.ts  — Multi-dataset
│   ├── exact-search.ts, latent-clusters.ts  — Exhaustive/clustering
│   ├── stability-selection.ts  — Bootstrap + StARS
│   ├── streaming-discovery.ts  — OnlinePC
│   └── drift-detection.ts  — Causal drift
├── infer/           — Causal inference + Bayesian networks
│   ├── causal-inference.ts  — CausalAnalysis pipeline class
│   ├── backdoor.ts  — 5 backdoor variants
│   ├── do-calculus.ts  — Full ID algorithm
│   ├── effect-estimation.ts  — ATE/CATE/ATT, IV, PS, DR
│   ├── sensitivity.ts  — E-value, partial R-squared
│   ├── refutation-*.ts  — 5 refutation methods
│   ├── bayesian-network.ts  — 5 inference engines
│   ├── causal-forest.ts  — Causal Forest
│   ├── double-ml.ts  — Double ML
│   ├── mediation.ts  — NDE/NIE
│   └── uplift.ts  — Uplift modeling
├── gcm/             — Generative Causal Modeling
│   ├── structural-causal-model.ts  — SCM
│   ├── auto-assign.ts  — Auto mechanism selection
│   ├── neural-mechanisms.ts  — FFN mechanisms
│   ├── nonlinear-mechanisms.ts  — GP/MLP/spline
│   ├── model-evaluation.ts  — R-squared, MSE, Shapley
│   ├── distribution-change.ts  — Mechanism shifts
│   ├── resit.ts  — RESIT test
│   └── graph-falsification.ts  — Structure validation
├── analyze/         — Root Cause Analysis
│   ├── rca.ts  — 4 RCA methods
│   └── circa.ts  — CIRCA pipeline
├── detect/          — Anomaly Detection
│   ├── bsts.ts, spectral-residual.ts, spot.ts
│   ├── stats-detector.ts, voting-detector.ts
├── server.ts        — HTTP API server
├── cli.ts           — CLI entry point
├── benchmark.ts     — Benchmark suite
├── streaming.ts     — Streaming pipeline
├── audit-trail.ts   — Audit logging
├── encrypted-store.ts  — Encrypted persistence
├── rate-limiter.ts  — Token bucket rate limiter
├── explainer.ts     — NL explanation generator
└── parallel/        — Web Worker parallelism
    ├── worker-pool.ts, ci-worker.ts, restart-runner.ts
```

### 5.2 CausalGraph Data Structure

The **CausalGraph** class in pipeline (distinct from the `CausalGraph` type in core) is the workhorse DAG data structure:

```typescript
class CausalGraph {
  // Structure
  addEdge(from: string, to: string, weight?: number): void;
  removeEdge(from: string, to: string): void;
  orientEdge(a: string, b: string, direction: 'a->b' | 'b->a'): void;

  // Traversal
  parents(node: string): string[];
  children(node: string): string[];
  ancestors(node: string): string[];
  descendants(node: string): string[];
  hasDirectedPath(from: string, to: string): boolean;

  // Causal operations
  dSeparated(x: string, y: string, conditioning: Set<string>): boolean;
  do(node: string): CausalGraph;  // Returns graph with incoming edges removed

  // Graph operations
  isAcyclic(): boolean;
  topologicalSort(): string[];
  structuralHammingDistance(other: CausalGraph): number;

  // Conversion
  toPDAG(): CausalGraph;
  pdag2dag(): CausalGraph;  // Dor-Tarsi 1992 sink-finding

  // Serialization
  toJSON(): object;
  static fromJSON(json: object): CausalGraph;
}
```

**Key implementation details:**
- **Adjacency**: `ml-matrix` Matrix for O(1) edge lookups
- **d-Separation**: Pearl (2009, pp. 16-17) strict algorithm with trail DFS, collider detection, and full conditioning set semantics
- **PDAG-to-DAG**: Dor & Tarsi (1992) sink-finding algorithm — finds sink in PDAG, orients all incident edges, removes sink, repeats
- **Cycle Detection**: 3-color DFS (white/gray/black)
- **Topological Sort**: Kahn's algorithm (in-degree counting)

### 5.3 Algorithm Design Patterns

All discovery algorithms follow a consistent pattern:

```typescript
// Each algorithm exports a stateless function
function pcAlgorithm(
  data: Matrix,
  varNames: string[],
  config?: PCConfig,
): { graph: CausalGraph } {
  // 1. Preprocess (normalize, handle missing)
  // 2. Compute correlation matrix (cached)
  // 3. Run algorithm (skeleton + orientation)
  // 4. Return result
}

// Config interfaces allow algorithm parameterization
interface PCConfig {
  alpha?: number;        // Significance level (default: 0.05)
  correction?: 'bonferroni' | 'fdr' | 'none';
  maxConditioning?: number;
  ciTest?: 'fisher-z' | 'chi-square' | 'g-square';
}
```

**Why stateless functions instead of classes?**
- Pure functions are trivially testable (input -> output, no side effects)
- No constructor dependency injection needed
- Easy to wrap with Web Workers (serialize input, deserialize output)
- Composability: `fuseGraphs([pcAlgorithm(data), gesAlgorithm(data)])`

### 5.4 Pipeline Stage Execution

The `PipelineStage` enum defines the ordered execution model:

```
INGEST → DETECT → GRAPH → ANALYZE → INFER → VALIDATE
```

Each stage transforms data and passes it to the next:

```
Raw Data
  ↓ INGEST   (standardize, impute, discretize)
Clean Data
  ↓ DETECT   (BSTS, SR, SPOT, Stats, Voting)
Anomalies
  ↓ GRAPH    (PC, GES, NOTEARS, LiNGAM, FCI...)
Causal Graph
  ↓ ANALYZE  (HeuristicPathRCA, CIRCA, HTRCA...)
Root Causes
  ↓ INFER    (Backdoor, IV, PS, do-Calculus...)
Causal Effects
  ↓ VALIDATE (Refutation, Sensitivity, Stability)
Validated Results
```

---

## 6. Storage Architecture

### 6.1 Three-Tier Design

The storage layer provides three interchangeable backends implementing the same interface contracts:

| Tier | Package | When to Use |
|------|---------|-------------|
| **Embedded** | `storage-embed` | Single-server deployments, zero external deps |
| **Browser** | `storage-browser` | Client-side analytics, offline-first apps |
| **Remote** | `storage-remote` | Multi-tenant enterprise, high-availability clusters |

### 6.2 Embedded Storage

```
┌─────────────────────────────────────┐
│  EmbedRelationalStore                │
│  ┌─────────────────────────────────┐│
│  │  node:sqlite DatabaseSync        ││
│  │  - WAL journal mode              ││
│  │  - synchronous=NORMAL            ││
│  │  - Prepared statements            ││
│  │                                   ││
│  │  Tables:                          ││
│  │  - metrics (id, ts, name, value) ││
│  │  - cpt (graph_id, node, values)  ││
│  │  - regression_models (params)    ││
│  │  - rca_results (results, meta)   ││
│  │  - analysis_state (checkpoints)  ││
│  └─────────────────────────────────┘│
├─────────────────────────────────────┤
│  EmbedGraphStore                     │
│  ┌─────────────────────────────────┐│
│  │  OverGraph (Rust/napi-rs)        ││
│  │  - LSM-tree persistent storage    ││
│  │  - Versioned graph storage        ││
│  │  - Bidirected edge support (PAG) ││
│  │  - Graph similarity via core     ││
│  │  - Label: g_{id}_v{version}      ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
```

### 6.3 Browser Storage

```
┌─────────────────────────────────────┐
│  WasmRelationalStore                 │
│  ┌─────────────────────────────────┐│
│  │  wa-sqlite WASM                  ││
│  │  - DirectSqlitePort (main thread)││
│  │  - WorkerSqlitePort (Web Worker) ││
│  │  - OPFS (Origin Private FS)      ││
│  │  - Persistent across sessions    ││
│  └─────────────────────────────────┘│
├─────────────────────────────────────┤
│  WasmGraphStore                      │
│  ┌─────────────────────────────────┐│
│  │  SQLite adjacency tables         ││
│  │  - nodes (id, name, metadata)    ││
│  │  - edges (source, target, weight)││
│  │  - Graph similarity in JS        ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
```

**Key design decision — why adjacency tables instead of a graph library in browser?**
The browser constraints (single-threaded WASM, limited memory) make a full graph database impractical. Using SQLite adjacency tables with indexed queries provides comparable query performance for graphs up to ~1000 nodes, which covers the vast majority of browser use cases.

### 6.4 Remote Storage

```
┌─────────────────────────────────────┐
│  RemoteRelationalStore               │
│  ┌─────────────────────────────────┐│
│  │  pg.Client (PG-wire protocol)    ││
│  │  - Prepared statements            ││
│  │  - mTLS support                  ││
│  │  - Testable via PgClientLike      ││
│  │  - Compatible with pg-mem, pglite││
│  └─────────────────────────────────┘│
├─────────────────────────────────────┤
│  RemoteGraphStore                    │
│  ┌─────────────────────────────────┐│
│  │  neo4j-driver-lite (Bolt v4/v5)  ││
│  │  - 5 auth types                   ││
│  │  - TLS + mTLS                     ││
│  │  - UNWIND batched writes          ││
│  │  - Exponential backoff retry      ││
│  │  - Testable via DriverLike        ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
```

### 6.5 Interface Testability

Every storage backend exposes a testable interface (`PgClientLike`, `DriverLike`) that enables in-memory substitution during testing:

```typescript
// Production
const store = new RemoteRelationalStore({ host: 'prod-db', ... });

// Test (zero external deps)
const store = new RemoteRelationalStore({
  clientLike: new PgMemClient(),
});
```

This eliminates the need for Docker containers in CI for most tests.

---

## 7. Visual Architecture

### 7.1 Web Components (Lit 3)

All visualization components are **framework-agnostic Custom Elements** built on Lit 3:

```html
<!-- Works in any framework or plain HTML -->
<ca-causal-graph
  .data="${graphData}"
  @node-click="${handleNodeClick}">
</ca-causal-graph>

<ca-time-series
  .data="${timeSeriesData}"
  .theme="${lightTheme}">
</ca-time-series>

<ca-root-cause-ranking
  .data="${rankingData}"
  @cause-hover="${handleCauseHover}">
</ca-root-cause-ranking>
```

**Design rationale for Web Components:**
- **Zero framework dependency** — works in React, Vue, Angular, Svelte, plain HTML
- **Shadow DOM encapsulation** — styles don't leak, styles can't be broken
- **Standards-based** — Custom Elements v1, no proprietary runtime
- **Tree-shakeable** — Lit is 6KB gzipped, only used functions are included

### 7.2 Canvas2D Renderer

The `<ca-causal-graph>` component uses Canvas 2D for rendering, with a **pluggable** renderer interface:

```typescript
interface GraphRenderer {
  render(canvas: HTMLCanvasElement, data: GraphVisualizationData, theme: Theme): void;
  hitTest(x: number, y: number, data: GraphVisualizationData): string | null;
  dispose(): void;
}
```

The default `Canvas2DRenderer` implementation provides:
- **Layered Sugiyama layout** with caching for stable animations
- **Directed edges** with arrowheads (computed via atan2)
- **Color-coded nodes**: root_cause (amber), anomaly (red), healthy (green)
- **Hit testing** via Euclidean distance (<14px threshold)
- **Responsive sizing** via ResizeObserver

The pluggable interface enables future WebGL renderers for 500+ node graphs.

### 7.3 Accessibility

All components meet WCAG 2.1 Level AA:

```
<ca-causal-graph role="img" aria-label="Causal graph showing 5 nodes and 6 edges">
  <div class="sr-only" aria-live="polite">
    <!-- Screen reader announcements -->
  </div>
</ca-causal-graph>
```

- **Screen reader support**: aria-label, aria-live regions, role attributes
- **Keyboard navigation**: Tab to focus, Arrow keys to navigate, Enter to select
- **Color independence**: Node shapes (radius variations) in addition to colors
- **Focus indicators**: Visible outlines on focused nodes

---

## 8. HTTP API and CLI Architecture

### 8.1 Server Architecture

The HTTP server is built on **Node.js stdlib `http`/`https` modules** — zero framework dependencies:

```
┌──────────────────────────────────────────────────┐
│  CausalityServer                                  │
│  ┌──────────────────────────────────────────────┐│
│  │  Middleware Chain (no framework)              ││
│  │                                              ││
│  │  Request → RateLimiter → Auth(mTLS+Token)    ││
│  │         → Router → Handler → Audit → Response ││
│  │                                              ││
│  │  Endpoints:                                   ││
│  │  GET  /health     Combined probe             ││
│  │  GET  /ready      Readiness                  ││
│  │  GET  /live       Liveness                   ││
│  │  GET  /metrics    Prometheus                  ││
│  │  POST /v1/discover  Causal discovery         ││
│  │  POST /v1/analyze   RCA analysis             ││
│  │  POST /v1/estimate  Effect estimation        ││
│  │  GET  /v1/openapi.json  OpenAPI 3.1 spec     ││
│  └──────────────────────────────────────────────┘│
└──────────────────────────────────────────────────┘
```

### 8.2 Authentication Layering

Two independent authentication layers, both optional:

1. **mTLS** (transport layer): When `tls.requestCert: true`, client must present a valid certificate signed by the configured CA. Rejected at TLS handshake — never reaches application code.
2. **Bearer Token** (application layer): When `CAUSALITY_API_TOKEN` is set, all `/v1/*` endpoints require `Authorization: Bearer <token>`. Health endpoints remain public.

### 8.3 CLI Architecture

The CLI is a minimal entry point using Node.js stdlib:

```bash
npx causal-analyzer serve --port 3000 --tls-cert cert.pem --tls-key key.pem
npx causal-analyzer benchmark --config bench-config.json
```

Architecture: CLI parses arguments → instantiates CausalityServer → delegates all logic.

---

## 9. Design Patterns

| Pattern | Where Used | Purpose |
|---------|-----------|---------|
| **Strategy** | Logger, GraphRenderer, SqlitePort, PgClientLike | Pluggable implementations behind interfaces |
| **Interface Segregation** | IMetricStore, IModelStore, IResultStore, ITransactionStore | Fine-grained contracts; consumers depend only on what they need |
| **Dependency Injection** | CausalityAnalyzerConfig | Runtime injection of storage/logging/telemetry |
| **Decorator** | @RegisterDetector, @RegisterGraph, @RegisterAnalyzer | Declarative plugin auto-registration |
| **Facade** | Telemetry class | OTel API abstraction; no-op by default, injectable |
| **Builder** | CausalAnalysis.ingest().model().identify().estimate().refute() | Chainable pipeline construction |
| **Template Method** | BaseConfig.validate() / getSchema() | Subclasses define schemas; base handles validation |
| **Null Object** | NoopLogger, NoopCounter, NoopSpan | Zero-overhead when optional deps unavailable |
| **Worker** | WorkerPool, SqliteWorker | Offload computation to separate threads |
| **Singleton** | PluginRegistry, Telemetry | Static global state for cross-cutting concerns |
| **Command** | PipelineStage enum | Ordered execution stages with typed data flow |

---

## 10. Security Architecture

### 10.1 Defense in Depth

```
Layer 1: Network — mTLS (TLS 1.3, client certificate validation)
Layer 2: Application — Bearer Token (Authorization header)
Layer 3: Storage — AES-256-GCM (at-rest encryption)
Layer 4: Audit — SHA-256 trail (append-only operation log)
Layer 5: Rate Limiting — Token bucket (per-IP request throttling)
```

### 10.2 mTLS Configuration

```typescript
interface MtlsConfig {
  cert: string;      // Server certificate (PEM)
  key: string;       // Server private key (PEM)
  ca?: string;       // CA certificate for client validation
  requestCert?: boolean;      // Require client certificates
  rejectUnauthorized?: boolean; // Reject invalid certificates
}
```

### 10.3 Encrypted Persistence

The `EncryptedStore` wrapper provides AES-256-GCM encryption for any stored data:

```typescript
const store = new EncryptedStore({
  underlying: embedRelationalStore,
  encryptionKey: process.env.ENCRYPTION_KEY,
});
```

- **Algorithm**: AES-256-GCM (authenticated encryption with associated data)
- **Key derivation**: PBKDF2 with random salt
- **IV**: Cryptographically random, 12 bytes, stored with ciphertext
- **Authentication tag**: 16 bytes, verified on decryption

---

## 11. Observability Architecture

### 11.1 OpenTelemetry Facade

The Telemetry class is a **zero-overhead facade** over the OpenTelemetry API:

```typescript
// Default: all no-op (zero overhead)
const span = Telemetry.startSpan('pcAlgorithm');
span.setAttribute('num_nodes', 10);
span.end();  // No-op — no OTel SDK loaded

// When OTel is needed: inject real implementations
Telemetry.init({
  tracer: opentelemetry.trace.getTracer('causality-analyzer'),
  meter: opentelemetry.metrics.getMeter('causality-analyzer'),
});

// Now spans are real!
const span = Telemetry.startSpan('pcAlgorithm');
// → Exported to OTel Collector → Jaeger/Tempo/...
```

**Design rationale:** OpenTelemetry SDKs are large (500KB+). By defaulting to no-op, users who don't need distributed tracing pay zero cost. Users who do can inject their OTel SDK and get full tracing without modifying application code.

### 11.2 Prometheus Metrics

```
GET /metrics  →  Prometheus-compatible text format

# HELP causality_requests_total Total API requests
# TYPE causality_requests_total counter
causality_requests_total{endpoint="/v1/discover",method="POST"} 42

# HELP causality_algorithm_duration_seconds Algorithm execution time
# TYPE causality_algorithm_duration_seconds histogram
causality_algorithm_duration_seconds_bucket{algorithm="pc",le="0.1"} 15
causality_algorithm_duration_seconds_bucket{algorithm="pc",le="0.5"} 38
```

### 11.3 Health Probes

Kubernetes-compatible health endpoints:

- **`GET /health`**: Combined status (200 if all OK, 503 if any degraded)
- **`GET /live`**: Liveness probe (200 if process is alive)
- **`GET /ready`**: Readiness probe (200 if dependencies available)

### 11.4 Structured Logging

```typescript
// ConsoleLogger outputs JSON to stdout
logger.info('pc_algorithm_complete', {
  numNodes: 10,
  numEdges: 8,
  duration: 45.2,
});
// → {"level":"info","message":"pc_algorithm_complete","numNodes":10,...}
```

---

## 12. Data Flow Diagrams

### 12.1 Root Cause Analysis Flow

```
  ┌─────────┐     ┌─────────────┐     ┌──────────────┐
  │ Metrics │────→│ Standardize  │────→│ PC Algorithm │
  │ (raw)   │     │ (z-score)   │     │ (discovery)  │
  └─────────┘     └─────────────┘     └──────┬───────┘
                                              │
                    ┌─────────────────────────┘
                    ▼
            ┌──────────────┐
            │ CausalGraph  │
            │ (DAG)        │
            └──────┬───────┘
                   │
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
┌─────────┐  ┌──────────┐  ┌──────────┐
│  SPOT   │  │ Spectral │  │  Voting  │
│Detector │  │ Residual │  │ Detector │
└────┬────┘  └────┬─────┘  └────┬─────┘
     │            │              │
     └────────────┼──────────────┘
                  ▼
          ┌──────────────┐
          │  Anomalies   │
          │  (regions)   │
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │ CIRCA/RCA    │
          │ Engine       │
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │ Ranked Root  │
          │ Causes       │
          └──────────────┘
```

### 12.2 Causal Inference Flow

```
┌──────────┐    ┌───────────┐    ┌───────────┐    ┌──────────┐
│   Data   │───→│  Model    │───→│ Identify  │───→│ Estimate │
│          │    │ (DAG)     │    │ (Estimand)│    │ (ATE)    │
└──────────┘    └───────────┘    └───────────┘    └────┬─────┘
                                                        │
                         ┌──────────────────────────────┘
                         ▼
                  ┌───────────┐
                  │  Refute   │
                  │ Bootstrap │
                  │ Placebo   │
                  │ Subset    │
                  └─────┬─────┘
                        │
                        ▼
                 ┌───────────┐
                 │Sensitivity│
                 │ E-value   │
                 │ Partial R²│
                 └───────────┘
```

---

## 13. Extensibility

### 13.1 Adding a New Discovery Algorithm

1. **Implement the algorithm** in `packages/pipeline/src/graph/my-algorithm.ts`:

```typescript
export function myAlgorithm(
  data: Matrix,
  varNames: string[],
  config?: MyConfig,
): { graph: CausalGraph } {
  // Algorithm logic
  return { graph };
}
```

2. **Export from barrel** in `packages/pipeline/src/graph/index.ts`:

```typescript
export { myAlgorithm } from './my-algorithm.js';
```

3. **Register plugin** (optional, for dynamic discovery):

```typescript
@RegisterGraph('my_algorithm')
export class MyAlgorithmDiscoverer implements IGraphDiscoverer {
  discover(data: Matrix, varNames: string[]): CausalGraph {
    return myAlgorithm(data, varNames).graph;
  }
}
```

4. **Write tests** (mandatory):

```typescript
describe('myAlgorithm', () => {
  it('recovers known DAG', () => {
    const { graph } = myAlgorithm(knownData, vars);
    expect(graph.structuralHammingDistance(trueGraph)).toBeLessThan(3);
  });
});
```

### 13.2 Adding a New Storage Backend

1. **Create package** `packages/storage-mydb/package.json`
2. **Implement interfaces** from core:

```typescript
class MyRelationalStore implements IRelationalStore {
  // IMetricStore
  async insertMetrics(...) { ... }
  // IModelStore
  async saveCPT(...) { ... }
  // IResultStore
  async saveResult(...) { ... }
  // ITransactionStore
  async beginTransaction() { ... }
}
```

3. **Inject via DI** — no pipeline code changes needed:

```typescript
const config: CausalityAnalyzerConfig = {
  relationalStore: new MyRelationalStore(config),
  graphStore: new MyGraphStore(config),
};
```

### 13.3 Adding a New Visual Renderer

1. **Implement GraphRenderer**:

```typescript
class WebGLRenderer implements GraphRenderer {
  render(canvas, data, theme) {
    const gl = canvas.getContext('webgl2');
    // WebGL rendering logic for 500+ node graphs
  }
  hitTest(x, y, data) { ... }
  dispose() { ... }
}
```

2. **Use in component**:

```html
<ca-causal-graph .renderer="${new WebGLRenderer()}"></ca-causal-graph>
```

---

This architecture document describes the design principles, patterns, and trade-offs that make Causality Analyzer a production-ready, extensible, and maintainable causal AI library. For implementation details of specific algorithms, see the [TypeDoc API reference](https://agentix-e.github.io/causality-analyzer/api/). For usage instructions, see the [User Guide](./user-guide.md).
