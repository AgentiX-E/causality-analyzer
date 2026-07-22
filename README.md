# Causality Analyzer

> A production-grade causal AI platform for AIOps — causal discovery, root cause analysis, effect estimation, counterfactual reasoning, and visualization in pure TypeScript.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![CI](https://github.com/AgentiX-E/causality-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/AgentiX-E/causality-analyzer/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-report-blue)](https://agentix-e.github.io/causality-analyzer/coverage/)

## Overview

Causality Analyzer combines a complete causal inference stack with AIOps-specific root cause analysis pipelines. It answers three critical questions for incident response:

1. **What happened?** — Anomaly detection across metrics, traces, and logs
2. **Why did it happen?** — Causal discovery + root cause analysis with confidence intervals
3. **What if?** — Counterfactual reasoning and intervention simulation

Built as a pnpm monorepo with dependency injection, it supports embedded storage (SQLite + OverGraph) and remote storage (PostgreSQL + Neo4j) with full mTLS for enterprise deployments.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Causality Analyzer                       │
├───────────┬───────────┬──────────────┬──────────┬──────────┤
│   core    │ pipeline  │ storage-embed│storage-  │  visual  │
│           │           │              │ remote   │          │
├───────────┼───────────┼──────────────┼──────────┼──────────┤
│ Types     │ Detection │ SQLite       │PostgreSQL│ Web      │
│ Interfaces│ Discovery │ OverGraph    │ Neo4j    │Components│
│ Math      │ RCA       │              │ mTLS     │ uPlot    │
│ Registry  │ Inference │              │          │ Canvas   │
│ Config    │ GCM       │              │          │          │
└───────────┴───────────┴──────────────┴──────────┴──────────┘

Data Flow:
  Raw Metrics → Standardize → Detect Anomalies → Causal Discovery (PC/FCI)
  → RCA (Bayesian/HT/RandomWalk) → Effect Estimation → Counterfactuals
  → Visualization → Storage
```

## Key Features

- **Causal Discovery** — PC algorithm (stable variant), FCI with R1-R4 orientation rules
- **Root Cause Analysis** — BayesianRCA, RandomWalkRCA, HTRCA, FPGrowthRCA, CIRCA pipeline
- **Causal Effect Estimation** — Backdoor adjustment, Frontdoor, IV/2SLS, Propensity Score, Doubly Robust
- **Sensitivity Analysis** — E-value, partial R², robustness value with plain-English interpretation
- **do-Calculus** — Pearl's identification rules + ID algorithm (Tian & Pearl, Shpitser & Pearl)
- **Structural Causal Models** — Additive noise, PostNonlinear (sigmoid), auto mechanism assignment
- **Counterfactual Inference** — Abduction-Action-Prediction framework, Shapley anomaly attribution
- **Distribution Change Detection** — Mechanism change attribution with bootstrap confidence intervals
- **Graph Validation** — CI-based graph falsification, local Markov condition testing
- **Mediation Analysis** — Natural direct/indirect effects, arrow strength quantification
- **Deterministic Reproducibility** — All stochastic algorithms accept optional seed via `createRNG()`
- **Enterprise Security** — Full mTLS on both Bolt (Neo4j) and PG-wire (PostgreSQL)
- **TypeScript Native** — Strict type safety, dependency injection, framework-agnostic design

## Packages

| Package | Version | Description |
|---------|---------|-------------|
| [`@agentix-e/causality-analyzer-core`](./packages/core) | [![npm](https://img.shields.io/badge/version-0.1.0-blue)](https://www.npmjs.com/package/@agentix-e/causality-analyzer-core) | Types, interfaces, ColumnarTable, math, plugin registry |
| [`@agentix-e/causality-analyzer-pipeline`](./packages/pipeline) | [![npm](https://img.shields.io/badge/version-0.1.0-blue)](https://www.npmjs.com/package/@agentix-e/causality-analyzer-pipeline) | Detection, causal discovery, RCA, inference, GCM, visualization data |
| [`@agentix-e/causality-analyzer-storage-embed`](./packages/storage-embed) | [![npm](https://img.shields.io/badge/version-0.1.0-blue)](https://www.npmjs.com/package/@agentix-e/causality-analyzer-storage-embed) | SQLite (better-sqlite3) + OverGraph embedded stores |
| [`@agentix-e/causality-analyzer-storage-remote`](./packages/storage-remote) | [![npm](https://img.shields.io/badge/version-0.1.0-blue)](https://www.npmjs.com/package/@agentix-e/causality-analyzer-storage-remote) | PostgreSQL (pg) + Neo4j (neo4j-driver-lite) with mTLS |
| [`@agentix-e/causality-analyzer-visual`](./packages/visual) | [![npm](https://img.shields.io/badge/version-0.1.0-blue)](https://www.npmjs.com/package/@agentix-e/causality-analyzer-visual) | Lit 3 Web Components for causal graphs + time series |

## Quick Start

### Installation

```bash
git clone https://github.com/AgentiX-E/causality-analyzer.git
cd causality-analyzer
pnpm install
pnpm run --filter @agentix-e/causality-analyzer-core build
```

### Basic Usage: Anomaly Detection

```typescript
import { StatsDetector } from '@agentix-e/causality-analyzer-pipeline';

const detector = new StatsDetector({ method: 'zscore' });
detector.train([[1, 2], [1.1, 2.1], [0.9, 1.9]]);

const result = detector.update([5.0, 8.0]);
console.log(result.isAnomalous); // true
console.log(result.scores);      // z-scores per metric
```

### Basic Usage: Causal Discovery

```typescript
import { Matrix } from 'ml-matrix';
import { pcAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

// 3 variables, 500 observations
const data = new Matrix(500, 3);
// ... populate data ...

const { graph } = pcAlgorithm(data, ['CPU', 'Memory', 'Latency']);
console.log(graph.edges);
// [{ source: 'CPU', target: 'Latency', ... }, ...]
```

### Basic Usage: Root Cause Analysis

```typescript
import { CausalGraph, BayesianRCA } from '@agentix-e/causality-analyzer-pipeline';

const graph = new CausalGraph(['Memory', 'CPU', 'Latency']);
graph.addEdge('Memory', 'CPU');
graph.addEdge('CPU', 'Latency');

const rca = new BayesianRCA();
rca.train(graph, new Set(['CPU', 'Latency']), data);
const result = rca.findRootCauses(['CPU', 'Latency']);

console.log(result.rootCauses[0].name);  // 'Memory'
console.log(result.rootCauses[0].score); // posterior probability
```

### Basic Usage: Causal Effect Estimation

```typescript
import { adjustBackdoor } from '@agentix-e/causality-analyzer-pipeline';

const nodeIndex = new Map([['Treatment', 0], ['Outcome', 1], ['Confounder', 2]]);
const { ate, se, adjustors } = adjustBackdoor(graph, 'Treatment', 'Outcome', data, nodeIndex);

console.log(`ATE = ${ate.toFixed(3)} ± ${(se * 1.96).toFixed(3)}`);
// ATE = 0.742 ± 0.128
```

### Basic Usage: Counterfactual Reasoning

```typescript
import { CausalGraph, StructuralCausalModel } from '@agentix-e/causality-analyzer-pipeline';

const scm = new StructuralCausalModel(graph);
scm.train(data);

// What would latency be if we had increased memory allocation?
const noise = scm.abduct({ Memory: 0.5, CPU: 0.8, Latency: 120 });
const cf = scm.counterfactual(noise, { Memory: 1.0 });
console.log(`Counterfactual latency: ${cf.Latency?.toFixed(0)} ms`);
```

## Project Structure

```
causality-analyzer/
├── packages/
│   ├── core/                       # Foundation layer
│   │   └── src/
│   │       ├── index.ts            # Barrel exports
│   │       ├── types/index.ts      # CausalEdge, CausalGraph, RCAResult, etc.
│   │       ├── interfaces/index.ts # IRelationalStore, IGraphStore
│   │       ├── table/index.ts      # ColumnarTable (zero-copy columnar data)
│   │       ├── math.ts             # solveLinear, normalTail, erf, colMean, createRNG
│   │       ├── registry/index.ts   # PluginRegistry (detectors, graphs, analyzers)
│   │       ├── config/index.ts     # BaseConfig with Zod validation
│   │       └── di/index.ts         # Dependency injection config
│   │
│   ├── pipeline/                   # Causal analysis engine
│   │   └── src/
│   │       ├── index.ts            # Barrel exports (all sub-packages)
│   │       ├── data/standardizer.ts     # zscore, minmax, robust, discretize
│   │       ├── detect/stats-detector.ts # Z-score / MAD / IQR anomaly detection
│   │       ├── detect/spectral-residual.ts # FFT-based anomaly detection
│   │       ├── detect/spot.ts           # SPOT/DSPOT extreme value detectors
│   │       ├── detect/voting-detector.ts # Ensemble voting (majority/max/weighted)
│   │       ├── graph/causal-graph.ts     # Graph data structure (DAG/PDAG/CPDAG)
│   │       ├── graph/pc.ts              # PC algorithm (constraint-based)
│   │       ├── graph/advanced-discovery.ts # FCI, Grow-Shrink, targeted discovery
│   │       ├── analyze/rca.ts            # BayesianRCA, RandomWalkRCA, HTRCA, FPGrowthRCA
│   │       ├── analyze/circa.ts          # CIRCA pipeline: RHTScorer + DAScorer
│   │       ├── infer/causal-inference.ts # Backdoor/frontdoor ID, refutation
│   │       ├── infer/effect-estimation.ts # Backdoor, frontdoor, IV, PS, DR estimators
│   │       ├── infer/sensitivity.ts      # E-value, partial R², robustness value
│   │       ├── infer/do-calculus.ts      # do-calculus rules + ID algorithm
│   │       ├── infer/mediation.ts        # NDE/NIE, arrow strength
│   │       ├── infer/cate-fairness.ts    # CATE, IPW, counterfactual fairness
│   │       ├── infer/bootstrap-ci.ts     # Bootstrap CI + parallel execution
│   │       ├── gcm/structural-causal-model.ts # SCM with counterfactuals
│   │       ├── gcm/model-evaluation.ts   # R², MSE, Shapley RCA, bootstrap CI
│   │       ├── gcm/nonlinear-mechanisms.ts # PostNonlinear, auto-assign, relevance
│   │       ├── gcm/distribution-change.ts # Mechanism change detection + attribution
│   │       ├── gcm/graph-falsification.ts # CI-based falsification + LMC testing
│   │       ├── viz/viz-data.ts           # Visualization data builders
│   │       └── viz/fusion.ts             # Multi-modal RCA fusion
│   │
│   ├── storage-embed/              # Embedded storage
│   │   └── src/
│   │       ├── embed-relational-store.ts # SQLite via better-sqlite3
│   │       └── embed-graph-store.ts      # OverGraph LSM-tree graph store
│   │
│   ├── storage-remote/             # Remote storage (enterprise)
│   │   └── src/
│   │       ├── remote-relational-store.ts # PostgreSQL via pg.Client
│   │       ├── remote-graph-store.ts      # Neo4j via neo4j-driver-lite
│   │       └── types.ts                   # MtlsConfig, TrustStrategy
│   │
│   └── visual/                     # Web Components
│       └── src/
│           └── components/
│               ├── ca-causal-graph.ts      # Force-directed causal graph
│               ├── ca-time-series.ts       # Time series with anomaly bands
│               └── ca-root-cause-ranking.ts # Ranked root cause list
│
├── docs/
│   ├── adr/             # Architecture Decision Records
│   ├── audit-*.md       # Audit reports
│   └── user-guide.md    # Comprehensive user guide
├── .github/workflows/   # CI/CD (lint, typecheck, test, browser, Neo4j mTLS)
└── typedoc.json         # API documentation config
```

## Development

```bash
# Install
pnpm install

# Build foundation
pnpm run --filter @agentix-e/causality-analyzer-core build

# Quality gates (all packages)
pnpm -r lint
pnpm -r typecheck
pnpm -r test

# Generate API docs
pnpm docs
```

CI runs on every PR: lint → typecheck → unit tests → browser tests → Neo4j mTLS integration tests.

## Documentation

- [**User Guide**](./docs/user-guide.md) — From zero to production with Causality Analyzer
- [**API Reference**](./docs/api/) — TypeDoc-generated API documentation
- [**Architecture Decisions**](./docs/adr/) — Key design rationales
- [**Changelog**](./CHANGELOG.md) — Full release history
- [**Contributing**](./CONTRIBUTING.md) — Development workflow and standards

## References

| Resource | Link |
|----------|------|
| PC Algorithm | Spirtes, Glymour & Scheines (2000). *Causation, Prediction, and Search.* |
| FCI Algorithm | Zhang (2008). *On the completeness of orientation rules* |
| CIRCA | Li et al. (KDD 2022). *Causal Inference-Based Root Cause Analysis* |
| DoWhy | [py-why/dowhy](https://github.com/py-why/dowhy) |
| Intel Causal Discovery Lab | [IntelLabs/causality-lab](https://github.com/IntelLabs/causality-lab) |
| SPOT/DSPOT | Siffer et al. (KDD 2017). *Anomaly Detection in Streams* |

## License

MIT — see [LICENSE](LICENSE) for details.
