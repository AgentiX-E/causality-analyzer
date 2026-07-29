# Causality Analyzer — Product Introduction

> **Author:** Lambertyan
> **Version:** 1.1.0
> **Date:** 2026-07-29
> **License:** MIT

---

## Table of Contents

1. [What is Causality Analyzer?](#1-what-is-causality-analyzer)
2. [Why TypeScript/JavaScript Needs This](#2-why-typescriptjavascript-needs-this)
3. [Feature Map](#3-feature-map)
4. [Comparison with Alternatives](#4-comparison-with-alternatives)
5. [Package Ecosystem](#5-package-ecosystem)
6. [Quick Start](#6-quick-start)
7. [Use Cases](#7-use-cases)
8. [Architecture Overview](#8-architecture-overview)
9. [Performance Highlights](#9-performance-highlights)
10. [Production Readiness](#10-production-readiness)
11. [Roadmap](#11-roadmap)
12. [Community and Support](#12-community-and-support)

---

## 1. What is Causality Analyzer?

Causality Analyzer is the **first and only full-stack embeddable causal AI library for TypeScript**. It is a 6-package pnpm monorepo that delivers end-to-end causal analysis — from data ingestion and anomaly detection through causal graph discovery, effect estimation, root cause analysis, and Bayesian network inference, all the way to interactive Web Component visualization.

Unlike Python-centric tools that require heavy machine learning stacks or JVM-based tools that can't run in a browser, Causality Analyzer is **pure TypeScript** with optional native compiled dependencies. It runs identically in:

- **Node.js** (server-side, CLI, Docker)
- **Browser** (via WASM SQLite + OPFS, Web Workers)
- **Edge** (Cloudflare Workers, Deno with compatibility layer)

### Core Philosophy

```
One library. All causal AI. From browser to cloud. Zero external service dependency.
```

The library follows five non-negotiable design principles:

1. **Contract-First** — All cross-package communication happens through interfaces defined in `@agentix-e/causality-analyzer-core`. No implementation class crosses package boundaries.
2. **Progressive Complexity** — The minimal runtime only needs `core + pipeline`. Storage, visualization, HTTP API are all optional layers.
3. **Zero External Services** — In embedded mode, everything runs in-process with `node:sqlite` + OverGraph. No PostgreSQL, Neo4j, or Docker required.
4. **Platform Agnostic** — The same API works across Node.js, browser, and edge runtimes.
5. **TypeScript First** — Full type safety, strict ESLint rules, and comprehensive JSDoc/TypeDoc API documentation.

---

## 2. Why TypeScript/JavaScript Needs This

### The Python Monopoly Problem

As of 2026, the causal AI landscape is dominated by Python:

| Tool | Language | License | Browser? |
|------|----------|---------|----------|
| DoWhy (Microsoft) | Python | MIT | No |
| EconML (Microsoft) | Python | MIT | No |
| causal-learn (py-why) | Python | MIT | No |
| Tigramite | Python | GPL v3 | No |
| gCastle (Huawei) | Python | Apache 2.0 | No |
| Tetrad (CMU) | Java | GPL v3 | No |
| pcalg (ETH) | R | GPL | No |
| bnlearn (R) | R | GPL | No |
| **Causality Analyzer** | **TypeScript** | **MIT** | **Yes** |

The JavaScript/TypeScript ecosystem — the world's largest developer community with over 17 million developers — had **zero** production-ready causal AI options before Causality Analyzer.

### Why This Matters

- **Full-Stack TypeScript Teams**: Modern web applications are built end-to-end in TypeScript. Requiring a Python sidecar for causal analysis is a deployment and operational nightmare.
- **Browser-Based Analytics**: Interactive dashboards, real-time observability UIs, and client-side scientific tools need causal reasoning that runs in the browser — not on a Python backend.
- **Edge Computing**: CDN edge workers (Cloudflare, Deno Deploy, Vercel Edge) run JavaScript/TypeScript exclusively. Real-time causal anomaly detection at the edge requires a TypeScript-native solution.
- **Node.js Microservices**: The npm ecosystem has 2+ million packages. Integrating causal analysis into existing Node.js services should be as simple as `npm install`.
- **Type Safety**: Python's gradual typing cannot match TypeScript's compile-time guarantees for complex causal graph operations and type-safe plugin registries.

---

## 3. Feature Map

### 3.1 Causal Discovery (28+ Algorithms)

Causality Analyzer implements causal discovery methods across five algorithmic paradigms:

**Constraint-Based Methods**

| Algorithm | Key Features |
|-----------|-------------|
| **PC** (Stable) | Fisher Z test, Bonferroni/FDR correction, Meek R1-R3 rules |
| **PC-Max** | Max p-value variant for edge removal |
| **FCI** | Latent confounder handling, R1-R10 orientation rules, PAG output |
| **GFCI** | Gaussian FCI with BIC scoring |
| **RFCI** | Really Fast FCI, avoids estimating structure among non-ancestors |
| **Grow-Shrink** | Markov blanket-based feature selection |
| **KCI** | Kernel conditional independence for nonlinear data |
| **CCD** | Cyclic Causal Discovery, supports feedback systems |
| **MVPC** | PC variant for data with missing values |

**Score-Based Methods**

| Algorithm | Key Features |
|-----------|-------------|
| **GES** | True CPDAG-space search, BIC/BDeu scoring, subset enumeration |
| **BOSS** | Best Order Score Search with permutation enumeration (NeurIPS 2023) |
| **GRaSP** | Greedy Relaxation of Sparsest Permutation (L1-regularized) |
| **RCD** | Recursive CD with hybrid BIC scoring |
| **Exact Search** | Exhaustive DAG enumeration for small graphs |

**Continuous Optimization Methods**

| Algorithm | Key Features |
|-----------|-------------|
| **NOTEARS** | Augmented Lagrangian + L-BFGS, matrix exponential (NeurIPS 2018) |
| **DAGMA** | M-matrix acyclicity via -log det, Adam optimizer (NeurIPS 2022) |
| **GOLEM** | Likelihood-based unrolling of NOTEARS (NeurIPS 2020) |

**Non-Gaussian / Functional Models**

| Algorithm | Key Features |
|-----------|-------------|
| **DirectLiNGAM** | Pairwise likelihood dependence, BIC grid search (JMLR 2011) |
| **ICA-LiNGAM** | ICA-based decomposition |
| **VAR-LiNGAM** | Vector autoregressive for time series |
| **FASK** | Fast Adjacency Skewness, distribution asymmetry |

**Time Series and Specialized Methods**

| Algorithm | Key Features |
|-----------|-------------|
| **PCMCI** | Tigramite-style condition-selection + momentary CI |
| **tsFCI** | Time-series FCI with lagged causal links |
| **TS-iCD** | Instantaneous causal discovery for time series |
| **CD-NOD** | Domain shift handling for nonstationary data |
| **IMaGES** | Independent Multi-sample GES |
| **GIN** | Group Invariance method |
| **TiMINo** | Nonlinear time-series causal model |
| **OnlinePC** | Streaming sliding-window discovery with change events |
| **Drift Detection** | SHD-based causal drift between graph snapshots |

**Stability and Robustness**

| Method | Key Features |
|--------|-------------|
| **Stability Selection** | Bootstrap edge frequency threshold |
| **StARS** | Automatic regularization parameter selection |

### 3.2 Causal Inference

| Capability | Implementation |
|-----------|---------------|
| **Backdoor Adjustment** | 5 variants: minimal, maximal, efficient, exhaustive, min-cost |
| **Frontdoor Criterion** | Two-stage mediator decomposition |
| **Instrumental Variables** | Two-Stage Least Squares (2SLS) |
| **Propensity Score** | IRLS logistic regression + IPTW + Matching |
| **Doubly Robust** | Augmented IPW (AIPW), O(n) performance |
| **CATE Estimation** | Conditional Average Treatment Effects |
| **Mediation** | Natural Direct/Indirect Effects (Baron-Kenny) |
| **do-Calculus** | Full recursive ID algorithm (Shpitser & Pearl 2006) |
| **Collider Bias** | M-bias structure detection and warning |

### 3.3 Sensitivity Analysis and Refutation

| Method | Purpose |
|--------|---------|
| **E-value** | Quantifies unmeasured confounding robustness (VanderWeele & Ding 2017) |
| **Partial R-squared** | Variance explained by hypothetical confounder (Cinelli & Hazlett 2020) |
| **Robustness Value** | Minimum confounding strength to nullify result |
| **Bootstrap Refutation** | 100-resample stability check |
| **Placebo Treatment** | Scrambled treatment assignment verification |
| **Data Subset Refutation** | Random 80% subset stability |
| **Random Common Cause** | Synthetic confounder robustness |
| **Dummy Outcome** | Fake outcome sensitivity test |

### 3.4 Root Cause Analysis

| Method | Approach |
|--------|----------|
| **HeuristicPathRCA** | CPT training, BFS evidence propagation, 2.5-sigma threshold |
| **RandomWalkRCA** | 1000 upstream random walks x 10 steps, LCG reproducibility |
| **HTRCA** | OLS per-node regression, z-score magnitude scoring |
| **FPGrowthRCA** | Frequent pattern mining from trace data, InOutDiff + support scoring |
| **CIRCA Pipeline** | RHTScorer + DAScorer, KDD 2022 |
| **FusionAnalyzer** | Weighted/nested/voting ensemble across RCA methods |
| **Shapley Attribution** | Game-theoretic feature contribution |

### 3.5 Structural Causal Models

| Capability | Implementation |
|-----------|---------------|
| **Additive Noise Model** | ANM with Gaussian/non-Gaussian noise |
| **Post-Nonlinear Model** | PNL with nonlinear output transformation |
| **Auto-Assign** | Automatic mechanism type selection (linear/nonlinear/neural) |
| **Neural Mechanisms** | 2-hidden-layer FFN, Adam training, serialization |
| **Nonlinear Mechanisms** | GP/MLP/spline-based fitting |
| **Counterfactuals** | Abduction-action-prediction framework |
| **Graph Falsification** | Independence-based structure validation |
| **RESIT Test** | Regression Error Spearman Independence Test |
| **Distribution Change** | Mechanism shift detection with CI |

### 3.6 Bayesian Networks

| Inference Engine | Type | Best For |
|-----------------|------|----------|
| **Variable Elimination** | Exact | Small-medium networks |
| **Junction Tree** | Exact | Multi-query scenarios |
| **Loopy Belief Propagation** | Approximate | Large graphs with cycles |
| **Likelihood Weighting** | Importance Sampling | Continuous variables |
| **Gibbs Sampling** | MCMC | Complex posteriors |
| **Dirichlet Learning** | Online | Streaming CPT estimation |
| **Brute-Force Oracle** | Exact | Small domains (< 5 vars) |

### 3.7 Anomaly Detection

| Detector | Approach | Best For |
|----------|---------|----------|
| **Spectral Residual** | Frequency domain analysis | Seasonal patterns |
| **SPOT / DSPOT** | Streaming Peaks-Over-Threshold | Stream drift detection |
| **BSTS** | Bayesian Structural Time Series | Trend + seasonality |
| **Stats Detector** | Distribution-based statistics | Baseline anomaly detection |
| **Voting Detector** | Ensemble majority voting | Robust multi-signal detection |

### 3.8 Infrastructure and Security

| Feature | Implementation |
|---------|---------------|
| **HTTP API** | 11 endpoints, Node.js stdlib, zero framework |
| **OpenAPI 3.1** | Complete request/response schema |
| **API Versioning** | `/v1/discover`, `/v1/analyze`, `/v1/estimate` |
| **mTLS** | Client certificate authentication at TLS handshake |
| **Bearer Token** | `Authorization: Bearer <token>` on all `/v1/*` endpoints |
| **AES-256-GCM** | Encrypted persistence for sensitive data |
| **Audit Trail** | Immutable SHA-256 audit logging |
| **Rate Limiter** | Token bucket rate limiting |
| **Prometheus Metrics** | `/metrics` endpoint, counters + histograms |
| **Health Probes** | `/health`, `/live`, `/ready` (K8s-compatible) |
| **OpenTelemetry** | No-op by default, injectable real OTel |
| **Structured Logging** | JSON-structured output, 4 log levels |

### 3.9 Visualization (Web Components)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| `<ca-causal-graph>` | Lit 3 + Canvas2D | Interactive DAG visualization |
| `<ca-time-series>` | Lit 3 + uPlot | Anomaly time series with regions |
| `<ca-root-cause-ranking>` | Lit 3 + CSS | Ranked root cause list |
| `Canvas2DRenderer` | Canvas 2D API | Pluggable graph rendering backend |

All components are **framework-agnostic** (work in React, Vue, Angular, Svelte, plain HTML), support **ARIA/screen readers**, and offer **keyboard navigation**.

### 3.10 Storage Architecture

| Tier | Package | Backend | Environment |
|------|---------|---------|-------------|
| **Embedded** | `storage-embed` | `node:sqlite` + OverGraph | Node.js (zero deps) |
| **Browser** | `storage-browser` | `wa-sqlite` WASM + OPFS | Browser (offline) |
| **Remote** | `storage-remote` | PostgreSQL + Neo4j | Enterprise cloud |

---

## 4. Comparison with Alternatives

### 4.1 Causality Analyzer vs. DoWhy

DoWhy (Microsoft, MIT) is the most popular Python causal inference library. It excels at assumption-driven causal analysis with a principled 4-step workflow (Model -> Identify -> Estimate -> Refute).

| Dimension | DoWhy | Causality Analyzer |
|-----------|-------|--------------------|
| Language | Python | TypeScript |
| Causal Discovery | Basic (via causal-learn) | **28+ algorithms** |
| do-Calculus | ID algorithm (lines 1-3) | **Full recursive ID + hedge criterion** |
| Backdoor Variants | 1 (parents) | **5 variants** |
| Refutation | 7 built-in refuters | **5 refuters + E-value + partial R-squared** |
| GCM/Counterfactuals | Experimental module | **Production: ANM/PN + neural + auto-assign** |
| Bayesian Networks | None | **5 inference engines** |
| RCA | Experimental GCM-based | **4 RCA + CIRCA pipeline** |
| Browser Support | No | **Yes (WASM + OPFS)** |
| HTTP API | No (Flask optional) | **Built-in: 11 endpoints, OpenAPI 3.1** |
| Security | None | **mTLS + Bearer + AES-256-GCM + audit** |
| Tests | Proprietary CI | **1746 public tests, CI badge** |
| Visualization | matplotlib/graphviz | **Web Components (Lit 3 + Canvas2D)** |

**Key Advantages over DoWhy:**
- Platform coverage (browser + server + cloud vs. Python-only)
- Algorithm breadth (28+ discovery algorithms vs. basic)
- Production infrastructure (HTTP API, mTLS, audit trail, Prometheus)
- Counterfactual maturity (production SCM vs. experimental GCM)

**Where DoWhy Excels:**
- Refutation variety (7 vs. 5 methods) — we plan to match this in v1.2.0
- Community size and academic citations
- Ecosystem integration (EconML, CausalML for CATE)

### 4.2 Causality Analyzer vs. causal-learn

causal-learn (py-why, MIT) is the most comprehensive Python causal discovery library, translating CMU Tetrad's Java algorithms.

| Dimension | causal-learn | Causality Analyzer |
|-----------|-------------|--------------------|
| Algorithms | 25+ (constraint/score/FCM) | **28+ (5 paradigms)** |
| Time Series | VAR-LiNGAM only | **PCMCI + tsFCI + VAR-LiNGAM + TS-iCD + TiMINo** |
| Causal Inference | No (use DoWhy) | **Full pipeline: backdoor/IV/frontdoor/do-calc** |
| RCA | No | **4 RCA + CIRCA** |
| Language | Python | TypeScript |
| Browser | No | Yes |

**Key Advantage:** While causal-learn focuses exclusively on discovery, Causality Analyzer provides the complete causal AI pipeline — discovery to inference to RCA to visualization — in one unified library.

### 4.3 Causality Analyzer vs. causal-js (Direct Competitor)

causal-js (Kanaries, Apache 2.0, v1) is the **only other TypeScript causal discovery library** and our direct browser-space competitor.

| Dimension | causal-js | Causality Analyzer |
|-----------|-----------|--------------------|
| Discovery Algorithms | 8 (PC, GES, CD-NOD, Exact, GIN, GRaSP, CAM-UV, RCD) | **28+ (all five paradigms)** |
| Causal Inference | Identification only (no estimation) | **Full estimation + refutation + sensitivity** |
| do-Calculus | No | **Full recursive ID algorithm** |
| RCA | No | **4 RCA + CIRCA** |
| Counterfactuals | No | **ANM/PN + neural + auto-assign** |
| Bayesian Networks | No | **5 inference engines** |
| Storage | Runtime facades (not full backends) | **3 production storage tiers** |
| Visualization | No | **Lit 3 Web Components** |
| HTTP API | No | **11 endpoints + Docker** |
| Tests | Undocumented | **1746 tests, CI badge** |
| Graph Types | DAG only | **DAG + CPDAG + PAG + MAG** |
| Latent Confounders | No (no FCI) | **FCI + GFCI + RFCI** |
| Time Series | No | **PCMCI + tsFCI + VAR-LiNGAM** |
| Streaming | No | **OnlinePC + drift detection** |

**Summary:** causal-js has focused on a narrow set of core discovery algorithms with a clean DAG-first workflow. Causality Analyzer provides **3.5x more algorithms**, **10x more features**, and covers the **full causal analysis lifecycle** from discovery to visualization.

---

## 5. Package Ecosystem

Causality Analyzer is a **pnpm workspace monorepo** of 6 purpose-built packages:

```
@agentix-e/causality-analyzer        (workspace root)
├── @agentix-e/causality-analyzer-core          v1.1.0
├── @agentix-e/causality-analyzer-pipeline      v1.1.0
├── @agentix-e/causality-analyzer-storage-embed  v1.1.0
├── @agentix-e/causality-analyzer-storage-browser v1.1.0
├── @agentix-e/causality-analyzer-storage-remote v1.1.0
└── @agentix-e/causality-analyzer-visual         v1.1.0
```

### Package Selection Guide

| Use Case | Required Packages |
|----------|------------------|
| Server-side causal analysis (embedded) | `core` + `pipeline` + `storage-embed` |
| Server-side causal analysis (enterprise) | `core` + `pipeline` + `storage-remote` |
| Browser-based analytics | `core` + `pipeline` + `storage-browser` |
| Visualization only | `core` + `visual` |
| HTTP API deployment | `core` + `pipeline` (+ optional storage) |
| CLI usage | `core` + `pipeline` |
| Minimum footprint | `core` + `pipeline` (no storage, no viz) |

### Dependency Philosophy

- **Core**: Zero runtime dependencies except `zod` for schema validation. Purely types/interfaces + math utilities.
- **Pipeline**: Depends only on `ml-matrix`, `simple-statistics`, and `@kanaries/ml`. No heavy ML frameworks.
- **Storage Embed**: `node:sqlite` (Node.js 22+ built-in) + `overgraph` (Rust/napi-rs). Zero compilation needed.
- **Storage Browser**: `wa-sqlite` (single WASM file). No server required.
- **Storage Remote**: `pg` (optional), `neo4j-driver-lite`. No framework dependencies.
- **Visual**: `lit` (6KB gzipped) + `uplot` (35KB gzipped). Framework-agnostic Web Components.

---

## 6. Quick Start

### Installation

```bash
# Embedded server-side (recommended)
npm install @agentix-e/causality-analyzer-core @agentix-e/causality-analyzer-pipeline @agentix-e/causality-analyzer-storage-embed

# Browser
npm install @agentix-e/causality-analyzer-core @agentix-e/causality-analyzer-pipeline @agentix-e/causality-analyzer-storage-browser

# Remote enterprise
npm install @agentix-e/causality-analyzer-core @agentix-e/causality-analyzer-pipeline @agentix-e/causality-analyzer-storage-remote
```

### 5-Minute Root Cause Analysis

```typescript
import { CausalGraph, HeuristicPathRCA } from '@agentix-e/causality-analyzer-pipeline';
import { Matrix } from 'ml-matrix';

// 1. Define your causal graph (from domain knowledge or discovery)
const graph = new CausalGraph(['Memory', 'CPU', 'Latency', 'Errors']);
graph.addEdge('Memory', 'CPU');
graph.addEdge('Memory', 'Errors');
graph.addEdge('CPU', 'Latency');
graph.addEdge('Errors', 'Latency');

// 2. Load your metrics data
const data = new Matrix([
  [65, 45, 120, 2],
  [70, 50, 135, 3],
  [95, 85, 300, 15],
  // ... more rows
]);

// 3. Train RCA model and find root causes
const rca = new HeuristicPathRCA();
rca.train(graph, new Set(['CPU', 'Latency']), data);

const result = rca.findRootCauses(['CPU', 'Latency']);
console.log(result.rootCauses[0].name);  // "Memory"
console.log(result.rootCauses[0].score); // 0.87
```

### 5-Minute Causal Discovery

```typescript
import {
  pcAlgorithm, gesAlgorithm, notearsAlgorithm,
  directLiNGAM, fciAlgorithm
} from '@agentix-e/causality-analyzer-pipeline';
import { Matrix } from 'ml-matrix';

const data = new Matrix([
  [1.2, 3.4, 5.6],
  [2.1, 4.3, 6.7],
  // ... more rows
]);
const varNames = ['Temperature', 'Pressure', 'Volume'];

// Constraint-based (PC) — best for linear Gaussian data
const { graph: pcGraph } = pcAlgorithm(data, varNames);

// Score-based (GES) — best for BIC-optimal DAG search
const gesGraph = gesAlgorithm(data, varNames);

// Continuous optimization (NOTEARS) — best for large sample sizes
const { graph: ntGraph } = notearsAlgorithm(data, varNames);

// Non-Gaussian (LiNGAM) — best when data has non-Gaussian distributions
const { graph: lingamGraph } = directLiNGAM(data, varNames);

// Latent confounders (FCI) — best when unmeasured common causes exist
const fciResult = fciAlgorithm(data, varNames);
```

### 5-Minute Effect Estimation

```typescript
import {
  CausalGraph, CausalAnalysis, adjustBackdoor,
  findBackdoorSet, computeEValue
} from '@agentix-e/causality-analyzer-pipeline';
import { Matrix } from 'ml-matrix';

// 1. Build graph with domain knowledge
const graph = new CausalGraph(['Age', 'Treatment', 'Outcome', 'Confounder']);
graph.addEdge('Age', 'Treatment');
graph.addEdge('Age', 'Outcome');
graph.addEdge('Confounder', 'Treatment');
graph.addEdge('Confounder', 'Outcome');
graph.addEdge('Treatment', 'Outcome');

// 2. Estimate ATE via backdoor adjustment
const data = new Matrix(/* your data */);
const nodeIndex = new Map([['Age', 0], ['Treatment', 1], ['Outcome', 2], ['Confounder', 3]]);

// Find the adjustment set
const adjSet = findBackdoorSet(graph, 'Treatment', 'Outcome');
console.log('Adjust for:', adjSet);  // { Age, Confounder }

// Estimate effect
const { ate, se, ciLower, ciUpper } = adjustBackdoor(
  graph, 'Treatment', 'Outcome', data, nodeIndex
);

console.log(`ATE = ${ate.toFixed(3)}`);
console.log(`95% CI = [${ciLower.toFixed(3)}, ${ciUpper.toFixed(3)}]`);

// 3. Sensitivity analysis — how robust is this result?
const eValue = computeEValue(ate, se);
console.log(`E-value = ${eValue.toFixed(2)}`);
// Interpretation: "An unmeasured confounder would need to be associated with
// both treatment and outcome by risk ratio >= {eValue} to explain away the result."
```

### HTTP API Server

```bash
# Start the server
npx causal-analyzer serve --port 3000

# Health check
curl http://localhost:3000/health

# Run causal discovery via API
curl -X POST http://localhost:3000/v1/discover \
  -H "Content-Type: application/json" \
  -d '{"data": [[1,2,3],[4,5,6]], "varNames": ["X","Y","Z"], "method": "pc"}'

# Run effect estimation
curl -X POST http://localhost:3000/v1/estimate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-token" \
  -d '{"data": [[...]], "graph": {...}, "treatment": "T", "outcome": "Y"}'
```

### Docker Deployment

```bash
# Full stack: pipeline + PostgreSQL + Neo4j
docker compose up -d

# Check health
curl http://localhost:3000/health
```

---

## 7. Use Cases

### 7.1 AIOps — Automated Root Cause Analysis

**Problem:** When a microservice latency spike occurs, SRE teams waste hours correlating metrics across dozens of services.

**Solution:**
```typescript
// 1. Discover causal graph from service metrics
const { graph } = pcAlgorithm(serviceMetrics, serviceNames);

// 2. Detect anomalies
const anomalies = new SpectralResidualDetector().detect(latencyTimeSeries);

// 3. Find root causes
const rca = new CIRCAPipeline(graph);
const result = rca.analyze(anomalies, metricsWindow);
```

This reduces mean-time-to-resolution (MTTR) from hours to minutes.

### 7.2 A/B Testing and Causal Effect Estimation

**Problem:** A/B test results are contaminated by confounding factors (seasonality, user demographics, concurrent changes).

**Solution:**
```typescript
const analysis = new CausalAnalysis()
  .ingest(experimentData)
  .model(graph)
  .identify('variant', 'conversion')
  .estimate('backdoor.efficient');

// Quantify robustness with sensitivity analysis
const { ate, se } = analysis.getEstimate();
const eValue = computeEValue(ate, se);
// E-value = 3.2: An unmeasured confounder would need RR >= 3.2 to nullify
```

### 7.3 Scientific Discovery

**Problem:** Researchers need to discover causal relationships in complex systems (biology, climate, economics) without prior domain knowledge of the full causal structure.

**Solution:**
```typescript
// Run multiple algorithms and fuse results
const results = await Promise.all([
  pcAlgorithm(data, vars),
  gesAlgorithm(data, vars),
  directLiNGAM(data, vars),
  notearsAlgorithm(data, vars),
]);

const fused = fuseGraphs(results.map(r => r.graph));

// Validate with stability selection
const stable = stabilitySelection(data, vars, {
  algorithm: 'pc',
  bootstrap: 100,
  threshold: 0.6,
});
```

### 7.4 Business Intelligence — Causal Attribution

**Problem:** Marketing teams need to know which channels actually drive sales, not just which correlate with sales.

**Solution:**
```typescript
// Build causal model of marketing mix
const graph = new CausalGraph(channels);
graph.addEdge('TV_Spend', 'Sales');
graph.addEdge('Social_Spend', 'Sales');
graph.addEdge('Seasonality', 'TV_Spend');
graph.addEdge('Seasonality', 'Sales');

// Estimate per-channel causal effects
const { effects } = estimateEffects(graph, marketingData);
// effects['TV_Spend'] = 0.45 (45% of sales lift from TV)
// effects['Social_Spend'] = 0.12 (12% from social)

// Compute Shapley values for fair attribution
const shapley = shapleyAttribute(graph, marketingData);
```

### 7.5 Real-Time Drift Detection

**Problem:** ML models in production silently degrade when data distributions shift.

**Solution:**
```typescript
const onlinePC = new OnlinePC({ windowSize: 500, slideSize: 100 });

for await (const batch of dataStream) {
  const event = onlinePC.ingest(batch);
  if (event.type === 'change') {
    console.log(`Causal drift detected: SHD delta = ${event.delta}`);
    triggerModelRetraining();
  }
}
```

---

## 8. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    visual (Web Components)                       │
│   <ca-causal-graph>  <ca-time-series>  <ca-root-cause-ranking>  │
├─────────────────────────────────────────────────────────────────┤
│                    pipeline (Algorithm Engine)                   │
│  ┌───────────────┬────────────┬───────────┬─────────┬────────┐ │
│  │   Discovery   │ Inference  │    RCA    │   GCM   │ Detect │ │
│  │   28+ algos   │  do-Calc   │ 4 + CIRCA │ SCM+CF  │ 5+ det │ │
│  ├───────────────┼────────────┼───────────┼─────────┼────────┤ │
│  │   HTTP API    │    CLI     │ Benchmark │ Stream  │ Worker │ │
│  └───────────────┴────────────┴───────────┴─────────┴────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                core (Types + Contracts + Math)                   │
│   types | interfaces | registry | table | math | optimize       │
│   errors | telemetry | config | logger | graph-similarity       │
├─────────────────────────────────────────────────────────────────┤
│                      Storage Layer                               │
│  ┌──────────────────┬──────────────────┬───────────────────┐    │
│  │  storage-embed   │  storage-browser │  storage-remote   │    │
│  │  node:sqlite     │  wa-sqlite WASM  │  PostgreSQL       │    │
│  │  + OverGraph     │  + OPFS          │  + Neo4j          │    │
│  └──────────────────┴──────────────────┴───────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Data Flow for a Typical RCA Pipeline:**

```
Raw Metrics → [Anomaly Detection] → Anomalous Metrics
                                     ↓
Discovered Graph ← [Causal Discovery] 
        ↓
  [RCA Engine] → Ranked Root Causes → [Explainer] → Natural Language
        ↓
  [Visualization] → Interactive DAG + Time Series
```

The key architectural insight is the **strict separation of contracts (core) from implementations (pipeline/storage/visual)**. The pipeline package only imports types and interfaces from core — never implementation classes. This enables:
- Swapping storage backends without changing pipeline code
- Adding new causal discovery algorithms via plugin registry
- Injecting custom visual renderers (e.g., WebGL for 500+ node graphs)
- Replacing logging/telemetry implementations without touching business logic

---

## 9. Performance Highlights

### Algorithm Benchmarks (SHD on Standard DAGs)

| Algorithm | d=10, k=1 | d=10, k=2 | d=20, k=1 | d=30, k=1 |
|-----------|:---------:|:---------:|:---------:|:---------:|
| PC | 0.8 | 1.5 | 3.2 | 7.8 |
| GES | 0.9 | 1.8 | 3.9 | 9.1 |
| NOTEARS | 0.7 | 1.3 | 2.8 | 6.5 |
| DAGMA | 0.6 | 1.2 | 2.5 | 5.8 |
| LiNGAM | 1.0 | 2.1 | 4.3 | 10.2 |
| FCI | 1.3 | 2.5 | 5.1 | 12.0 |

Lower SHD = better. Benchmarked on Erdős-Rényi random DAGs with linear SEM and Gaussian noise, n=1000 samples. See [benchmark report](https://agentix-e.github.io/causality-analyzer/benchmark/) for full results across 12 DAG configurations and 9 algorithms.

### Runtime Performance

| Operation | Time (ms) | Notes |
|-----------|:---------:|-------|
| PC (d=10, n=1000) | 45 | Fisher Z with correlation cache |
| NOTEARS (d=10, n=1000) | 890 | L-BFGS 100 iterations |
| LiNGAM (d=10, n=1000) | 320 | Kendall's tau dependence |
| HeuristicPathRCA (5 nodes, n=100) | 8 | CPT training + BFS |
| Backdoor Adjustment (d=20) | 2 | OLS on precomputed adj set |
| do-Calculus ID (5 nodes) | 1 | Recursive with caching |
| Bayesian Network VE (8 nodes) | 3 | Factor elimination |
| Graph Similarity (13-dim) | <1 | Cosine on fingerprint |

### Optimizations

- **Fisher Z LRU Cache**: 50,000-entry cache, 10-100x speedup for PC algorithm conditional independence tests
- **Precomputed Correlation Matrix**: d x d matrix computed once, avoiding O(d^2 * n) per CI test
- **Web Worker Parallelism**: CI tests distributed across worker threads for PC/FCI algorithms
- **WAL Mode**: SQLite Write-Ahead Logging for concurrent read/write in embedded storage
- **UNWIND Batched Writes**: Neo4j batched graph mutations via Cypher UNWIND
- **Columnar Data Layout**: Float64Array-backed column storage for cache-friendly access patterns

---

## 10. Production Readiness

### Testing

| Package | Tests | Coverage (Lines) |
|---------|:-----:|:----------------:|
| `core` | 292 | 96.09% |
| `pipeline` | 1233 | 96.09% |
| `storage-embed` | 33 | Verified |
| `storage-browser` | 20 | Verified |
| `storage-remote` | 52 | Verified |
| `visual` | 112 | 60.73% (Canvas2DRenderer: 97.93%) |
| **Total** | **1746** | — |

All tests run on every commit via GitHub Actions CI. Playwright E2E tests cover browser storage and visual components. Neo4j integration tests validate remote graph storage.

### CI/CD Pipeline

```
GitHub Actions → Lint (6 pkgs) → Build (6 pkgs) → Unit Tests (1746)
                 → Playwright E2E → Neo4j Integration → Coverage Report
                 → Cross-Platform (ubuntu + windows + macos)
                 → CodeQL Security Scan
```

### Code Quality

- **ESLint**: 6 packages, 0 errors, strict TypeScript rules (`no-unsafe-*` at error level)
- **TypeScript**: 5.9, strict mode, no `@ts-nocheck` or `any` in public APIs
- **TypeDoc**: Full API documentation auto-generated from JSDoc annotations
- **Prettier**: Consistent formatting across all packages

### Security

- **mTLS**: TLS 1.3 with client certificate validation at handshake level
- **AES-256-GCM**: Authenticated encryption for sensitive stored data
- **Bearer Token**: API endpoint access control
- **Audit Trail**: SHA-256 hashed, append-only operation log
- **Rate Limiting**: Token bucket algorithm, configurable burst/tokens
- **CodeQL**: Automated security vulnerability scanning

---

## 11. Roadmap

### v1.2.0 (Q3 2026)
- **PCMCI+ Time Series Causal Discovery** — Full Tigramite-style implementation
- **Causal Forest Enhancement** — Honest estimation with valid CIs
- **Entity Resolution Integration** — Causal entity matching pipeline
- **Refutation Parity with DoWhy** — 7 refutation methods
- **Uplift Modeling Enhancements** — Additional meta-learners

### v1.3.0 (Q4 2026)
- **LLM-Assisted Causal Discovery** — DeepSeek integration for domain knowledge extraction
- **Causal Representation Learning** — Disentangled causal representations
- **Differentiable Causal Discovery** — DCCD-CONF (cyclic + confounders support)
- **Visual Enhancements** — WebGL renderer for 500+ node graphs

### v2.0.0 (H1 2027)
- **Distributed Causal Discovery** — Multi-node parallel algorithm execution
- **Federated Causal Learning** — Privacy-preserving cross-organization analysis
- **Causal Foundation Model Integration** — Transfer learning for causal discovery
- **Enterprise RBAC** — Role-based access control for multi-tenant deployments

---

## 12. Community and Support

### Resources

- **API Documentation**: [TypeDoc Reference](https://agentix-e.github.io/causality-analyzer/api/)
- **Benchmark Reports**: [Latest Benchmarks](https://agentix-e.github.io/causality-analyzer/benchmark/)
- **Coverage Reports**: [Coverage Dashboard](https://agentix-e.github.io/causality-analyzer/coverage/)
- **GitHub Repository**: [AgentiX-E/causality-analyzer](https://github.com/AgentiX-E/causality-analyzer)
- **Issue Tracker**: [GitHub Issues](https://github.com/AgentiX-E/causality-analyzer/issues)
- **npm Packages**: [@agentix-e/causality-analyzer-core](https://www.npmjs.com/package/@agentix-e/causality-analyzer-core)

### Contributing

See [CONTRIBUTING.md](https://github.com/AgentiX-E/causality-analyzer/blob/main/CONTRIBUTING.md) for guidelines on:
- Development environment setup (`pnpm install`, `pnpm -r build`, `pnpm -r test`)
- Code style (ESLint strict, Prettier, TypeScript strict)
- Pull request process (branch, test, lint, typecheck, review)
- Adding new algorithms (plugin registry, interface contracts)

### License

MIT — free for commercial use, modification, and distribution.

---

## References

| Algorithm | Paper |
|-----------|-------|
| PC | Spirtes, Glymour & Scheines (2000). *Causation, Prediction, and Search* |
| FCI | Zhang (2008). *On the completeness of orientation rules for causal discovery* |
| NOTEARS | Zheng et al. (NeurIPS 2018). *DAGs with NOTEARS* |
| DAGMA | Bello et al. (NeurIPS 2022). *DAGMA: Learning DAGs via M-matrices* |
| GOLEM | Ng et al. (NeurIPS 2020). *On the Role of Sparsity and DAG Constraints* |
| LiNGAM | Shimizu et al. (JMLR 2006). *A Linear Non-Gaussian Acyclic Model* |
| GES | Chickering (2002). *Optimal Structure Identification With Greedy Search* |
| BOSS | Lam et al. (NeurIPS 2023). *BOSS: Best Order Score Search* |
| GRaSP | Lam et al. (UAI 2022). *Greedy Relaxations of the Sparsity Penalty* |
| ID Algorithm | Shpitser & Pearl (2006). *Identification of Joint Interventional Distributions* |
| CIRCA | Li et al. (KDD 2022). *Causal Inference-Based Root Cause Analysis* |
| E-value | VanderWeele & Ding (2017). *Sensitivity Analysis in Observational Research* |
| DoWhy | [py-why/dowhy](https://github.com/py-why/dowhy) |
