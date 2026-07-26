# Causality Analyzer v1.0.0

> The most complete causal AI library for TypeScript — modular packages for anomaly detection, causal discovery, root cause analysis, effect estimation, counterfactual reasoning, and visualization. Enterprise-grade security, CI-verified quality, DoWhy cross-validated correctness.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![CI](https://github.com/AgentiX-E/causality-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/AgentiX-E/causality-analyzer/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-99.8%25_core_%7C_95.3%25_pipeline-brightgreen)](https://agentix-e.github.io/causality-analyzer/coverage/)
[![Tests](https://img.shields.io/badge/tests-1182%20passed-brightgreen)](.)

## Why Causality Analyzer?

| Feature | DoWhy (Python) | causal-js (TS) | Causality Analyzer |
|---------|:---:|:---:|:---:|
| Causal Discovery | ✗ | 10 algos | **10 algos** (PC/FCI/GES/LiNGAM/NOTEARS/Grow-Shrink/KCI/RCD/CD-NOD/MVPC/GRaSP/TS-ICD) |
| Causal Inference | ✓ | ✗ | **Backdoor (5 variants)/IV/PS/DR/Frontdoor** |
| Root Cause Analysis | ✗ | ✗ | **4 RCA + CIRCA pipeline + OnlinePC streaming** |
| do-Calculus | ✓ | ✗ | **Full recursive ID algorithm + hedge criterion** |
| SCM + Counterfactuals | ✓ | ✗ | **ANM/PN + auto-assign + abduction framework** |
| Bayesian Networks | ✗ | ✗ | **5 inference engines + Dirichlet learning** |
| Sensitivity + Refutation | ✓ | ✗ | **E-value + partial R² + 5 refuters (Bootstrap/Placebo/DataSubset/RandomCommonCause/DummyOutcome)** |
| Enterprise Security | ✗ | ✗ | **mTLS + AES-256-GCM + audit trail (SHA-256)** |
| Web Components | ✗ | ✗ | **Lit 3 + Canvas2D + uPlot + ARIA** |
| HTTP API + Docker | ✗ | ✗ | **7 REST endpoints + compose (PostgreSQL + Neo4j)** |
| Streaming | ✗ | ✗ | **Sliding window OnlinePC + drift detection** |
| Uplift Modeling | ✗ | ✗ | **Qini curve + AUUC + uplift@k** |
| Model Serialization | ✗ | ✗ | **CausalGraph + SCM JSON** |
| TypeScript Native | ✗ | ✓ | **Strict mode (99% core / 95% pipeline) + DI + ISP** |
| CI Quality | Basic | None | **recommendedTypeChecked ESLint, 5 job pipeline, d-sep validated** |

## Quick Start

```bash
npm install @agentix-e/causality-analyzer-core @agentix-e/causality-analyzer-pipeline
```

### 5-Minute RCA

```typescript
import { CausalGraph, HeuristicPathRCA } from '@agentix-e/causality-analyzer-pipeline';
import { Matrix } from 'ml-matrix';

const graph = new CausalGraph(['Memory', 'CPU', 'Latency']);
graph.addEdge('Memory', 'CPU');
graph.addEdge('CPU', 'Latency');

const data = new Matrix(100, 3); // your metrics
const rca = new HeuristicPathRCA();
rca.train(graph, new Set(['CPU', 'Latency']), data);
const result = rca.findRootCauses(['CPU', 'Latency']);

console.log(result.rootCauses[0].name);  // "Memory"
```

### 5-Minute RCA

```typescript
import { pcAlgorithm, gesAlgorithm, notearsAlgorithm, directLiNGAM, fciAlgorithm, kciTest } from '@agentix-e/causality-analyzer-pipeline';
import { Matrix } from 'ml-matrix';

// PC algorithm (constraint-based)
const { graph } = pcAlgorithm(data, ['X', 'Y', 'Z']);

// NOTEARS (neural/continuous optimization)
const { graph: dag } = notearsAlgorithm(rawData, ['X', 'Y', 'Z']);

// GES (score-based)
const dag2 = gesAlgorithm(data, ['X', 'Y', 'Z']);

// LiNGAM (non-Gaussian ICA)
const { graph: dag3 } = directLiNGAM(data, ['X', 'Y', 'Z']);
```

### Effect Estimation

```typescript
import { adjustBackdoor, findBackdoorSet } from '@agentix-e/causality-analyzer-pipeline';

const adj = findBackdoorSet(graph, 'Treatment', 'Outcome'); // {Confounder}
const { ate, se } = adjustBackdoor(graph, 'Treatment', 'Outcome', data, nodeIndex);
console.log(`ATE = ${ate.toFixed(3)} ± ${(se * 1.96).toFixed(3)}`);
```

### Sensitivity Analysis

```typescript
import { computeEValue, computePartialR2 } from '@agentix-e/causality-analyzer-pipeline';

const eValue = computeEValue(ate, se);        // "How strong must an unmeasured confounder be?"
const r2 = computePartialR2(ate, se, n);       // "How much variance would it explain?"
```

### HTTP API Server

```bash
npx causal-analyzer serve --port 3000
# GET  /health /ready /live /metrics
# POST /discover /analyze /estimate
```

### Docker

```bash
docker compose up -d  # pipeline + PostgreSQL + Neo4j
```

## Feature Map

**Causal Discovery**: PC (stable), FCI (R1-R10), **BOSS** (permutation+GST, NeurIPS 2023), **GFCI** (hybrid PAG), **RFCI** (fast PAG), GES (BIC), LiNGAM (non-Gaussian), NOTEARS (neural+L-BFGS), Grow-Shrink, KCI (kernel), RCD (hybrid BIC), CD-NOD (domain shifts), MVPC (missing values), GRaSP (L1-regularized), TS-ICD (time-series), targeted discovery, OnlinePC (streaming)

**Root Cause Analysis**: HeuristicPathRCA, RandomWalkRCA, HTRCA, FPGrowthRCA, CIRCA pipeline, Shapley attribution, FusionAnalyzer (weighted/nested/voting)

**Causal Inference**: Backdoor adjustment (5 variants: minimal/maximal/efficient/exhaustive/mincost), Frontdoor, IV/2SLS, Propensity Score (IRLS), PS Matching, Doubly Robust (O(n) performance), CATE, IPW, Mediation (Baron-Kenny)

**Sensitivity + Refutation**: E-value (Cohen's d conversion), Partial R² (Cinelli & Hazlett 2020), Robustness value, Bootstrap, Placebo treatment, Data subset, Random Common Cause, Dummy Outcome refutation

**do-Calculus**: Full recursive ID Algorithm (Shpitser & Pearl 2006), c-component decomposition, hedge criterion, Pearl's 3 rules

**SCM + GCM**: Additive noise, PostNonlinear, auto-assign mechanisms, counterfactuals, Shapley RCA, mechanism change detection, graph falsification

**Bayesian Networks**: Variable Elimination, Junction Tree, Loopy BP, Likelihood Weighting, Gibbs Sampling, online Dirichlet learning, brute-force oracle

**Uplift Modeling**: Qini curve, AUUC (normalized), uplift@k, model comparison

**Stability + Robustness**: Stability Selection (bootstrap+edge threshold), StARS (auto regularization selection)

**Infrastructure**: Audit trail (SHA-256), AES-256-GCM encryption, Prometheus metrics, Rate limiter, mTLS, L-BFGS/Adam optimizers, structured error hierarchy

**CI Tests**: Fisher Z, Chi-Square, G-Square + Kernel CI (KCI)

**Streaming + Drift**: OnlinePC sliding-window discovery, Welford incremental covariance, stability scoring, graph drift detection (SHD-based)

**Visualization**: Canvas2D causal DAG (directed edges + arrowheads), uPlot time series, Lit 3 Web Components, screen-reader ARIA support, keyboard navigation

## Packages

| Package | Description |
|---------|-------------|
| `@agentix-e/causality-analyzer-core` | Types, interfaces, math, ColumnarTable, plugin registry, L-BFGS/Adam optimizers |
| `@agentix-e/causality-analyzer-pipeline` | Detection, discovery, RCA, inference, GCM, visualization data, HTTP server |
| `@agentix-e/causality-analyzer-storage-embed` | SQLite + OverGraph embedded stores |
| `@agentix-e/causality-analyzer-storage-remote` | PostgreSQL + Neo4j with mTLS |
| `@agentix-e/causality-analyzer-visual` | Lit 3 Web Components for causal graphs + time series |

## Development

```bash
pnpm install
pnpm run --filter @agentix-e/causality-analyzer-core build
pnpm -r test       # 1278 tests (254 core + 1024 pipeline)
pnpm -r typecheck  # strict TypeScript (exactOptionalPropertyTypes, noUncheckedIndexedAccess)
pnpm -r lint       # ESLint recommendedTypeChecked
```

## References

| Algorithm | Paper |
|-----------|-------|
| PC | Spirtes, Glymour & Scheines (2000). *Causation, Prediction, and Search* |
| FCI | Zhang (2008). *On the completeness of orientation rules for causal discovery* |
| NOTEARS | Zheng et al. (NeurIPS 2018). *DAGs with NOTEARS* |
| ID Algorithm | Shpitser & Pearl (2006). *Identification of Joint Interventional Distributions* |
| CIRCA | Li et al. (KDD 2022). *Causal Inference-Based Root Cause Analysis* |
| SPOT | Siffer et al. (KDD 2017). *Anomaly Detection in Streams with Extreme Value Theory* |
| GES | Chickering (2002). *Optimal Structure Identification With Greedy Search* |
| LiNGAM | Shimizu et al. (JMLR 2006). *A Linear Non-Gaussian Acyclic Model* |
| GRaSP | Lam et al. (UAI 2022). *Greedy Relaxations of the Sparsity Penalty* |
| TS-ICD | Rohekar et al. (ICML 2023). *Time-Series Iterative Causal Discovery* |
| MVPC | Tu et al. (2019). *Causal Discovery in the Presence of Missing Data* |
| RCD | Ma et al. (2022). *Reinforced Causal Discovery* |
| CD-NOD | Zhang et al. (2017). *Causal Discovery from Nonstationary Data* |
| E-value | VanderWeele & Ding (2017). *Sensitivity Analysis in Observational Research* |
| DoWhy | [py-why/dowhy](https://github.com/py-why/dowhy) |
| causal-js | [Kanaries/causal-js](https://github.com/Kanaries/causal-js) |
| Intel Causal Lab | [IntelLabs/causality-lab](https://github.com/IntelLabs/causality-lab) |

## License

MIT
