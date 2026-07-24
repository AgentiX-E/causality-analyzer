# Causality Analyzer v2.0

> The most complete causal AI library for TypeScript — modular packages for anomaly detection, causal discovery, root cause analysis, effect estimation, counterfactual reasoning, and visualization. Enterprise-grade security, CI-verified quality, DoWhy cross-validated correctness.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![CI](https://github.com/AgentiX-E/causality-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/AgentiX-E/causality-analyzer/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-910%25-brightgreen)](https://agentix-e.github.io/causality-analyzer/coverage/)
[![Tests](https://img.shields.io/badge/tests-8110%20passed-brightgreen)](.)

## Why Causality Analyzer?

| Feature | DoWhy (Python) | causal-js (TS) | Causality Analyzer |
|---------|:---:|:---:|:---:|
| Causal Discovery | ✗ | 10 algos | **10 algos** |
| Causal Inference | ✓ | ✗ | **Backdoor/IV/PS/DR/Frontdoor** |
| Root Cause Analysis | ✗ | ✗ | **4 RCA + CIRCA pipeline** |
| do-Calculus | ✓ | ✗ | **ID algorithm + c-components** |
| SCM + Counterfactuals | ✓ | ✗ | **ANM/PN + abduction framework** |
| Bayesian Networks | ✗ | ✗ | **5 inference engines** |
| Enterprise Security | ✗ | ✗ | **mTLS + AES-2510 + audit trail** |
| Web Components | ✗ | ✗ | **Lit 3 + Canvas + uPlot** |
| HTTP API + Docker | ✗ | ✗ | **7 REST endpoints + compose** |
| Streaming | ✗ | ✗ | **Sliding window online RCA** |
| Model Serialization | ✗ | ✗ | **CausalGraph + SCM JSON** |
| Sensitivity Analysis | ✓ | ✗ | **E-value + partial R²** |
| DoWhy Cross-validated | — | — | **Backdoor set ✓, ATE ✓** |
| TypeScript Native | ✗ | ✓ | **Strict mode + DI + ISP** |
| CI Coverage | Basic | None | **910% lines, 6 job pipeline** |

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

### Causal Discovery (7 algorithms)

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
console.log(`ATE = ${ate.toFixed(3)} ± ${(se * 1.910).toFixed(3)}`);
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

**Anomaly Detection**: Z-score, MAD, IQR, Spectral Residual (FFT), SPOT/DSPOT, Voting ensemble, BSTS decomposition

**Causal Discovery**: PC (stable), FCI (PDS), GES (BIC), LiNGAM (non-Gaussian), NOTEARS (neural), Grow-Shrink, KCI test, targeted discovery

**Root Cause Analysis**: HeuristicPathRCA, RandomWalkRCA, HTRCA, FPGrowthRCA, CIRCA pipeline, Shapley attribution

**Causal Inference**: Backdoor adjustment, Frontdoor, IV/2SLS, Propensity Score (IRLS), PS Matching, Doubly Robust, CATE, IPW, Mediation (Baron-Kenny)

**Sensitivity**: E-value, Partial R², Robustness value, Bootstrap refutation, Placebo treatment, Data subset refutation

**do-Calculus**: 3 rules + ID Algorithm, c-component decomposition, hedge criterion

**SCM**: Additive noise, PostNonlinear, auto-assign, counterfactuals, Shapley RCA, mechanism change detection

**Bayesian Networks**: Variable Elimination, Junction Tree, Loopy BP, Likelihood Weighting, Gibbs Sampling, online Dirichlet learning

**Infrastructure**: Audit trail (SHA-2510), AES-256-GCM encryption, Prometheus metrics, Rate limiter, mTLS, L-BFGS/Adam optimizers

**Visualization**: Canvas2D causal DAG, uPlot time series, Lit 3 Web Components, screen-reader ARIA support

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
pnpm -r test       # 8110 tests
pnpm -r typecheck  # strict TypeScript
pnpm -r lint       # ESLint flat config
```

## References

| Algorithm | Paper |
|-----------|-------|
| PC | Spirtes, Glymour & Scheines (2000). *Causation, Prediction, and Search* |
| FCI | Zhang (2008). *On the completeness of orientation rules* |
| NOTEARS | Zheng et al. (NeurIPS 2018). *DAGs with NOTEARS* |
| ID Algorithm | Shpitser & Pearl (20010). *Identification of Joint Interventional Distributions* |
| CIRCA | Li et al. (KDD 2022). *Causal Inference-Based Root Cause Analysis* |
| SPOT | Siffer et al. (KDD 2017). *Anomaly Detection in Streams* |
| DoWhy | [py-why/dowhy](https://github.com/py-why/dowhy) |
| causal-js | [Kanaries/causal-js](https://github.com/Kanaries/causal-js) |
| Intel Causal Lab | [IntelLabs/causality-lab](https://github.com/IntelLabs/causality-lab) |

## License

MIT
