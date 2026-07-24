# @agentix-e/causality-analyzer-pipeline

> Complete causal AI pipeline — anomaly detection, causal discovery, root cause analysis, effect estimation, counterfactual reasoning, and model evaluation.

[![npm](https://img.shields.io/badge/version-1.0.0-blue)](https://www.npmjs.com/package/@agentix-e/causality-analyzer-pipeline)

## Overview

`@agentix-e/causality-analyzer-pipeline` is the causal reasoning engine. It implements the full stack from raw metric ingestion to counterfactual "what-if" analysis, including academic-grade sensitivity testing and graph validation.

### Architecture

```
Raw Data → Standardize → Detect Anomalies
                                ↓
              Causal Discovery (PC / FCI / Targeted)
                                ↓
              Root Cause Analysis (Bayesian / HT / RandomWalk / CIRCA)
                                ↓
              Effect Estimation (Backdoor / Frontdoor / IV / PS / DR)
                                ↓
              Sensitivity & Refutation (E-value / pR² / Bootstrap)
                                ↓
              Counterfactual Reasoning (SCM → Abduction → Action → Prediction)
                                ↓
              Model Evaluation (R² / MSE / Shapley RCA / Distribution Change)
```

## Installation

```bash
npm install @agentix-e/causality-analyzer-pipeline
npm install @agentix-e/causality-analyzer-core  # peer dependency
```

## Quick Start

### 1. Anomaly Detection

```typescript
import { StatsDetector, SpectralResidualDetector, VotingDetector } from '@agentix-e/causality-analyzer-pipeline';

// Z-score detector
const detector = new StatsDetector({ method: 'zscore' });
detector.train(normalData);
const result = detector.update([5.2, 8.1]);  // anomalous!

// Ensemble voting
const ensemble = new VotingDetector({
  detectors: [statsDetector, srDetector],
  strategy: 'majority',
});
const voted = ensemble.detect(dataPoints);
```

### 2. Causal Discovery

```typescript
import { Matrix } from 'ml-matrix';
import { pcAlgorithm, fciAlgorithm, targetedDiscovery } from '@agentix-e/causality-analyzer-pipeline';

// PC algorithm (no latent confounders)
const { graph } = pcAlgorithm(data, ['CPU', 'Memory', 'Latency']);

// FCI algorithm (with latent confounders)
const { pagEdges } = fciAlgorithm(data, nodeNames);

// Targeted: only find parents of 'Latency'
const parents = targetedDiscovery(data, ['Latency'], nodeNames);
```

### 3. Root Cause Analysis

```typescript
import { CausalGraph, BayesianRCA, HTRCA, CIRCAPipeline } from '@agentix-e/causality-analyzer-pipeline';

// Bayesian Network RCA
const rca = new BayesianRCA();
rca.train(graph, anomalousNodes, data);
const result = rca.findRootCauses(['CPU', 'Latency']);

// Hypothesis Testing RCA (regression residuals)
const ht = new HTRCA();
ht.train(graph, data);
const htResult = ht.findRootCauses(['CPU', 'Latency'], data);

// CIRCA Pipeline (KDD 2022)
const circa = new CIRCAPipeline(graph);
const circaResult = circa.analyze(anomalyData, ['CPU', 'Latency']);
```

### 4. Causal Effect Estimation

```typescript
import {
  adjustBackdoor, estimateIV, estimatePSMatching, estimateDoublyRobust
} from '@agentix-e/causality-analyzer-pipeline';

// Backdoor adjustment
const { ate, se } = adjustBackdoor(graph, 'Treatment', 'Outcome', data, nodeIndex);

// Instrumental Variables (2SLS)
const ivResult = estimateIV(data, treatmentIdx, outcomeIdx, ivIdx);

// Propensity Score Matching
const psmResult = estimatePSMatching(data, treatmentIdx, outcomeIdx, [confounderIdx]);

// Doubly Robust
const drResult = estimateDoublyRobust(data, treatmentIdx, outcomeIdx, [confounderIdx]);
```

### 5. Sensitivity Analysis

```typescript
import { eValueSensitivity, robustnessValue } from '@agentix-e/causality-analyzer-pipeline';

const { eValue, interpretation } = eValueSensitivity(0.8);
// "E-value=4.22: strong robustness — only very strong unmeasured confounding..."

const { rv } = robustnessValue(0.8, 0.1, 1000);
// "RV=3.15: ROBUST — causal conclusion is well-supported"
```

### 6. Counterfactual Reasoning

```typescript
import { CausalGraph, StructuralCausalModel } from '@agentix-e/causality-analyzer-pipeline';

const scm = new StructuralCausalModel(graph);
scm.train(data);

// What would latency be if we doubled memory?
const noise = scm.abduct({ Memory: 0.5, CPU: 0.8, Latency: 120 });
const cf = scm.counterfactual(noise, { Memory: 1.0 });

// Shapley-based anomaly attribution
import { shapleyAttribute } from '@agentix-e/causality-analyzer-pipeline';
const shapleyRCA = shapleyAttribute(scm, anomalousObservation, 5);
```

## API Reference

📚 Full TypeDoc API: `pnpm docs` from the monorepo root.

### Module Index

| Module | Key Exports |
|--------|------------|
| `data/standardizer` | `standardize`, `discretize`, `extractWindows`, `imputeMean` |
| `detect/stats-detector` | `StatsDetector` (zscore/mad/iqr) |
| `detect/spectral-residual` | `SpectralResidualDetector` (FFT-based) |
| `detect/spot` | `SPOTDetector`, `DSPOTDetector` (extreme value) |
| `detect/voting-detector` | `VotingDetector` (majority/max/weighted) |
| `graph/causal-graph` | `CausalGraph` (DAG/PDAG/CPDAG) |
| `graph/pc` | `pcAlgorithm`, `fisherZTest` |
| `graph/advanced-discovery` | `fciAlgorithm`, `growShrink`, `targetedDiscovery` |
| `analyze/rca` | `BayesianRCA`, `RandomWalkRCA`, `HTRCA`, `FPGrowthRCA` |
| `analyze/circa` | `RHTScorer`, `DAScorer`, `CIRCAPipeline` |
| `infer/causal-inference` | `CausalAnalysis`, `identifyBackdoor`, `identifyFrontdoor`, `refutePlaceboTreatment`, `refuteBootstrap` |
| `infer/effect-estimation` | `adjustBackdoor`, `estimateFrontdoor`, `estimateIV`, `estimatePSMatching`, `estimateDoublyRobust` |
| `infer/sensitivity` | `eValueSensitivity`, `partialRSensitivity`, `robustnessValue` |
| `infer/do-calculus` | `identifyByDoCalculus` (3 rules + ID algorithm) |
| `infer/mediation` | `naturalDirectEffect`, `arrowStrength` |
| `infer/cate-fairness` | `estimateCATE`, `estimateIPW`, `checkFairness` |
| `infer/bootstrap-ci` | `bootstrapATE`, `bootstrapATEParallel`, `parallelBootstrap` |
| `gcm/structural-causal-model` | `StructuralCausalModel`, `cateToRCA` |
| `gcm/model-evaluation` | `evaluateMechanismR2`, `evaluateMSE`, `shapleyAttribute`, `bootstrapRCA` |
| `gcm/nonlinear-mechanisms` | `PostNonlinearMechanism`, `fitLogisticPNL`, `autoAssignMechanisms`, `parentRelevance` |
| `gcm/distribution-change` | `detectMechanismChanges`, `distributionChangeRobust`, `changeAttributionCI` |
| `gcm/graph-falsification` | `falsifyGraph`, `lmcFalsification` |
| `viz/viz-data` | `buildGraphVizData`, `buildTimeseriesVizData`, `buildRankingVizData` |
| `viz/fusion` | `FusionAnalyzer` (metric + trace + log) |

## Deterministic Reproducibility

All stochastic algorithms accept an optional `seed` parameter for reproducible results:

```typescript
// With seed → deterministic
const result = shapleyAttribute(scm, obs, 5, seed);
const ci = bootstrapRCA(scm, obs, 200, 0.05, seed);

// Without seed → non-deterministic (uses Math.random)
const result2 = shapleyAttribute(scm, obs, 5);
```

## License

MIT
