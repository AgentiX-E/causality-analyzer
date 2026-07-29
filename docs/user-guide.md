# Causality Analyzer — Detailed User Guide

> **Author:** Lambertyan
> **Version:** 1.1.0
> **Date:** 2026-07-29
> **License:** MIT

---

## Table of Contents

1. [Installation](#1-installation)
2. [Getting Started](#2-getting-started)
3. [Causal Discovery](#3-causal-discovery)
4. [Causal Inference](#4-causal-inference)
5. [Root Cause Analysis](#5-root-cause-analysis)
6. [Bayesian Networks](#6-bayesian-networks)
7. [Generative Causal Modeling](#7-generative-causal-modeling)
8. [Anomaly Detection](#8-anomaly-detection)
9. [Storage Configuration](#9-storage-configuration)
10. [HTTP API Server](#10-http-api-server)
11. [Visualization](#11-visualization)
12. [Advanced Topics](#12-advanced-topics)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Installation

### 1.1 Prerequisites

- **Node.js** >= 22 (for embedded storage: `node:sqlite` DatabaseSync)
- **npm** >= 10, **pnpm** >= 9, or **yarn** >= 4

### 1.2 Package Selection

Choose packages based on your deployment environment:

| Use Case | Packages | Command |
|----------|----------|---------|
| Server-side, embedded (minimum) | `core` + `pipeline` | `npm i @agentix-e/causality-analyzer-core @agentix-e/causality-analyzer-pipeline` |
| Server-side, embedded (with persistence) | add `storage-embed` | `npm i @agentix-e/causality-analyzer-storage-embed` |
| Server-side, enterprise | add `storage-remote` | `npm i @agentix-e/causality-analyzer-storage-remote` |
| Browser | `core` + `pipeline` + `storage-browser` | `npm i @agentix-e/causality-analyzer-storage-browser` |
| Visualization | add `visual` | `npm i @agentix-e/causality-analyzer-visual` |
| Full stack (all 6) | all packages | `npm i @agentix-e/causality-analyzer-{core,pipeline,storage-embed,storage-browser,storage-remote,visual}` |

### 1.3 Verifying Installation

```typescript
import { CausalGraph, pcAlgorithm } from '@agentix-e/causality-analyzer-pipeline';
import { ColumnarTable } from '@agentix-e/causality-analyzer-core';

console.log('Causality Analyzer installed successfully');
```

---

## 2. Getting Started

### 2.1 5-Minute Root Cause Analysis

This example demonstrates the full RCA workflow: define a causal graph, detect anomalies, and identify root causes.

```typescript
import {
  CausalGraph, HeuristicPathRCA, CIRCAPipeline,
  SpectralResidualDetector,
} from '@agentix-e/causality-analyzer-pipeline';
import { Matrix } from 'ml-matrix';

// Step 1: Define the causal graph
// This can come from domain knowledge or causal discovery (see Section 3)
const graph = new CausalGraph([
  'IngressLatency', 'AuthService', 'DatabasePool', 'CacheHitRate', 'P99Latency',
]);

graph.addEdge('IngressLatency', 'AuthService');
graph.addEdge('AuthService', 'DatabasePool');
graph.addEdge('DatabasePool', 'P99Latency');
graph.addEdge('CacheHitRate', 'DatabasePool');
graph.addEdge('CacheHitRate', 'P99Latency');

// Step 2: Load historical normal-behavior metrics
const normalData = new Matrix([
  [5.2, 12.1, 45, 0.87, 120],
  [4.8, 11.9, 42, 0.85, 115],
  // ... more normal rows
]);

// Step 3: Train RCA model
const rca = new HeuristicPathRCA();
rca.train(graph, new Set(['P99Latency', 'AuthService']), normalData);

// Step 4: Analyze anomalous window data
const anomalousData = new Matrix([
  [25.0, 150, 98, 0.23, 850],   // High latency, low cache hit rate
]);
const result = rca.findRootCauses(['P99Latency', 'AuthService']);

console.log('Root Causes (ranked):');
for (const rc of result.rootCauses) {
  console.log(`  ${rc.rank}. ${rc.name} (score: ${rc.score.toFixed(3)})`);
}
// Output:
//   1. CacheHitRate (score: 0.860)
//   2. IngressLatency (score: 0.540)
```

### 2.2 5-Minute Causal Discovery

Discover causal structure from data when you lack domain knowledge:

```typescript
import {
  pcAlgorithm, gesAlgorithm, notearsAlgorithm, directLiNGAM,
} from '@agentix-e/causality-analyzer-pipeline';
import { Matrix } from 'ml-matrix';

const data = new Matrix([
  [1.0, 0.5, 2.1],
  [1.2, 0.6, 2.4],
  [0.9, 0.5, 1.9],
  [1.1, 0.7, 2.3],
  // ... more rows (at least 100 recommended)
]);
const varNames = ['Revenue', 'AdSpend', 'Visitors'];

// Constraint-based: PC (best for linear Gaussian)
const { graph: pcGraph } = pcAlgorithm(data, varNames, { alpha: 0.05 });
console.log('PC discovered edges:', pcGraph.edges.length);

// Score-based: GES (best for BIC-optimal DAG)
const gesGraph = gesAlgorithm(data, varNames);

// Continuous optimization: NOTEARS (best for large n)
const { graph: ntGraph } = notearsAlgorithm(data, varNames, {
  lambda1: 0.1,
  maxIter: 100,
});

// Non-Gaussian: LiNGAM (best for non-normal data)
const { graph: lingamGraph } = directLiNGAM(data, varNames);

// Compare results
for (const edge of pcGraph.edges) {
  console.log(`${edge.source} -> ${edge.target} (weight: ${edge.weight})`);
}
```

### 2.3 5-Minute Effect Estimation

Estimate causal effects and validate with sensitivity analysis:

```typescript
import {
  CausalGraph, findBackdoorSet, adjustBackdoor,
  computeEValue, computePartialR2,
  refuteBootstrap, refutePlaceboTreatment,
} from '@agentix-e/causality-analyzer-pipeline';
import { Matrix } from 'ml-matrix';

// Step 1: Build causal DAG from domain knowledge
const graph = new CausalGraph([
  'Age', 'Education', 'Income', 'Treatment', 'Outcome',
]);
graph.addEdge('Age', 'Treatment');
graph.addEdge('Age', 'Outcome');
graph.addEdge('Education', 'Treatment');
graph.addEdge('Education', 'Outcome');
graph.addEdge('Income', 'Treatment');
graph.addEdge('Income', 'Outcome');
graph.addEdge('Treatment', 'Outcome');

// Step 2: Load data
const data = new Matrix(/* your data matrix */);
const nodeIndex = new Map([
  ['Age', 0], ['Education', 1], ['Income', 2],
  ['Treatment', 3], ['Outcome', 4],
]);

// Step 3: Find adjustment set and estimate ATE
const adjSet = findBackdoorSet(graph, 'Treatment', 'Outcome');
console.log('Adjust for:', adjSet);  // { Age, Education, Income }

const { ate, se, ciLower, ciUpper } = adjustBackdoor(
  graph, 'Treatment', 'Outcome', data, nodeIndex,
);

console.log(`ATE = ${ate.toFixed(3)} (SE = ${se.toFixed(3)})`);
console.log(`95% CI = [${ciLower.toFixed(3)}, ${ciUpper.toFixed(3)}]`);

// Step 4: Sensitivity analysis
const eValue = computeEValue(ate, se);
console.log(`E-value = ${eValue.toFixed(2)}`);
// E-value = 2.5 means: an unmeasured confounder would need
// risk ratio >= 2.5 with both treatment and outcome to explain away

const r2 = computePartialR2(ate, se, data.rows);
console.log(`Partial R^2 = ${r2.toFixed(3)}`);

// Step 5: Refutation tests
const bootstrapResult = refuteBootstrap(
  graph, 'Treatment', 'Outcome', data, nodeIndex, { nBootstraps: 200 }
);
console.log(`Bootstrap p-value: ${bootstrapResult.pValue}`);

const placeboResult = refutePlaceboTreatment(
  graph, 'Treatment', 'Outcome', data, nodeIndex,
);
console.log(`Placebo test passed: ${placeboResult.effectCloseToZero}`);
```

---

## 3. Causal Discovery

### 3.1 Algorithm Selection Guide

Choosing the right algorithm depends on your data characteristics:

| Data Characteristic | Recommended Algorithm(s) |
|---------------------|-------------------------|
| Linear, Gaussian, no latent | **PC** (fastest), GES (most accurate) |
| Linear, Gaussian, possible latent | **FCI**, GFCI (handles hidden confounders) |
| Nonlinear relationships | **KCI** (kernel test), NOTEARS (nonlinear SEM) |
| Non-Gaussian distributions | **DirectLiNGAM** (asymmetry-based), ICA-LiNGAM |
| Large sample size (>1000) | **NOTEARS**, **DAGMA**, GOLEM (continuous optimization) |
| Small sample size (<200) | **PC** with Fisher Z, **GES** (well-calibrated) |
| Time series data | **PCMCI**, tsFCI, VAR-LiNGAM |
| Mixed data types | **MVPC** (handles missing values) |
| Domain shifts/nonstationary | **CD-NOD** |
| Unknown best algorithm | Run multiple and **fuse** results, or use **Stability Selection** |

### 3.2 Constraint-Based Methods

#### PC Algorithm (Stable PC)

The most widely-used constraint-based algorithm. PC works by testing conditional independence relationships to construct a skeleton, then orienting edges using Meek's rules.

```typescript
import { pcAlgorithm } from '@agentix-e/causality-analyzer-pipeline';
import { Matrix } from 'ml-matrix';

const { graph } = pcAlgorithm(data, varNames, {
  alpha: 0.05,          // Significance level (default)
  correction: 'fdr',    // Multiple testing correction: 'bonferroni' | 'fdr' | 'none'
  ciTest: 'fisher-z',   // CI test: 'fisher-z' | 'chi-square' | 'g-square'
  maxConditioning: 3,   // Maximum conditioning set size (limits complexity)
  verbose: false,       // Enable progress logging
});

// Access results
console.log('Nodes:', graph.nodes);
console.log('Edges:', graph.edges.length);
console.log('Is acyclic:', graph.isAcyclic());
```

**When to use:** Linear Gaussian data, moderate dimensionality (d < 50), uncertain about graph structure.

**Performance tip:** The Fisher Z test uses a 50,000-entry LRU cache. For dense graphs where conditioning sets are large, enable `correction: 'bonferroni'` to reduce false positives.

#### FCI Algorithm (Handling Latent Confounders)

FCI extends PC to handle unmeasured common causes (latent confounders). It outputs a **Partial Ancestral Graph (PAG)** instead of a DAG.

```typescript
import { fciAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const result = fciAlgorithm(data, varNames, {
  alpha: 0.05,
  depth: -1,  // -1 = unlimited conditioning set depth
});

// FCI produces a PAG with special edge types:
// A o-> B: A is potentially an ancestor of B
// A <-> B: A and B have a latent common cause
// A o-o B: unknown relationship
console.log('Latent confounder edges:', result.latentEdges);
```

**When to use:** When you suspect unmeasured confounding (common in observational studies, economics, epidemiology).

**Variants:**
- **GFCI** (Gaussian FCI): Adds BIC scoring to FCI for better performance on Gaussian data
- **RFCI** (Really Fast FCI): Faster variant that avoids estimating structure among non-ancestors

```typescript
import { gfciAlgorithm, rfciAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const gfciResult = gfciAlgorithm(data, varNames);
const rfciResult = rfciAlgorithm(data, varNames);
```

#### Grow-Shrink Algorithm

Markov blanket-based feature selection for causal discovery:

```typescript
import { growShrink } from '@agentix-e/causality-analyzer-pipeline';

const blanket = growShrink(data, varNames, 'TargetVariable', { alpha: 0.05 });
console.log('Markov blanket:', blanket);
```

#### Kernel CI (KCI) Test

Nonlinear conditional independence testing using kernel methods:

```typescript
import { kciTest } from '@agentix-e/causality-analyzer-pipeline';

const isIndependent = kciTest(data, 'X', 'Y', ['Z1', 'Z2'], {
  kernelX: 'gaussian',
  kernelY: 'gaussian',
  kernelZ: 'gaussian',
  epsilon: 1e-3,
});
console.log('X independent of Y given Z1,Z2:', isIndependent);
```

**When to use:** When relationships are nonlinear and Fisher Z/Chi-square tests are inappropriate.

### 3.3 Score-Based Methods

#### GES Algorithm

Greedy Equivalence Search performs forward phase (adding edges that improve BIC score) followed by backward phase (removing edges). Operates in CPDAG space.

```typescript
import { gesAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const graph = gesAlgorithm(data, varNames, {
  score: 'bic',          // 'bic' | 'bdeu' (for discrete data)
  maxDegree: 0,          // 0 = unlimited
  useCovarianceBIC: true, // Use covariance-matrix BIC (faster for large d)
});
```

**When to use:** Linear Gaussian data, when you want the BIC-optimal Markov equivalence class.

**Performance:** True CPDAG-space search with subset enumeration validates each edge addition/removal via clique condition checks and semi-directed path validity tests.

#### BOSS Algorithm (NeurIPS 2023)

Best Order Score Search — enumerates permutations and applies greedy sparse transforms:

```typescript
import { bossAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const graph = bossAlgorithm(data, varNames, {
  maxIterations: 1000,
  useGST: true,  // Greedy Sparse Transform for scoring
});
```

**When to use:** When GES gets stuck in local optima; BOSS explores more of the order space.

#### GRaSP Algorithm

Greedy Relaxation of the Sparsest Permutation — L1-regularized scoring:

```typescript
import { graspAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const graph = graspAlgorithm(data, varNames, {
  lambda: 0.1,  // L1 regularization strength
});
```

**When to use:** When the true graph is believed to be sparse.

### 3.4 Continuous Optimization Methods

#### NOTEARS Algorithm (NeurIPS 2018)

Reformulates DAG learning as a continuous optimization problem using the trace exponential acyclicity constraint:

```typescript
import { notearsAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const { graph, W } = notearsAlgorithm(data, varNames, {
  lambda1: 0.1,     // L1 regularization for sparsity
  lossType: 'l2',   // 'l2' | 'logistic' (for binary data)
  maxIter: 100,     // Maximum iterations
  hTol: 1e-8,       // Acyclicity tolerance
  rhoMax: 1e16,     // Augmented Lagrangian max penalty
  verbose: false,
});

console.log('Weight matrix norm:', W.norm());
```

**How it works:** NOTEARS minimizes `F(W) = loss(W) + lambda1 * ||W||_1` subject to `h(W) = trace(exp(W * W)) - d = 0`. The constraint is enforced via augmented Lagrangian with L-BFGS as the inner optimizer.

**When to use:** Large sample sizes, when you want the globally optimal DAG (not just Markov equivalence class).

**Performance:** The matrix exponential is computed via 15-term Taylor expansion + repeated squaring (scaling-and-squaring method). For node counts d > 30, NOTEARS becomes computationally expensive.

#### DAGMA Algorithm (NeurIPS 2022)

Uses M-matrix-based acyclicity constraint instead of trace exponential:

```typescript
import { dagmaAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const { graph } = dagmaAlgorithm(data, varNames, {
  lambda1: 0.02,
  lossType: 'l2',
  T: 4,            // Number of outer iterations
  muInit: 1e-3,    // Initial ALM penalty
  verbose: false,
});

// DAGMA is generally more stable and faster than NOTEARS for d > 15
```

**Key difference from NOTEARS:** DAGMA minimizes `-log det(sI - W*W) + d*log(s)` as the acyclicity penalty, which has better gradient properties. Uses Adam optimizer with 3-stage schedule.

#### GOLEM Algorithm (NeurIPS 2020)

Unrolls NOTEARS with a likelihood-based objective:

```typescript
import { golemAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const { graph } = golemAlgorithm(data, varNames, {
  lambda1: 0.02,
  lambda2: 5,
  equalVar: true,   // Assume equal noise variance
  numIter: 1e5,
});
```

**When to use:** When you want NOTEARS-like performance with better statistical properties (GOLEM directly maximizes data likelihood rather than using squared loss).

### 3.5 Non-Gaussian / Functional Models

#### DirectLiNGAM

Causal discovery by exploiting non-Gaussianity of data distributions:

```typescript
import { directLiNGAM } from '@agentix-e/causality-analyzer-pipeline';

const { graph, causalOrder } = directLiNGAM(data, varNames, {
  prune: true,       // BIC pruning of adjacency
  measure: 'pwling', // Dependence measure
});

console.log('Causal order:', causalOrder);
// Output: ['AdSpend', 'Visitors', 'Revenue']
// Interpretation: AdSpend → Visitors → Revenue

// The causal order is the unique strength of LiNGAM
// — it identifies the full causal ordering, not just the Markov equivalence class.
```

**When to use:** Data with non-Gaussian distributions (common in economics, marketing, social science). LiNGAM uniquely identifies the full causal DAG (not just the equivalence class) under non-Gaussianity.

**Performance notes:**
- For n <= 800: Full pairwise Kendall's tau dependence measure
- For n > 800: Adaptive sampling strategy
- Two-stage OLS for stable adjacency estimation
- BIC-optimal Lasso grid search for pruning

**Variants:**

```typescript
import { icaLiNGAM, varLingam } from '@agentix-e/causality-analyzer-pipeline';

// ICA-LiNGAM: Uses Independent Component Analysis for causal ordering
const icaResult = icaLiNGAM(data, varNames);

// VAR-LiNGAM: Vector Autoregressive LiNGAM for time series
const varResult = varLingam(timeSeriesData, varNames, { lag: 2 });
console.log('Lagged causal effects:', varResult.laggedEffects);
```

#### FASK

Fast Adjacency Skewness — uses distribution asymmetry for causal direction:

```typescript
import { faskAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const graph = faskAlgorithm(data, varNames, {
  alpha: 0.05,
  depth: -1,
});
```

### 3.6 Time Series Discovery

#### PCMCI Algorithm

Tigramite-style time series causal discovery with condition-selection:

```typescript
import { pcmciAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const result = pcmciAlgorithm(timeSeriesData, varNames, {
  tauMax: 5,         // Maximum time lag
  alpha: 0.05,       // Significance level
  ciTest: 'parcorr', // CI test for time series
});

// PCMCI identifies lagged causal links
for (const edge of result.edges) {
  console.log(
    `${edge.source}(t-${edge.lag}) -> ${edge.target}(t), ` +
    `strength: ${edge.strength}`
  );
}
```

**When to use:** Time series data with potential lagged and instantaneous causal effects.

#### tsFCI and TS-iCD

```typescript
import { tsFciAlgorithm, tsIcdAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

// tsFCI: FCI for time series with lagged links and latent confounders
const tsFciResult = tsFciAlgorithm(timeSeriesData, varNames, {
  maxLag: 5,
  alpha: 0.05,
});

// TS-iCD: Instantaneous Causal Discovery for time series
const tsIcdResult = tsIcdAlgorithm(timeSeriesData, varNames, {
  maxLag: 3,
});
```

### 3.7 Specialized Discovery Methods

#### CD-NOD (Nonstationary Data)

```typescript
import { cdnodAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const graph = cdnodAlgorithm(multiDomainData, varNames, {
  domainVar: 'Environment',  // Variable indicating domain/environment
  alpha: 0.05,
});
```

**When to use:** When data comes from multiple domains/environments with distribution shifts.

#### MVPC (Missing Values)

```typescript
import { mvpcAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const { graph } = mvpcAlgorithm(dataWithMissing, varNames, {
  alpha: 0.05,
  // Automatically handles NaN entries via test-wise deletion
});
```

**When to use:** Data with missing values — MVPC handles NaN entries natively.

#### CCD (Cyclic Causal Discovery)

```typescript
import { ccdAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const graph = ccdAlgorithm(data, varNames, {
  alpha: 0.05,
});
```

**When to use:** When feedback loops/cycles may exist in the system (e.g., economics, climate systems).

### 3.8 Robustness and Validation

#### Stability Selection

Validate discovered edges through bootstrap resampling:

```typescript
import { stabilitySelection } from '@agentix-e/causality-analyzer-pipeline';

const result = stabilitySelection(data, varNames, {
  algorithm: 'pc',     // Which algorithm to run per bootstrap
  bootstrap: 200,      // Number of bootstrap samples
  threshold: 0.6,      // Edge frequency threshold (0.6 = edge appears in 60% of bootstraps)
  alpha: 0.05,
});

console.log('Stable edges:');
for (const [edge, freq] of result.edgeStability) {
  if (freq >= result.threshold) {
    console.log(`  ${edge}: ${(freq * 100).toFixed(1)}%`);
  }
}
```

#### StARS (Stability Approach to Regularization Selection)

Automatically select the optimal regularization parameter:

```typescript
import { starsSelection } from '@agentix-e/causality-analyzer-pipeline';

const result = starsSelection(data, varNames, {
  algorithm: 'notears',
  lambdaGrid: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
  bootstrap: 50,
  beta: 0.05,  // Instability threshold
});

console.log('Optimal lambda:', result.optimalLambda);
console.log('Graph at optimal lambda:', result.graph);
```

#### Algorithm Fusion

Combine results from multiple discovery algorithms:

```typescript
import { pcAlgorithm, gesAlgorithm, notearsAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const results = await Promise.all([
  pcAlgorithm(data, varNames),
  gesAlgorithm(data, varNames),
  notearsAlgorithm(data, varNames),
]);

// Weighted fusion: edges appearing in more algorithms get higher confidence
const fused = fusionAnalyzer.fuseGraphs(
  results.map(r => r.graph),
  { method: 'voting', threshold: 2 }  // 2 out of 3 algorithms must agree
);
```

### 3.9 Streaming Discovery

#### OnlinePC

Real-time causal structure discovery from streaming data:

```typescript
import { OnlinePC } from '@agentix-e/causality-analyzer-pipeline';

const online = new OnlinePC({
  windowSize: 500,     // Sliding window size
  slideSize: 100,      // Slide amount per update
  alpha: 0.05,
});

// Subscribe to change events
online.onChange((event) => {
  console.log(`Causal drift at t=${event.timestamp}`);
  console.log(`  SHD delta: ${event.shdDelta}`);
  console.log(`  Added edges: ${event.addedEdges.length}`);
  console.log(`  Removed edges: ${event.removedEdges.length}`);
});

// Feed data in streaming fashion
for await (const batch of dataStream) {
  const state: StreamingGraphState = online.ingest(batch);
  console.log(`Current edges: ${state.graph.edges.length}`);
  console.log(`Stability score: ${state.stabilityScore}`);
}
```

#### Drift Detection

Detect causal drift between graph snapshots:

```typescript
import { detectCausalDrift } from '@agentix-e/causality-analyzer-pipeline';

const result = detectCausalDrift(graphAtTime1, graphAtTime2, {
  shdThreshold: 3,  // Minimum SHD to trigger drift detection
});

if (result.driftDetected) {
  console.log(`Causal drift detected! SHD = ${result.shd}`);
  console.log('Added:', result.added);
  console.log('Removed:', result.removed);
  console.log('Reversed:', result.reversed);
}
```

---

## 4. Causal Inference

### 4.1 The CausalAnalysis Pipeline

The recommended approach for causal inference uses the builder-pattern `CausalAnalysis` pipeline:

```typescript
import { CausalAnalysis } from '@agentix-e/causality-analyzer-pipeline';

const analysis = new CausalAnalysis()
  .ingest(data, varNames)          // Load data
  .model(graph)                     // Specify causal DAG
  .identify('Treatment', 'Outcome') // Find identification strategy
  .estimate('backdoor.efficient')   // Estimate effect
  .refute([                         // Validate
    'bootstrap',
    'placebo',
    'data_subset',
  ]);

const { ate, se, ciLower, ciUpper } = analysis.getEstimate();
console.log(`ATE = ${ate.toFixed(3)} ± ${(se * 1.96).toFixed(3)}`);

for (const refutation of analysis.getRefutations()) {
  console.log(`${refutation.method}: passed = ${refutation.passed}`);
}
```

### 4.2 Backdoor Adjustment (5 Variants)

The backdoor criterion is the most common identification strategy. Causality Analyzer provides five variants for finding adjustment sets:

```typescript
import {
  findBackdoorAdjustmentSet,
  findMinimalBackdoorSet,
  findEfficientBackdoorSet,
  findExhaustiveBackdoorSets,
  findMinCostEfficientBackdoorSet,
} from '@agentix-e/causality-analyzer-pipeline';

// 1. Minimal: Parents of treatment (Pearl's canonical set, smallest)
const minimal = findMinimalBackdoorSet(graph, 'T', 'Y');

// 2. Maximal: All admissible ancestors (most conservative)
const maximal = findBackdoorAdjustmentSet(graph, 'T', 'Y');

// 3. Efficient: Greedy backward selection (smallest statistically efficient set)
const efficient = findEfficientBackdoorSet(graph, 'T', 'Y', data, nodeIndex);

// 4. Exhaustive: All valid minimal sets (for sensitivity analysis)
const allSets = findExhaustiveBackdoorSets(graph, 'T', 'Y');
console.log(`Found ${allSets.length} valid backdoor sets`);

// 5. MinCostEfficient: Data-adaptive, minimizes variance inflation
const { set: minCostSet, varianceInflation } =
  findMinCostEfficientBackdoorSet(graph, 'T', 'Y', data, nodeIndex);
console.log(`Best set: ${[...minCostSet]}, variance inflation: ${varianceInflation}`);
```

**When to use each variant:**
- **Minimal**: When you trust your graph is complete (no omitted confounders)
- **Maximal**: When you want maximum robustness (but potentially higher variance)
- **Efficient**: When you want the smallest set that still achieves unbiased estimation
- **Exhaustive**: When you want to sensitivity-test across all valid adjustment strategies
- **MinCostEfficient**: When you want to minimize estimation variance (best for small n)

All variants are verified via strict d-separation check in the `G_{X-bar}` graph.

### 4.3 Effect Estimation Methods

#### Backdoor Adjustment via OLS

```typescript
import { adjustBackdoor } from '@agentix-e/causality-analyzer-pipeline';

const { ate, se, ciLower, ciUpper, residuals } = adjustBackdoor(
  graph, 'Treatment', 'Outcome', data, nodeIndex,
  { adjustmentSet: new Set(['Age', 'Education']) } // Optional: override
);
```

#### Frontdoor Adjustment

When a mediator blocks all paths and there are no unblocked backdoor paths:

```typescript
import { estimateFrontdoor } from '@agentix-e/causality-analyzer-pipeline';

const { ate, se } = estimateFrontdoor(
  graph, 'Treatment', 'Mediator', 'Outcome', data, nodeIndex,
);
```

#### Instrumental Variables (2SLS)

When treatment is confounded but you have a valid instrument:

```typescript
import { estimateIV } from '@agentix-e/causality-analyzer-pipeline';

const { ate, se, firstStageFStat } = estimateIV(
  graph, 'Instrument', 'Treatment', 'Outcome', data, nodeIndex,
);

console.log(`First-stage F-statistic: ${firstStageFStat}`);
if (firstStageFStat < 10) {
  console.warn('Weak instrument! F-stat < 10 may indicate bias.');
}
```

#### Propensity Score Methods

```typescript
import { estimatePropensityScore, estimatePSMatching } from '@agentix-e/causality-analyzer-pipeline';

// Inverse Probability of Treatment Weighting (IPTW)
const { ate, weights } = estimatePropensityScore(
  graph, 'Treatment', 'Outcome', data, nodeIndex,
  { method: 'iptw', caliper: 0.2 }
);

// Propensity Score Matching
const { ate, matchedPairs } = estimatePSMatching(
  graph, 'Treatment', 'Outcome', data, nodeIndex,
  { caliper: 0.2, ratio: 1 }  // 1:1 matching
);
```

#### Doubly Robust Estimation (AIPW)

Combines propensity score and outcome regression — consistent if either model is correct:

```typescript
import { estimateDoublyRobust } from '@agentix-e/causality-analyzer-pipeline';

const { ate, se } = estimateDoublyRobust(
  graph, 'Treatment', 'Outcome', data, nodeIndex,
  {
    outcomeModel: 'linear',  // or 'randomForest'
    psModel: 'logistic',
  }
);
```

### 4.4 do-Calculus (ID Algorithm)

The full recursive ID algorithm (Shpitser & Pearl 2006) for when backdoor/frontdoor are insufficient:

```typescript
import { identifyCausalEffect } from '@agentix-e/causality-analyzer-pipeline';

const result = identifyCausalEffect(graph, ['Treatment'], ['Outcome']);

switch (result.type) {
  case 'backdoor':
    console.log('Identified via backdoor, adjust for:', result.adjustmentSet);
    break;
  case 'frontdoor':
    console.log('Identified via frontdoor, mediator:', result.mediator);
    break;
  case 'id_algorithm':
    console.log('Identified via ID algorithm:', result.expression);
    break;
  case 'not_identifiable':
    console.warn('Causal effect is not identifiable from observational data');
    break;
}
```

**How it works:** The ID algorithm recursively applies Pearl's three rules of do-calculus, decomposing the query into identifiable sub-problems via c-component factorization. It automatically detects the hedge criterion when no identification is possible.

### 4.5 Sensitivity Analysis

#### E-value (VanderWeele & Ding 2017)

```typescript
import { computeEValue } from '@agentix-e/causality-analyzer-pipeline';

const eValue = computeEValue(ate, se);
console.log(`E-value = ${eValue.toFixed(2)}`);

// Interpretation:
// "An unmeasured confounder would need to be associated with both the treatment
// and the outcome by a risk ratio of at least {eValue} each, above and beyond
// the measured confounders, to fully explain away the observed effect."
```

#### Partial R-squared (Cinelli & Hazlett 2020)

```typescript
import { computePartialR2, computeRobustnessValue } from '@agentix-e/causality-analyzer-pipeline';

const r2 = computePartialR2(ate, se, n);
console.log(`Partial R^2 = ${r2.toFixed(3)}`);
// "An unmeasured confounder explaining {r2*100}% of the residual variance of
// both treatment and outcome would suffice to bring the estimate to zero."

const rv = computeRobustnessValue(ate, se, n);
console.log(`Robustness value = ${rv.toFixed(3)}`);
// The minimum confounding strength (in R^2 terms) needed to change the conclusion.
```

### 4.6 Refutation Methods

```typescript
import {
  refuteBootstrap,
  refutePlaceboTreatment,
  refuteDataSubset,
  refuteRandomCommonCause,
  refuteDummyOutcome,
} from '@agentix-e/causality-analyzer-pipeline';

// Bootstrap refutation: Is the estimate stable under resampling?
const bootstrap = refuteBootstrap(graph, 'T', 'Y', data, nodeIndex, {
  nBootstraps: 200,
});
// bootstrap.passed: true if CI excludes zero consistently

// Placebo treatment: Does scrambling treatment nullify the effect?
const placebo = refutePlaceboTreatment(graph, 'T', 'Y', data, nodeIndex);
// placebo.passed: true if new effect ≈ 0

// Data subset: Is the estimate stable across random 80% subsets?
const subset = refuteDataSubset(graph, 'T', 'Y', data, nodeIndex, {
  fractions: [0.7, 0.8, 0.9],
  nRepeats: 10,
});
// subset.passed: true if effects are consistent across subsets

// Random common cause: Is the estimate robust to synthetic confounders?
const randomCC = refuteRandomCommonCause(graph, 'T', 'Y', data, nodeIndex);

// Dummy outcome: Does replacing outcome nullify the effect?
const dummy = refuteDummyOutcome(graph, 'T', 'Y', data, nodeIndex);
```

### 4.7 Mediation Analysis

```typescript
import { analyzeMediation } from '@agentix-e/causality-analyzer-pipeline';

const { nde, nie, totalEffect, proportionMediated } = analyzeMediation(
  graph, 'Treatment', 'Mediator', 'Outcome', data, nodeIndex,
);

console.log(`Natural Direct Effect: ${nde.toFixed(3)}`);
console.log(`Natural Indirect Effect: ${nie.toFixed(3)}`);
console.log(`Total Effect: ${totalEffect.toFixed(3)}`);
console.log(`Proportion Mediated: ${(proportionMediated * 100).toFixed(1)}%`);
```

### 4.8 Uplift Modeling

```typescript
import { computeUpliftMetrics } from '@agentix-e/causality-analyzer-pipeline';

const metrics = computeUpliftMetrics(
  predictedUplift,    // Array of predicted CATE per individual
  actualTreatment,    // Array of actual treatment assignment (0/1)
  actualOutcome,      // Array of actual outcomes
);

console.log(`Qini coefficient: ${metrics.qini}`);
console.log(`AUUC (normalized): ${metrics.auuc}`);
console.log(`Uplift@10%: ${metrics.upliftAtK}`);
```

---

## 5. Root Cause Analysis

### 5.1 HeuristicPathRCA

CPT-based root cause scoring with BFS evidence propagation:

```typescript
import { CausalGraph, HeuristicPathRCA } from '@agentix-e/causality-analyzer-pipeline';
import { Matrix } from 'ml-matrix';

const graph = new CausalGraph(serviceNames);
// ... build graph from discovery or domain knowledge

const rca = new HeuristicPathRCA();

// Train on normal operation data
rca.train(graph, new Set(['Latency', 'ErrorRate']), normalData);

// Find root causes for an anomalous window
const result = rca.findRootCauses(['Latency', 'ErrorRate']);

console.log('Ranked Root Causes:');
for (const rc of result.rootCauses) {
  console.log(`  #${rc.rank}: ${rc.name} (confidence: ${rc.confidence})`);
  for (const evidence of rc.evidence) {
    console.log(`    - ${evidence.type}: score=${evidence.score}`);
  }
}
```

**How it works:**
1. Trains Conditional Probability Tables (CPTs) from normal-behavior data
2. During analysis, identifies nodes exceeding 2.5-sigma deviation
3. Uses BFS from anomalous nodes toward root nodes
4. Scores root nodes by probability of generating the observed anomaly pattern

### 5.2 RandomWalkRCA

Upstream random walks for root cause discovery:

```typescript
import { RandomWalkRCA } from '@agentix-e/causality-analyzer-pipeline';

const rca = new RandomWalkRCA({
  numWalks: 1000,      // Number of random walk repetitions
  walkLength: 10,      // Steps per walk
  backtrackProb: 0.15, // Probability of walking backward (downstream)
  seed: 42,            // LCG seed for reproducibility
});

rca.train(graph, anomalousNodes, data);
const result = rca.findRootCauses(['Latency']);

// Root causes are scored by how frequently random walks
// from anomalous nodes end up visiting them.
```

### 5.3 HTRCA (Hypothesis Testing RCA)

Per-node OLS residual scoring:

```typescript
import { HTRCA } from '@agentix-e/causality-analyzer-pipeline';

const rca = new HTRCA();
rca.train(graph, anomalousNodes, data);
const result = rca.findRootCauses(['Latency', 'CPU']);

// Each node gets a z-score based on how much its observed value
// deviates from the OLS prediction given its parents' values.
// max(z)/5 normalization produces scores in [0, 1].
```

### 5.4 FPGrowthRCA

Frequent pattern mining from trace data:

```typescript
import { FPGrowthRCA } from '@agentix-e/causality-analyzer-pipeline';

const rca = new FPGrowthRCA({
  minSupport: 0.1,     // Minimum pattern support
  inOutDiffWeight: 0.4, // Weight for InOutDiff score
  patternWeight: 0.6,    // Weight for pattern support score
});

rca.train(graph, anomalousNodes, traceData);
const result = rca.findRootCauses(['P99Latency']);
```

**How it works:** Mines frequent itemsets from system trace data (e.g., "DB slow AND Cache miss AND CPU spike"). Patterns that appear significantly more during anomalies than normal operation get higher scores.

### 5.5 CIRCA Pipeline (KDD 2022)

The Causal Inference-based Root Cause Analysis pipeline:

```typescript
import { CIRCAPipeline, RHTScorer, DAScorer } from '@agentix-e/causality-analyzer-pipeline';

// Option 1: Full CIRCA pipeline
const circa = new CIRCAPipeline({
  rhtConfig: { aggregation: 'max' },  // 'max' | 'mean' | 'sum'
  daConfig: {
    parentPenalty: 0.3,    // Penalty: parent_z * 0.3
    childBonus: 0.1,       // Bonus: child_z * 0.1
  },
});

const result = circa.analyze(normalData, anomalousData, graph, anomalousNodes);

// Option 2: Use components individually
const rhtScorer = new RHTScorer();
const rawScores = rhtScorer.score(graph, normalData, anomalousData);

const daScorer = new DAScorer({ parentPenalty: 0.3, childBonus: 0.1 });
const adjustedScores = daScorer.adjust(rawScores, graph);

// RHT (Reversed HT): OLS regression on normal data, scores by residual
// z-score during failure window.
//
// DA (Descendant Adjustment):
// - Reduces scores for nodes with anomalous parents (they might just
//   be downstream effects)
// - Boosts scores for nodes with anomalous children (they explain
//   downstream behavior)
```

### 5.6 Fusion Analyzer

Ensemble multiple RCA methods:

```typescript
import { FusionAnalyzer } from '@agentix-e/causality-analyzer-pipeline';

const fusion = new FusionAnalyzer({
  method: 'weighted',  // 'weighted' | 'nested' | 'voting'
  weights: {
    heuristicPath: 0.35,
    randomWalk: 0.25,
    ht: 0.25,
    fpGrowth: 0.15,
  },
});

const result = fusion.analyze(data, graph, anomalousNodes);
```

---

## 6. Bayesian Networks

### 6.1 Building Bayesian Networks

```typescript
import { CausalGraph } from '@agentix-e/causality-analyzer-pipeline';

// Define structure
const bn = new CausalGraph(['Cloud', 'Sprinkler', 'Rain', 'WetGrass']);
bn.addEdge('Cloud', 'Rain');
bn.addEdge('Cloud', 'Sprinkler');
bn.addEdge('Rain', 'WetGrass');
bn.addEdge('Sprinkler', 'WetGrass');

// Learn CPTs from data
import { estimateCPTs } from '@agentix-e/causality-analyzer-pipeline';

const cpts = estimateCPTs(bn, data, {
  method: 'mle',           // 'mle' | 'dirichlet'
  alpha: 1,                // Dirichlet prior strength
  discretize: 'equalWidth', // 'equalWidth' | 'equalFreq'
  bins: 5,
});
```

### 6.2 Inference Engines

#### Variable Elimination (Exact)

Best for small-to-medium networks:

```typescript
import { variableElimination } from '@agentix-e/causality-analyzer-pipeline';

const result = variableElimination(bn, cpts, 'WetGrass', {
  evidence: new Map([['Rain', 'yes']]),
  eliminationOrder: ['Cloud', 'Sprinkler', 'Rain'],
});

console.log(`P(WetGrass | Rain=yes) = ${result.probability}`);
```

#### Junction Tree (Exact, Multi-Query)

Best when you need multiple queries on the same network:

```typescript
import { junctionTreeInference } from '@agentix-e/causality-analyzer-pipeline';

const jt = junctionTreeInference(bn, cpts);
const result1 = jt.query('WetGrass', new Map([['Rain', 'yes']]));
const result2 = jt.query('Sprinkler', new Map([['WetGrass', 'yes']]));
// Second query reuses the same junction tree — no recomputation.
```

#### Loopy Belief Propagation (Approximate)

Best for large graphs with cycles:

```typescript
import { loopyBeliefPropagation } from '@agentix-e/causality-analyzer-pipeline';

const result = loopyBeliefPropagation(bn, cpts, 'WetGrass', {
  evidence,
  maxIterations: 50,
  tolerance: 1e-4,
});
```

#### Likelihood Weighting (Importance Sampling)

Best for continuous variables:

```typescript
import { likelihoodWeighting } from '@agentix-e/causality-analyzer-pipeline';

const result = likelihoodWeighting(bn, cpts, 'WetGrass', {
  evidence,
  numSamples: 10000,
});
```

#### Gibbs Sampling (MCMC)

Best for complex posteriors:

```typescript
import { gibbsSampling } from '@agentix-e/causality-analyzer-pipeline';

const result = gibbsSampling(bn, cpts, 'WetGrass', {
  evidence,
  numSamples: 10000,
  burnIn: 1000,
  thinning: 10,
});
```

### 6.3 Dirichlet Learning (Online)

Update CPTs incrementally with streaming data:

```typescript
import { DirichletLearner } from '@agentix-e/causality-analyzer-pipeline';

const learner = new DirichletLearner(bn, { alpha: 1 });

// Process data in batches
for (const batch of dataBatches) {
  learner.update(batch);
}

const cpts = learner.getCPTs();

// Access learning statistics
console.log('Samples processed:', learner.sampleCount);
```

### 6.4 Factor Operations

Low-level factor operations for custom inference:

```typescript
import {
  factorMultiply, factorMarginalize, factorReduce, factorNormalize,
} from '@agentix-e/causality-analyzer-pipeline';

const f1 = { variables: ['A', 'B'], values: [0.1, 0.2, 0.3, 0.4] };
const f2 = { variables: ['B', 'C'], values: [0.5, 0.5, 0.5, 0.5] };

// Multiply factors
const f3 = factorMultiply(f1, f2);

// Marginalize out a variable
const f4 = factorMarginalize(f3, 'B');

// Reduce a factor given evidence
const f5 = factorReduce(f4, 'C', 'yes');
```

---

## 7. Generative Causal Modeling

### 7.1 Structural Causal Models

Build and train Structural Causal Models (SCMs):

```typescript
import { StructuralCausalModel } from '@agentix-e/causality-analyzer-pipeline';

const scm = new StructuralCausalModel(graph);

// Auto-assign causal mechanisms based on data characteristics
scm.autoAssignMechanisms(data);

// Or manually specify mechanisms
scm.assignMechanism('X', 'additiveNoise');
scm.assignMechanism('Y', 'postNonlinear');
scm.assignMechanism('Z', 'neural');

// Train all mechanisms
scm.train(data);

// Serialize for later use
const json = scm.toJSON();
// ... save to disk ...
const restored = StructuralCausalModel.fromJSON(json);
```

### 7.2 Computing Counterfactuals

```typescript
import { StructuralCausalModel } from '@agentix-e/causality-analyzer-pipeline';

const scm = new StructuralCausalModel(graph);
scm.autoAssignMechanisms(data);
scm.train(data);

// Abduction-Action-Prediction framework:
// 1. Abduction: Infer noise terms from factual observation
// 2. Action: Perform do-operation (intervention)
// 3. Prediction: Forward-propagate with modified noise to get counterfactual

const counterfactual = scm.computeCounterfactual(
  dataRow,                        // Actual observation
  new Map([['Treatment', 1.0]]),  // Intervention: set Treatment = 1.0
);

console.log('Actual outcome:', dataRow[outcomeIdx]);
console.log('Counterfactual (if Treatment=1):', counterfactual[outcomeIdx]);
```

### 7.3 Neural Causal Mechanisms

Train feedforward neural networks as causal mechanisms:

```typescript
import { FFNMechanism, trainFFNMechanism } from '@agentix-e/causality-analyzer-pipeline';

const mechanism = new FFNMechanism({
  hiddenLayers: [32, 16],  // Two hidden layers
  activation: 'relu',
  learningRate: 0.001,
  epochs: 500,
});

// parentData: matrix of parent node values
// childData: vector of child node values
mechanism.train(parentData, childData);

// Forward pass: predict child given parents
const prediction = mechanism.predict(newParentValues);

// Inverse pass: infer parent noise given child
const noise = mechanism.inverse(newParentValues, childValue);
```

### 7.4 Model Evaluation

```typescript
import {
  evaluateMechanismR2, evaluateMSE,
  shapleyAttribute, bootstrapRCA,
} from '@agentix-e/causality-analyzer-pipeline';

// Per-mechanism R-squared
const r2ByNode = evaluateMechanismR2(scm, testData);
for (const [node, r2] of r2ByNode) {
  console.log(`${node}: R^2 = ${r2.toFixed(3)}`);
}

// Mean Squared Error
const mseByNode = evaluateMSE(scm, testData);

// Shapley attribution: how much does each node contribute to the outcome?
const shapley = shapleyAttribute(graph, data, 'Outcome');
for (const [node, value] of shapley) {
  console.log(`${node}: Shapley value = ${value.toFixed(3)}`);
}
```

### 7.5 RESIT Test (Regression Error Spearman Independence Test)

Validate causal ordering by testing residual independence:

```typescript
import { resitTest } from '@agentix-e/causality-analyzer-pipeline';

const result = resitTest(data, varNames, graph);
console.log(`RESIT p-value: ${result.pValue}`);
// p > 0.05: No evidence against the causal ordering
// p < 0.05: The causal ordering may be wrong
```

### 7.6 Graph Falsification

Test structural assumptions by comparing implied vs. observed independencies:

```typescript
import { falsifyGraph } from '@agentix-e/causality-analyzer-pipeline';

const result = falsifyGraph(graph, data, {
  alpha: 0.05,
  maxConditioning: 3,
});

if (result.falsified) {
  console.log('Graph structure is rejected by the data:');
  for (const violation of result.violations) {
    console.log(
      `  ${violation.x} should be independent of ${violation.y} ` +
      `given ${violation.z}, but p-value = ${violation.pValue}`
    );
  }
  console.log('Suggested edge removals:', result.suggestions);
}
```

---

## 8. Anomaly Detection

### 8.1 Spectral Residual Detector

Best for data with seasonal patterns:

```typescript
import { SpectralResidualDetector } from '@agentix-e/causality-analyzer-pipeline';

const detector = new SpectralResidualDetector({
  windowSize: 64,      // FFT window
  threshold: 3.0,      // Z-score threshold for anomaly
  delay: 7,            // Look-ahead for point adjustment
});

const result = detector.detect(timeSeries);
console.log('Anomalies:', result.anomalyIndices);
```

### 8.2 SPOT/DSPOT (Streaming POT)

Streaming Peaks-Over-Threshold for drift detection:

```typescript
import { SPOTDetector, DSPOTDetector } from '@agentix-e/causality-analyzer-pipeline';

// SPOT: For detecting upward anomalies
const spot = new SPOTDetector({
  q: 1e-4,     // Risk parameter
  nInit: 1000, // Initial calibration samples
});

// DSPOT: For detecting downward AND upward anomalies
const dspot = new DSPOTDetector({
  q: 1e-4,
  nInit: 1000,
});

// Streaming mode
for (const value of streamData) {
  const isAnomaly = dspot.ingest(value);
  if (isAnomaly) {
    console.log('Anomaly detected at value', value);
  }
}
```

### 8.3 Stats Detector

Distribution-based statistical anomaly detection:

```typescript
import { StatsDetector } from '@agentix-e/causality-analyzer-pipeline';

const detector = new StatsDetector({
  method: 'zscore',    // 'zscore' | 'iqr' | 'mad' | 'percentile'
  threshold: 3.0,      // For zscore method
  windowSize: 100,     // Rolling window for baseline statistics
});

const result = detector.detect(data);
```

### 8.4 Voting Detector

Ensemble majority voting across detectors:

```typescript
import { VotingDetector, SpectralResidualDetector, SPOTDetector } from '@agentix-e/causality-analyzer-pipeline';

const votingDetector = new VotingDetector({
  detectors: [
    new SpectralResidualDetector(),
    new SPOTDetector({ q: 1e-4, nInit: 1000 }),
    new StatsDetector({ method: 'zscore' }),
  ],
  strategy: 'majority',  // 'majority' | 'weighted' | 'unanimous'
  weights: [1, 2, 1],    // For 'weighted' strategy
});

const result = votingDetector.detect(data);
```

---

## 9. Storage Configuration

### 9.1 Embedded Mode (Zero External Dependencies)

```typescript
import { EmbedRelationalStore, EmbedGraphStore } from '@agentix-e/causality-analyzer-storage-embed';
import { CausalAnalysis } from '@agentix-e/causality-analyzer-pipeline';

const relationalStore = new EmbedRelationalStore('/data/metrics.db');
const graphStore = new EmbedGraphStore('/data/graphs.db');

const analysis = new CausalAnalysis({
  relationalStore,
  graphStore,
});

// Everything runs in-process — no PostgreSQL, no Neo4j, no Docker
```

### 9.2 Browser Mode (WASM SQLite + OPFS)

```typescript
import {
  WasmRelationalStore, WasmGraphStore,
} from '@agentix-e/causality-analyzer-storage-browser';

// In a browser environment:
const relationalStore = await WasmRelationalStore.create({
  // Automatically uses OPFS for persistence
  // Falls back to in-memory if OPFS unavailable
});

const graphStore = await WasmGraphStore.create();

// Offline persistence: data survives page refreshes
// Web Worker: all SQLite operations run in a worker for non-blocking UI
```

### 9.3 Remote Mode (PostgreSQL + Neo4j)

```typescript
import { RemoteRelationalStore, RemoteGraphStore } from '@agentix-e/causality-analyzer-storage-remote';

// PostgreSQL
const relationalStore = new RemoteRelationalStore({
  host: 'postgres.internal',
  port: 5432,
  database: 'causality',
  user: 'app',
  password: process.env.PG_PASSWORD,
  // mTLS (optional)
  ssl: {
    ca: fs.readFileSync('/certs/ca.pem'),
    cert: fs.readFileSync('/certs/client.pem'),
    key: fs.readFileSync('/certs/client-key.pem'),
  },
});

// Neo4j via Bolt protocol
const graphStore = new RemoteGraphStore({
  uri: 'bolt://neo4j.internal:7687',
  auth: {
    type: 'basic',
    username: 'neo4j',
    password: process.env.NEO4J_PASSWORD,
  },
  // mTLS (optional)
  trust: 'TRUST_CUSTOM_CA_SIGNED_CERTIFICATES',
  trustedCertificates: [fs.readFileSync('/certs/neo4j-ca.pem').toString()],
});
```

---

## 10. HTTP API Server

### 10.1 Starting the Server

```bash
# Basic HTTP
npx causal-analyzer serve --port 3000

# HTTPS with authentication
npx causal-analyzer serve \
  --port 443 \
  --tls-cert /etc/ssl/certs/server.pem \
  --tls-key /etc/ssl/private/server-key.pem \
  --api-token "${API_TOKEN}"

# With mTLS (mutual TLS)
npx causal-analyzer serve \
  --tls-cert server.pem \
  --tls-key server-key.pem \
  --tls-ca client-ca.pem \
  --tls-request-cert \
  --tls-reject-unauthorized
```

### 10.2 API Endpoint Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | None | Combined health/liveness/readiness |
| `GET` | `/ready` | None | Readiness probe |
| `GET` | `/live` | None | Liveness probe |
| `GET` | `/metrics` | None | Prometheus metrics |
| `GET` | `/v1/openapi.json` | Bearer | OpenAPI 3.1 specification |
| `POST` | `/v1/discover` | Bearer | Run causal discovery |
| `POST` | `/v1/analyze` | Bearer | Run RCA pipeline |
| `POST` | `/v1/estimate` | Bearer | Run effect estimation |

### 10.3 API Request Examples

#### Causal Discovery

```bash
curl -X POST http://localhost:3000/v1/discover \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -d '{
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    "varNames": ["X", "Y", "Z"],
    "method": "pc",
    "config": {
      "alpha": 0.05,
      "ciTest": "fisher-z"
    }
  }'
```

Response:
```json
{
  "success": true,
  "data": {
    "graph": {
      "nodes": ["X", "Y", "Z"],
      "edges": [
        {"source": "X", "target": "Y", "weight": 0.8},
        {"source": "Y", "target": "Z", "weight": 0.9}
      ]
    }
  }
}
```

#### Effect Estimation

```bash
curl -X POST http://localhost:3000/v1/estimate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -d '{
    "data": [[...]],
    "graph": {"nodes": [...], "edges": [...]},
    "treatment": "T",
    "outcome": "Y",
    "method": "backdoor"
  }'
```

### 10.4 Docker Deployment

```bash
# Full stack: pipeline + PostgreSQL + Neo4j
docker compose up -d

# Health check
curl http://localhost:3000/health

# Check logs
docker compose logs -f pipeline
```

Environment variables for Docker:
```
CAUSALITY_API_TOKEN=your-secure-token
CAUSALITY_PG_HOST=postgres
CAUSALITY_NEO4J_URI=bolt://neo4j:7687
CAUSALITY_ENCRYPTION_KEY=$(openssl rand -hex 32)
CAUSALITY_TLS_CERT=/run/secrets/server-cert
CAUSALITY_TLS_KEY=/run/secrets/server-key
```

---

## 11. Visualization

### 11.1 ca-causal-graph

Interactive DAG visualization Web Component:

```html
<!-- In HTML -->
<ca-causal-graph id="graph" style="width: 800px; height: 600px;"></ca-causal-graph>
```

```typescript
// In JavaScript/TypeScript
import '@agentix-e/causality-analyzer-visual';
import { buildGraphVizData } from '@agentix-e/causality-analyzer-pipeline';

const graphEl = document.getElementById('graph');
const vizData = buildGraphVizData(result.graph, {
  highlightAnomalous: ['P99Latency'],
  highlightRootCauses: ['CacheHitRate'],
});

graphEl.data = vizData;

graphEl.addEventListener('node-click', (event) => {
  const nodeName = event.detail.nodeName;
  console.log('Clicked:', nodeName);
});
```

**CSS Custom Properties for theming:**
```css
ca-causal-graph {
  --ca-primary: #3b82f6;
  --ca-anomaly: #ef4444;
  --ca-root-cause: #f59e0b;
  --ca-healthy: #22c55e;
  --ca-edge-weight: #94a3b8;
  --ca-background: #ffffff;
}
```

### 11.2 ca-time-series

Anomaly time series with highlighted regions:

```html
<ca-time-series id="ts" style="width: 100%; height: 400px;"></ca-time-series>
```

```typescript
import { buildTimeseriesVizData } from '@agentix-e/causality-analyzer-pipeline';

const tsEl = document.getElementById('ts');
const tsData = buildTimeseriesVizData(latencyTimeSeries, anomalyRegions);

tsEl.data = tsData;
```

### 11.3 ca-root-cause-ranking

Ranked list of root causes:

```html
<ca-root-cause-ranking id="ranking" style="width: 400px;"></ca-root-cause-ranking>
```

```typescript
import { buildRankingVizData } from '@agentix-e/causality-analyzer-pipeline';

const rankingEl = document.getElementById('ranking');
const rankingData = buildRankingVizData(rcaResult);

rankingEl.data = rankingData;

rankingEl.addEventListener('cause-hover', (event) => {
  highlightPropagationPath(event.detail.causeName);
});
```

### 11.4 Accessibility

All components support:
- **Screen readers**: ARIA labels, live regions for dynamic content
- **Keyboard navigation**: Tab focus, Arrow key navigation, Enter to select
- **High contrast mode**: Respects `prefers-contrast` media query
- **Reduced motion**: Respects `prefers-reduced-motion` media query

---

## 12. Advanced Topics

### 12.1 Web Worker Parallelism

Offload CI test computation to worker threads:

```typescript
import { WorkerPool } from '@agentix-e/causality-analyzer-pipeline';

const pool = new WorkerPool({
  numWorkers: navigator.hardwareConcurrency || 4,
  workerScript: './ci-worker.js',
});

const results = await pool.map(
  conditionalIndependenceTests,
  (test) => pool.run(test),
);

await pool.terminate();
```

### 12.2 Streaming Pipeline

Real-time causal analysis for telemetry data:

```typescript
import { StreamingPipeline } from '@agentix-e/causality-analyzer-pipeline';

const pipeline = new StreamingPipeline({
  windowSize: 500,
  slideSize: 100,
  discoveryAlgorithm: 'pc',
  rcaMethod: 'circa',
});

pipeline.onResult((result) => {
  console.log(`Window ${result.windowId}:`);
  console.log('  Root causes:', result.rootCauses);
  console.log('  Graph:', result.graph.edges.length, 'edges');
});

for await (const batch of telemetryStream) {
  pipeline.ingest(batch);
}
```

### 12.3 LLM-Enhanced Explanation (DeepSeek)

Generate natural language explanations of causal findings:

```typescript
import { generateCausalExplanation } from '@agentix-e/causality-analyzer-pipeline';

const explanation = await generateCausalExplanation({
  graph: result.graph,
  rcaResult: rcaResult,
  context: 'Microservice observability incident; P99 latency spike detected.',
  apiKey: process.env.DEEPSEEK_API_KEY,
  model: 'deepseek-chat',
  maxTokens: 500,
});

console.log(explanation);
// → "The P99 latency spike at 14:23 UTC was primarily caused by a decline in
//    CacheHitRate (confidence: 0.86), which cascaded through DatabasePool → P99Latency.
//    CacheHitRate dropped from 87% to 23%, resulting in a 6x increase in database
//    queries. Recommended action: Investigate cache eviction policies..."
```

### 12.4 Graph Similarity Analysis

Compare causal graphs using structural fingerprints:

```typescript
import { graphSimilarity } from '@agentix-e/causality-analyzer-core';

const similarity = graphSimilarity(graphA, graphB);
console.log(`Graph similarity: ${(similarity * 100).toFixed(1)}%`);

// Graph similarity uses a 13-dimensional structural fingerprint:
// - Dimensions 0-4: node count, edge density, root/leaf/v-structure ratios
// - Dimensions 5-9: out-degree distribution bins [0, 1, 2, 3, 4+]
// - Dimensions 10-12: max depth, avg depth, graph diameter
//
// Similarity = cosine(fingerprint(graphA), fingerprint(graphB))
```

### 12.5 Domain Knowledge Integration

Apply expert knowledge to constrain or guide discovery:

```typescript
import { pcAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const domainKnowledge = {
  forbiddenEdges: [
    ['Outcome', 'Treatment'],  // Outcome cannot cause treatment
    ['Age', 'AdSpend'],        // Age cannot cause ad spend (logically)
  ],
  requiredEdges: [
    ['Treatment', 'Outcome'],  // Treatment must affect outcome
  ],
  rootNodes: ['Age', 'Seasonality'],     // These have no parents
  leafNodes: ['Revenue', 'Conversion'],   // These have no children
  temporalLags: [
    ['AdSpend', 'Revenue', 1],  // Ad spend affects revenue with 1-period lag
  ],
};

const { graph } = pcAlgorithm(data, varNames, {
  alpha: 0.05,
  domainKnowledge,
});

// The algorithm respects all domain constraints:
// - Forbidden edges are never added
// - Required edges are never removed
// - Root/leaf constraints guide orientation
// - Temporal lags inform direction
```

### 12.6 Model Serialization

Save and restore models for later analysis:

```typescript
// Save causal graph
const graphJson = graph.toJSON();
fs.writeFileSync('graph.json', JSON.stringify(graphJson));

// Load causal graph
const graph = CausalGraph.fromJSON(JSON.parse(fs.readFileSync('graph.json')));

// Save SCM with trained mechanisms
const scmJson = scm.toJSON();
fs.writeFileSync('scm.json', JSON.stringify(scmJson));

// Load SCM
const scm = StructuralCausalModel.fromJSON(JSON.parse(fs.readFileSync('scm.json')));

// Serialize specific mechanisms
const mechanism = scm.getMechanism('Y'); // FFNMechanism
const mechJson = mechanism.toJSON();
```

### 12.7 Programmatic Pipeline Execution

Build and execute custom analysis pipelines:

```typescript
import { PipelineStage } from '@agentix-e/causality-analyzer-core';

// Manual pipeline execution with checkpointing
const pipeline = {
  data: rawData,
  normalizedData: undefined,
  anomalies: undefined,
  graph: undefined,
  rootCauses: undefined,
  estimates: undefined,
};

// Stage 1: INGEST — Standardize and impute
pipeline.normalizedData = standardize(pipeline.data);

// Stage 2: DETECT — Find anomalies
const detector = new SpectralResidualDetector();
pipeline.anomalies = detector.detect(pipeline.normalizedData);

// Stage 3: GRAPH — Discover causal structure
const { graph } = pcAlgorithm(pipeline.normalizedData, varNames);
pipeline.graph = graph;

// Stage 4: ANALYZE — Find root causes
const rca = new CIRCAPipeline();
pipeline.rootCauses = rca.analyze(
  pipeline.normalizedData, anomalousWindow, pipeline.graph, anomalousNodes,
);

// Stage 5: INFER — Estimate effects
pipeline.estimates = adjustBackdoor(
  pipeline.graph, 'CacheHitRate', 'P99Latency',
  pipeline.normalizedData, nodeIndex,
);

// Stage 6: VALIDATE — Refute and check sensitivity
const refutation = refuteBootstrap(
  pipeline.graph, 'CacheHitRate', 'P99Latency',
  pipeline.normalizedData, nodeIndex,
);
```

---

## 13. Troubleshooting

### 13.1 Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ConvergenceError: NOTEARS did not converge` | Data scale too large or lambda too small | Standardize data to zero-mean unit-variance; increase `lambda1` |
| `StoreError: CONNECTION_FAILED` | Database unreachable | Check host/port/credentials; verify VPN/firewall |
| `ValidationError: CausalGraph has cycles` | Graph contains feedback loop | Use `graph.isAcyclic()` first; consider CCD algorithm for cyclic data |
| `Singular matrix in solveLinear` | Perfect collinearity in adjustment variables | Remove redundant variables; use `solveLinearSafe()` |
| `IdentifiedEstimand: not_identifiable` | Causal effect not identifiable | Check if valid backdoor/frontdoor/IV set exists; use do-calculus ID algorithm |

### 13.2 Performance Tuning

**Discovery speed:**
- Use `alpha: 0.01` instead of `0.05` for fewer CI tests
- Set `maxConditioning: 2` to limit conditioning set depth
- Use `correction: 'fdr'` instead of `'bonferroni'` for faster convergence
- Pre-standardize data (z-score) before running algorithms

**Memory usage:**
- For d > 50 nodes, use stochastic methods (GES, NOTEARS) instead of exact enumeration
- Avoid `maxConditioning: -1` on large graphs (unlimited depth)
- Use `Float64Array` for all numeric data (default behavior)

**Server throughput:**
- Increase `WINDOW_SIZE` in rate limiter config
- Use embedded storage for single-server deployments (lower latency than remote)
- Enable WAL mode on SQLite for concurrent reads

### 13.3 Dependency Verification

```bash
# Verify all packages install and build
pnpm install
pnpm -r build

# Run tests to verify your environment
pnpm -r test

# Verify specific package
pnpm --filter @agentix-e/causality-analyzer-pipeline test
```

---

This user guide covers the complete API surface of Causality Analyzer v1.1.0. For API reference documentation, see the [TypeDoc generated docs](https://agentix-e.github.io/causality-analyzer/api/). For architecture details, see the [Architecture Document](./architecture.md). For installation support, open an issue on [GitHub](https://github.com/AgentiX-E/causality-analyzer/issues).
