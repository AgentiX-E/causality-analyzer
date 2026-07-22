# Causality Analyzer — User Guide

> From zero to production: anomaly detection, causal discovery, root cause analysis, effect estimation, and counterfactual reasoning.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation & Setup](#2-installation--setup)
3. [Core Concepts](#3-core-concepts)
4. [Anomaly Detection](#4-anomaly-detection)
5. [Causal Discovery](#5-causal-discovery)
6. [Root Cause Analysis](#6-root-cause-analysis)
7. [Causal Effect Estimation](#7-causal-effect-estimation)
8. [Sensitivity & Refutation](#8-sensitivity--refutation)
9. [Counterfactual Analysis](#9-counterfactual-analysis)
10. [Model Evaluation & Validation](#10-model-evaluation--validation)
11. [Data Storage](#11-data-storage)
12. [Visualization](#12-visualization)
13. [Enterprise Deployment](#13-enterprise-deployment)
14. [Troubleshooting & FAQ](#14-troubleshooting--faq)

---

## 1. Introduction

Causality Analyzer helps you answer three critical questions during incident response:

| Question | Toolkit |
|----------|---------|
| **What happened?** | Anomaly detection across metrics |
| **Why did it happen?** | Causal discovery → root cause analysis |
| **What if?** | Counterfactual reasoning |

Unlike correlation-based monitoring tools, Causality Analyzer discovers the actual causal structure of your system and quantifies how confident you should be in the findings.

### When to Use Each Approach

| Scenario | Recommended Approach |
|----------|---------------------|
| Simple monitoring, well-understood system | StatsDetector (z-score) |
| Unusual patterns, periodic signals | SpectralResidualDetector |
| Extreme events with heavy tails | SPOT / DSPOT |
| Multiple detectors, need consensus | VotingDetector |
| No prior causal graph | PC algorithm (automatic discovery) |
| Suspected hidden confounders | FCI algorithm |
| Only care about one metric's causes | Targeted Discovery |
| Known service topology | BayesianRCA / HTRCA |
| Unknown topology + anomaly data | CIRCA pipeline |
| Need to quantify treatment effect | Backdoor adjustment / IV |
| Unbalanced failure data | IPW / Doubly Robust |
| Need worst-case robustness | E-value sensitivity |
| "What would have happened if..." | Counterfactual inference |
| Validate your causal graph | Graph falsification |

---

## 2. Installation & Setup

### Prerequisites

- Node.js ≥ 20
- pnpm ≥ 10

### Install from npm

```bash
# Foundation
npm install @agentix-e/causality-analyzer-core

# Causal analysis
npm install @agentix-e/causality-analyzer-pipeline

# Storage (choose one)
npm install @agentix-e/causality-analyzer-storage-embed   # SQLite + OverGraph
npm install @agentix-e/causality-analyzer-storage-remote   # PostgreSQL + Neo4j
npm install pg neo4j-driver-lite                            # optional: remote drivers

# Visualization
npm install @agentix-e/causality-analyzer-visual
```

### Development Setup

```bash
git clone https://github.com/AgentiX-E/causality-analyzer.git
cd causality-analyzer
pnpm install
pnpm run --filter @agentix-e/causality-analyzer-core build
pnpm -r test   # 420+ tests, all should pass
```

---

## 3. Core Concepts

### Causal Graph

A **CausalGraph** represents variable nodes and directed edges (A → B means A causes B). The library supports DAGs (fully directed), PDAGs (partially directed), and PAGs (ancestral graphs for latent confounders).

```typescript
import { CausalGraph } from '@agentix-e/causality-analyzer-pipeline';

// Manual specification
const graph = new CausalGraph(['Memory', 'CPU', 'Latency']);
graph.addEdge('Memory', 'CPU');   // Memory → CPU
graph.addEdge('CPU', 'Latency');  // CPU → Latency
graph.addEdge('Memory', 'Latency'); // Memory → Latency (direct + mediated)
```

### ColumnarTable

All data is represented as zero-copy columnar tables for performance:

```typescript
import { ColumnarTable } from '@agentix-e/causality-analyzer-core';

const table = ColumnarTable.fromColumnar({
  cpu: new Float64Array([0.5, 0.8, 0.9]),
  mem: new Float64Array([0.6, 0.7, 0.85]),
  latency: new Float64Array([100, 120, 250]),
});

const cpuColumn = table.column('cpu');  // Float64Array — zero copy
```

### Deterministic Reproducibility

Every stochastic algorithm accepts an optional `seed` parameter:

```typescript
// With seed → same result every time
const result = shapleyAttribute(scm, obs, 5, 42);

// Without seed → uses internal RNG
const result2 = shapleyAttribute(scm, obs, 5);
```

---

## 4. Anomaly Detection

### When One Detector Is Enough

**StatsDetector (Z-score)** — for normally distributed metrics:

```typescript
import { StatsDetector } from '@agentix-e/causality-analyzer-pipeline';

const detector = new StatsDetector({ method: 'zscore' });
detector.train(normalData);  // learn baseline mean and std

const result = detector.update([5.0, 8.1]);
if (result.isAnomalous) {
  console.log('Alert!', result.scores);  // z-scores per metric
  console.log(result.labels);            // [1, 0, ...] per metric
}
```

**StatsDetector methods:**
- `'zscore'` — parametric, fast, assumes normality
- `'mad'` — robust to outliers, uses median absolute deviation
- `'iqr'` — non-parametric, uses inter-quartile range

### When You Need to Detect Unusual Patterns

**SpectralResidualDetector** — for periodic/seasonal data:

```typescript
import { SpectralResidualDetector } from '@agentix-e/causality-analyzer-pipeline';

const detector = new SpectralResidualDetector({ windowSize: 64 });
detector.init(initialData);    // warm up FFT buffer

// Streaming
for (const point of stream) {
  const result = detector.update(point);
  if (result.isAnomalous) handleAlert(result);
}

// Batch
const results = detector.detect(batchData);
```

### When Extreme Events Matter

**SPOT / DSPOT** — for metrics with heavy tails (rare but extreme anomalies):

```typescript
import { SPOTDetector } from '@agentix-e/causality-analyzer-pipeline';

const detector = new SPOTDetector({ initSize: 100, q: 1e-4 });
// DSPOT variant: handles gradual drift
import { DSPOTDetector } from '@agentix-e/causality-analyzer-pipeline';
const driftDetector = new DSPOTDetector({ driftWindow: 100 });
```

### When Multiple Detectors Vote

```typescript
import { VotingDetector } from '@agentix-e/causality-analyzer-pipeline';

const ensemble = new VotingDetector({
  detectors: [statsDetector, srDetector, spotDetector],
  strategy: 'majority',       // or 'maximum', 'weighted'
  minAgreement: 2,            // at least 2 must agree
  weights: { metric: 0.5 },   // for 'weighted' strategy
});
```

---

## 5. Causal Discovery

### PC Algorithm — When You Have No Prior Graph

```typescript
import { Matrix } from 'ml-matrix';
import { pcAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

// Prepare data (n rows × m columns)
const data = new Matrix(1000, 4); // 1000 observations, 4 metrics
// ... fill data ...

const { graph, sepSet } = pcAlgorithm(data, ['CPU', 'Memory', 'Disk', 'Latency'], {
  alpha: 0.05,     // significance level
  stable: true,    // stable-PC variant (recommended)
  maxDegree: -1,   // unlimited conditioning set size
});

console.log(graph.edges);
// [{ source: 'Memory', target: 'CPU', weight: 1, directed: true }, ...]
```

### FCI Algorithm — When You Suspect Hidden Confounders

```typescript
import { fciAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const { pagEdges } = fciAlgorithm(data, nodeNames, { alpha: 0.05 });
// pagEdges shows "X→Y", "X↔Y" (latent), "X∘→Y" (ancestral)
for (const [key, type] of pagEdges) {
  if (type.includes('↔')) console.log(`${key} may have latent confounder`);
}
```

### Targeted Discovery — When You Only Care About One Metric

```typescript
import { targetedDiscovery } from '@agentix-e/causality-analyzer-pipeline';

// Only find what causes Latency
const parents = targetedDiscovery(data, ['Latency'], nodeNames);
console.log(parents.get('Latency'));  // ['CPU', 'Memory']
```

### Grow-Shrink — Markov Blanket Discovery

```typescript
import { growShrink } from '@agentix-e/causality-analyzer-pipeline';

// Find variables that make CPU independent of all others
const blanket = growShrink(data, 2/*CPU*/, nodeNames);
```

---

## 6. Root Cause Analysis

### When You Have a Known Causal Graph

**BayesianRCA** — exact inference via variable elimination:

```typescript
import { BayesianRCA } from '@agentix-e/causality-analyzer-pipeline';

const rca = new BayesianRCA();
rca.train(graph, new Set(['CPU', 'Latency']), data);
const result = rca.findRootCauses(['CPU', 'Latency']);

console.log(result.rootCauses[0].name);   // The most likely root cause
console.log(result.rootCauses[0].score);  // Posterior probability P(root|evidence)
```

**HTRCA** — regression-based hypothesis testing:

```typescript
import { HTRCA } from '@agentix-e/causality-analyzer-pipeline';

const rca = new HTRCA();
rca.train(graph, data);
const result = rca.findRootCauses(['CPU', 'Latency'], data);
// Uses OLS residuals: normal behavior → small residuals, failure → large residuals
```

**RandomWalkRCA** — graph-based random walk (reproducible with seed):

```typescript
import { RandomWalkRCA } from '@agentix-e/causality-analyzer-pipeline';

const rca = new RandomWalkRCA(42);  // seed = 42 for reproducibility
rca.train(graph);
const result = rca.findRootCauses(['CPU', 'Latency'], 10, 1000);
```

### When You Don't Have a Known Graph — CIRCA

```typescript
import { CIRCAPipeline } from '@agentix-e/causality-analyzer-pipeline';

const circa = new CIRCAPipeline(graph, {
  rht: { tauMax: 5, aggregator: 'max' },
  da: { bonus: 0.5, malus: 0.3 },
});

const result = circa.analyze(anomalyData, ['CPU', 'Latency']);
// RHTScorer: fits regression β on normal data, computes residual z-scores on anomaly data
// DAScorer: corrects for anomaly propagation (descendant adjustment)
```

### Multi-Modal Fusion

```typescript
import { FusionAnalyzer } from '@agentix-e/causality-analyzer-pipeline';

const fusion = new FusionAnalyzer({ strategy: 'weighted' });
const fused = fusion.fuse(metricRCA, traceRCA, logRCA);
```

---

## 7. Causal Effect Estimation

### Backdoor Adjustment — When You Can Observe All Confounders

```typescript
import { adjustBackdoor } from '@agentix-e/causality-analyzer-pipeline';

const nodeIndex = new Map([
  ['Confounder', 0], ['Treatment', 1], ['Outcome', 2],
]);

const { ate, se, adjustors } = adjustBackdoor(
  graph, 'Treatment', 'Outcome', data, nodeIndex,
);

// ATE = 0.742 ± 0.128 (95% CI)
console.log(`${ate.toFixed(3)} ± ${(se * 1.96).toFixed(3)}`);
```

### Frontdoor Adjustment — When You Can't Observe Confounders But Have Mediators

```typescript
import { estimateFrontdoor } from '@agentix-e/causality-analyzer-pipeline';

const { ate, se } = estimateFrontdoor(graph, 'Treatment', 'Outcome', data, nodeIndex, ['Mediator']);
// ATE = β(T→M) × β(M→Y)
```

### Instrumental Variables — When Treatment Is Endogenous

```typescript
import { estimateIV } from '@agentix-e/causality-analyzer-pipeline';

// data layout: [IV, Treatment, Outcome]
const { ate, se } = estimateIV(data, 1/*treatment*/, 2/*outcome*/, 0/*IV*/);
```

### Propensity Score Matching — Best for Binary Treatment with Covariates

```typescript
import { estimatePSMatching } from '@agentix-e/causality-analyzer-pipeline';

const { ate, se } = estimatePSMatching(
  data, 2/*treatment*/, 3/*outcome*/, [0, 1]/*covariates*/,
  42/*optional seed*/,
);
```

### Doubly Robust — When You Want Both Propensity Score + Outcome Model

```typescript
import { estimateDoublyRobust } from '@agentix-e/causality-analyzer-pipeline';

const { ate, se } = estimateDoublyRobust(data, 2/*treatment*/, 3/*outcome*/, [0, 1]/*covariates*/);
```

### IPW — When Treatment Groups Are Unbalanced

```typescript
import { estimateIPW } from '@agentix-e/causality-analyzer-pipeline';

const { ate, se } = estimateIPW(data, 0/*treatment*/, 1/*outcome*/);
```

### CATE — When Different Instances Have Different Effects

```typescript
import { estimateCATE } from '@agentix-e/causality-analyzer-pipeline';

const { cateFn, baselineATE } = estimateCATE(data, 1/*treatment*/, 2/*outcome*/, [0]/*features*/);

const effectForHighLoad = cateFn([0.9]);
const effectForLowLoad = cateFn([0.1]);
```

---

## 8. Sensitivity & Refutation

### E-value — How Strong Would an Unmeasured Confounder Need to Be?

```typescript
import { eValueSensitivity } from '@agentix-e/causality-analyzer-pipeline';

const { eValue, interpretation } = eValueSensitivity(0.8);
// "E-value=4.22: strong robustness — only very strong unmeasured confounding..."

// Interpret E-values:
//   > 3: strong robustness (confounder needs RR > 3 to explain away)
//   1.5-3: moderate robustness
//   < 1.5: weak — small confounders could change conclusion
```

### Partial R² — What Fraction of Variance Must a Confounder Explain?

```typescript
import { partialRSensitivity } from '@agentix-e/causality-analyzer-pipeline';

const { r2Treatment, r2Outcome, interpretation } = partialRSensitivity(0.8, 0.1, 1000);
// r2Treatment > 0.1 → robust; r2Treatment < 0.01 → highly sensitive
```

### Robustness Value — Combined Metric

```typescript
import { robustnessValue } from '@agentix-e/causality-analyzer-pipeline';

const { rv, interpretation } = robustnessValue(0.8, 0.1, 1000);
// RV > 2: ROBUST, RV > 1.5: MODERATE, RV < 1.5: SENSITIVE
```

### Refutation Tests

```typescript
import {
  refutePlaceboTreatment, refuteDataSubset, refuteBootstrap,
} from '@agentix-e/causality-analyzer-pipeline';

// Placebo: scramble treatment → effect should disappear
const placebo = refutePlaceboTreatment(data, 0, 1, 50, 42/*seed*/);
console.log(placebo.isRobust); // true = conclusion is robust

// Data subset: check stability across random subsets
const subset = refuteDataSubset(data, 0, 1, 0.8, 20, 42/*seed*/);

// Bootstrap: resample with replacement
const boot = refuteBootstrap(data, 0, 1, 100, 42/*seed*/);
```

---

## 9. Counterfactual Analysis

### What-If Reasoning

```typescript
import { CausalGraph, StructuralCausalModel } from '@agentix-e/causality-analyzer-pipeline';

// Step 1: Train SCM on normal data
const scm = new StructuralCausalModel(graph);
scm.train(normalData);

// Step 2: Abduction — infer noise from an anomalous observation
const observation = { Memory: 0.5, CPU: 0.95, Latency: 350 };
const noise = scm.abduct(observation);

// Step 3: Counterfactual — "What if we had allocated more memory?"
const cf = scm.counterfactual(noise, { Memory: 1.5 });
console.log(cf.CPU);      // Predicted CPU if memory were 1.5
console.log(cf.Latency);  // Predicted latency if memory were 1.5
```

### Shapley-Based Root Cause Attribution

```typescript
import { shapleyAttribute } from '@agentix-e/causality-analyzer-pipeline';

// Compute Shapley values for each node's contribution to the anomaly
const rootCauses = shapleyAttribute(scm, observation, 5, 42/*seed*/);

rootCauses.forEach(rc => {
  console.log(`${rc.name}: Shapley=${rc.score.toFixed(3)}, confidence=${rc.confidence}`);
});
```

### Mechanism Change Detection

```typescript
import { detectMechanismChanges } from '@agentix-e/causality-analyzer-pipeline';

const before = [/* observations before deployment */];
const after = [/* observations after deployment */];

const changes = detectMechanismChanges(scm, before, after);
for (const c of changes) {
  if (c.changed) console.log(`${c.node}: mechanism changed (p=${c.pValue.toFixed(3)})`);
}
```

---

## 10. Model Evaluation & Validation

### Graph Validation

```typescript
import { falsifyGraph, lmcFalsification } from '@agentix-e/causality-analyzer-pipeline';

// Is my causal graph consistent with the data?
const result = falsifyGraph(graph, data, nodeNames);
if (result.falsified) {
  console.log(`Missing edges:`, result.missingEdges);
  console.log(`Spurious edges:`, result.spuriousEdges);
}

// Per-node Markov condition check
const lmc = lmcFalsification(graph, data, nodeNames);
```

### Model Fit Assessment

```typescript
import { evaluateMechanismR2, evaluateMSE } from '@agentix-e/causality-analyzer-pipeline';

// R² per node — how well does each mechanism fit the data?
const r2s = evaluateMechanismR2(scm, data, nodeMap);
console.log('CPU R²:', r2s.get('CPU'));  // 1 = perfect, 0 = mean only

// MSE per mechanism
const mses = evaluateMSE(scm, data, nodeMap);
```

### Auto Mechanism Selection

```typescript
import { autoAssignMechanisms } from '@agentix-e/causality-analyzer-pipeline';

// Automatically pick linear, PNL, or empirical per node
const assignments = autoAssignMechanisms(graph, data, nodeNames);
for (const [node, info] of assignments) {
  console.log(`${node}: ${info.type} (R²=${info.r2.toFixed(2)})`);
}
```

### Confidence Intervals

```typescript
import { bootstrapRCA, bootstrapATE } from '@agentix-e/causality-analyzer-pipeline';

// For RCA scores
const rcaCI = bootstrapRCA(scm, observations, 200, 0.05, 42);

// For any ATE estimator
const ateCI = bootstrapATE(data, d => adjustBackdoor(graph, 'T', 'Y', d, nodeIndex).ate, 200, 0.05, 42);
console.log(`ATE: ${ateCI.ate.toFixed(3)} [${ateCI.ciLow.toFixed(3)}, ${ateCI.ciHigh.toFixed(3)}]`);
```

---

## 11. Data Storage

### Embedded (Zero Configuration)

```typescript
// SQLite for relational data — perfect for development and single-node deployment
import { EmbedRelationalStore } from '@agentix-e/causality-analyzer-storage-embed';
const store = new EmbedRelationalStore({ dbPath: ':memory:' });

// OverGraph for causal graphs
import { EmbedGraphStore } from '@agentix-e/causality-analyzer-storage-embed';
const graphStore = new EmbedGraphStore({ dbPath: './graphs' });
```

### Remote (Enterprise)

```typescript
// PostgreSQL — production, replication, backups
import { RemoteRelationalStore } from '@agentix-e/causality-analyzer-storage-remote';
const pgStore = new RemoteRelationalStore({
  connectionString: 'postgresql://user:pass@host:5432/db',
  mtls: { cert: pemCert, key: pemKey, ca: pemCA },
});

// Neo4j — graph-native storage with Bolt protocol
import { RemoteGraphStore } from '@agentix-e/causality-analyzer-storage-remote';
const neoStore = new RemoteGraphStore({
  uri: 'neo4j+s://host:7687',
  auth: { type: 'basic', user: 'neo4j', password: 'secret' },
  mtls: { cert: pemCert, key: pemKey },
  maxPoolSize: 8,
});
```

### Switching Backends

All stores implement the same interfaces (`IRelationalStore`, `IGraphStore`), so you can switch with zero code changes:

```typescript
function runAnalysis(store: IRelationalStore) { /* ... */ }

// Dev
runAnalysis(new EmbedRelationalStore({ dbPath: ':memory:' }));

// Prod
runAnalysis(new RemoteRelationalStore({ connectionString: '...', mtls: {...} }));
```

---

## 12. Visualization

### Causal Graph

```html
<ca-causal-graph></ca-causal-graph>
```

```typescript
const el = document.querySelector('ca-causal-graph');
el.data = buildGraphVizData(nodes, edges, rootCauses, anomalousNodes);
```

### Time Series with Anomaly Bands

```html
<ca-time-series></ca-time-series>
```

```typescript
const el = document.querySelector('ca-time-series');
el.data = buildTimeseriesVizData(metricData, timestamps, anomalousIndices, 'Memory');
// Anomaly regions render automatically as shaded bands
```

### Root Cause Ranking

```html
<ca-root-cause-ranking></ca-root-cause-ranking>
```

```typescript
const el = document.querySelector('ca-root-cause-ranking');
el.data = buildRankingVizData(rootCauses, paths);
```

---

## 13. Enterprise Deployment

### CI/CD

```yaml
# .github/workflows/ci.yml — quality gates on every PR
lint → typecheck (5 packages) → unit tests → browser tests → Neo4j mTLS tests
```

### mTLS Configuration

Both Bolt and PG-wire backends use the same PEM-based config:

```typescript
const mtls: MtlsConfig = {
  ca: fs.readFileSync('/etc/ssl/ca.crt', 'utf8'),
  cert: fs.readFileSync('/etc/ssl/client.crt', 'utf8'),
  key: fs.readFileSync('/etc/ssl/client.key', 'utf8'),
};
```

### Reproducibility in Regulated Environments

```typescript
// All stochastic algorithms accept seed for audit-trail reproducibility
const HARDCODED_SEED = 0xBEEF;

const result = shapleyAttribute(scm, obs, 5, HARDCODED_SEED);
const ci = bootstrapRCA(scm, obs, 200, 0.05, HARDCODED_SEED);
const placebo = refutePlaceboTreatment(data, 0, 1, 50, HARDCODED_SEED);
```

---

## 14. Troubleshooting & FAQ

### "My discovered graph has wrong edges"

1. Increase sample size — PC/FCI need adequate data per conditioning set
2. Lower alpha (try 0.01 for more conservative discovery)
3. Enable stable-PC variant: `{ stable: true }`
4. Use FCI if you suspect latent confounders
5. Validate with `falsifyGraph` to check consistency

### "RCA returns unexpected root causes"

1. Verify the causal graph is a DAG — use `falsifyGraph`
2. Check for collider bias: conditioning on a collider opens non-causal paths
3. Try multiple RCA methods — if they disagree, investigate the graph structure
4. Use `shapleyAttribute` for model-based attribution (requires trained SCM)

### "ATE has huge standard error"

1. Increase sample size (more data = smaller SE)
2. Add relevant covariates to reduce residual variance
3. Try Doubly Robust (combines PS + outcome model for efficiency)
4. Use `bootstrapATE` to get accurate CIs
5. Check for extreme propensity scores (< 0.05 or > 0.95) — trim with IPW

### "Sensitivity analysis says my result is fragile"

1. This is actually good — it means the tool is honest
2. Add more covariates to your adjustment set
3. Use `refutePlaceboTreatment` with more simulations
4. Consider using IV if treatment endogeneity is the issue
5. NEVER ignore a fragile result — it means you need more data or better controls

### "Do I need PostgreSQL and Neo4j?"

No. For development and single-node production:
- Use `EmbedRelationalStore` (SQLite)
- Use `EmbedGraphStore` (OverGraph)

Remote stores are for enterprise deployments that need replication, failover, or existing PostgreSQL/Neo4j infrastructure.
