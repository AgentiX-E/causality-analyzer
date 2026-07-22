# Root Cause Analysis

## BayesianRCA

**Family:** Probabilistic graphical model  
**Method:** Variable elimination (exact inference)

Models the system as a Bayesian network. During training, estimates Conditional Probability Tables (CPT) from data using data-driven anomaly thresholds (>2.5σ). During inference, computes posterior probability P(root=1 | anomalous evidence) via simplified likelihood propagation.

### When to Use

- Known causal graph topology
- Need probabilistic confidence scores
- System fits well with linear relationships

### Scenario: Memory → CPU → Latency chain

```typescript
import { CausalGraph, BayesianRCA } from '@agentix-e/causality-analyzer-pipeline';

const graph = new CausalGraph(['Memory', 'CPU', 'Latency']);
graph.addEdge('Memory', 'CPU');
graph.addEdge('CPU', 'Latency');

const rca = new BayesianRCA();
rca.train(graph, new Set(['CPU', 'Latency']), data); // CPTs from data
const result = rca.findRootCauses(['CPU', 'Latency']);

// result.rootCauses[0].name === 'Memory' (only root node)
// result.rootCauses[0].score — posterior probability
```

### Key Characteristics

- **Strengths:** Exact probabilistic inference, calibrated confidence scores
- **Limitations:** Requires CPT estimation from data; linear CPT for root nodes

---

## HTRCA (Hypothesis Testing RCA)

**Family:** Regression-based  
**Method:** OLS residual analysis

Fits linear regression models X_i = β₀ + Σβⱼ × PAⱼ on normal data. During failure, computes residual z-scores — nodes whose observed values deviate significantly from their parent-based predictions are flagged.

### When to Use

- Known causal graph
- Need to detect which specific node behavior changed
- Want statistical significance for each flag

### Scenario: Detecting regression residual spikes

```typescript
import { HTRCA } from '@agentix-e/causality-analyzer-pipeline';

const rca = new HTRCA();
rca.train(graph, normalData); // fit regressions on normal period

// During failure
const result = rca.findRootCauses(['CPU', 'Latency'], failureData);
// Each node gets a z-score: how many σ from predicted value
```

### Key Characteristics

- **Strengths:** Interpretable z-scores, per-node evidence
- **Limitations:** Linear relationships, sensitive to outliers in training

---

## RandomWalkRCA

**Family:** Graph traversal  
**Method:** Weighted random walk with restart

Simulates random walks starting from anomalous nodes, walking upstream (toward parents). Root nodes (no incoming edges) visited most frequently are identified as root causes. Now supports seeded reproducibility.

### When to Use

- Known causal graph
- Quick screening (no training data needed)
- Need deterministic results (set seed)

### Scenario: Reproducible random walk

```typescript
import { RandomWalkRCA } from '@agentix-e/causality-analyzer-pipeline';

const rca = new RandomWalkRCA(42); // seed for reproducibility
rca.train(graph); // sets edge weights
const result = rca.findRootCauses(['CPU', 'Latency'], 10/*steps*/, 1000/*repeats*/);
// Same result every time with same seed
```

### Key Characteristics

- **Strengths:** No training needed, fast, reproducible
- **Limitations:** Edge weights uniform unless specified; heuristic confidence

---

## FPGrowthRCA

**Family:** Pattern mining  
**Method:** FP-Growth frequent itemset mining

Analyzes trace data (service invocation patterns) to find frequently co-occurring anomaly patterns. The FP-Tree structure enables efficient mining of multi-item frequent patterns.

### When to Use

- Trace data available (service invocation logs)
- Need pattern-based RCA (not metric-based)
- Suspicious patterns involving multiple services

### Scenario: Mining anomalous trace patterns

```typescript
import { FPGrowthRCA } from '@agentix-e/causality-analyzer-pipeline';

const transactions = traceData.map(trace =>
  new Set(trace.services.filter(s => s.anomalous))
);
const rca = new FPGrowthRCA(0.1); // minSupport = 10%
const result = rca.analyze(graph, transactions, abnormalTraces);
// Finds multi-service anomaly patterns
```

---

## CIRCA Pipeline

**Family:** Causal inference + scoring  
**Reference:** Li et al. (KDD 2022). *Causal Inference-Based Root Cause Analysis*

Two-stage pipeline:
1. **RHTScorer** (Regression-based Hypothesis Testing): Fits regression on normal data, scores nodes by residual deviation during failure.
2. **DAScorer** (Descendant Adjustment): Corrects scores for anomaly propagation — nodes whose anomalous parents explain their deviation get penalty.

### When to Use

- Unknown or uncertain causal graph
- Need topological correction for anomaly propagation
- Production systems with SLI-based analysis

### Scenario: KDD 2022-style analysis

```typescript
import { CIRCAPipeline } from '@agentix-e/causality-analyzer-pipeline';

const circa = new CIRCAPipeline(graph, {
  rht: { tauMax: 5, aggregator: 'max' },
  da: { bonus: 0.5, malus: 0.3 },
});

const result = circa.analyze(anomalyData, ['Latency']);
// RHTScorer computes z-scores; DAScorer adjusts for propagation
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rht.tauMax` | 5 | Maximum time lag for parent inclusion |
| `rht.aggregator` | `max` | Z-score aggregation: max/mean/sum |
| `da.bonus` | 0.5 | Bonus for root nodes |
| `da.malus` | 0.3 | Penalty for propagated anomalies |

---

## Choosing an RCA Method

| Criteria | Recommended |
|----------|------------|
| Known graph + need probabilities | BayesianRCA |
| Known graph + need per-node z-scores | HTRCA |
| Known graph + quick screening | RandomWalkRCA |
| Trace data with anomaly patterns | FPGrowthRCA |
| Uncertain graph + topological correction | CIRCA |
| Multiple methods disagree → consensus | `FusionAnalyzer` |

[← Back to User Guide](../user-guide.md)

---

## FusionAnalyzer

**Family:** Ensemble / Multi-modal fusion

Combines RCA results from multiple modalities (metric, trace, log) into a single consolidated analysis. Three strategies available for different use cases.

### Strategies

| Strategy | Behavior | Best For |
|----------|----------|----------|
| `weighted` | Weighted average by modality confidence | Balanced integration with domain weights |
| `nested` | Metric RCA defines scope → Trace RCA refines within scope | Hierarchical analysis (coarse → fine) |
| `voting` | Majority vote across modalities | Conservative — requires cross-modal agreement |

### Scenario: Combine Metric + Trace RCA

```typescript
import { FusionAnalyzer } from '@agentix-e/causality-analyzer-pipeline';

const fusion = new FusionAnalyzer({
  strategy: 'weighted',
  weights: { metric: 0.5, trace: 0.35, log: 0.15 },
});

const consolidated = fusion.fuse(metricRCA, traceRCA, logRCA);
console.log(consolidated.rootCauses); // Top-5 across all modalities
```

### When to Use Each Strategy

| Situation | Strategy |
|-----------|----------|
| All modalities equally reliable | `weighted` with equal weights |
| Metric RCA is more trustworthy | `weighted` with high metric weight |
| Want fine-grained trace analysis within metric scope | `nested` |
| Require consensus across modalities | `voting` |

[← Back to User Guide](../user-guide.md)
