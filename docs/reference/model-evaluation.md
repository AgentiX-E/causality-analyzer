# Model Evaluation & Validation

## Mechanism Fit (R²)

Evaluates how well each causal mechanism explains its target variable. Uses OLS-based R² (coefficient of determination):

R² = 1 − SS_res / SS_tot

where SS_res is the residual sum of squares from the fitted SCM, and SS_tot is the total sum of squares from the mean.

### Scenario: "Which causal relationships are well-modeled?"

```typescript
import { evaluateMechanismR2 } from '@agentix-e/causality-analyzer-pipeline';

const r2s = evaluateMechanismR2(scm, data, nodeMap);
for (const [node, r2] of r2s) {
  console.log(`${node}: R²=${r2.toFixed(3)}`);
  if (r2 < 0.3) console.log('  → Consider nonlinear mechanism');
  if (r2 < 0) console.log('  → Model worse than mean — check graph structure');
}
```

---

## Mechanism Fit (MSE)

Mean Squared Error per mechanism — lower is better.

```typescript
import { evaluateMSE } from '@agentix-e/causality-analyzer-pipeline';

const mses = evaluateMSE(scm, data, nodeMap);
```

---

## Auto Mechanism Assignment

Automatically assigns the best mechanism type to each node based on linear R².

```typescript
import { autoAssignMechanisms } from '@agentix-e/causality-analyzer-pipeline';

const assignments = autoAssignMechanisms(graph, data, nodeNames);

// For PNL-assigned nodes, use fitLogisticPNL
import { fitLogisticPNL } from '@agentix-e/causality-analyzer-pipeline';
for (const [node, info] of assignments) {
  if (info.type === 'postnonlinear') {
    const pnl = fitLogisticPNL(data, nodeIdx, parentIdx);
  }
}
```

### Assignment Rules

| R² Range | Mechanism Type | Why |
|----------|---------------|-----|
| > 0.7 | Linear (AdditiveNoise) | Linear model fits well |
| 0.3 – 0.7 | PostNonlinear (sigmoid) | Some nonlinearity — logistic transform helps |
| < 0.3 | Empirical | Linear model fails — use non-parametric |

---

## Graph Falsification

Tests whether a causal graph is consistent with the data using conditional independence tests with Bonferroni correction.

### Scenario: "Is my manually specified graph correct?"

```typescript
import { falsifyGraph, lmcFalsification } from '@agentix-e/causality-analyzer-pipeline';

// Global falsification
const result = falsifyGraph(graph, data, nodeNames);
if (result.falsified) {
  console.log('Missing edges:', result.missingEdges);
  console.log('Spurious edges:', result.spuriousEdges);
}

// Per-node Markov condition check
const lmc = lmcFalsification(graph, data, nodeNames);
for (const [node, r] of lmc) {
  if (r.violated) console.log(`${node}: LMC violated — graph may be incorrect`);
}
```

---

## Bootstrap Confidence Intervals

Provides uncertainty quantification for any ATE estimator.

### Scenario: "How precise is my effect estimate?"

```typescript
import { bootstrapATE, bootstrapATEParallel } from '@agentix-e/causality-analyzer-pipeline';

// Sequential
const ci = bootstrapATE(data, d => adjustBackdoor(g, 'T', 'Y', d, idx).ate, 200, 0.05, 42);
console.log(`ATE: ${ci.ate.toFixed(3)} [${ci.ciLow.toFixed(3)}, ${ci.ciHigh.toFixed(3)}]`);

// Parallel (multi-threaded)
const ciPar = await bootstrapATEParallel(data, estimator, 400, 4, 0.05, 42);
```

### For RCA Scores

```typescript
import { bootstrapRCA } from '@agentix-e/causality-analyzer-pipeline';

const rcaCI = bootstrapRCA(scm, observations, 200, 0.05, 42);
for (const [node, ci] of rcaCI) {
  console.log(`${node}: ${ci.mean.toFixed(3)} [${ci.ciLow.toFixed(3)}, ${ci.ciHigh.toFixed(3)}]`);
}
```

---

## Feature Importance

### Parent Relevance

Shapley-based quantification of each parent's contribution to predictions.

```typescript
import { parentRelevance } from '@agentix-e/causality-analyzer-pipeline';

const relevance = parentRelevance(graph, data, nodeNames, 'Latency', 42);
// { 'CPU': 0.65, 'DiskIO': 0.25, 'Memory': 0.10 }
// → CPU is the dominant predictor of Latency
```

### Arrow Strength

Regression-based edge strength: |β_edge| / Σ|β_incoming|

```typescript
import { arrowStrength } from '@agentix-e/causality-analyzer-pipeline';

const strengths = arrowStrength(graph, data, nodeNames);
// { 'Memory→CPU': 0.8, 'CPU→Latency': 0.65, 'Memory→Latency': 0.35 }
```

---

## Distribution Change Attribution

When the distribution of observations shifts, attributes the shift to specific mechanisms.

```typescript
import { distributionChangeRobust, changeAttributionCI } from '@agentix-e/causality-analyzer-pipeline';

const attrib = distributionChangeRobust(scm, before, after);
for (const [node, a] of attrib) {
  console.log(`${node}: ${(a.contribution * 100).toFixed(1)}% of total shift`);
}

// With confidence intervals
const ci = changeAttributionCI(scm, before, after, 200, 42);
```

[← Back to User Guide](../user-guide.md)
