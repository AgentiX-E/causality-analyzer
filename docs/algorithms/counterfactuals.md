# Counterfactual Analysis

## Structural Causal Model (SCM)

**Reference:** Pearl (2009). *Causality: Models, Reasoning, and Inference*

A Structural Causal Model represents each variable X_i as a function of its parents plus independent noise:

X_i = f_i(PA_i, ε_i)

Causality Analyzer implements Pearl's three-step counterfactual framework:

1. **Abduction** — Given an observation, infer the noise terms ε
2. **Action** — Apply the do-operator (set variables to intervention values)
3. **Prediction** — Forward-simulate with inferred noise

### Scenario: "What would have happened if..."

```typescript
import { CausalGraph, StructuralCausalModel } from '@agentix-e/causality-analyzer-pipeline';

// Step 0: Train SCM on normal-period data
const scm = new StructuralCausalModel(graph);
scm.train(normalData);

// Step 1: Abduction — infer noise from anomalous observation
const observation = { Memory: 0.3, CPU: 0.95, Latency: 450 };
const noise = scm.abduct(observation);
// noise = { Memory: ε_M, CPU: ε_C, Latency: ε_L }

// Step 2+3: Counterfactual — "What if Memory were 1.5 instead?"
const cf = scm.counterfactual(noise, { Memory: 1.5 });
console.log(cf.CPU);      // Predicted CPU at Memory=1.5
console.log(cf.Latency);  // Predicted Latency at Memory=1.5
```

### Mechanism Types

| Type | Formula | When Auto-Assigned |
|------|---------|-------------------|
| Additive Noise | X = β₀ + ΣβPA + ε | R² > 0.7 |
| PostNonlinear | X = sigmoid(β₀ + ΣβPA + ε) | 0.3 ≤ R² ≤ 0.7 |
| Empirical | Non-parametric quantiles | R² < 0.3 |

---

## Shapley Anomaly Attribution

**Family:** Game-theoretic  
**Reference:** Budhathoki et al. (ICML 2022). *Causal structure-based root cause analysis*

Quantifies each node's marginal contribution to the total anomaly using Shapley values from cooperative game theory.

### Scenario: Fair attribution of anomaly responsibility

```typescript
import { shapleyAttribute } from '@agentix-e/causality-analyzer-pipeline';

const rootCauses = shapleyAttribute(scm, observation, 5, 42/*seed*/);
// Each node gets a Shapley value: fair fraction of total anomaly
// Reproducible with seed; Monte Carlo permutation approximation

rootCauses.forEach(rc => {
  console.log(`${rc.name}: Shapley=${rc.score.toFixed(3)}, conf=${rc.confidence}`);
});
```

---

## Mechanism Change Detection

Detects whether the causal mechanism itself changed (e.g., deployment, config change) vs. normal data drift.

### Scenario: "Is the deployment causing problems, or is it just data drift?"

```typescript
import { detectMechanismChanges } from '@agentix-e/causality-analyzer-pipeline';

const before = [/* observations before deployment */];
const after = [/* observations after deployment */];

const changes = detectMechanismChanges(scm, before, after);

for (const c of changes) {
  if (c.changed) {
    console.log(`${c.node}: MECHANISM CHANGED (p=${c.pValue.toFixed(4)}, Z=${c.zScore.toFixed(2)})`);
    // This suggests the deployment changed how this node behaves
  }
}
```

### Interpreting Results

- `changed=true, noiseShift>0`: Mechanism now produces higher values than predicted
- `changed=true, noiseShift<0`: Mechanism now produces lower values
- `changed=false`: No evidence of mechanism change (likely data drift, not deployment issue)

[← Back to User Guide](../user-guide.md)

---

## Counterfactual Fairness

**Reference:** Kusner et al. (NeurIPS 2017). *Counterfactual Fairness*

Ensures that RCA decisions are fair across protected groups (teams, regions, instance types). A decision is counterfactually fair if it would have been the same had the protected attribute been different, given all other observed variables.

### Scenario: Is the RCA biased against a specific team?

```typescript
import { checkFairness } from '@agentix-e/causality-analyzer-pipeline';

const rootCauses = [
  { name: 'team-a-svc1', score: 0.9 },
  { name: 'team-a-svc2', score: 0.85 },
  { name: 'team-b-svc1', score: 0.3 },
  { name: 'team-b-svc2', score: 0.2 },
];

const protectedGroups = {
  'team-a': ['team-a-svc1', 'team-a-svc2'],
  'team-b': ['team-b-svc1', 'team-b-svc2'],
};

const fairness = checkFairness(rootCauses, protectedGroups);
console.log(fairness.fair);           // false — significant disparity
console.log(fairness.disparity);      // > 0.5 — team-a services scored much higher
console.log(fairness.explanation);    // Human-readable fairness report
```

### Interpreting Results

| disparity | Interpretation |
|-----------|---------------|
| < 0.2 | Fair — no significant score disparity |
| 0.2–0.5 | Moderate — warrants investigation |
| > 0.5 | High — possible systematic bias, review RCA methodology |

[← Back to User Guide](../user-guide.md)
