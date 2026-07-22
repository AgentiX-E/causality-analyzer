# Causal Effect Estimation

## Backdoor Adjustment

**Family:** Identification + estimation  
**Reference:** Pearl (1995). *Causal diagrams for empirical research*

Computes the Average Treatment Effect (ATE) by adjusting for variables that satisfy the backdoor criterion. Pearl's criterion: Z blocks all backdoor paths from X to Y without opening new paths. Z must not contain descendants of X.

### Scenario: Controlled experiment equivalent from observational data

```typescript
import { adjustBackdoor, findBackdoorSet } from '@agentix-e/causality-analyzer-pipeline';

// Step 1: Find valid adjustors
const adjustors = findBackdoorSet(graph, 'Memory', 'Latency');
// ['CPU'] — blocks Memory ← ? → Latency paths

// Step 2: Estimate ATE
const nodeIndex = new Map([['Memory',0],['CPU',1],['Latency',2]]);
const { ate, se, adjustors: used } = adjustBackdoor(graph, 'Memory', 'Latency', data, nodeIndex);

console.log(`ATE = ${ate.toFixed(3)} ± ${(se * 1.96).toFixed(3)} (95% CI)`);
```

### Interpretability

- `ate > 0`: Increasing treatment increases outcome (e.g., more memory → higher latency? — check model)
- `ate < 0`: Increasing treatment decreases outcome (e.g., more memory → lower latency)
- `se`: Standard error → CI width

---

## Frontdoor Adjustment

**Family:** Identification (mediator-based)  
**Reference:** Pearl (1993). *Aspects of graphical models connected with causality*

When backdoor is blocked but a mediator is available:
P(Y|do(X)) = Σ_m P(m|x) Σ_x' P(Y|x',m) P(x')

### Scenario: Can't observe confounder, but mediator is observable

```typescript
import { estimateFrontdoor } from '@agentix-e/causality-analyzer-pipeline';

const mediators = ['DiskIO']; // observable mediator between Memory and Latency
const { ate, se } = estimateFrontdoor(graph, 'Memory', 'Latency', data, nodeMap, mediators);
// ATE = β(Memory→DiskIO) × β(DiskIO→Latency)
```

---

## Instrumental Variables (2SLS)

**Family:** Identification (instrument-based)  
**Reference:** Angrist & Imbens (1995). *Two-Stage Least Squares*

When treatment is endogenous (correlated with unobserved confounders), use an instrument Z that:
1. Affects treatment X
2. Does NOT directly affect outcome Y
3. Is independent of unobserved confounders

### Scenario: Randomized experiment proxy

```typescript
import { estimateIV } from '@agentix-e/causality-analyzer-pipeline';

// Z = randomized A/B test flag
// X = actual feature adoption (endogenous)
// Y = latency (outcome)
const { ate, se } = estimateIV(data, 1/*X*/, 2/*Y*/, 0/*Z*/, []/*covariates*/);
```

---

## Propensity Score Matching

**Family:** Semi-parametric  
**Reference:** Rosenbaum & Rubin (1983). *The central role of the propensity score*

1. Estimate propensity scores via logistic regression
2. Match each treated unit to nearest control by propensity score
3. Compute ATE = mean(Y_treated - Y_matched_control)

### Scenario: Binary treatment with many covariates

```typescript
import { estimatePSMatching, estimatePropensityScore } from '@agentix-e/causality-analyzer-pipeline';

// Estimate scores
const scores = estimatePropensityScore(data, treatmentIdx, covariateIndices);

// Nearest-neighbor matching
const { ate, se } = estimatePSMatching(data, treatmentIdx, outcomeIdx, covariateIndices, 42/*seed*/);
```

### When to Use

- Binary treatment (deployed vs not deployed)
- Many covariates → dimensionality reduction via propensity score
- Unbalanced groups (few failures, many normal periods)

---

## Doubly Robust

**Family:** Semi-parametric (doubly robust)  
**Reference:** Robins & Rotnitzky (1995). *Semiparametric efficiency*

Combines propensity score weighting with outcome regression. Consistent if EITHER the propensity model OR the outcome model is correctly specified — hence "doubly" robust.

### Scenario: Conservative ATE when model uncertainty is high

```typescript
import { estimateDoublyRobust } from '@agentix-e/causality-analyzer-pipeline';

const { ate, se } = estimateDoublyRobust(data, treatmentIdx, outcomeIdx, covariateIndices);
// DR = (1/n) Σ [μ₁(Z) − μ₀(Z) + T(Y−μ₁)/π − (1−T)(Y−μ₀)/(1−π)]
```

---

## IPW (Inverse Probability Weighting)

**Family:** Weighting estimator  

Reweights observations by inverse propensity score to create a pseudo-population where treatment is independent of covariates.

### Scenario: Rare failure events (unbalanced data)

```typescript
import { estimateIPW } from '@agentix-e/causality-analyzer-pipeline';

// 95% normal, 5% failure → IPW corrects the imbalance
const { ate, se } = estimateIPW(data, treatmentIdx, outcomeIdx, covariateIndices);
```

---

## CATE (Conditional ATE)

**Family:** Heterogeneous effect estimation  

Estimates how the treatment effect varies with covariates: CATE(x) = E[Y(1)-Y(0) | X=x]

### Scenario: "Which pods are most affected?"

```typescript
import { estimateCATE } from '@agentix-e/causality-analyzer-pipeline';

const { cateFn, baselineATE } = estimateCATE(data, treatmentIdx, outcomeIdx, [podCPUidx, podMemIdx]);

// Per-instance effects
const effectHighCPU = cateFn([0.9, 0.5]);  // high CPU pod
const effectLowCPU = cateFn([0.1, 0.5]);   // low CPU pod
```

---

## Choosing an Estimator

| Criteria | Recommended |
|----------|------------|
| Can observe all confounders | Backdoor adjustment |
| Can't observe confounders, have mediator | Frontdoor |
| Have a valid instrument | IV (2SLS) |
| Binary treatment + many covariates | Propensity Score Matching |
| Uncertain about model specification | Doubly Robust |
| Unbalanced groups | IPW |
| Need per-instance effects | CATE |

[← Back to User Guide](../user-guide.md)

---

## do-Calculus Identification

**Family:** Graph-theoretic  
**Reference:** Pearl (1995). *Causal diagrams for empirical research*

When backdoor, frontdoor, and IV all fail, do-calculus provides the most general framework for determining if an effect is identifiable. Applies Pearl's three rules systematically via the ID algorithm (Tian & Pearl 2002, Shpitser & Pearl 2006).

### Scenario: General identifiability check

```typescript
import { identifyByDoCalculus } from '@agentix-e/causality-analyzer-pipeline';

const result = identifyByDoCalculus(graph, 'Treatment', 'Outcome');
console.log(result.identifiable);      // true / false
console.log(result.expressionType);    // 'backdoor' | 'frontdoor' | 'id_algorithm'
console.log(result.adjustmentSet);     // variables needed for estimation
```

### How It Works

1. Check backdoor criterion → adjust if satisfied
2. Check frontdoor criterion → identify via mediators
3. Apply ID algorithm — recursive graph manipulation to find identifiable decomposition
4. Return `not_identifiable` if no method succeeds

---

## Mediation Analysis

**Family:** Path-specific effect decomposition  
**Reference:** Baron & Kenny (1986), Pearl (2001)

Decomposes the total causal effect into Natural Direct Effect (NDE) and Natural Indirect Effect (NIE) through a mediator.

### NDE / NIE

NDE = E[Y(x, M(x*)) − Y(x*, M(x*))]  
NIE = E[Y(x, M(x)) − Y(x, M(x*))]

### Scenario: How much of Memory's effect on Latency goes through CPU?

```typescript
import { naturalDirectEffect } from '@agentix-e/causality-analyzer-pipeline';

const result = naturalDirectEffect(data, memoryIdx, latencyIdx, cpuIdx);
console.log(`NDE: ${result.nde.toFixed(3)} (direct path)`);
console.log(`NIE: ${result.nie.toFixed(3)} (through CPU)`);
console.log(`Proportion mediated: ${(result.proportionMediated * 100).toFixed(0)}%`);
```

### Arrow Strength

Normalized edge strength: |β_edge| / Σ|β_incoming| — quantifies causal flow along each edge.

```typescript
import { arrowStrength } from '@agentix-e/causality-analyzer-pipeline';

const strengths = arrowStrength(graph, data, nodeNames);
// { 'Memory→CPU': 0.8, 'CPU→Latency': 0.6, 'Memory→Latency': 0.4 }
```

---

## Choosing an Identification Strategy

| Criteria | Method | Reference |
|----------|--------|-----------|
| Confounders observable | Backdoor adjustment | [↑ Backdoor](#backdoor-adjustment) |
| Mediator observable, confounders not | Frontdoor | [↑ Frontdoor](#frontdoor-adjustment) |
| Valid instrument available | IV (2SLS) | [↑ IV](#instrumental-variables-2sls) |
| None of the above — check automatically | do-Calculus | [↑ do-Calculus](#do-calculus-identification) |
| Decompose direct vs mediated effects | Mediation (NDE/NIE) | [↑ Mediation](#mediation-analysis) |
