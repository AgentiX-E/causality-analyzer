# Causal Discovery

## PC Algorithm

**Family:** Constraint-based  
**Reference:** Spirtes, Glymour & Scheines (2000). *Causation, Prediction, and Search.*

The Peter-Clark algorithm discovers causal structure from observational data by iteratively testing conditional independence between variable pairs. It produces a Completed Partially Directed Acyclic Graph (CPDAG) representing the Markov equivalence class.

### Algorithm Steps

1. **Skeleton estimation:** Start with complete undirected graph. For each pair (i,j), test i ⟂ j | S for conditioning sets S of increasing size. Remove edge if independence found.
2. **V-structure orientation:** For each unshielded triple i—k—j, if k was NOT in the separating set of (i,j), orient as i → k ← j.
3. **Meek's rules (R1-R3):** Apply orientation propagation rules to maximize directed edges.

### Scenarios

**Scenario 1: Full causal discovery (no prior knowledge)**

```typescript
import { Matrix } from 'ml-matrix';
import { pcAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

const data = new Matrix(1000, 5); // 1000 obs × 5 variables
const { graph, sepSet } = pcAlgorithm(data, ['CPU', 'Mem', 'Disk', 'Net', 'Latency'], {
  alpha: 0.05,    // significance threshold
  stable: true,   // stable-PC for order-independent results
  maxDegree: -1,  // unlimited conditioning set
});
```

**Scenario 2: Conservative discovery (fewer false positives)**

```typescript
const { graph } = pcAlgorithm(data, nodeNames, {
  alpha: 0.01,     // stricter threshold
  stable: true,
  maxDegree: 3,    // limit conditioning set size → faster
});
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.05 | Significance level for CI tests |
| `stable` | true | Stable-PC variant (recommended) |
| `maxDegree` | -1 | Max conditioning set size (-1 = unlimited) |

---

## FCI Algorithm

**Family:** Constraint-based (with latent confounders)  
**Reference:** Zhang (2008). *On the completeness of orientation rules*

Fast Causal Inference extends PC to handle latent (unobserved) confounders. It produces a Partial Ancestral Graph (PAG) that distinguishes direct causation from latent confounding.

### Edge Types in PAG

| Edge | Meaning |
|------|---------|
| A → B | A is an ancestor of B |
| A ↔ B | A and B share a latent common cause |
| A ∘→ B | A is not a descendant of B |
| A ∘–∘ B | No information about relationship |

### Scenarios

**Scenario: Microservice dependency with shared infrastructure**

```typescript
import { fciAlgorithm } from '@agentix-e/causality-analyzer-pipeline';

// 4 services sharing a kernel/network layer (latent)
const { pagEdges } = fciAlgorithm(data, ['svcA', 'svcB', 'svcC', 'svcD'], { alpha: 0.05 });

for (const [key, type] of pagEdges) {
  if (type.includes('↔')) {
    console.log(`${key}: likely shares latent cause (kernel, network, etc.)`);
  } else if (type.includes('→')) {
    console.log(`${key}: direct causal link`);
  }
}
```

### Additional Orientation Rules

FCI uses orientation rules R4-R10 (beyond PC's R1-R3) for discriminating paths and uncovering additional direction information in the presence of latent variables.

---

## Targeted Discovery

**Family:** Constraint-based (targeted)  
**Reference:** Grow-Shrink + conditional independence filtering

Instead of discovering the full causal graph, finds only the causal parents of specified target variables. Uses Grow-Shrink Markov blanket discovery as a pre-filter.

### When to Use

- Only care about what causes one metric (e.g., Latency)
- Large systems (100+ metrics) where full discovery is expensive
- Incremental discovery (add variables one at a time)

### Scenario: "What causes Latency?"

```typescript
import { targetedDiscovery } from '@agentix-e/causality-analyzer-pipeline';

const parents = targetedDiscovery(data, ['Latency'], allNodeNames);
console.log('Latency parents:', parents.get('Latency'));
// ['CPU', 'DiskIO', 'NetworkOut'] — only 3 variables instead of full graph
```

---

## Grow-Shrink

**Family:** Markov blanket discovery  
**Reference:** Margaritis & Thrun (1999)

Discovers the Markov blanket of a target variable — the minimal set of variables that makes the target conditionally independent of all others.

### Algorithm

1. **Grow:** Add variables that are dependent on target given current blanket
2. **Shrink:** Remove variables that become independent given rest of blanket

### Scenario: "What do I need to know to predict CPU?"

```typescript
import { growShrink } from '@agentix-e/causality-analyzer-pipeline';

const blanket = growShrink(data, cpuIndex, nodeNames);
// Returns the Markov blanket: all variables needed to predict CPU
```

[← Back to User Guide](../user-guide.md)
