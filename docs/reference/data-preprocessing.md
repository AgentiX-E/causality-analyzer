# Data Preprocessing

## Standardization

Prepares metric data for causal analysis by centering and scaling. Supports three methods with different robustness properties.

### Methods

| Method | Formula | Use When |
|--------|---------|----------|
| `zscore` | `(x - μ) / σ` | Normally distributed metrics |
| `minmax` | `(x - min) / (max - min)` | Bounded outputs needed [0,1] |
| `robust` | `(x - median) / (Q3 - Q1)` | Outlier-heavy metrics |

### Scenario: Prepare latency and CPU for PC algorithm

```typescript
import { ColumnarTable } from '@agentix-e/causality-analyzer-core';
import { standardize } from '@agentix-e/causality-analyzer-pipeline';

const table = ColumnarTable.fromRows([
  { cpu: 0.5, latency: 100 },
  { cpu: 0.8, latency: 250 },
  { cpu: 0.9, latency: 500 },
]);

const normalized = standardize(table, { method: 'zscore' });
// All columns now have μ≈0, σ≈1
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `'zscore'` | Standardization method |
| `columns` | all | Which columns to transform |

---

## Discretization

Bins continuous values into integer labels for CPT estimation and Bayesian inference.

### Scenario: Discretize latency into 5 bins for root cause analysis

```typescript
import { discretize } from '@agentix-e/causality-analyzer-pipeline';

const binned = discretize(table, { bins: 5 });
// Aligned with standardize: each row gets a replacement array
```

---

## Imputation

Fills missing (NaN) values with column means.

```typescript
import { imputeMean } from '@agentix-e/causality-analyzer-pipeline';

const clean = imputeMean(table);
```

---

## Windowing

Extracts sliding windows for time-series analysis.

```typescript
import { extractWindows } from '@agentix-e/causality-analyzer-pipeline';

for (const window of extractWindows(table, { size: 10, step: 5 })) {
  // Process each 10-row window with 5-row overlap
}
```

[← Back to User Guide](../user-guide.md)
