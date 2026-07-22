# Anomaly Detection

## StatsDetector

**Family:** Statistical / Parametric  
**Reference:** Standard Z-score, MAD, and IQR methods

Detects anomalies by comparing each incoming metric value to a learned baseline distribution. Three methods are available, each with different robustness properties.

### Methods

| Method | Formula | Best For | Robust to Outliers |
|--------|---------|----------|-------------------|
| `zscore` | `|x - μ| / σ` | Normally distributed metrics | No |
| `mad` | `1.4826 × |x - median| / MAD` | Metrics with occasional spikes | Yes |
| `iqr` | `(x - Q1) / (Q3 - Q1)` | Non-parametric, any distribution | Yes |

### Scenarios

**Scenario 1: CPU monitoring (normal distribution) → use `zscore`**

```typescript
const detector = new StatsDetector({ method: 'zscore' });
detector.train(cpuHistory); // 100+ normal observations
const result = detector.update([currentCPU]);
// Anomalous if |z| > 3 (3σ rule)
```

**Scenario 2: Latency monitoring (long-tailed) → use `mad`**

```typescript
const detector = new StatsDetector({ method: 'mad' });
detector.train(latencyHistory); // median ± 1.4826 × MAD
// MAD is resistant to the occasional extreme latency spike
```

**Scenario 3: Throughput monitoring (unknown distribution) → use `iqr`**

```typescript
const detector = new StatsDetector({ method: 'iqr' });
detector.train(throughputHistory);
// Non-parametric: works for any distribution shape
```

### When NOT to Use

- Periodic/seasonal data → use `SpectralResidualDetector` instead
- Rare extreme events → use `SPOTDetector` instead
- Need ensemble consensus → use `VotingDetector` instead

---

## SpectralResidualDetector

**Family:** Signal Processing / Frequency Domain  
**Reference:** Ren et al. (KDD 2019), "Time-Series Anomaly Detection Service at Microsoft"

Transforms the time series into the frequency domain via FFT, computes the spectral residual (difference between log spectrum and its smoothed version), and detects anomalies via saliency mapping.

### When to Use

- Metrics with periodic/seasonal patterns (daily, hourly cycles)
- When point-based detectors miss contextual anomalies
- Streaming and batch modes both supported

### Scenarios

**Scenario: Daily traffic pattern anomaly**

```typescript
const detector = new SpectralResidualDetector({ windowSize: 64 });
detector.init(historicalTraffic); // learn baseline spectrum

// Streaming mode
for (const point of liveTraffic) {
  const result = detector.update(point);
  if (result.isAnomalous) alert('Unusual traffic pattern');
}

// Batch mode
const results = detector.detect(batchTraffic);
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `windowSize` | 64 | FFT window size (power of 2) |
| `threshold` | 3 | Anomaly score threshold |

---

## SPOTDetector / DSPOTDetector

**Family:** Extreme Value Theory  
**Reference:** Siffer et al. (KDD 2017), "Anomaly Detection in Streams with Extreme Value Theory"

Models the tail distribution of a metric using the Generalized Pareto Distribution (GPD) via the Peaks-Over-Threshold method. SPOT handles stationary tails; DSPOT adds drift adaptation via exponential moving average.

### When to Use

- Rare but extreme events (P99.9+ anomalies)
- Metrics with heavy-tailed distributions
- DSPOT: when the baseline is slowly drifting (e.g., memory leak, gradual degradation)

### Scenarios

**Scenario 1: P99.9 latency threshold**

```typescript
const detector = new SPOTDetector({ initSize: 1000, q: 1e-3 });
```

**Scenario 2: Gradual memory leak**

```typescript
import { DSPOTDetector } from '@agentix-e/causality-analyzer-pipeline';
const detector = new DSPOTDetector({ driftWindow: 200, q: 1e-4 });
// Automatically subtracts local EMA before threshold comparison
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initSize` | 1000 | Number of observations for initial GPD fit |
| `q` | 1e-4 | Quantile threshold (lower = more conservative) |
| `driftWindow` | 100 | (DSPOT only) Window for EMA drift estimation |

---

## VotingDetector

**Family:** Ensemble / Meta-learning

Combines multiple independent detectors into a consensus vote, reducing false positives from any single detector.

### Strategies

| Strategy | Behavior |
|----------|----------|
| `majority` | Flags anomalous if ≥ `minAgreement` detectors agree |
| `maximum` | Selects the detector with the highest anomaly score |
| `weighted` | Weighted average of detection scores |

### Scenarios

**Scenario: Multi-signal monitoring with consensus**

```typescript
const ensemble = new VotingDetector({
  detectors: [
    new StatsDetector({ method: 'zscore' }),
    new SpectralResidualDetector({ windowSize: 64 }),
    new SPOTDetector({ q: 1e-4 }),
  ],
  strategy: 'majority',
  minAgreement: 2, // at least 2/3 must agree
});
```

**Scenario: Weighted by domain expertise**

```typescript
const ensemble = new VotingDetector({
  detectors: [statsDetector, srDetector, spotDetector],
  strategy: 'weighted',
  weights: { statsDetector: 0.5, srDetector: 0.3, spotDetector: 0.2 },
});
```

[← Back to User Guide](../user-guide.md)
