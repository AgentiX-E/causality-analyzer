# Backward Compatibility (COMPAT.md)

Causality Analyzer follows [Semantic Versioning 2.0.0](https://semver.org/).

## v2.0 Migration Guide

### RCA: HeuristicPathRCA → BayesianRCA

```typescript
// v1.0 (deprecated but still works)
const rca = new HeuristicPathRCA();
rca.train(graph, anomalies, data);

// v2.0 (recommended — true Bayesian inference)
const rca = new BayesianRCA({ engine: 'variable_elimination' });
rca.train(graph, data);
```

### SCM: train() signature

```typescript
// v1.0: only data
scm.train(data);

// v2.0: optional mechanism types (backward compatible)
scm.train(data);                          // auto-assign
scm.train(data, { Y: 'post_nonlinear' }); // manual specification
```

### Metrics: Prometheus format

```typescript
// v1.0: JSON output only
const counters = metrics.exportCounters();

// v2.0: add Prometheus text format
const promText = metrics.toPrometheus(); // scrape-ready text
```

### Rate Limiting: Token Bucket

```typescript
// v1.0: queue-based only
const limiter = new RateLimiter({ maxBufferSize: 100 });

// v2.0: add rate-based TokenBucket
const bucket = new TokenBucket({ rate: 100, capacity: 200 });
bucket.tryConsume(1); // returns { accepted, utilization }
```

### New Exports (no breaking changes)

| v2.0 Module | Exports |
|-------------|---------|
| `./infer/metalearners.js` | `sLearner, tLearner, xLearner, rLearner, upliftTree, upliftForest` |
| `./graph/notears.js` | `notearsAlgorithm, golemAlgorithm` |
| `./analyze/bayesian-rca.js` | `BayesianRCA` |
| `./gcm/structural-causal-model.js` | `CausalModel` (new unified API) |
| `./parallel.js` | `parallelMap, chunkedPC` |
| `./observability.js` | `HealthTracker` (new), `MetricsRegistry.toPrometheus()` |

### Deprecated (will be removed in v3.0)

- `HeuristicPathRCA` — use `BayesianRCA` instead
- `exportCounters()` / `exportHistograms()` — use `toPrometheus()` for production

### Supported Node.js Versions

| Version | Status |
|---------|--------|
| 20.x | ✅ Supported |
| 22.x | ✅ Supported (primary) |
| 24.x | ✅ Supported |
| 18.x | ❌ End of life |
