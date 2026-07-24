# Changelog

## [2.0.0] — 2026-07-24

### Enterprise Infrastructure (I5)

- **HTTP REST API**: 7 endpoints (health/ready/live/metrics/discover/analyze/estimate), zero external dependencies
- **Docker**: Multi-stage Dockerfile + docker-compose (pipeline + PostgreSQL + Neo4j)
- **Streaming**: `StreamingPipeline` with sliding-window online RCA
- **Model Serialization**: `CausalGraph.fromJSON()`, `StructuralCausalModel.toJSON()/fromJSON()`
- **Health Checks**: K8s-compatible liveness/readiness probes

### Algorithm Enhancement (I6)

- **NOTEARS**: Continuous-optimization DAG learning (Zheng et al., NeurIPS 2018)
- **L-BFGS / Adam**: Numerical optimizers in core package
- **Worker Threads**: `WorkerPool` for parallel computation

### Cross-validation (I7)

- **Benchmark Suite**: 4 canonical DAGs × 5 algorithms, SHD/TPR/FPR metrics
- **DoWhy Cross-validation**: 14 tests, backdoor set matches DoWhy on 5+ graph types
- **ATE Numerical Validation**: Effect estimates within tolerance on known-coefficient data

### Algorithm Correctness Fixes (I4)

- **Backdoor Criterion**: Unified implementation with d-separation verification; Fixed `||` → `&&` bug
- **C-Component**: Removed v-structure hack; only bidirected edges
- **ID Algorithm**: Recursive implementation with c-component factorization and hedge criterion
- **LLM Explainer**: DeepSeek API-powered NL explanations with graceful fallback
- **Coverage**: Lines 96.09% | Statements 96.09% | Functions 96.68% | Branches 86.04%

## [1.0.0] — 2026-07-23

### Breaking Changes

- **`BayesianRCA` → `HeuristicPathRCA`**: Old name kept as deprecated alias
- **d-separation reimplemented**: Strict Pearl (2009) d-separation
- **pdag2dag fixed**: Correctly implements Dor-Tarsi (1992)
- **IPW propensity scores**: IRLS logistic regression replaces constant marginal probability
- **MAD center**: Uses median (correct) instead of mean

### Added

- 7 causal discovery algorithms: PC, FCI, GES, LiNGAM, Grow-Shrink, KCI, targeted
- do-calculus ID algorithm (Shpitser & Pearl 2006)
- Backdoor/Frontdoor/IV/PS/Doubly Robust estimators
- SPOT/DSPOT streaming anomaly detection
- CIRCA root cause analysis pipeline
- Sensitivity analysis (E-value, partial R², robustness value)
- 5 Bayesian Network inference engines
- AuditLogger, MetricsRegistry (Prometheus), RateLimiter, EncryptedStore (AES-256-GCM)
- ASIA benchmark validation
