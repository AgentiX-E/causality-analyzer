# Changelog

All notable changes to Causality Analyzer.

## [2.0.0] — 2026-07-24

### Major Features

#### Causal Discovery (7 algorithms)
- **PC** (constraint-based): Fisher Z + stable variant + Meek R1-R3
- **FCI** (latent confounders): Possible-D-SEP + Meek R1-R4
- **GES** (score-based, Chickering 2002): BIC-optimized forward/backward
- **LiNGAM** (non-Gaussian, Shimizu 2011): DirectLiNGAM causal ordering
- **NOTEARS** (continuous, Zheng 2018): Augmented Lagrangian + L1 proximal
- **GOLEM** (gradient, Ng 2020): Adam optimizer + log-det likelihood
- **KCI test**: kernel-based nonlinear CI testing (RBF + permutation p-value)

#### Root Cause Analysis (5 methods)
- **BayesianRCA**: 6-engine exact/approximate inference (VE/JT/LBP/LW/Gibbs/BF)
- **HeuristicPathRCA**: path-based heuristic scoring (deprecated in favor of BayesianRCA)
- **HTRCA**: hypothesis testing via OLS residuals
- **RandomWalkRCA**: weighted graph random walk
- **FPGrowthRCA**: frequent pattern mining trace RCA
- **CIRCA Pipeline**: RHT z-score + descending adjustment

#### Causal Effect Estimation (8 methods)
- Backdoor adjustment, Frontdoor, IV/2SLS, Propensity Score (IRLS), PSM, Doubly Robust
- **S-Learner**: single model with treatment-as-feature
- **T-Learner**: separate outcome models per arm
- **X-Learner**: cross-predict counterfactuals → meta-model
- **R-Learner**: orthogonalized residual-on-residual (Nie & Wager 2021)
- **UpliftTree**: decision tree maximizing |Δμ_L - Δμ_R|
- **UpliftForest**: bootstrap ensemble + OOB prediction + feature importance

#### SCM & Counterfactuals
- **PostNonlinearMechanism**: sigmoid-based nonlinear causal mechanism
- **Auto-assign**: R² heuristic for mechanism selection
- **Counterfactual reasoning**: Abduction-Action-Prediction (Pearl 2009)
- **Interventional samples**: Monte Carlo from P(Y|do(X))
- **Counterfactual samples**: noise-resampling distribution

#### do-Calculus (Shpitser & Pearl 2006)
- Pearl's 3 rules + ID algorithm
- Complete c-component decomposition
- Hedge criterion detection

#### Sensitivity Analysis
- E-value, partial R², robustness value with plain-English interpretation

#### Infrastructure
- **Licensed**: MIT
- **Security**: SECURITY.md with vulnerability reporting SLA
- **CODEOWNERS**: automated PR review routing
- **Prometheus metrics**: `MetricsRegistry.toPrometheus()` text format
- **Token Bucket**: rate limiting with burst/refill/waitTime
- **Health SLI**: p50/p90/p99 latency + error rate tracker
- **Key Rotation**: `EncryptedStore.rotateKey()` + `generateKey()`
- **.env support**: `BaseConfig.fromEnv()` with `CA_` prefix
- **Migration**: `migrate()` + `getSchemaVersion()` on embed store
- **mTLS**: full mTLS on Neo4j Bolt + PostgreSQL
- **Audit trail**: SHA-256 hash-chained (RFC 6962 pattern)
- **AES-256-GCM**: encrypted storage wrapper
- **Rate limiter**: drop_oldest/drop_newest/block strategies
- **Structured logging**: Logger interface + ConsoleLogger/NoopLogger
- **Error hierarchy**: 6 typed error classes with 20 error codes
- **Plugin registry**: decorator-based detector/graph/analyzer registration

#### CLI
- `discover`: 5 algorithm options with config file + env vars
- `analyze`: 3 RCA methods with external graph file support
- `benchmark`: 5 standardized performance tests
- `serve`: REST API with /health + /metrics (Prometheus)

#### Parallel Computing
- `parallelMap`: Worker Threads distributed task execution
- `parallelPermutationTest`: parallelized KCI permutation testing
- `chunkedPC`: chunked causal discovery with voting aggregation

#### Storage
- **Embed**: SQLite + OverGraph LSM-tree
- **Remote**: PostgreSQL + Neo4j with connection pooling
- **Migration**: idempotent schema versioning

#### Visualization
- `<ca-causal-graph>`: Canvas2D force-directed DAG (Lit 3)
- `<ca-time-series>`: uPlot-based time series with anomaly bands
- `<ca-root-cause-ranking>`: ranked list with keyboard navigation

#### Quality Infrastructure
- **1008 tests**: unit + integration + browser E2E + Neo4j mTLS + fuzz + perf
- **Zero @ts-nocheck**: strict type safety across all packages
- **CI/CD**: lint → typecheck → test → browser → Neo4j → audit (zero skip)
- **npm Provenance**: sigstore signatures on all releases
- **CodeQL**: continuous security scanning
- **Cross-platform**: ubuntu/macos/windows CI matrix

#### Conformance
- ASIA benchmark (Lauritzen & Spiegelhalter 1988): 8-node DAG validation
- Shpitser & Pearl (2006) Figures 1-5: do-calculus identification
- BayesianRCA engine verification: all 5+1 engines produce consistent rankings
- MetaLearner agreement: S/T/X/R learners agree on effect direction

### Breaking Changes (v1.0 → v2.0)

- **`BayesianRCA` becomes the default RCA engine** — `HeuristicPathRCA` remains available but deprecated
- **`StructuralCausalModel.train()` signature changed**: accepts optional `mechanismTypes` parameter
- **`MetricsRegistry`** enhanced with `toPrometheus()`, `setGauge()`, `setLabels()`
- **`RateLimiter`** now co-exists with new `TokenBucket` class
- **`EncryptedStore`** added `rotateKey()` and `generateKey()` static method
- **New packages**: `parallel.ts`, `metalearners.ts`, `notears.ts`, `bayesian-rca.ts`
- **`CausalModel`** unified API added (DoWhy-compatible pattern)
- **CLI** expanded from 4 commands to 4 commands with richer options

### Migration from v1.0

1. Replace `new HeuristicPathRCA()` → `new BayesianRCA({ engine: 'variable_elimination' })`
2. Update `scm.train(data)` → `scm.train(data)` — backward compatible, add `mechanismTypes` for PostNonlinear
3. Use `MetricsRegistry.toPrometheus()` for Prometheus scraping
4. Use `TokenBucket` for rate-based limiting, `RateLimiter` for queue-based
5. See `COMPAT.md` for detailed migration guide

## [1.0.0] — 2026-07-23

### Breaking Changes

- **`BayesianRCA` → `HeuristicPathRCA`**: The class is a heuristic path-scoring engine, not Bayesian inference. Old name kept as deprecated alias; will be removed in v2.0.
- **d-separation reimplemented**: `CausalGraph.dSeparated()` now implements strict Pearl (2009) d-separation. Previous implementation used moralized graph separation which gave incorrect results for colliders. Behavior change — previous test results relying on the buggy implementation may need review.
- **pdag2dag fixed**: Was a no-op (dead loop); now correctly implements Dor-Tarsi (1992). Any code relying on PDAG→DAG conversion will get different (correct) orientiation.
- **IPW propensity scores**: Previously used constant marginal probability for all observations. Now uses IRLS logistic regression. ATE estimates will differ.
- **MAD center changed**: StatsDetector MAD now uses median (correct) instead of mean. Anomaly thresholds will shift.
- **NaN handling in SCM/RHTScorer**: Previously skipped individual columns, causing biased OLS matrices. Now skips entire rows. Regression coefficients will differ when NaN values are present.
- **Voting fusion tie-breaking**: Now sorts by vote count first, then score. Previous sorting by score only. Root cause rankings in voting mode will change.
- **CIRCAPipeline toJSON()**: Previously serialized all rootCauses (ignoring top-5 slice). Now correctly serializes only top 5.
- **Shapley V(S)**: Previously used 50/50 blending for counterfactuals. Now uses proper conditional expectation. Attribution rankings will differ.

### Added

#### Causal Discovery (4 algorithms)
- **PC** (constraint-based): Fisher Z + stable variant
- **FCI** (latent confounders): Possible-D-SEP search + Meek rules R1-R3
- **GES** (score-based, Chickering 2002): BIC-optimized forward/backward search
- **LiNGAM** (non-Gaussian, Shimizu et al. 2011): DirectLiNGAM causal ordering

#### Conditional Independence (2 methods)
- **Fisher Z test**: linear Gaussian CI testing
- **KCI test**: kernel-based nonlinear CI testing (RBF + permutation p-value)

#### Causal Identification
- **do-calculus ID algorithm** (Shpitser & Pearl 2006): c-component decomposition
- **Backdoor adjustment**: correct confounding adjustment
- **Frontdoor adjustment**: mediation-based identification
- **Propensity score matching**: IRLS logistic regression + IPW
- **Doubly robust estimation**: combined propensity + outcome model

#### Anomaly Detection & RCA
- **CIRCA**: RHT z-score + descending adjustment for root cause ranking
- **HeuristicPathRCA**: path-based heuristic scoring
- **Shapley RCA**: Monte Carlo Shapley-value anomaly attribution
- **SPOT/DSPOT**: streaming POT threshold detection
- **Spectral Residual**: FFT-based anomaly detection
- **VotingDetector**: ensemble anomaly detection

#### Causal Inference
- **CATE**: Conditional Average Treatment Effect with feature centering
- **Mediation analysis**: Baron-Kenny natural direct/indirect effects
- **Sensitivity analysis**: E-value, partial R², robustness value
- **Bootstrap CI**: sequential + parallel (Promise.all chunking)

#### Graph Operations
- Strict Pearl (2009) d-separation with collider activation
- Dor-Tarsi (1992) PDAG→DAG conversion
- c-component decomposition for latent confounder handling
- SHD (Structural Hamming Distance), topological sort, do-surgery
- Domain knowledge application (forbid/require/rootLeaf)

#### Quality Infrastructure
- **AuditLogger**: immutable JSON audit trail
- **MetricsRegistry**: Prometheus-compatible counters + histograms
- **RateLimiter**: drop_oldest/drop_newest/block overflow strategies
- **EncryptedStore**: AES-256-GCM storage encryption
- **Typed error hierarchy**: 6 error classes
- **Shared constants**: 20+ tunable parameters with literature references

#### Benchmarks & Validation
- **ASIA benchmark** (Lauritzen & Spiegelhalter 1988): standard DAG validation
- **Performance benchmarks**: 8 algorithmic operations with documented budgets
- **Fuzz testing**: property-based random DAG validation (440 tests total)

### Fixed
- P0-1: d-separation collider handling (was moralized graph, now Pearl 2009)
- P0-2: pdag2dag Dor-Tarsi sink-finding algorithm
- P0-3: CATE dimension bug (k = 2+2p) + feature centering
- P0-4: IPW propensity score fitting (was constant, now IRLS)
- P0-5: CIRCAPipeline toJSON closure (captured full, now top 5)
- P0-7: SCM + RHTScorer NaN row-level skipping
- P1-1: PC non-stable variant (removed, always stable)
- P1-2: MAD center (was mean, now median)
- P1-5: Shapley V(S) arbitrary blending (now proper conditional expectation)
- P1-6: Graph falsification CI testing
- P1-7: Voting fusion tie-breaking
