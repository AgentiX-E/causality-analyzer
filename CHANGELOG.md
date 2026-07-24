# Changelog

All notable changes to Causality Analyzer.

## [1.0.0] — 2026-07-24

### I12: Deep Learning Causal Discovery
- **NOTEARS** (Zheng et al. NeurIPS 2018): Continuous optimization for DAG learning
  - Padé(3,3) matrix exponential with scaling & squaring
  - Augmented Lagrangian with L-BFGS inner optimization
  - l1 regularization for sparsity control
  - 11 test cases: chain, fork, collider, 4-node, domain knowledge, lambda sweep

### I11: Effect Estimation & Refutation Enhancement
- **DML** (Chernozhukov et al. 2018): Double ML with K-fold cross-fitting
- **Meta-Learners**: S-Learner, T-Learner, X-Learner for heterogeneous ATE
- **RidgeRegressor** / **LogisticClassifier**: built-in ML backends
- **Extended Refutation** (3 new methods):
  - `refuteRandomCommonCause` — sensitivity to unobserved confounding
  - `refuteDummyOutcome` — null effect verification
  - `refuteUnobservedConfounder` — parametric sensitivity analysis
  - `comprehensiveRefutation` — all 6 methods in one call
- 28 test cases: DML + MetaLearners + Refutation + cross-method agreement

### I10: Core Implementation Deep Fixes
- **do-calculus**: complete hedge criterion (Shpitser & Pearl 2006), Union-Find c-components
- **JunctionTree**: Hugin collect+distribute message passing, Kruskal MST clique tree
- Both: correct induced subgraph, separator management, edge case handling

### I9: Discovery Breadth — 4→9 Algorithm Suite
- **GIN** (Xie et al. UAI 2020): Group Independence-based discovery
- **CD-NOD** (Huang et al. AAAI 2020): Non-stationary domain-varying Fisher Z
- **GRaSP** (Lam et al. AIStats 2022): Greedy permutation search with BIC
- **CAM-UV** (Bühlmann et al. 2014): Additive models with unobserved confounders
- **ExactSearch** (Yuan & Malone JMLR 2013): A* optimal DAG search
- 46 test cases: ASIA benchmark, continuous/discrete/nonlinear scenarios

### I8: Release Readiness
- Version badges unified: 5 packages → 1.0.0
- Performance test budgets sandbox/CI-safe (fisherZ: 150ms, kci: 250ms, dsep: 50ms)
- Removed placeholder `_quality.yml.nodeversion` file

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
