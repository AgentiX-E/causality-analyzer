# Changelog

All notable changes to Causality Analyzer.

## [Unreleased]

### Added (I7-I8)
- **Causal Effect Estimation**: Backdoor adjustment, frontdoor formula, IV/2SLS, propensity score matching, doubly robust estimation
- **Shapley RCA**: Monte Carlo Shapley-value anomaly attribution for SCMs
- **Model Evaluation**: R², MSE per mechanism; bootstrap confidence intervals
- **Neo4j mTLS CI**: Full mutual TLS integration tests with Dockerized Neo4j

### Added (I5-I6)
- **Shared Math Library**: `solveLinear`, `normalTail`, `erf`, `normalCDF` extracted from pipeline → `core/src/math.ts`
- **CI Quality**: ESLint + Prettier enforcement, full typecheck on all 5 packages
- **Pipeline barrel exports**: Complete re-exports for graph, analyze, infer, gcm, viz sub-packages

### Fixed (I5)
- **P0-2**: HTRCA intercept computation — uses per-column means (was first-row/n)
- **P0-3**: FPGrowthRCA — full recursive FP-Tree mining (was singleton-only)
- **P0-4**: `refutePlaceboTreatment` — empirical p-value (was hardcoded 0.5/0.01)
- **P0-6**: BayesianRCA CPT — data-driven per-row anomaly detection
- **P0-7**: SpectralResidual — removed buffer state leak in `detect()`
- **P1-1**: RandomWalkRCA — seed-based reproducibility via LCG
- **P1-3**: Fusion voting strategy implemented
- **P1-7**: `readMetrics` — honors `MetricQuery.metrics` filter
- **P1-8/9**: EmbedGraphStore — graph-specific label lookup, version-aware loading
- **P1-10/11**: Time series chart — unified time axis, anomaly region rendering
- **P1-12**: Graph renderer — topology-aware BFS layered layout

### Changed
- All storage interfaces (`IRelationalStore`, `IGraphStore`) now include `close(): void`
- `BaseConfig.getSchema()` is now abstract — subclasses must provide schemas
- `ColumnarTable.fromRows` collects all row keys (was first-row-only)
- neo4j-driver-lite moved from optionalDependencies → dependencies

## [0.1.0] — 2026-06

### Added (I1-I4)
- **Monorepo**: 5 packages — core, pipeline, storage-embed, storage-remote, visual
- **CIRCA Pipeline**: RHTScorer + DAScorer for causal RCA
- **RCA Algorithms**: BayesianRCA, RandomWalkRCA, HTRCA, FPGrowthRCA
- **Causal Inference**: `identifyBackdoor`, `identifyFrontdoor`, `estimateLinearRegression`, refutation methods
- **Anomaly Detection**: StatsDetector, SpectralResidual, SPOT/DSPOT, VotingDetector
- **Causal Discovery**: PC algorithm with Fisher's Z-test
- **GCM**: StructuralCausalModel with counterfactual inference
- **Storage**: SQLite-backed embed store, pg.Client + neo4j-driver-lite remote stores
- **Visualization**: Lit 3 Web Components (`<ca-causal-graph>`, `<ca-time-series>`, `<ca-root-cause-ranking>`)
- **CI**: GitHub Actions with test, browser-test, Neo4j-test jobs
