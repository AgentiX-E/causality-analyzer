# Changelog

## [1.1.0] — 2026-07-26

### Type Safety (I1)
- **no-unsafe-* rules → error**: 6条ESLint规则从warn升级为error，零妥协类型安全
- **Array.from 替代**: 所有`Array.from<T>()`模式替换为`new Array<T>().fill() as T[]`
- **类型化边界**: neo4j-driver-lite `_require` → `loadNeo4j()` 类型接口
- **SQLite 类型**: `EmbedRelationalStore` 完整 `PreparedStatements` 类型定义

### API Productionization (I1)
- **OpenAPI 3.1**: `GET /v1/openapi.json` — 完整请求/响应schema + security schemes
- **Bearer Token 认证**: `CAUSALITY_API_TOKEN` 环境变量启用
- **API 版本化**: 业务端点 `/v1/discover` `/v1/analyze` `/v1/estimate`

### Algorithm Precision (I2)
- **GES backward phase**: 修复无向边删除时parent集合计算bug
- **LiNGAM dependence**: 自适应策略(N≤800全量Kendall's tau, >800采样) + 两阶段稳定OLS
- **精度阈值测试**: `i91-precision-thresholds.test.ts` — 6 DAGs × 6 algorithms = 16个显式阈值

### Test Coverage (I3)
- **Visual 覆盖率**: 35.26% → 60.73% (Canvas2DRenderer: 17.89% → 97.93%)
- **OpenTelemetry**: `Telemetry` facade — 零开销no-op默认, 注入真实OTel无缝切换
- **Telemetry 测试**: 15个测试覆盖所有接口 + DI注入

### Modern ML Mechanisms (I4)
- **FFNMechanism**: 2-hidden-layer FFN因果机制, Adam训练, 序列化支持
- **Neural 测试**: 6个测试覆盖训练/预测/反演/序列化

### Extended Benchmarks (I5)
- **12 DAG 配置**: Small(≤11) + Medium(12-20) + Large(≥27节点) 全面基准
- **9 algorithms**: PC, GES, LiNGAM, NOTEARS, BOSS, DAGMA, GOLEM, FCI, GFCI

### Documentation (I6)
- **部署指南**: `docs/production-deployment.md` — 安全加固, 容量规划, 监控, 灾备
- **OpenTelemetry 集成**: 生产环境分布式追踪配置示例

## [1.0.0] — 2026-07-25

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
