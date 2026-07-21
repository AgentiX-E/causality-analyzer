# Final Audit — Post I5-I10 (2026-07-22)

## 三个核心问题的诚实答案

### Q1: 企业级和工业级水准？

**接近，但未达到。** 评分 B+ (82/100)。

I5-I10 解决了原始审计的 29 个缺陷，新增了因果效应估计、Shapley RCA、模型评估、FCI 算法和完整文档。但终审发现 3 个正确性缺陷和 6 个质量问题残留：

| # | 严重度 | 位置 | 问题 |
|---|--------|------|------|
| 1 | 🔴 | `effect-estimation.ts:32-49` | `findBackdoorSet` 构建 `result` 数组后丢弃，返回无关表达式 |
| 2 | 🔴 | `model-evaluation.ts:110-123` | `evaluateMSE` 根节点预测为空 if-block，非根节点用均值而非拟合机制 |
| 3 | 🔴 | `advanced-discovery.ts:106-138` | FCI 仅实现 R1-R3，缺少 FCI 专有 R4-R10 + Possible-D-SEP |
| 4 | 🟡 | 多处 | 非确定性算法：Shapley、bootstrap、PS matching 均用无种子 `Math.random()` |
| 5 | 🟡 | `pc.ts:58-78` | `solveLinear` 重复（`invertMatrix` 未迁移到 `core/src/math.ts`） |
| 6 | 🟡 | 多处 | 死代码：`val()` 未使用、`logLik` 未使用、`covIdx` 参数未使用 |
| 7 | 🟡 | `effect-estimation.ts` + `model-evaluation.ts` | `colMean()` 重复定义 |
| 8 | 🟡 | CI | `--no-frozen-lockfile`、缺少 Prettier 配置 |
| 9 | 🟡 | 全仓库 | 250+ `!` 非空断言，无 ESLint 规则约束 |

### Q2: 全面超越参考项目？

**在广度上是，在深度上否。**

| 维度 | DoWhy | Causality Analyzer |
|------|-------|-------------------|
| 因果效应估计 | backdoor/frontdoor/IV + 8 estimators + EconML | backdoor/frontdoor/IV/PS/DR ✅ |
| Shapley RCA | Shapley symmetrization (ICML 2022) | Monte Carlo Shapley ✅ |
| FCI | 完整 R1-R10 + PDS | R1-R3 only ❌ |
| 模型评估 | KL, R², NMSE, CRPS, 图验证 | R² + MSE (broken) ❌ |
| 反事实 | 可逆 SCM + 点估计 | SCM counterfactual ✅ |
| Bootstrap CI | 全接口支持 | bootstrapRCA ✅ |
| AIOps 专用 RCA | 无 | CIRCA/Bayesian/RandomWalk/HT/FPGrowth ✅ |
| 存储后端 | 无 | SQLite + PG + Neo4j + OverGraph ✅ |
| mTLS | 无 | Bolt + PG-wire ✅ |
| Docker CI | 无 | Neo4j mTLS ✅ |
| 文档 | 40+ notebooks | TypeDoc + ADR×3 ✅ |

**独有优势：** AIOps 管线、4 存储后端、mTLS、Web Components、Docker CI。
**劣势：** FCI 和大模型评估不够深入。

### Q3: 业界最优和行业标杆？

**在 AIOps 因果分析这一细分子领域是独特的，但整个因果推断软件的绝对深度不如 DoWhy。**

---

## I11: 最终收尾 — 9 缺陷修复

| 缺陷 | 修复 |
|------|------|
| `findBackdoorSet` 丢弃结果 | 删除废代码，直接返回 `result`（或实现正确的 backdoor criterion） |
| `evaluateMSE` root if-block | 实现根节点均值预测 |
| FCI R4-R10 缺失 | 补充 discriminating path 规则 |
| Math.random() 无种子 | 添加 `seed?: number` 参数 + LCG |
| `pc.ts` invertMatrix 重复 | 删除，改用 `core/src/math.ts` 的 `solveLinear` |
| 死代码 `val()`, `logLik`, `covIdx` | 删除 |
| `colMean()` 重复 | 提取到 `core/src/math.ts` |
| CI `--no-frozen-lockfile` | 恢复 `--frozen-lockfile` |
| Prettier 缺失 | 添加 `.prettierrc` + CI 检查 |
