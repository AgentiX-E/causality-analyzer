# IHDP CATE Estimation Benchmark
**Date:** 2026-07-31
**Repetitions:** 100 (standard: 100)

| Estimator | PEHE (ours) | PEHE (published) | ATE Error | Time (ms/rep) |
|-----------|-------------|-------------------|-----------|---------------|
| ForestDRLearner | 2.933 | 0.500 | 1.211 | 111 |
| CausalForestDML | 4.090 | 0.430 | 1.257 | 4762 |
| SLearner | 7.300 | 0.690 | 6.804 | 2 |
| NonParamDML | 9.160 | 0.510 | 1.379 | 4345 |
| LinearDML | 9.160 | 0.460 | 1.379 | 4310 |
| TLearner | 228.427 | 0.720 | 4.142 | 0 |
| LinearDRLearner | 432.687 | 0.520 | 432.637 | 7 |
| XLearner | 12124.404 | 0.630 | 39.448 | 6 |

*Published values from Curth & van der Schaar (2021), Table 1.*