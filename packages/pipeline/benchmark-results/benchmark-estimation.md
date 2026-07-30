# IHDP CATE Estimation Benchmark
**Date:** 2026-07-30
**Repetitions:** 10 (standard: 100)

| Estimator | PEHE (ours) | PEHE (published) | ATE Error | Time (ms/rep) |
|-----------|-------------|-------------------|-----------|---------------|
| ForestDRLearner | 2.865 | 0.500 | 1.068 | 116 |
| CausalForestDML | 4.193 | 0.430 | 1.001 | 1549 |
| SLearner | 7.058 | 0.690 | 6.546 | 2 |
| NonParamDML | 10.632 | 0.510 | 1.411 | 1087 |
| LinearDML | 10.632 | 0.460 | 1.411 | 1083 |
| TLearner | 279.928 | 0.720 | 3.918 | 1 |
| LinearDRLearner | 372.152 | 0.520 | 372.005 | 11 |
| XLearner | 16097.351 | 0.630 | 46.635 | 12 |

*Published values from Curth & van der Schaar (2021), Table 1.*