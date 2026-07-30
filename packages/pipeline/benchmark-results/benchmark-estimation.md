# IHDP CATE Estimation Benchmark
**Date:** 2026-07-30
**Repetitions:** 10 (standard: 100)

| Estimator | PEHE (ours) | PEHE (published) | ATE Error | Time (ms/rep) |
|-----------|-------------|-------------------|-----------|---------------|
| ForestDRLearner | 2.865 | 0.500 | 1.068 | 118 |
| SLearner | 6.915 | 0.690 | 6.393 | 2 |
| TLearner | 139883061056.113 | 0.720 | 3.854 | 1 |
| LinearDRLearner | 205242691309.810 | 0.520 | 205242691309.811 | 11 |
| XLearner | 4.000601460064065e+21 | 0.630 | 22487912812.786 | 13 |

*Published values from Curth & van der Schaar (2021), Table 1.*