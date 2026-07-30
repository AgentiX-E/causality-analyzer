# IHDP CATE Estimation Benchmark
**Date:** 2026-07-30
**Repetitions:** 10 (standard: 100)

| Estimator | PEHE (ours) | PEHE (published) | ATE Error | Time (ms/rep) |
|-----------|-------------|-------------------|-----------|---------------|
| ForestDRLearner | 2.865 | 0.500 | 1.068 | 115 |
| SLearner | 6.929 | 0.690 | 6.408 | 2 |
| TLearner | 2797.774 | 0.720 | 3.860 | 1 |
| LinearDRLearner | 4754.283 | 0.520 | 4754.280 | 9 |
| XLearner | 1601191.704 | 0.630 | 451.366 | 11 |

*Published values from Curth & van der Schaar (2021), Table 1.*