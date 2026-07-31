# Hyperparameter Tuning Results
**Date:** 2026-07-31
**Trials:** 2

| Algorithm | Dataset | Best SHD | F1 | Best Params |
|-----------|---------|----------|-----|-------------|
| BOSS | ASIA | 2.0 | 0.875 | `{"numStarts":3,"maxParents":4}` |
| BOSS | Child | 6.5 | 0.863 | `{"numStarts":3,"maxParents":4}` |
| BOSS | Sachs | 12.5 | 0.604 | `{"numStarts":3,"maxParents":4}` |
| GES | ASIA | 5.5 | 0.673 | `{"penaltyDiscount":1.5}` |
| GES | Child | 20.5 | 0.593 | `{"penaltyDiscount":2.5}` |
| GES | Sachs | 8.0 | 0.742 | `{"penaltyDiscount":1}` |
| NOTEARS | ASIA | 7.5 | 0.616 | `{"lambda1":0.005,"wThreshold":0.2}` |
| NOTEARS | Child | 43.5 | 0.155 | `{"lambda1":0.002,"wThreshold":0.2}` |
| NOTEARS | Sachs | 19.0 | 0.296 | `{"lambda1":0.001,"wThreshold":0.2}` |