## I40: GOLEM λ₂ Sensitivity Analysis

λ₂ sweep on ASIA (5000 samples, 7 values 0.5-10.0):
- λ₂=0.5: SHD=14 (under-constrained)
- λ₂=1.0-10.0: SHD=11, TPR=0.375, edges=9 (identical plateau)
- Default λ₂=5.0 confirmed optimal; no further tuning gains.
- Bottleneck: mathjs expm/det precision, not parameter selection.
