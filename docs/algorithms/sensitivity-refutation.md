# Sensitivity Analysis & Refutation

## E-value Sensitivity

**Reference:** VanderWeele & Ding (2017). *Sensitivity Analysis in Observational Research*

Computes the minimum strength of association (on the risk ratio scale) that an unmeasured confounder would need to have with both treatment and outcome to fully explain away the observed effect.

### Interpreting E-values

| E-value | Interpretation |
|---------|---------------|
| > 3 | Strong robustness |
| 1.5 – 3 | Moderate robustness |
| < 1.5 | Weak — small confounders could change the conclusion |

### Scenario: "How confident should I be in this RCA?"

```typescript
import { eValueSensitivity } from '@agentix-e/causality-analyzer-pipeline';

const { eValue, interpretation } = eValueSensitivity(0.8);
// E-value=4.22 — even strong unmeasured confounders unlikely to negate

const { eValue: weak } = eValueSensitivity(0.15);
// E-value=1.3 — warning: small confounders could explain this away
```

---

## Partial R² Sensitivity

**Reference:** Cinelli & Hazlett (2020). *Making Sense of Sensitivity*

Computes the minimum fraction of residual variance an unmeasured confounder must explain in both treatment and outcome to reduce the effect below a threshold.

### Scenario: Quantifying sensitivity

```typescript
import { partialRSensitivity } from '@agentix-e/causality-analyzer-pipeline';

const { r2Treatment, r2Outcome, interpretation } = partialRSensitivity(0.8, 0.1, 1000);

// r2Treatment > 0.1 → robust (confounder needs to explain >10% of treatment variance)
// r2Treatment < 0.01 → fragile (confounder explaining <1% could alter the result)
```

---

## Robustness Value

Combined metric integrating E-value and partial R²:

RV = E-value / (1 + partialR2_treatment)

### Scenario: Single-number robustness score

```typescript
import { robustnessValue } from '@agentix-e/causality-analyzer-pipeline';

const { rv, interpretation } = robustnessValue(0.8, 0.1, 1000);

// RV > 2: ROBUST
// RV in [1.5, 2]: MODERATE
// RV < 1.5: SENSITIVE
```

---

## Refutation Methods

### Placebo Treatment

Scrambles the treatment assignment — if the effect disappears, the original conclusion is robust.

```typescript
import { refutePlaceboTreatment } from '@agentix-e/causality-analyzer-pipeline';

const result = refutePlaceboTreatment(data, treatmentIdx, outcomeIdx, 50, 42);
// pValue < 0.05 → placebo doesn't reproduce the effect → robust
```

### Data Subset Refutation

Tests stability across random subsets — if consistent, the effect is not driven by a few outliers.

```typescript
import { refuteDataSubset } from '@agentix-e/causality-analyzer-pipeline';

const result = refuteDataSubset(data, treatmentIdx, outcomeIdx, 0.8, 20, 42);
```

### Bootstrap Refutation

Resamples with replacement to estimate the sampling distribution.

```typescript
import { refuteBootstrap } from '@agentix-e/causality-analyzer-pipeline';

const result = refuteBootstrap(data, treatmentIdx, outcomeIdx, 100, 42);
// Returns CI and p-value for the ATE
```

### When to Refute

| Scenario | Action |
|----------|--------|
| All 3 refuters agree (p > 0.05) | Conclusion is robust |
| Placebo fails but subset passes | Check treatment assignment quality |
| Subset fails but placebo passes | Data may have influential outliers |
| All fail (p < 0.05) | Conclusion is NOT reliable — collect more data or add covariates |

[← Back to User Guide](../user-guide.md)
