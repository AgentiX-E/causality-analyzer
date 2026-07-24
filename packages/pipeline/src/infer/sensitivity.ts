/**
 * Sensitivity Analysis for causal estimates.
 *
 * Quantifies how robust an estimated causal effect is to unmeasured
 * confounding. Critical for AIOps where complete observability is rare.
 *
 * Methods:
 *   eValueSensitivity — how strong would an unmeasured confounder need to be?
 *   partialRSensitivity — what fraction of residual variance must confounder explain?
 *   robustnessValue — combined robustness metric (E-value × partial R² relative)
 *
 * References:
 *   VanderWeele & Ding (2017). "Sensitivity Analysis in Observational Research"
 *   Cinelli & Hazlett (2020). "Making Sense of Sensitivity"
 *
 * @packageDocumentation
 */

/**
 * E-value sensitivity analysis.
 *
 * The E-value is the minimum strength of association (on the risk ratio scale)
 * that an unmeasured confounder would need to have with both treatment and
 * outcome to fully explain away the observed effect.
 *
 * For linear models, the ATE is first converted to a standardized effect size
 * (Cohen's d), then to an approximate risk ratio for E-value computation.
 * This follows Mathur & VanderWeele (2020) methodology.
 *
 * For RR > 1: E-value = RR + sqrt(RR * (RR - 1))
 * For RR < 1: E-value = 1/RR + sqrt(1/RR * (1/RR - 1))
 *
 * @param ate — estimated Average Treatment Effect (linear scale)
 * @param outcomeStd — standard deviation of outcome (required for proper SMD conversion).
 *                     If not provided, falls back to using ATE directly (for backward compat).
 *
 * @returns eValue and interpretation. eValue > 1 means the effect is robust
 *          to confounding below that strength threshold.
 */
export function eValueSensitivity(
  ate: number,
  outcomeStd?: number,
): { eValue: number; interpretation: string } {
  const absATE = Math.abs(ate);
  if (absATE < 1e-10) {
    return { eValue: 1, interpretation: 'No detectable effect — E-value undefined' };
  }

  // Convert ATE to approximate risk ratio
  let adjustedRR: number;
  if (outcomeStd !== undefined && outcomeStd > 0) {
    // Proper standardized mean difference (Cohen's d)
    const smd = absATE / outcomeStd;
    // Convert SMD to approximate RR using the formula from
    // Mathur & VanderWeele (2020): RR ≈ exp(1.81 × SMD) for binary outcomes
    // For continuous outcomes, use the conservative bound
    adjustedRR = Math.exp(1.81 * smd);
  } else {
    // Fallback: use ATE directly as an approximate effect size
    // This is only valid when the outcome is approximately on a risk-ratio-compatible scale.
    // For proper E-value, always provide outcomeStd.
    const rr = Math.exp(absATE);
    adjustedRR = rr > 1 ? rr : 1 / rr;
  }

  // Ensure adjustedRR is at least 1 (E-value formula requires RR ≥ 1)
  adjustedRR = Math.max(1, adjustedRR);

  // E-value = RR + sqrt(RR * (RR - 1))  — VanderWeele & Ding (2017)
  const ev = adjustedRR + Math.sqrt(adjustedRR * Math.max(0, adjustedRR - 1));

  let interpretation: string;
  if (ev < 1.5) {
    interpretation = `E-value=${ev.toFixed(2)}: weak robustness — a confounder with risk ratio ${ev.toFixed(2)} could explain the effect`;
  } else if (ev < 3) {
    interpretation = `E-value=${ev.toFixed(2)}: moderate robustness — requires moderately strong unmeasured confounding`;
  } else {
    interpretation = `E-value=${ev.toFixed(2)}: strong robustness — only very strong unmeasured confounding could explain the effect`;
  }

  return { eValue: ev, interpretation };
}

/**
 * Partial R² sensitivity analysis.
 *
 * Computes the minimum fraction of residual variance in both treatment
 * and outcome that an unmeasured confounder must explain to reduce the
 * estimated effect below a given threshold.
 *
 * Implements the method from Cinelli & Hazlett (2020):
 * - R²_{Y~Z|X,D} ≈ t² / (t² + df) where t = ATE / SE
 * - The confounder must explain this fraction of the residual variance
 *   in BOTH treatment and outcome to fully explain away the effect.
 *
 * @param ate — estimated ATE
 * @param se — standard error of ATE
 * @param n — sample size
 * @param threshold — effect threshold below which result is not meaningful (default: 0)
 *
 * @returns { r2Treatment, r2Outcome } — minimum partial R² values needed
 */
export function partialRSensitivity(
  ate: number,
  se: number,
  n: number,
  threshold: number = 0,
): { r2Treatment: number; r2Outcome: number; interpretation: string } {
  if (n < 3) {
    return { r2Treatment: 0, r2Outcome: 0, interpretation: 'Insufficient data for sensitivity analysis (n < 3)' };
  }

  const absATE = Math.abs(ate);
  if (absATE < 1e-10) {
    return { r2Treatment: 0, r2Outcome: 0, interpretation: 'Estimated effect is zero — sensitivity analysis not meaningful' };
  }

  // t-statistic for the estimated effect
  const tStat = absATE / Math.max(se, 1e-10);

  // Reduction factor needed to bring effect below threshold
  const reduction = Math.max(0, absATE - Math.abs(threshold));

  // Cinelli & Hazlett (2020) §3.2:
  //   R²_{Y~Z|X,D} = t²_{α/2,d-2} / (d-2 + t²_{α/2,d-2})
  // where d = effective degrees of freedom.
  // For a properly specified model with k covariates, df = n - k - 1.
  // Here we use df = n - 2 as a conservative lower bound (single treatment + intercept).
  const df = n - 2;
  const r2Obs = (tStat * tStat) / (tStat * tStat + df);

  // The unmeasured confounder must explain r2Min fraction of the RESIDUAL variance
  // in both the treatment and the outcome equations (Cinelli & Hazlett 2020, §4).
  // r2Treatment ≈ r2Outcome ≈ R²_{Y~Z|X,D} for a confounder of equal strength.
  //
  // The "benchmark" R² needed to explain the ENTIRE effect (r2Min × r2Obs) is
  // what we report. If an unmeasured confounder explains at least this much of
  // the residual variance in both treatment and outcome, the adjusted estimate
  // crosses the threshold.
  const benchmark = reduction / Math.max(absATE, 1e-10);
  const r2Treatment = benchmark * r2Obs;
  const r2Outcome = benchmark * r2Obs; // same R² for treatment and outcome (Cinelli & Hazlett 2020 §4.2)

  let interpretation: string;
  if (r2Treatment < 0.01 && r2Outcome < 0.01) {
    interpretation = 'Highly sensitive — even weak confounders could change the conclusion';
  } else if (r2Treatment < 0.1) {
    interpretation = `Sensitive — confounder explaining ${(r2Treatment * 100).toFixed(1)}% of variance could alter results`;
  } else {
    interpretation = `Robust — confounder would need to explain ${(r2Treatment * 100).toFixed(1)}% of treatment variance`;
  }

  return { r2Treatment, r2Outcome, interpretation };
}

/**
 * Combined robustness value.
 *
 * Integrates E-value and partial R² into a single interpretable metric.
 *
 * RV = E-value × (1 - partialR2_treatment)
 *
 * This captures both: how strong a confounder must be (E-value) and
 * how constrained the opportunity for confounding is (partial R²).
 *
 * RV > 2: robust
 * RV > 1.5: moderate
 * RV < 1.5: sensitive
 */
export function robustnessValue(
  ate: number,
  se: number,
  n: number,
  outcomeStd?: number,
): { rv: number; interpretation: string } {
  const ev = eValueSensitivity(ate, outcomeStd);
  const pr2 = partialRSensitivity(ate, se, n);

  // RV = E-value × (1 - r2) — higher is more robust
  // When r2Treatment is small, RV ≈ E-value (no constraint).
  // When r2Treatment is large (lots of residual variance), RV decreases.
  const rv = ev.eValue * (1 - pr2.r2Treatment);

  let interpretation: string;
  if (rv > 2) {
    interpretation = `RV=${rv.toFixed(2)}: ROBUST — causal conclusion is well-supported`;
  } else if (rv > 1.5) {
    interpretation = `RV=${rv.toFixed(2)}: MODERATE — conclusion is plausible but warrants caution`;
  } else {
    interpretation = `RV=${rv.toFixed(2)}: SENSITIVE — conclusion is fragile to unmeasured confounding`;
  }

  return { rv, interpretation };
}
