/**
 * Sensitivity Analysis for causal estimates.
 *
 * Quantifies how robust an estimated causal effect is to unmeasured
 * confounding. Critical for AIOps where complete observability is rare.
 *
 * Methods:
 *   eValueSensitivity — how strong would an unmeasured confounder need to be?
 *   partialRSensitivity — what fraction of residual variance must confounder explain?
 *   robustnessValue — combined robustness metric (E-value adjusted for partial R²)
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
 * For ATE > 0: E-value = ATE + sqrt(ATE * (ATE - 1))
 * For ATE < 0: E-value = 1 / (|ATE| + sqrt(|ATE| * (|ATE| - 1)))
 *
 * Returns { eValue, conclusion } where eValue > 1 means the effect is robust
 * to confounding below that strength threshold.
 */
export function eValueSensitivity(ate: number): { eValue: number; interpretation: string } {
  const absATE = Math.abs(ate);
  if (absATE < 1e-10) {
    return { eValue: 1, interpretation: 'No detectable effect — E-value undefined' };
  }

  // Convert ATE to risk ratio scale: RR = exp(ATE) for log-linear, or use RR = (ATE + 1) for linear
  const rr = Math.exp(ate);
  const adjustedRR = rr > 1 ? rr : 1 / rr;

  // E-value = RR + sqrt(RR * (RR - 1))
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
 * Parameters:
 *   ate — estimated ATE
 *   se — standard error of ATE
 *   n — sample size
 *   threshold — effect threshold below which result is not meaningful (default: 0)
 *
 * Returns { r2Treatment, r2Outcome } — minimum partial R² values needed
 */
export function partialRSensitivity(
  ate: number, se: number, n: number, threshold: number = 0,
): { r2Treatment: number; r2Outcome: number; interpretation: string } {
  if (n < 2) {
    return { r2Treatment: 0, r2Outcome: 0, interpretation: 'Insufficient data for sensitivity analysis' };
  }

  const absATE = Math.abs(ate);
  const tStat = absATE / Math.max(se, 1e-10);

  // Partial R² needed to reduce effect to threshold
  const reduction = Math.max(0, absATE - Math.abs(threshold));
  const r2Min = reduction / Math.max(absATE, 1e-10);

  // Cinelli & Hazlett (2020) formula: R² ~ t² / (t² + df)
  const df = n - 2;
  const r2Obs = (tStat * tStat) / (tStat * tStat + df);

  // Confounder must explain at least r2Min of residual variance
  const r2Treatment = r2Min * r2Obs;
  const r2Outcome = r2Min * 0.5; // conservative: half of treatment R²

  let interpretation: string;
  if (r2Treatment < 0.01 && r2Outcome < 0.01) {
    interpretation = 'Highly sensitive — even weak confounders could change the conclusion';
  } else if (r2Treatment < 0.1) {
    interpretation = `Sensitive — confounder explaining ${(r2Treatment*100).toFixed(1)}% of variance could alter results`;
  } else {
    interpretation = `Robust — confounder would need to explain ${(r2Treatment*100).toFixed(1)}% of treatment variance`;
  }

  return { r2Treatment, r2Outcome, interpretation };
}

/**
 * Combined robustness value.
 *
 * Integrates E-value and partial R² into a single interpretable metric.
 *
 * RV = E-value / (1 + partialR2_treatment)
 *
 * RV > 2: robust
 * RV > 1.5: moderate
 * RV < 1.5: sensitive
 */
export function robustnessValue(ate: number, se: number, n: number): { rv: number; interpretation: string } {
  const ev = eValueSensitivity(ate);
  const pr2 = partialRSensitivity(ate, se, n);

  const rv = ev.eValue / (1 + pr2.r2Treatment);

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
