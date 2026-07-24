/**
 * Uplift Modeling — Qini curve, AUUC, uplift@k.
 *
 * Provides evaluation metrics for heterogeneous treatment effect models
 * and uplift (persuasion) modeling. Essential for CATE validation and
 * decision-making under causal constraints.
 *
 * References:
 *   Radcliffe (2007). "Using Control Groups to Target on Predicted Lift"
 *   Gutierrez & Gérardy (2017). "Causal Inference and Uplift Modelling"
 *
 * @packageDocumentation
 */

// ── Data Structures ─────────────────────────────────────────────────

/** A single observation for uplift evaluation */
export interface UpliftObservation {
  /** Predicted uplift score (higher = more likely to respond to treatment) */
  score: number;
  /** Actual treatment assignment (1 = treated, 0 = control) */
  treatment: number;
  /** Actual outcome */
  outcome: number;
}

/** Uplift curve data point */
export interface UpliftCurvePoint {
  /** Cumulative proportion of population targeted */
  proportion: number;
  /** Cumulative uplift (ATE × proportion) */
  cumulativeUplift: number;
  /** Cumulative incremental gain */
  incrementalGain: number;
}

/** Complete uplift evaluation */
export interface UpliftEvaluation {
  /** AUUC — Area Under Uplift Curve (normalized) */
  auuc: number;
  /** Qini coefficient */
  qiniCoefficient: number;
  /** Uplift at top 10% */
  upliftAt10: number;
  /** Uplift at top 20% */
  upliftAt20: number;
  /** Uplift at top 50% */
  upliftAt50: number;
  /** Full curve data points for plotting */
  curve: UpliftCurvePoint[];
}

// ── Uplift Evaluation ──────────────────────────────────────────────

/**
 * Compute uplift evaluation metrics from observations.
 *
 * The Qini curve sorts observations by predicted uplift score (descending),
 * then computes cumulative uplift as:
 *   Qini(k) = (Σ Y_treated - Σ Y_control × ratio) / N
 * where ratio adjusts for treatment/control imbalance.
 *
 * @param observations — array of {score, treatment, outcome}
 * @param normalize — whether to normalize AUUC by random model AUUC
 */
export function evaluateUplift(
  observations: UpliftObservation[],
  normalize: boolean = true,
): UpliftEvaluation {
  if (observations.length === 0) {
    return {
      auuc: 0, qiniCoefficient: 0,
      upliftAt10: 0, upliftAt20: 0, upliftAt50: 0,
      curve: [],
    };
  }

  // Sort by predicted score descending
  const sorted = [...observations].sort((a, b) => b.score - a.score);

  const n = sorted.length;
  const nTreated = sorted.filter(o => o.treatment > 0.5).length;
  const nControl = n - nTreated;
  if (nTreated === 0 || nControl === 0) {
    // Degenerate: all observations in same group
    // AUUC is undefined — return zeros
    return {
      auuc: 0, qiniCoefficient: 0,
      upliftAt10: 0, upliftAt20: 0, upliftAt50: 0,
      curve: [],
    };
  }

  const ratio = nControl / nTreated;

  // Cumulative sums
  let cumTreatedOutcome = 0;
  let cumControlOutcome = 0;
  let cumTreated = 0;
  let cumControl = 0;

  const curve: UpliftCurvePoint[] = [];
  let totalArea = 0;
  let randomArea = 0;

  const overallATE = (() => {
    const ty = sorted.reduce((s, o) => s + (o.treatment > 0.5 ? o.outcome : 0), 0) / nTreated;
    const cy = sorted.reduce((s, o) => s + (o.treatment <= 0.5 ? o.outcome : 0), 0) / nControl;
    return ty - cy;
  })();

  for (let k = 1; k <= n; k++) {
    const obs = sorted[k - 1]!;
    if (obs.treatment > 0.5) {
      cumTreatedOutcome += obs.outcome;
      cumTreated++;
    } else {
      cumControlOutcome += obs.outcome;
      cumControl++;
    }

    const proportion = k / n;
    // Qini: incremental uplift per treated observation
    const uplift = cumTreated > 0 ? cumTreatedOutcome / cumTreated : 0;
    const controlAdj = cumControl > 0 ? cumControlOutcome / cumControl : 0;
    const cumulativeUplift = (uplift - controlAdj) * proportion;

    // Random baseline: overall ATE × proportion
    const randomUplift = overallATE * proportion;

    curve.push({
      proportion,
      cumulativeUplift,
      incrementalGain: cumulativeUplift - randomUplift,
    });

    // Trapezoidal integration for area
    if (k > 1) {
      const prev = curve[k - 2]!;
      const width = proportion - prev.proportion;
      totalArea += (prev.cumulativeUplift + cumulativeUplift) * width / 2;
      randomArea += overallATE * proportion * width;
    }
  }

  // AUUC: area under uplift curve, normalized by random model
  const auuc = normalize && Math.abs(randomArea) > 1e-10
    ? totalArea / randomArea
    : totalArea;

  // Qini coefficient: (model_area - random_area) / (perfect_area - random_area)
  // Perfect lift model: all treated ranked before control.
  const treatedRatio = nTreated / n;
  const perfectUpliftArea = overallATE * treatedRatio * (1 - treatedRatio / 2);
  const qiniCoefficient = Math.abs(perfectUpliftArea) > 1e-10
    ? (totalArea - randomArea) / (perfectUpliftArea - randomArea)
    : 0;

  // Uplift at key percentiles
  const upliftAt = (pct: number): number => {
    const idx = Math.min(n - 1, Math.floor(n * pct));
    if (idx < 0) return 0;
    return curve[idx]!.cumulativeUplift;
  };

  return {
    auuc,
    qiniCoefficient,
    upliftAt10: upliftAt(0.1),
    upliftAt20: upliftAt(0.2),
    upliftAt50: upliftAt(0.5),
    curve,
  };
}

// ── Uplift Curve Utilities ─────────────────────────────────────────

/**
 * Compute the uplift at a specific top-k percentage of the population.
 * Sorts by predicted score, computes uplift for the top proportion.
 */
export function upliftAtK(
  observations: UpliftObservation[],
  k: number, // proportion, e.g. 0.1 for top 10%
): number {
  const sorted = [...observations].sort((a, b) => b.score - a.score);
  const cutoff = Math.max(1, Math.floor(sorted.length * k));
  const topK = sorted.slice(0, cutoff);

  const treated = topK.filter(o => o.treatment > 0.5);
  const control = topK.filter(o => o.treatment <= 0.5);

  if (treated.length === 0 || control.length === 0) return 0;

  const treatedMean = treated.reduce((s, o) => s + o.outcome, 0) / treated.length;
  const controlMean = control.reduce((s, o) => s + o.outcome, 0) / control.length;

  return treatedMean - controlMean;
}

/**
 * Normalized Uplift Comparison: compare two uplift models.
 *
 * Returns the ratio of AUUC(modelA) / AUUC(modelB).
 * > 1 means modelA is better; < 1 means modelB is better.
 */
export function compareUpliftModels(
  observationsA: UpliftObservation[],
  observationsB: UpliftObservation[],
): number {
  const evalA = evaluateUplift(observationsA);
  const evalB = evaluateUplift(observationsB);

  if (Math.abs(evalB.auuc) < 1e-10) return evalA.auuc > 0 ? Infinity : 0;
  return evalA.auuc / evalB.auuc;
}
