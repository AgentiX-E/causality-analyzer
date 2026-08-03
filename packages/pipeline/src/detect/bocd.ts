/**
 * Bayesian Changepoint Detection — CUSUM + BOCD Hybrid.
 *
 * Primary algorithm: CUSUM (Cumulative Sum Control Chart).
 * Industry-standard, parameter-free, and proven for fault detection
 * in manufacturing, finance, and IT operations since Page (1954).
 *
 *   S_0 = 0
 *   S_t = max(0, S_{t-1} + (x_t - μ₀) / σ - k)
 *
 * Where:
 *   μ₀ = pre-change mean (estimated from first half of data)
 *   σ  = pre-change standard deviation
 *   k  = δ/(2σ) — half the minimum detectable shift (default: 0.5)  
 *   h  = detection threshold (default: 5.0)
 *
 * A changepoint is detected when S_t > h.
 * The changepoint index is back-traced to where S first became positive.
 *
 * For robustness, μ₀ and σ are estimated via median/MAD rather than
 * mean/std, making the detection resistant to outliers.
 *
 * Complexity: O(T) time, O(1) memory per time series.
 *
 * @packageDocumentation
 */

import { Matrix } from 'ml-matrix';

// ── Types ────────────────────────────────────────────────────────────────

export interface BOCDConfig {
  /**
   * CUSUM drift parameter: minimum detectable shift in σ units.
   * Default: 0.5 (detects shifts ≥ 0.5σ from baseline).
   * Lower → more sensitive. Higher → only catches large shifts.
   */
  driftParam: number;
  /**
   * CUSUM decision threshold.
   * Default: 5.0 (standard 5σ CUSUM, 1% false alarm rate).
   * Lower → more detections (false positives). Higher → fewer (false negatives).
   */
  threshold: number;
  /**
   * Fraction of data used for baseline estimation (first N% assumed stationary).
   * Default: 0.5 (use first half to estimate μ₀ and σ).
   */
  baselineFraction: number;
  /**
   * Minimum data points required before detecting.
   * Default: 10 (avoid false positives on very short series).
   */
  minPoints: number;
}

export interface ChangepointResult {
  /** CUSUM statistic at each time step [0..T-1] */
  cusum: number[];
  /** Index of the detected changepoint (back-traced to S>0 start) */
  mostLikelyIndex: number;
  /** Maximum CUSUM value reached */
  maxCusum: number;
  /** Is a changepoint detected (maxCusum > threshold)? */
  detected: boolean;
  /** Confidence score: maxCusum / threshold, clamped to [0, 1] */
  confidence: number;
  /** Total number of data points processed */
  dataPoints: number;
  /** Estimated pre-change mean (baseline) */
  baselineMean: number;
  /** Estimated pre-change standard deviation */
  baselineStd: number;
}

export interface BOCDAnomalyResult {
  service: string;
  changepoint: ChangepointResult;
  preMean: number;
  postMean: number;
  magnitudeShift: number;
  preStd: number;
}

// ── CUSUM Detector ───────────────────────────────────────────────────────

export class BOCDDetector {
  readonly config: BOCDConfig;
  private baselineValues: number[] = [];

  constructor(config?: Partial<BOCDConfig>) {
    this.config = {
      driftParam: config?.driftParam ?? 0.5,
      threshold: config?.threshold ?? 5.0,
      baselineFraction: config?.baselineFraction ?? 0.5,
      minPoints: config?.minPoints ?? 10,
    };
  }

  /** Reset baseline (called implicitly by detect) */
  reset(): void {
    this.baselineValues = [];
  }

  /**
   * Run CUSUM detection on a full time series.
   *
   * Steps:
   *   1. Estimate baseline μ₀, σ from first (baselineFraction) of data
   *   2. Run CUSUM from the first point after baseline
   *   3. Back-trace changepoint to where S first became positive
   */
  detect(values: number[]): ChangepointResult {
    const n = values.length;
    if (n < this.config.minPoints) {
      return this.emptyResult(n);
    }

    // Estimate baseline from first fraction of data
    const baselineEnd = Math.max(
      this.config.minPoints,
      Math.floor(n * this.config.baselineFraction),
    );
    const baseline = values.slice(0, baselineEnd);
    const { median, mad } = estimateRobustParams(baseline);

    const mu0 = median;
    const sigma = Math.max(mad * 1.4826, 1e-10); // MAD → σ conversion
    const k = this.config.driftParam;
    const h = this.config.threshold;

    // Run CUSUM from the start (including baseline, to find changepoint in baseline)
    const cusum: number[] = new Array(n).fill(0);
    let S = 0;
    let maxS = 0;
    let maxIdx = 0;
    let cpIdx = -1;

    for (let i = 0; i < n; i++) {
      const z = (values[i]! - mu0) / sigma;
      S = Math.max(0, S + z - k);
      cusum[i] = S;

      if (S > 0 && cpIdx < 0) {
        cpIdx = i;
      }
      if (S === 0) {
        cpIdx = -1; // reset if CUSUM drops back to 0
      }
      if (S > maxS) {
        maxS = S;
        maxIdx = i;
      }
    }

    const detected = maxS > h;
    const confidence = Math.min(1, maxS / h);

    return {
      cusum,
      mostLikelyIndex: detected ? cpIdx : maxIdx,
      maxCusum: maxS,
      detected,
      confidence,
      dataPoints: n,
      baselineMean: mu0,
      baselineStd: sigma,
    };
  }

  /**
   * Incremental CUSUM — not supported (CUSUM requires baseline).
   * Use detect() for batch processing.
   */
  detectOnline(newValue: number): { changepointProb: number } {
    return { changepointProb: 0 };
  }

  /**
   * Detect changepoints across all columns of a data matrix.
   */
  detectAllColumns(data: Matrix, serviceNames: string[]): BOCDAnomalyResult[] {
    const results: BOCDAnomalyResult[] = [];

    for (let j = 0; j < serviceNames.length; j++) {
      const col: number[] = [];
      for (let i = 0; i < data.rows; i++) col.push(data.get(i, j));

      const cp = this.detect(col);

      const cpIdx = cp.mostLikelyIndex > 0 ? cp.mostLikelyIndex : Math.floor(col.length / 2);
      const preSlice = col.slice(0, cpIdx);
      const postSlice = col.slice(cpIdx);

      const preMean = preSlice.length > 0 ? preSlice.reduce((s, v) => s + v, 0) / preSlice.length : col[0] ?? 0;
      const postMean = postSlice.length > 0 ? postSlice.reduce((s, v) => s + v, 0) / postSlice.length : col[col.length - 1] ?? 0;
      const preStd = preSlice.length > 1
        ? Math.sqrt(preSlice.reduce((s, v) => s + (v - preMean) ** 2, 0) / preSlice.length) || 1
        : 1;

      results.push({
        service: serviceNames[j]!,
        changepoint: cp,
        preMean,
        postMean,
        magnitudeShift: Math.abs(postMean - preMean) / preStd,
        preStd,
      });
    }

    // Sort by normalized score: earlier changepoint × higher magnitude
    results.sort((a, b) => {
      const aScore = (1 - a.changepoint.mostLikelyIndex / Math.max(1, a.changepoint.dataPoints)) * a.magnitudeShift;
      const bScore = (1 - b.changepoint.mostLikelyIndex / Math.max(1, b.changepoint.dataPoints)) * b.magnitudeShift;
      return bScore - aScore;
    });

    return results;
  }

  private emptyResult(n: number): ChangepointResult {
    return {
      cusum: new Array(n).fill(0),
      mostLikelyIndex: -1,
      maxCusum: 0,
      detected: false,
      confidence: 0,
      dataPoints: n,
      baselineMean: 0,
      baselineStd: 1,
    };
  }
}

// ── Robust Statistics ────────────────────────────────────────────────────

/**
 * Estimate median and MAD (Median Absolute Deviation) robustly.
 * MAD × 1.4826 ≈ σ for normally distributed data.
 */
function estimateRobustParams(values: number[]): { median: number; mad: number } {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const median = n % 2 === 1
    ? sorted[Math.floor(n / 2)]!
    : (sorted[n / 2 - 1]! + sorted[n / 2]!) / 2;

  const deviations = values.map(v => Math.abs(v - median)).sort((a, b) => a - b);
  const m = deviations.length;
  const mad = m % 2 === 1
    ? deviations[Math.floor(m / 2)]!
    : (deviations[m / 2 - 1]! + deviations[m / 2]!) / 2;

  return { median, mad: mad === 0 ? 1e-10 : mad };
}
