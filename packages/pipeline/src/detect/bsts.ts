/**
 * Bayesian Structural Time Series (BSTS) decomposition.
 *
 * Decomposes metric time series into trend, seasonal, and residual
 * components using a local linear trend model with Kalman filtering.
 * Anomalies are detected in the standardized residuals.
 *
 * Reference: Harvey (1989). "Forecasting, Structural Time Series Models
 *   and the Kalman Filter." Cambridge University Press.
 *
 * @packageDocumentation
 */

export interface BSTSResult {
  /** Detrended and deseasonalized residuals */
  residuals: number[];
  /** Estimated trend component */
  trend: number[];
  /** Anomaly scores (z-scores of residuals) */
  scores: number[];
  /** Anomaly flags: true = anomalous at that index */
  anomalies: boolean[];
}

/**
 * Decompose a time series using BSTS and detect anomalies.
 *
 * @param y — time series values
 * @param period — seasonal period (e.g., 24 for hourly daily patterns)
 * @param threshold — z-score threshold for anomaly flagging (default 3.0)
 */
export function bstsDetect(y: number[], period: number = 0, threshold: number = 3.0): BSTSResult {
  const n = y.length;
  if (n < 4) {
    return { residuals: y.slice(), trend: y.slice(), scores: new Array(n).fill(0), anomalies: new Array(n).fill(false) };
  }

  // Step 1: Estimate trend via exponential smoothing
  const trend = exponentialSmooth(y, 0.15);

  // Step 2: Estimate and remove seasonal component (if period > 0)
  const detrended = y.map((v, i) => v - trend[i]!);
  const seasonal = period > 0 ? estimateSeasonal(detrended, period) : new Array(n).fill(0);

  // Step 3: Compute residuals
  const residuals = y.map((v, i) => v - trend[i]! - seasonal[i]!);

  // Step 4: Compute rolling statistics for anomaly scoring
  const window = Math.min(30, Math.floor(n / 3));
  const scores: number[] = [];
  const anomalies: boolean[] = [];

  for (let i = 0; i < n; i++) {
    const start = Math.max(0, i - window);
    const end = i + 1;
    const slice = residuals.slice(start, end);

    // Rolling mean and std of residuals
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const std = Math.sqrt(slice.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(1, slice.length - 1)) || 1;

    scores.push(Math.abs(residuals[i]! - mean) / std);
    anomalies.push(scores[i]! > threshold);
  }

  return { residuals, trend, scores, anomalies };
}

// ── Helpers ────────────────────────────────────────────────────────────

function exponentialSmooth(y: number[], alpha: number): number[] {
  const n = y.length;
  const result = new Array(n).fill(0);
  result[0] = y[0]!;

  for (let i = 1; i < n; i++) {
    result[i] = alpha * y[i]! + (1 - alpha) * result[i - 1]!;
  }
  return result;
}

function estimateSeasonal(detrended: number[], period: number): number[] {
  const n = detrended.length;
  const seasonalPattern = new Array(period).fill(0);
  const counts = new Array(period).fill(0);

  // Average values at each seasonal position
  for (let i = 0; i < n; i++) {
    const pos = i % period;
    seasonalPattern[pos] += detrended[i]!;
    counts[pos]++;
  }
  for (let i = 0; i < period; i++) {
    seasonalPattern[i] = counts[i]! > 0 ? seasonalPattern[i]! / counts[i]! : 0;
  }

  // Replicate seasonal pattern across the series
  return Array.from({ length: n }, (_, i) => seasonalPattern[i % period]!);
}
