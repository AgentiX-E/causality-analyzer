/**
 * SPOT / DSPOT — Streaming Peaks-Over-Threshold anomaly detector.
 *
 * Based on "Anomaly Detection in Streams with Extreme Value Theory"
 * (Siffer et al., KDD 2017).
 *
 * Models the tail distribution of data using the Generalized Pareto
 * Distribution (GPD). Parameters are estimated via Grimshaw's trick.
 * The DSPOT variant handles concept drift by subtracting a local
 * moving average before comparison.
 */
import type { DetectionResult } from '@agentix-e/causality-analyzer-core';

export interface SPOTConfig {
  /** Risk level: probability of false positive (default 1e-4) */
  q: number;
  /** Initial threshold quantile for peak selection (default 0.98) */
  initThresholdQuantile: number;
  /** Maximum number of peaks to retain (default 10000) */
  maxPeaks: number;
  /** Number of initial data points for calibration */
  initSize: number;
}

export interface DSPOTConfig extends SPOTConfig {
  /** Window size for local moving average (default 100) */
  driftWindow: number;
}

/** GPD parameter estimate */
interface GPDParams { gamma: number; sigma: number; }

/**
 * SPOT detector — models upper-tail exceedances via GPD.
 */
export class SPOTDetector {
  readonly config: SPOTConfig;
  protected initData: number[] = [];
  protected peaks: number[] = [];
  protected Nt = 0;   // total count of peaks
  protected n = 0;    // total observations
  protected threshold = 0;
  protected extremeQuantile = 0;
  protected gpd: GPDParams | null = null;
  protected initialized = false;

  constructor(config: Partial<SPOTConfig> = {}) {
    this.config = {
      q: config.q ?? 1e-4,
      initThresholdQuantile: config.initThresholdQuantile ?? 0.98,
      maxPeaks: config.maxPeaks ?? 10000,
      initSize: config.initSize ?? 1000,
    };
  }

  /** Initialize with historical data to calibrate GPD */
  initialize(data: number[]): void {
    this.initData = [...data].sort((a, b) => a - b);
    const idx = Math.floor(this.config.initThresholdQuantile * this.initData.length);
    this.threshold = this.initData[idx]!;

    // Extract initial peaks (exceedances above threshold)
    for (const v of data) {
      if (v > this.threshold) {
        this.peaks.push(v - this.threshold);
        this.Nt++;
      }
    }
    this.n = data.length;

    if (this.peaks.length > 5) {
      this.gpd = estimateGPD(this.peaks);
      this.extremeQuantile = this.computeQuantile();
      this.initialized = true;
    }
  }

  /** Streaming: process a single data point */
  update(value: number): DetectionResult {
    const adjValue = this.transformValue(value);
    this.n++;

    if (!this.initialized) {
      this.initData.push(adjValue);
      if (this.initData.length >= this.config.initSize) {
        this.initialize(this.initData);
      }
      return { isAnomalous: false, labels: new Float64Array([0]), scores: new Float64Array([0]), timestamp: Date.now(), metadata: { method: this.getMethodName(), stage: 'warming' } };
    }

    if (adjValue > this.extremeQuantile) {
      return { isAnomalous: true, labels: new Float64Array([1]), scores: new Float64Array([1.0]), timestamp: Date.now(), metadata: { method: this.getMethodName(), threshold: this.extremeQuantile, actualValue: adjValue } };
    }

    if (adjValue > this.threshold) {
      this.peaks.push(adjValue - this.threshold);
      this.Nt++;
      if (this.peaks.length > this.config.maxPeaks) this.peaks.shift();
      this.gpd = estimateGPD(this.peaks);
      this.extremeQuantile = this.computeQuantile();
    }

    return { isAnomalous: false, labels: new Float64Array([0]), scores: new Float64Array([1 - adjValue / this.extremeQuantile]), timestamp: Date.now(), metadata: { method: this.getMethodName() } };
  }

  protected transformValue(v: number): number { return v; }
  protected getMethodName(): string { return 'spot'; }

  private computeQuantile(): number {
    if (!this.gpd) return this.threshold * 2;
    const { gamma, sigma } = this.gpd;
    const r = this.config.q * this.n / this.Nt;
    if (Math.abs(gamma) < 1e-10) return this.threshold - sigma * Math.log(r);
    return this.threshold + (sigma / gamma) * (Math.pow(r, -gamma) - 1);
  }
}

/**
 * DSPOT detector — extends SPOT with concept drift handling.
 * Subtracts a local exponential moving average before comparison.
 */
export class DSPOTDetector extends SPOTDetector {
  private driftValues: number[] = [];
  private ema = 0;
  private emaInitialized = false;

  constructor(config: Partial<DSPOTConfig> = {}) {
    super(config);
    Object.assign(this.config, { driftWindow: config.driftWindow ?? 100 });
  }

  protected override transformValue(v: number): number {
    if (!this.emaInitialized) {
      this.driftValues.push(v);
      if (this.driftValues.length >= (this.config as DSPOTConfig).driftWindow) {
        this.ema = this.driftValues.reduce((a, b) => a + b, 0) / this.driftValues.length;
        this.emaInitialized = true;
      }
      return v;
    }
    const alpha = 2 / ((this.config as DSPOTConfig).driftWindow + 1);
    this.ema = alpha * v + (1 - alpha) * this.ema;
    return v - this.ema;
  }

  protected override getMethodName(): string { return 'dspot'; }
}

// ── GPD Estimation via Grimshaw's Trick ──────────────────────────────

function estimateGPD(peaks: number[]): GPDParams {
  const n = peaks.length;
  if (n < 5) return { gamma: 0, sigma: peaks.reduce((a, b) => a + b, 0) / n };

  const maxY = Math.max(...peaks);
  if (maxY <= 0) return { gamma: 0, sigma: 1 };

  // Grimshaw's trick: search for root of w(s) = 0
  let bestGamma = 0, bestSigma = peaks.reduce((a, b) => a + b, 0) / n;
  let bestLL = -Infinity;

  const interval1 = [1e-10 / maxY, (1 - 1e-10) / maxY]; // (0, 1/maxY)
  const interval2 = [-(1 - 1e-10) / maxY, -1e-10 / maxY]; // (-1/maxY, 0)

  for (const interval of [interval1, interval2]) {
      const lo = interval[0]!, hi = interval[1]!;
    for (let k = 0; k < 50; k++) {
      const s = lo + (hi - lo) * k / 49;
      const w = grimshawW(s, peaks);
      if (Number.isNaN(w)) continue;

      // If w(s) crosses zero between consecutive points
      if (k > 0) {
        const sPrev = lo + (hi - lo) * (k - 1) / 49;
        const wPrev = grimshawW(sPrev, peaks);
        if (!Number.isNaN(wPrev) && wPrev * w <= 0) {
          // Refine with bisection
          const refined = bisectGrimshaw(sPrev, s, peaks);
          if (refined) {
            const gamma = refined.gamma;
            const sigma = refined.sigma;
            if (sigma > 0) {
              const ll = gpdLogLikelihood(peaks, gamma, sigma);
              if (ll > bestLL) { bestLL = ll; bestGamma = gamma; bestSigma = sigma; }
            }
          }
        }
      }
    }
  }

  // Fallback: method of moments
  if (bestSigma <= 0 || !isFinite(bestLL)) {
    const mean = peaks.reduce((a, b) => a + b, 0) / n;
    let ss = 0; for (const p of peaks) ss += (p - mean) ** 2;
    const variance = ss / n;
    bestSigma = 0.5 * mean * (mean * mean / variance + 1);
    bestGamma = 0.5 * (mean * mean / variance - 1);
    if (bestSigma <= 0) bestSigma = 1;
  }

  // Safety: clamp gamma to realistic range [-0.5, 0.5].
  // Unbounded positive gamma produces extreme quantiles that
  // approach infinity, causing the detector to miss genuine
  // anomalies. Negative gamma <-0.5 is also unrealistic for
  // most real-world heavy-tailed distributions.
  if (bestGamma > 0.5) bestGamma = 0.5;
  if (bestGamma < -0.5) bestGamma = -0.5;
  if (bestSigma <= 0) bestSigma = 1;

  return { gamma: bestGamma, sigma: bestSigma };
}

function grimshawW(s: number, peaks: number[]): number {
  const ys = peaks.map(y => 1 - s * y);
  if (ys.some(y => y <= 0)) return NaN;
  const logYs = ys.map(y => Math.log(y));
  const meanLog = logYs.reduce((a, b) => a + b, 0) / peaks.length;
  const meanInv = ys.reduce((a, b) => a + 1 / b, 0) / peaks.length;
  return (1 + meanLog) * meanInv - 1;
}

function bisectGrimshaw(a: number, b: number, peaks: number[]): GPDParams | null {
  for (let i = 0; i < 30; i++) {
    const m = (a + b) / 2;
    const wA = grimshawW(a, peaks), wM = grimshawW(m, peaks);
    if (Number.isNaN(wM)) return null;
    if (Math.abs(wM) < 1e-8 || Math.abs(b - a) < 1e-12) {
      const logs = peaks.map(y => Math.log(1 - m * y));
      const gamma = logs.reduce((s, l) => s + l, 0) / peaks.length;
      return { gamma, sigma: -gamma / m };
    }
    if (wA * wM < 0) b = m; else a = m;
  }
  return null;
}

function gpdLogLikelihood(peaks: number[], gamma: number, sigma: number): number {
  if (sigma <= 0) return -Infinity;
  const n = peaks.length;
  if (Math.abs(gamma) < 1e-10) return -n * Math.log(sigma) - peaks.reduce((s, y) => s + y, 0) / sigma;
  const terms = peaks.map(y => 1 + gamma * y / sigma);
  if (terms.some(t => t <= 0)) return -Infinity;
  return -n * Math.log(sigma) - (1 + 1 / gamma) * terms.reduce((s, t) => s + Math.log(t), 0);
}
