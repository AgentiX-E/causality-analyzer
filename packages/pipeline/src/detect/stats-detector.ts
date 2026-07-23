import { CONSTANTS } from "../constants.js";
/**
 * Statistical anomaly detector: 3-sigma, MAD, IQR methods.
 *
 * Each detector computes per-metric thresholds from training data
 * and flags points exceeding those thresholds as anomalous.
 */
import type { DetectionResult } from '@agentix-e/causality-analyzer-core';

export type StatsMethod = 'zscore' | 'mad' | 'iqr';

export interface StatsDetectorConfig {
  method: StatsMethod;
  /** Number of standard deviations / MADs / IQR multipliers */
  threshold: number;
  /** Minimum data points required before detection begins */
  minSamples: number;
}

/**
 * Statistical anomaly detector using configurable deviation method.
 *
 * Streaming-ready: `update()` adds a point and returns detection result.
 * Batch mode: `train()` on historical data, `detect()` on test data.
 */
export class StatsDetector {
  private means: Float64Array | null = null;
  private scales: Float64Array | null = null;
  private nSamples = 0;
  private buffer: number[][] = [];
  readonly config: StatsDetectorConfig;

  constructor(config: Partial<StatsDetectorConfig> = {}) {
    this.config = {
      method: config.method ?? 'zscore',
      threshold: config.threshold ?? 3,
      minSamples: config.minSamples ?? 30,
    };
  }

  /** Train on historical data to establish baseline statistics */
  train(data: number[][]): void {
    if (data.length === 0) return;
    const nMetrics = data[0]!.length;
    this.means = new Float64Array(nMetrics);
    this.scales = new Float64Array(nMetrics);
    this.nSamples = data.length;
    const n = data.length;

    for (let m = 0; m < nMetrics; m++) {
      if (this.config.method === 'zscore') {
        // Single-pass: sum + sum-of-squares → mean + std
        let sum = 0, sumSq = 0;
        for (let r = 0; r < n; r++) { const v = data[r]![m]!; sum += v; sumSq += v * v; }
        const mean = sum / n;
        this.means![m] = mean;
        const variance = sumSq / n - mean * mean;
        this.scales![m] = Math.sqrt(Math.max(0, variance)) || 1;
      } else {
        // Extract column (one pass), sort for mad/iqr
        const vals = new Float64Array(n);
        let sum = 0;
        for (let r = 0; r < n; r++) { const v = data[r]![m]!; sum += v; vals[r] = v; }
        const mean = sum / n;
        this.means![m] = mean;

        if (this.config.method === 'mad') {
          // MAD = CONSTANTS.MAD_CONSISTENCY × median(|x_i - median(x)|)
          // The center must be the median, not the mean, for robustness.
          const sVals = Array.from(vals).sort((a, b) => a - b);
          const median = sVals[Math.floor(sVals.length / 2)]!;
          const absDevs = Array.from(vals, v => Math.abs(v - median)).sort((a, b) => a - b);
          this.scales![m] = CONSTANTS.MAD_CONSISTENCY * (absDevs[Math.floor(absDevs.length / 2)]! || 1);
        } else { // iqr
          const sorted = Array.from(vals).sort((a, b) => a - b);
          const q1 = sorted[Math.floor(sorted.length * 0.25)]!;
          const q3 = sorted[Math.floor(sorted.length * 0.75)]!;
          this.scales![m] = (q3 - q1) || 1;
        }
      }
    }
    this.buffer = [];
  }

  /** Streaming: add a single data point, return detection result */
  update(point: number[]): DetectionResult {
    this.buffer.push(point);
    if (this.buffer.length >= this.config.minSamples && !this.means) {
      this.train(this.buffer.splice(0));
    }
    if (!this.means) {
      return { isAnomalous: false, labels: new Float64Array(point.length), scores: new Float64Array(point.length), timestamp: Date.now(), metadata: { method: this.config.method, stage: 'warming' } };
    }

    const n = point.length;
    const labels = new Float64Array(n);
    const scores = new Float64Array(n);
    let anyAnomaly = false;

    for (let i = 0; i < n; i++) {
      const dev = Math.abs(point[i]! - this.means![i]!) / this.scales![i]!;
      scores[i] = dev;
      if (dev > this.config.threshold) { labels[i] = 1; anyAnomaly = true; }
    }

    return { isAnomalous: anyAnomaly, labels, scores, timestamp: Date.now(), metadata: { method: this.config.method } };
  }

  /** Detect anomalies in a batch of data points */
  detect(data: number[][]): DetectionResult[] {
    return data.map(row => this.update(row));
  }
}
