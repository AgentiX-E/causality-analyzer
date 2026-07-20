/**
 * Spectral Residual anomaly detector.
 *
 * Based on "Time-Series Anomaly Detection Service at Microsoft" (Ren et al., KDD 2019).
 * Transforms signal to frequency domain, computes spectral residual, then inverse
 * transforms to obtain a saliency map. Points exceeding the threshold are flagged.
 */
import type { DetectionResult } from '@agentix-e/causality-analyzer-core';

export interface SRConfig {
  /** Window size for spectral averaging (must be odd, default 3) */
  magWindow: number;
  /** Window size for local score averaging in time domain */
  scoreWindow: number;
  /** Detection threshold multiplier (default 3.0) */
  threshold: number;
  /** Minimum data points for FFT (power of 2, default 128) */
  minPoints: number;
}

export class SpectralResidualDetector {
  readonly config: SRConfig;
  private buffer: number[] = [];

  constructor(config: Partial<SRConfig> = {}) {
    this.config = {
      magWindow: config.magWindow ?? 3,
      scoreWindow: config.scoreWindow ?? 21,
      threshold: config.threshold ?? 3.0,
      minPoints: config.minPoints ?? 32,
    };
  }

  /** Streaming: add a value, detect if anomalous */
  update(value: number): DetectionResult {
    this.buffer.push(value);

    // Keep buffer at manageable size
    if (this.buffer.length > this.config.minPoints * 4) {
      this.buffer = this.buffer.slice(-this.config.minPoints * 2);
    }

    if (this.buffer.length < this.config.minPoints) {
      return { isAnomalous: false, labels: new Float64Array([0]), scores: new Float64Array([0]), timestamp: Date.now(), metadata: { method: 'spectral_residual', stage: 'warming' } };
    }

    const score = this.computeAnomalyScore();
    // Score close to 1 means anomalous; default threshold is 3*sem
    const isAnomaly = score > this.config.threshold * this.computeSEM();
    return { isAnomalous: isAnomaly, labels: new Float64Array([isAnomaly ? 1 : 0]), scores: new Float64Array([score]), timestamp: Date.now(), metadata: { method: 'spectral_residual' } };
  }

  private computeSEM(): number {
    // Compute standard error of the mean from recent scores
    const recent = this.buffer.slice(-10).map((_, i) => {
      const sub = this.buffer.slice(Math.max(0, i - 5), i + 5);
      return this.computeScoreForWindow(sub);
    });
    if (recent.length < 5) return 0.1;
    let sum = 0; for (const v of recent) sum += v;
    const mean = sum / recent.length;
    let ss = 0; for (const v of recent) ss += (v - mean) ** 2;
    return Math.sqrt(ss / recent.length) || 0.1;
  }

  private computeAnomalyScore(): number {
    return this.computeScoreForWindow(this.buffer);
  }

  private computeScoreForWindow(data: number[]): number {
    const n = data.length;
    // Pad to next power of 2
    const N = nextPowerOf2(n);
    const real = new Float64Array(N);
    const imag = new Float64Array(N);
    for (let i = 0; i < n; i++) real[i] = data[i]!;

    // FFT
    fft(real, imag);

    // Log amplitude spectrum
    const logAmp = new Float64Array(N / 2);
    for (let i = 0; i < N / 2; i++) {
      const amp = Math.sqrt(real[i]! ** 2 + imag[i]! ** 2);
      logAmp[i] = Math.log(amp + 1e-10);
    }

    // Spectral average (moving average of log amplitude)
    const avgLogAmp = new Float64Array(N / 2);
    const hw = Math.floor(this.config.magWindow / 2);
    for (let i = 0; i < N / 2; i++) {
      let sum = 0, cnt = 0;
      for (let j = Math.max(0, i - hw); j <= Math.min(N / 2 - 1, i + hw); j++) {
        sum += logAmp[j]!; cnt++;
      }
      avgLogAmp[i] = sum / cnt;
    }

    // Spectral residual
    const residual = new Float64Array(N / 2);
    for (let i = 0; i < N / 2; i++) residual[i] = logAmp[i]! - avgLogAmp[i]!;

    // Prepare for IFFT: use residual as magnitude, original phase
    for (let i = 0; i < N / 2; i++) {
      const phase = Math.atan2(imag[i]!, real[i]!);
      const mag = Math.exp(residual[i]!);
      real[i] = mag * Math.cos(phase);
      imag[i] = mag * Math.sin(phase);
      if (i > 0) { real[N - i] = real[i]!; imag[N - i] = -imag[i]!; }
    }

    // Inverse FFT
    ifft(real, imag);

    // Saliency map (absolute values)
    const saliency = new Float64Array(n);
    for (let i = 0; i < n; i++) saliency[i] = Math.abs(real[i]!) / N;

    // Local score average
    const scores = new Float64Array(n);
    const sw = this.config.scoreWindow;
    for (let i = 0; i < n; i++) {
      let sum = 0, cnt = 0;
      for (let j = Math.max(0, i - sw); j < Math.min(n, i + sw); j++) {
        if (j !== i) { sum += saliency[j]!; cnt++; }
      }
      const localAvg = cnt > 0 ? sum / cnt : saliency[i]!;
      scores[i] = localAvg > 0 ? (saliency[i]! - localAvg) / localAvg : 0;
    }

    // Return anomaly score for last point
    return Math.abs(scores[n - 1]!);
  }

  /** Batch detection on an array */
  detect(data: number[]): DetectionResult[] {
    const results: DetectionResult[] = [];
    const tmp = this.buffer;
    this.buffer = [];
    for (const v of data) results.push(this.update(v));
    this.buffer = tmp;
    return results;
  }
}

// ── FFT Utilities ─────────────────────────────────────────────────────

function nextPowerOf2(n: number): number {
  let p = 1; while (p < n) p *= 2; return p;
}

/** Cooley-Tukey radix-2 FFT (in-place) */
function fft(real: Float64Array, imag: Float64Array): void {
  const n = real.length;
  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j]!, real[i]!];
      [imag[i], imag[j]] = [imag[j]!, imag[i]!];
    }
  }
  // Danielson-Lanczos
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = -2 * Math.PI / len;
    const wReal = Math.cos(angle), wImag = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let curReal = 1, curImag = 0;
      for (let j = 0; j < half; j++) {
        const aReal = real[i + j]!, aImag = imag[i + j]!;
        const bReal = real[i + j + half]!, bImag = imag[i + j + half]!;
        const tReal = curReal * bReal - curImag * bImag;
        const tImag = curReal * bImag + curImag * bReal;
        real[i + j] = aReal + tReal; imag[i + j] = aImag + tImag;
        real[i + j + half] = aReal - tReal; imag[i + j + half] = aImag - tImag;
        const nReal = curReal * wReal - curImag * wImag;
        curImag = curReal * wImag + curImag * wReal;
        curReal = nReal;
      }
    }
  }
}

function ifft(real: Float64Array, imag: Float64Array): void {
  const n = real.length;
  for (let i = 0; i < n; i++) imag[i] = -imag[i]!;
  fft(real, imag);
  for (let i = 0; i < n; i++) { real[i]! /= n; imag[i] = -imag[i]! / n; }
}
