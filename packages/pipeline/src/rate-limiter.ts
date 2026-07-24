/**
 * Rate Limiter — backpressure for streaming anomaly detectors.
 *
 * Production AIOps pipelines ingest high-frequency metric streams.
 * When detectors cannot keep up, this module provides configurable
 * overflow strategies to prevent unbounded memory growth.
 *
 * Strategies:
 *   - drop_oldest: discard oldest buffered samples (default)
 *   - drop_newest: discard incoming samples when full
 *   - block: reject new samples with a full-buffer signal
 *
 * @packageDocumentation
 */

export type OverflowStrategy = 'drop_oldest' | 'drop_newest' | 'block' | 'sliding_window';

export interface RateLimiterConfig {
  /** Maximum number of buffered data points */
  maxBufferSize: number;
  /** Overflow strategy */
  strategy?: OverflowStrategy;
}

export interface RateLimitResult {
  /** Whether the point was accepted */
  accepted: boolean;
  /** Number of points dropped (cumulative since creation) */
  dropped: number;
  /** Current buffer utilization (0-1) */
  utilization: number;
}

/**
 * Bounded ring buffer with configurable overflow strategy.
 */
export class RateLimiter {
  private buffer: number[][] = [];
  private droppedCount = 0;
  private readonly maxSize: number;
  private readonly strategy: OverflowStrategy;

  constructor(config: RateLimiterConfig) {
    this.maxSize = Math.max(1, config.maxBufferSize);
    this.strategy = config.strategy ?? 'drop_oldest';
  }

  /** Push a data point. Returns acceptance status and metrics. */
  push(point: number[]): RateLimitResult {
    if (this.buffer.length < this.maxSize) {
      this.buffer.push(point);
      return { accepted: true, dropped: this.droppedCount, utilization: this.buffer.length / this.maxSize };
    }

    // Buffer full — apply overflow strategy
    this.droppedCount++;

    switch (this.strategy) {
      case 'drop_newest':
        return { accepted: false, dropped: this.droppedCount, utilization: 1 };
      case 'block':
        return { accepted: false, dropped: this.droppedCount, utilization: 1 };
      case 'sliding_window':
        // Like drop_oldest but keeps buffer at maxSize-1 to avoid re-shift overhead
        this.buffer.shift();
        this.buffer.push(point);
        return { accepted: true, dropped: this.droppedCount, utilization: 1 };
      case 'drop_oldest':
      default:
        this.buffer.shift();
        this.buffer.push(point);
        return { accepted: true, dropped: this.droppedCount, utilization: 1 };
    }
  }

  /** Drain all buffered points (for batch processing) */
  drain(): number[][] { const pts = this.buffer; this.buffer = []; return pts; }

  /** Number of points currently buffered */
  get size(): number { return this.buffer.length; }

  /** Total points dropped since creation */
  get dropped(): number { return this.droppedCount; }

  /** Reset the limiter state */
  reset(): void { this.buffer = []; this.droppedCount = 0; }
}
