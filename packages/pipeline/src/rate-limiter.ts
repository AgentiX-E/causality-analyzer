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

export type OverflowStrategy = 'drop_oldest' | 'drop_newest' | 'block';

export interface RateLimiterConfig {
  /** Maximum number of buffered data points (for streaming buffer mode) */
  maxBufferSize?: number;
  /** Overflow strategy (for data point buffer mode) */
  strategy?: OverflowStrategy;
  /** Maximum requests per window (for HTTP rate limiting) */
  maxRequests?: number;
  /** Window duration in ms (for HTTP rate limiting, default 60000) */
  windowMs?: number;
}

export interface RateLimitResult {
  /** Whether the point/request was accepted */
  accepted: boolean;
  /** Number of points dropped (cumulative since creation) */
  dropped: number;
  /** Current buffer utilization (0-1) */
  utilization: number;
}

export interface RateLimitCheckResult {
  /** Whether the request is allowed */
  allowed: boolean;
  /** Remaining requests in current window */
  remaining: number;
  /** Time until reset in ms */
  resetInMs: number;
}

/**
 * Bounded ring buffer with configurable overflow strategy.
 */
export class RateLimiter {
  private buffer: number[][] = [];
  private droppedCount = 0;
  private readonly maxSize: number;
  private readonly strategy: OverflowStrategy;

  // HTTP rate limiting state (token bucket per client key)
  private windowBuckets = new Map<string, { count: number; windowStart: number }>();
  private readonly maxRequests: number;
  private readonly windowMs: number;

  constructor(config: RateLimiterConfig) {
    this.maxSize = Math.max(1, config.maxBufferSize ?? 1000);
    this.strategy = config.strategy ?? 'drop_oldest';
    this.maxRequests = config.maxRequests ?? 100;
    this.windowMs = config.windowMs ?? 60000;
  }

  /**
   * HTTP request rate limiting check.
   * Uses a sliding window per client key (e.g., IP address).
   */
  check(key: string): RateLimitCheckResult {
    const now = Date.now();
    const bucket = this.windowBuckets.get(key);

    if (!bucket || now - bucket.windowStart >= this.windowMs) {
      // New window
      this.windowBuckets.set(key, { count: 1, windowStart: now });
      return { allowed: true, remaining: this.maxRequests - 1, resetInMs: this.windowMs };
    }

    bucket.count++;
    if (bucket.count > this.maxRequests) {
      const resetInMs = this.windowMs - (now - bucket.windowStart);
      return { allowed: false, remaining: 0, resetInMs: Math.max(0, resetInMs) };
    }

    return {
      allowed: true,
      remaining: this.maxRequests - bucket.count,
      resetInMs: this.windowMs - (now - bucket.windowStart),
    };
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
