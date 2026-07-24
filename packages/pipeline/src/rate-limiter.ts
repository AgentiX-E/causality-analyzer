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
  /** Maximum number of buffered data points */
  maxBufferSize: number;
  /** Overflow strategy */
  strategy?: OverflowStrategy;
}

export interface TokenBucketConfig {
  /** Tokens per second refill rate */
  rate: number;
  /** Maximum burst capacity (tokens) */
  capacity: number;
  /** Initial tokens (default = capacity) */
  initialTokens?: number;
}

export interface RateLimitResult {
  accepted: boolean;
  dropped: number;
  utilization: number;
}

/**
 * Token Bucket rate limiter — governs throughput by consuming tokens.
 *
 * Unlike the queue-based RateLimiter (which limits buffer depth),
 * TokenBucket limits processing RATE. Tokens refill at `rate` per second,
 * up to `capacity` max. Each `tryConsume(n)` attempt consumes `n` tokens.
 * If insufficient, the request is rejected.
 *
 * Reference: Google SRE Book, Chapter 22 — "Addressing Cascading Failures"
 */
export class TokenBucket {
  private tokens: number;
  private readonly rate: number;
  private readonly capacity: number;
  private lastRefill: number;
  private rejected: number = 0;
  private accepted: number = 0;

  constructor(config: TokenBucketConfig) {
    this.rate = config.rate;
    this.capacity = config.capacity;
    this.tokens = config.initialTokens ?? config.capacity;
    this.lastRefill = Date.now();
  }

  /** Refill tokens based on elapsed time */
  private refill(): void {
    const now = Date.now();
    const elapsed = (now - this.lastRefill) / 1000;
    this.tokens = Math.min(this.capacity, this.tokens + elapsed * this.rate);
    this.lastRefill = now;
  }

  /**
   * Try to consume `n` tokens. Returns true if successful.
   * When n=0, returns true unconditionally.
   */
  tryConsume(n: number = 1): RateLimitResult {
    this.refill();
    const utilization = 1 - this.tokens / this.capacity;

    if (this.tokens >= n) {
      this.tokens -= n;
      this.accepted++;
      return { accepted: true, dropped: this.rejected, utilization: 1 - this.tokens / this.capacity };
    }

    this.rejected++;
    return { accepted: false, dropped: this.rejected, utilization };
  }

  /** Wait for n tokens to become available (approximate, non-blocking). Returns estimated wait ms. */
  waitTime(n: number = 1): number {
    this.refill();
    if (this.tokens >= n) return 0;
    const deficit = n - this.tokens;
    return Math.ceil((deficit / this.rate) * 1000);
  }

  /** Current token count */
  get availableTokens(): number {
    this.refill();
    return this.tokens;
  }

  /** Total requests accepted */
  get totalAccepted(): number { return this.accepted; }

  /** Total requests rejected */
  get totalRejected(): number { return this.rejected; }

  /** Reset limiter state */
  reset(): void {
    this.tokens = this.capacity;
    this.rejected = 0;
    this.accepted = 0;
    this.lastRefill = Date.now();
  }
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
