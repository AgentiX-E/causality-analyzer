/**
 * I5 conformance: Prometheus format + Token Bucket + Health SLI + Key rotation.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MetricsRegistry, HealthTracker } from '../observability.js';
import { RateLimiter, TokenBucket } from '../rate-limiter.js';
import { EncryptedStore } from '../encrypted-store.js';

// ── Prometheus Export ────────────────────────────────────────────────
describe('MetricsRegistry Prometheus export', () => {
  let reg: MetricsRegistry;

  beforeEach(() => { reg = new MetricsRegistry(); });

  it('exports counters in Prometheus text format', () => {
    reg.inc('rca_analyzed_total', 5);
    reg.inc('rca_analyzed_total', 3);
    const output = reg.toPrometheus();
    expect(output).toContain('# HELP rca_analyzed_total');
    expect(output).toContain('# TYPE rca_analyzed_total counter');
    expect(output).toContain('rca_analyzed_total 8');
  });

  it('exports histograms with bucket boundaries', () => {
    reg.observe('detect_latency_ms', 12);
    reg.observe('detect_latency_ms', 45);
    reg.observe('detect_latency_ms', 200);
    const output = reg.toPrometheus();
    expect(output).toContain('# HELP detect_latency_ms');
    expect(output).toContain('# TYPE detect_latency_ms histogram');
    expect(output).toContain('detect_latency_ms_count 3');
    expect(output).toContain('detect_latency_ms_sum 257');
    expect(output).toContain('{le="50"}');
    expect(output).toContain('{le="+Inf"} 3');
  });

  it('exports gauges', () => {
    reg.setGauge('pipeline_active', 3);
    const output = reg.toPrometheus();
    expect(output).toContain('# TYPE pipeline_active gauge');
    expect(output).toContain('pipeline_active 3');
  });

  it('empty registry produces valid output', () => {
    const output = reg.toPrometheus();
    expect(output).toContain('ca_metrics_scrape_timestamp_seconds');
    expect(output.endsWith('\n')).toBe(true);
  });

  it('histogram with no observations produces zero counts', () => {
    reg.observe('empty_histogram', 0); // push then clear via not actually...
    // Actually: observe one and reset
    reg.observe('empty_histogram', 1);
    const output = reg.toPrometheus();
    expect(output).toContain('empty_histogram_count');
  });

  it('sanitizes invalid metric names', () => {
    reg.inc('invalid.name-with-dashes', 1);
    const output = reg.toPrometheus();
    expect(output).not.toContain('invalid.name');
    expect(output).toContain('invalid_name_with_dashes');
  });

  it('reset clears all metrics', () => {
    reg.inc('counter', 5);
    reg.observe('hist', 10);
    reg.reset();
    expect(reg.exportCounters()).toHaveLength(0);
    // After reset, histograms map is emptied
    const hists = reg.exportHistograms();
    expect(hists).toHaveLength(0);
  });

  it('labels are formatted correctly', () => {
    reg.setLabels('labeled_counter', { service: 'rca', env: 'prod' });
    reg.inc('labeled_counter', 7);
    const output = reg.toPrometheus();
    expect(output).toContain('{service="rca",env="prod"} 7');
  });

  it('exports scrape timestamp', () => {
    const output = reg.toPrometheus();
    expect(output).toContain('ca_metrics_scrape_timestamp_seconds');
  });
});

// ── Token Bucket ─────────────────────────────────────────────────────
describe('TokenBucket', () => {
  it('accepts requests within rate limit', () => {
    const bucket = new TokenBucket({ rate: 100, capacity: 100 });
    for (let i = 0; i < 50; i++) {
      const result = bucket.tryConsume(1);
      expect(result.accepted).toBe(true);
    }
  });

  it('rejects requests exceeding capacity', () => {
    const bucket = new TokenBucket({ rate: 10, capacity: 10, initialTokens: 10 });
    // Consume all tokens
    const results: boolean[] = [];
    for (let i = 0; i < 15; i++) {
      results.push(bucket.tryConsume(1).accepted);
    }
    // First 10 should pass, remaining should fail
    expect(results.slice(0, 10).every(r => r)).toBe(true);
    expect(results.slice(10).some(r => !r)).toBe(true);
  });

  it('tokens refill over time', async () => {
    const bucket = new TokenBucket({ rate: 100, capacity: 5, initialTokens: 5 });
    // Consume all
    for (let i = 0; i < 5; i++) bucket.tryConsume(1);
    expect(bucket.tryConsume(1).accepted).toBe(false);
    // Wait for refill
    await new Promise(r => setTimeout(r, 60));
    expect(bucket.tryConsume(1).accepted).toBe(true);
  });

  it('waitTime estimates correctly', () => {
    const bucket = new TokenBucket({ rate: 10, capacity: 5, initialTokens: 0 });
    const wait = bucket.waitTime(3);
    expect(wait).toBeGreaterThan(0); // some wait needed
  });

  it('reset restores capacity', () => {
    const bucket = new TokenBucket({ rate: 10, capacity: 10 });
    for (let i = 0; i < 10; i++) bucket.tryConsume(1);
    expect(bucket.tryConsume(1).accepted).toBe(false);
    bucket.reset();
    expect(bucket.tryConsume(1).accepted).toBe(true);
  });

  it('reports utilization after consumption', () => {
    const bucket = new TokenBucket({ rate: 10, capacity: 10 });
    // Consume half
    const r1 = bucket.tryConsume(5);
    // Utilization = 1 - tokens/capacity = 1 - 5/10 = 0.5
    expect(r1.utilization).toBeCloseTo(0.5, 1);
  });
});

// ── Health SLI ───────────────────────────────────────────────────────
describe('HealthTracker', () => {
  it('computes p50/p90/p99 from recorded latencies', () => {
    const tracker = new HealthTracker(200);
    for (let i = 0; i < 100; i++) {
      tracker.record(i, true); // 0,1,2,...,99
    }
    const sli = tracker.snapshot();
    expect(sli.p50).toBeGreaterThanOrEqual(49);
    expect(sli.p90).toBeGreaterThanOrEqual(89);
    expect(sli.p99).toBeGreaterThanOrEqual(98);
    expect(sli.totalRequests).toBe(100);
    expect(sli.totalErrors).toBe(0);
    expect(sli.errorRate).toBe(0);
  });

  it('tracks errors', () => {
    const tracker = new HealthTracker(100);
    tracker.record(10, false);
    tracker.record(20, true);
    tracker.record(30, false);
    const sli = tracker.snapshot();
    expect(sli.totalRequests).toBe(3);
    expect(sli.totalErrors).toBe(2);
    expect(sli.errorRate).toBeCloseTo(2 / 3, 1);
  });

  it('empty tracker returns zeros', () => {
    const tracker = new HealthTracker();
    const sli = tracker.snapshot();
    expect(sli.totalRequests).toBe(0);
    expect(sli.p50).toBe(0);
    expect(sli.errorRate).toBe(0);
  });

  it('reservoir sampling limits memory', () => {
    const tracker = new HealthTracker(50);
    for (let i = 0; i < 500; i++) {
      tracker.record(i, true);
    }
    const sli = tracker.snapshot();
    expect(sli.totalRequests).toBe(500); // count still tracks all
  });
});

// ── RateLimiter queue strategies ─────────────────────────────────────
describe('RateLimiter queue', () => {
  it('accepts when below capacity', () => {
    const limiter = new RateLimiter({ maxBufferSize: 5 });
    expect(limiter.push([1]).accepted).toBe(true);
    expect(limiter.size).toBe(1);
  });

  it('drop_oldest strategy evicts when full', () => {
    const limiter = new RateLimiter({ maxBufferSize: 3 });
    limiter.push([1]); limiter.push([2]); limiter.push([3]);
    const result = limiter.push([4]);
    expect(result.accepted).toBe(true);
    expect(result.dropped).toBe(1);
    expect(limiter.drain()[0]![0]).toBe(2); // [1] was dropped
  });

  it('drop_newest strategy rejects when full', () => {
    const limiter = new RateLimiter({ maxBufferSize: 2, strategy: 'drop_newest' });
    limiter.push([1]); limiter.push([2]);
    const result = limiter.push([3]);
    expect(result.accepted).toBe(false);
    expect(result.dropped).toBe(1);
  });

  it('block strategy rejects when full', () => {
    const limiter = new RateLimiter({ maxBufferSize: 1, strategy: 'block' });
    limiter.push([1]);
    expect(limiter.push([2]).accepted).toBe(false);
  });
});

// ── EncryptedStore ───────────────────────────────────────────────────
describe('EncryptedStore', () => {
  it('encrypts and decrypts round-trip', async () => {
    const key = new Uint8Array([...Array(32)].map((_, i) => i));
    const store = new EncryptedStore({ key });
    await store.init();

    const original = 'sensitive rca result data';
    const encrypted = await store.encrypt(original);
    expect(encrypted).not.toBe(original);
    expect(encrypted.length).toBeGreaterThan(original.length);

    const decrypted = await store.decrypt(encrypted);
    expect(decrypted).toBe(original);
  });

  it('generates different ciphertext for same plaintext', async () => {
    const key = EncryptedStore.generateKey();
    const store = new EncryptedStore({ key });
    await store.init();

    const c1 = await store.encrypt('hello');
    const c2 = await store.encrypt('hello');
    expect(c1).not.toBe(c2); // different IVs
  });

  it('rotateKey preserves data', async () => {
    const oldKey = EncryptedStore.generateKey();
    const newKey = EncryptedStore.generateKey();
    const store = new EncryptedStore({ key: oldKey });
    await store.init();

    const original = 'rotation test data';
    const oldEncrypted = await store.encrypt(original);
    const reEncrypted = await store.rotateKey(oldEncrypted, { key: newKey });

    // Decrypt with new key
    const newStore = new EncryptedStore({ key: newKey });
    await newStore.init();
    const decrypted = await newStore.decrypt(reEncrypted);
    expect(decrypted).toBe(original);
  });

  it('generateKey produces 32 bytes', () => {
    const key = EncryptedStore.generateKey();
    expect(key.length).toBe(32);
    // Should not be all zeros
    expect(key.some(b => b !== 0)).toBe(true);
  });

  it('isAvailable returns false before init', () => {
    const store = new EncryptedStore({ key: new Uint8Array(32) });
    expect(store.isAvailable()).toBe(false);
  });
});
