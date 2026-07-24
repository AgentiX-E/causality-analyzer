/**
 * Observability — Structured audit logging + metrics instrumentation.
 *
 * Zero-dependency observability layer for production deployments.
 * Supports JSON-structured audit logs and Prometheus-compatible metrics.
 * All exports are optional — the library functions without this module.
 *
 * @packageDocumentation
 */

import type { Logger } from '@agentix-e/causality-analyzer-core';

// ── Audit Logger ───────────────────────────────────────────────────────

export interface AuditEntry {
  /** ISO 8601 timestamp */
  timestamp: string;
  /** Event type (e.g. 'rca.analyze', 'pipeline.detect') */
  event: string;
  /** Duration in milliseconds */
  durationMs: number;
  /** Whether the operation succeeded */
  success: boolean;
  /** Arbitrary context data */
  context?: Record<string, unknown>;
  /** Optional error message */
  error?: string;
}

/**
 * Immutable audit logger — produces JSON-streamable audit entries
 * for downstream consumption (files, databases, SIEM systems).
 * Optionally forwards entries through a structured Logger for real-time monitoring.
 */
export class AuditLogger {
  private entries: AuditEntry[] = [];
  private logger: Logger | null = null;

  constructor(logger?: Logger) {
    this.logger = logger ?? null;
  }

  /**
   * Record an auditable event.
   * Thread-safe for sequential use; not designed for concurrent writes.
   */
  log(event: string, durationMs: number, success: boolean, context?: Record<string, unknown>, error?: string): void {
    const entry: AuditEntry = {
      timestamp: new Date().toISOString(),
      event,
      durationMs: Math.round(durationMs * 100) / 100,
      success,
      context,
      error,
    };
    this.entries.push(entry);

    // Forward to structured logger for real-time monitoring
    if (this.logger) {
      const level = success ? 'info' : 'error';
      this.logger[level]?.(`[audit] ${event}`, {
        durationMs,
        success,
        ...context,
        ...(error ? { error } : {}),
      });
    }
  }

  /** Return all entries as a JSON-encodable array (immutable copy) */
  toJSON(): AuditEntry[] { return [...this.entries]; }

  /** Clear all audit entries (for testing or rotation) */
  clear(): void { this.entries = []; }

  /** Number of entries logged */
  get count(): number { return this.entries.length; }
}

// ── Metrics Instrumentation ────────────────────────────────────────────

/** A single named counter metric */
export interface MetricCounter {
  name: string;
  value: number;
  labels?: Record<string, string>;
}

/** A single histogram metric (summary statistics) */
export interface MetricHistogram {
  name: string;
  count: number;
  sum: number;
  min: number;
  max: number;
  labels?: Record<string, string>;
}

/**
 * Lightweight Prometheus-compatible metrics registry.
 * Supports counters and histograms for RCA pipeline observability.
 *
 * Includes Prometheus Exposition Format (text) export for direct
 * scraping by Prometheus/VictoriaMetrics/Grafana Agent.
 */
export class MetricsRegistry {
  private counters = new Map<string, number>();
  private histograms = new Map<string, number[]>();
  private gauges = new Map<string, number>();
  private labels = new Map<string, Record<string, string>>();

  /** Increment a named counter */
  inc(name: string, by: number = 1): void {
    this.counters.set(name, (this.counters.get(name) ?? 0) + by);
  }

  /** Set a gauge value */
  setGauge(name: string, value: number): void {
    this.gauges.set(name, value);
  }

  /** Observe a value for a named histogram */
  observe(name: string, value: number): void {
    const h = this.histograms.get(name) ?? [];
    h.push(value);
    this.histograms.set(name, h);
  }

  /** Attach labels to a metric */
  setLabels(name: string, labels: Record<string, string>): void {
    this.labels.set(name, labels);
  }

  /** Export all counters as Prometheus-compatible text */
  exportCounters(): MetricCounter[] {
    return [...this.counters.entries()].map(([name, value]) => ({ name, value }));
  }

  /** Export all histogram summaries */
  exportHistograms(): MetricHistogram[] {
    return [...this.histograms.entries()].map(([name, values]) => {
      if (values.length === 0) return { name, count: 0, sum: 0, min: 0, max: 0 };
      const sorted = [...values].sort((a, b) => a - b);
      return {
        name,
        count: values.length,
        sum: values.reduce((a, b) => a + b, 0),
        min: sorted[0]!,
        max: sorted[sorted.length - 1]!,
      };
    });
  }

  /**
   * Export metrics in Prometheus Exposition Format (text).
   *
   * Produces standards-compliant output for direct scraping:
   * ```
   * # HELP ca_rca_analyzed_total Total RCA analyses performed
   * # TYPE ca_rca_analyzed_total counter
   * ca_rca_analyzed_total 42
   * # HELP ca_detect_latency_ms RCA detection latency
   * # TYPE ca_detect_latency_ms histogram
   * ca_detect_latency_ms_count 100
   * ca_detect_latency_ms_sum 4523.5
   * ca_detect_latency_ms_bucket{le="1"} 10
   * ```
   */
  toPrometheus(): string {
    const lines: string[] = [];
    const now = Date.now();

    // Counters
    for (const [name, value] of this.counters) {
      const safeName = sanitizeMetricName(name);
      const labels = formatLabels(this.labels.get(name));
      lines.push(`# HELP ${safeName} ${safeName} counter`);
      lines.push(`# TYPE ${safeName} counter`);
      lines.push(`${safeName}${labels} ${value}`);
    }

    // Gauges
    for (const [name, value] of this.gauges) {
      const safeName = sanitizeMetricName(name);
      lines.push(`# HELP ${safeName} ${safeName} gauge`);
      lines.push(`# TYPE ${safeName} gauge`);
      lines.push(`${safeName} ${value}`);
    }

    // Histograms
    for (const [name, values] of this.histograms) {
      const safeName = sanitizeMetricName(name);
      const labels = formatLabels(this.labels.get(name));
      lines.push(`# HELP ${safeName} ${safeName} histogram`);
      lines.push(`# TYPE ${safeName} histogram`);
      if (values.length > 0) {
        const sorted = [...values].sort((a, b) => a - b);
        const sum = sorted.reduce((a, b) => a + b, 0);
        lines.push(`${safeName}_count${labels} ${sorted.length}`);
        lines.push(`${safeName}_sum${labels} ${sum}`);
        // Exponential buckets: 1, 5, 10, 50, 100, 500, 1000, 5000
        const buckets = [1, 5, 10, 50, 100, 500, 1000, 5000, Infinity];
        const counts = new Array(buckets.length).fill(0);
        for (const v of sorted) {
          for (let b = 0; b < buckets.length; b++) {
            if (v <= buckets[b]!) { counts[b]++; break; }
          }
        }
        let cum = 0;
        for (let b = 0; b < buckets.length - 1; b++) {
          cum += counts[b]!;
          lines.push(`${safeName}_bucket${labels}{le="${buckets[b]}"} ${cum}`);
        }
        lines.push(`${safeName}_bucket${labels}{le="+Inf"} ${sorted.length}`);
      } else {
        lines.push(`${safeName}_count${labels} 0`);
        lines.push(`${safeName}_sum${labels} 0`);
      }
    }

    // Metadata gauge
    lines.push(`# HELP ca_metrics_scrape_timestamp_seconds Last metrics scrape timestamp`);
    lines.push(`# TYPE ca_metrics_scrape_timestamp_seconds gauge`);
    lines.push(`ca_metrics_scrape_timestamp_seconds ${(now / 1000).toFixed(3)}`);

    return lines.join('\n') + '\n';
  }

  /** Reset all metrics */
  reset(): void { this.counters.clear(); this.histograms.clear(); this.gauges.clear(); this.labels.clear(); }
}

// ── Prometheus helpers ──────────────────────────────────────────────

function sanitizeMetricName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_]/g, '_').replace(/^[^a-zA-Z]/, '_');
}

function formatLabels(labels?: Record<string, string>): string {
  if (!labels || Object.keys(labels).length === 0) return '';
  const parts = Object.entries(labels)
    .map(([k, v]) => `${sanitizeMetricName(k)}="${v.replace(/"/g, '\\"')}"`)
    .join(',');
  return `{${parts}}`;
}

// ── Health SLI ──────────────────────────────────────────────────────
export interface HealthSLI {
  /** Requests per second (rolling window) */
  rps: number;
  /** p50 latency in ms */
  p50: number;
  /** p90 latency in ms */
  p90: number;
  /** p99 latency in ms */
  p99: number;
  /** Error rate (0-1) */
  errorRate: number;
  /** Total requests processed */
  totalRequests: number;
  /** Total errors */
  totalErrors: number;
}

/**
 * Health SLI tracker — rolling-window latency + error rate.
 *
 * Maintains a ring buffer of recent request latencies and outcomes.
 * Uses reservoir sampling for long-running metric retention.
 */
export class HealthTracker {
  private latencies: number[] = [];
  private errors = 0;
  private requests = 0;
  private readonly maxSamples: number;

  constructor(maxSamples = 1000) {
    this.maxSamples = maxSamples;
  }

  /** Record a request outcome */
  record(latencyMs: number, success: boolean): void {
    if (this.latencies.length >= this.maxSamples) {
      // Reservoir sample: replace random element
      const idx = Math.floor(Math.random() * (this.requests + 1));
      if (idx < this.maxSamples) this.latencies[idx] = latencyMs;
    } else {
      this.latencies.push(latencyMs);
    }
    this.requests++;
    if (!success) this.errors++;
  }

  /** Compute current SLI snapshot */
  snapshot(rollingWindowMs?: number): HealthSLI {
    let lats = [...this.latencies];
    if (rollingWindowMs && lats.length > 10) {
      lats = lats.slice(-Math.ceil(lats.length * 0.5)); // approximate rolling window
    }
    lats.sort((a, b) => a - b);
    const n = lats.length;
    return {
      rps: n > 0 ? n / 60 : 0, // approximate 60s window RPS
      p50: n > 0 ? percentile(lats, 0.50) : 0,
      p90: n > 0 ? percentile(lats, 0.90) : 0,
      p99: n > 0 ? percentile(lats, 0.99) : 0,
      errorRate: this.requests > 0 ? this.errors / this.requests : 0,
      totalRequests: this.requests,
      totalErrors: this.errors,
    };
  }

  reset(): void {
    this.latencies = [];
    this.errors = 0;
    this.requests = 0;
  }
}

function percentile(sorted: number[], p: number): number {
  const idx = Math.max(0, Math.min(sorted.length - 1, Math.ceil(sorted.length * p) - 1));
  return sorted[idx] ?? 0;
}
