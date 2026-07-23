/**
 * Observability — Structured audit logging + metrics instrumentation.
 *
 * Zero-dependency observability layer for production deployments.
 * Supports JSON-structured audit logs and Prometheus-compatible metrics.
 * All exports are optional — the library functions without this module.
 *
 * @packageDocumentation
 */

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
 */
export class AuditLogger {
  private entries: AuditEntry[] = [];

  /**
   * Record an auditable event.
   * Thread-safe for sequential use; not designed for concurrent writes.
   */
  log(event: string, durationMs: number, success: boolean, context?: Record<string, unknown>, error?: string): void {
    this.entries.push({
      timestamp: new Date().toISOString(),
      event,
      durationMs: Math.round(durationMs * 100) / 100,
      success,
      context,
      error,
    });
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
 */
export class MetricsRegistry {
  private counters = new Map<string, number>();
  private histograms = new Map<string, number[]>();

  /** Increment a named counter */
  inc(name: string, by: number = 1): void {
    this.counters.set(name, (this.counters.get(name) ?? 0) + by);
  }

  /** Observe a value for a named histogram */
  observe(name: string, value: number): void {
    const h = this.histograms.get(name) ?? [];
    h.push(value);
    this.histograms.set(name, h);
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

  /** Reset all metrics */
  reset(): void { this.counters.clear(); this.histograms.clear(); }
}
