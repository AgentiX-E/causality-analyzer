/**
 * OpenTelemetry-compatible telemetry abstraction.
 *
 * Provides lightweight tracing and metrics without hard dependency on
 * `@opentelemetry/api`. If OTel is installed, spans and metrics are
 * exported to the configured OTLP backend. Otherwise, a no-op
 * implementation ensures zero runtime overhead.
 *
 * Usage:
 *   const span = Telemetry.startSpan('discovery.pc', { nodes: 5 });
 *   try { ... } finally { span.end(); }
 *
 * Integration:
 *   - Pipeline algorithms: wrap discovery/inference/RCA in spans
 *   - HTTP server: wrap each request in a span
 *   - Storage: record query latency histograms
 *
 * @packageDocumentation
 */

// ── Types ────────────────────────────────────────────────────────────

export interface TelemetrySpan {
  readonly name: string;
  setAttribute(key: string, value: string | number | boolean): void;
  addEvent(name: string, attributes?: Record<string, string | number | boolean>): void;
  recordException(error: Error): void;
  end(): void;
}

export interface TelemetryTracer {
  startSpan(name: string, options?: TelemetrySpanOptions): TelemetrySpan;
}

export interface TelemetrySpanOptions {
  attributes?: Record<string, string | number | boolean>;
}

export interface TelemetryCounter {
  add(value: number, attributes?: Record<string, string | number>): void;
}

export interface TelemetryHistogram {
  record(value: number, attributes?: Record<string, string | number>): void;
}

export interface TelemetryMeter {
  createCounter(name: string, options?: { description?: string; unit?: string }): TelemetryCounter;
  createHistogram(name: string, options?: { description?: string; unit?: string }): TelemetryHistogram;
}

// ── No-Op Implementation (default, zero overhead) ─────────────────────

class NoopSpan implements TelemetrySpan {
  readonly name: string;
  constructor(name: string) { this.name = name; }
  setAttribute(_k: string, _v: string | number | boolean): void {}
  addEvent(_n: string, _a?: Record<string, string | number | boolean>): void {}
  recordException(_e: Error): void {}
  end(): void {}
}

class NoopTracer implements TelemetryTracer {
  startSpan(name: string, _options?: TelemetrySpanOptions): TelemetrySpan {
    return new NoopSpan(name);
  }
}

class NoopCounter implements TelemetryCounter {
  add(_v: number, _a?: Record<string, string | number>): void {}
}

class NoopHistogram implements TelemetryHistogram {
  record(_v: number, _a?: Record<string, string | number>): void {}
}

class NoopMeter implements TelemetryMeter {
  createCounter(_n: string, _o?: { description?: string; unit?: string }): TelemetryCounter { return new NoopCounter(); }
  createHistogram(_n: string, _o?: { description?: string; unit?: string }): TelemetryHistogram { return new NoopHistogram(); }
}

// ── Telemetry Facade ─────────────────────────────────────────────────

/**
 * Singleton telemetry facade.
 *
 * By default, all telemetry is no-op (zero overhead).
 * Call `Telemetry.init({ tracer, meter })` to enable OTel export.
 */
export class Telemetry {
  private static _tracer: TelemetryTracer = new NoopTracer();
  private static _meter: TelemetryMeter = new NoopMeter();

  /** Initialize telemetry with real tracer/meter (e.g., from @opentelemetry/api). */
  static init(config: { tracer?: TelemetryTracer; meter?: TelemetryMeter }): void {
    if (config.tracer) Telemetry._tracer = config.tracer;
    if (config.meter) Telemetry._meter = config.meter;
  }

  /** Start a new span. Always safe — returns no-op span when telemetry is disabled. */
  static startSpan(name: string, options?: TelemetrySpanOptions): TelemetrySpan {
    return Telemetry._tracer.startSpan(name, options);
  }

  /** Create a counter metric. */
  static createCounter(name: string, options?: { description?: string; unit?: string }): TelemetryCounter {
    return Telemetry._meter.createCounter(name, options);
  }

  /** Create a histogram metric. */
  static createHistogram(name: string, options?: { description?: string; unit?: string }): TelemetryHistogram {
    return Telemetry._meter.createHistogram(name, options);
  }

  /** Reset to no-op state (useful in tests). */
  static reset(): void {
    Telemetry._tracer = new NoopTracer();
    Telemetry._meter = new NoopMeter();
  }
}
