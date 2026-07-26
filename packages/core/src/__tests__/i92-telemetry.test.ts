/**
 * I3: Telemetry module tests.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { Telemetry } from '../telemetry.js';
import type { TelemetrySpan, TelemetryTracer, TelemetryMeter, TelemetryCounter, TelemetryHistogram, TelemetrySpanOptions } from '../telemetry.js';

describe('Telemetry', () => {
  beforeEach(() => {
    Telemetry.reset();
  });

  describe('default (no-op)', () => {
    it('startSpan returns a span without throwing', () => {
      const span = Telemetry.startSpan('test');
      expect(span).toBeDefined();
      expect(typeof span.end).toBe('function');
    });

    it('span.end does not throw', () => {
      const span = Telemetry.startSpan('test');
      expect(() => span.end()).not.toThrow();
    });

    it('span.setAttribute does not throw', () => {
      const span = Telemetry.startSpan('test');
      expect(() => span.setAttribute('key', 'value')).not.toThrow();
    });

    it('span.addEvent does not throw', () => {
      const span = Telemetry.startSpan('test');
      expect(() => span.addEvent('event', { k: 1 })).not.toThrow();
    });

    it('span.recordException does not throw', () => {
      const span = Telemetry.startSpan('test');
      expect(() => span.recordException(new Error('boom'))).not.toThrow();
    });

    it('createCounter returns a working counter', () => {
      const c = Telemetry.createCounter('requests');
      expect(typeof c.add).toBe('function');
      expect(() => c.add(1)).not.toThrow();
      expect(() => c.add(5, { status: '200' })).not.toThrow();
    });

    it('createHistogram returns a working histogram', () => {
      const h = Telemetry.createHistogram('latency', { description: 'ms', unit: 'ms' });
      expect(typeof h.record).toBe('function');
      expect(() => h.record(42)).not.toThrow();
      expect(() => h.record(100, { endpoint: '/discover' })).not.toThrow();
    });
  });

  describe('with custom tracer', () => {
    it('delegates to injected tracer', () => {
      const spans: TelemetrySpan[] = [];
      const customTracer: TelemetryTracer = {
        startSpan(name: string, _options?: TelemetrySpanOptions): TelemetrySpan {
          const s = {
            name, _ended: false,
            setAttribute() {}, addEvent() {}, recordException() {},
            end() { spans.push(this); this._ended = true; },
          } as TelemetrySpan & { _ended: boolean };
          return s;
        },
      };

      Telemetry.init({ tracer: customTracer });
      const span = Telemetry.startSpan('custom.span', { attributes: { key: 'val' } });
      span.end();

      expect(spans.length).toBe(1);
      expect(spans[0]!.name).toBe('custom.span');
    });

    it('delegates to injected meter', () => {
      const counts: Array<{ v: number; a?: Record<string, string | number> }> = [];
      const customMeter: TelemetryMeter = {
        createCounter(_name: string): TelemetryCounter {
          return { add(v: number, a?: Record<string, string | number>) { counts.push({ v, a }); } };
        },
        createHistogram(_name: string): TelemetryHistogram {
          return { record(v: number, a?: Record<string, string | number>) { counts.push({ v, a }); } };
        },
      };

      Telemetry.init({ meter: customMeter });
      const c = Telemetry.createCounter('test.counter');
      c.add(42, { label: 'test' });

      expect(counts.length).toBe(1);
      expect(counts[0]!.v).toBe(42);
    });
  });

  describe('reset', () => {
    it('resets to no-op after custom tracer', () => {
      const customTracer: TelemetryTracer = {
        startSpan(name: string, _options?: TelemetrySpanOptions): TelemetrySpan {
          return { name, setAttribute() {}, addEvent() {}, recordException() {}, end() {} };
        },
      };
      Telemetry.init({ tracer: customTracer });
      Telemetry.reset();

      // After reset, spans are no-ops
      const span = Telemetry.startSpan('after.reset');
      span.end();
      expect(span.name).toBe('after.reset');
    });
  });
});
