import { describe, it, expect } from 'vitest';
import { AuditLogger, MetricsRegistry } from '../observability.js';

describe('AuditLogger', () => {
  it('records an audit entry', () => {
    const l = new AuditLogger();
    l.log('rca.analyze', 150.5, true, { method: 'circa' });
    expect(l.count).toBe(1);
    const json = l.toJSON();
    expect(json.length).toBe(1);
    expect(json[0]!.event).toBe('rca.analyze');
    expect(json[0]!.success).toBe(true);
    expect(json[0]!.context).toEqual({ method: 'circa' });
  });

  it('records failure entries', () => {
    const l = new AuditLogger();
    l.log('pipeline.detect', 50, false, {}, 'timeout');
    expect(l.toJSON()[0]!.error).toBe('timeout');
  });

  it('clear removes all entries', () => {
    const l = new AuditLogger();
    l.log('test', 1, true);
    l.clear();
    expect(l.count).toBe(0);
  });

  it('toJSON returns independent copy', () => {
    const l = new AuditLogger();
    l.log('a', 1, true);
    const json = l.toJSON();
    json.pop();
    expect(l.count).toBe(1);
  });

  it('multiple entries maintain order', () => {
    const l = new AuditLogger();
    l.log('first', 1, true);
    l.log('second', 2, false);
    expect(l.toJSON()[0]!.event).toBe('first');
    expect(l.toJSON()[1]!.event).toBe('second');
  });
});

describe('MetricsRegistry', () => {
  it('increments counters', () => {
    const m = new MetricsRegistry();
    m.inc('rca.calls', 1);
    m.inc('rca.calls', 2);
    const counters = m.exportCounters();
    expect(counters.find(c => c.name === 'rca.calls')?.value).toBe(3);
  });

  it('observes histogram values', () => {
    const m = new MetricsRegistry();
    m.observe('rca.latency', 100);
    m.observe('rca.latency', 200);
    m.observe('rca.latency', 50);
    const hist = m.exportHistograms();
    const h = hist.find(c => c.name === 'rca.latency')!;
    expect(h.count).toBe(3);
    expect(h.min).toBe(50);
    expect(h.max).toBe(200);
  });

  it('empty histogram returns zero summary', () => {
    const m = new MetricsRegistry();
    m.observe('empty', 0); m.observe('empty', 0);
    const h = m.exportHistograms().find(c => c.name === 'empty')!;
    expect(h.count).toBe(2);
  });

  it('reset clears all metrics', () => {
    const m = new MetricsRegistry();
    m.inc('test', 5);
    m.reset();
    expect(m.exportCounters().length).toBe(0);
    expect(m.exportHistograms().length).toBe(0);
  });
});
