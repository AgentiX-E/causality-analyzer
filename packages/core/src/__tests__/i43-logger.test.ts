/**
 * I43: Logger tests — ConsoleLogger + NoopLogger + LogLevel.
 */
import { describe, it, expect, vi } from 'vitest';
import { Logger, ConsoleLogger, NoopLogger, LogLevel } from '../logger.js';

function captureOutput(fn: () => void): { log: string; error: string } {
  const out: { log: string; error: string } = { log: '', error: '' };
  const origLog = console.log;
  const origError = console.error;
  console.log = vi.fn((...args: any[]) => { out.log += args.join(' ') + '\n'; });
  console.error = vi.fn((...args: any[]) => { out.error += args.join(' ') + '\n'; });
  try { fn(); } finally {
    console.log = origLog;
    console.error = origError;
  }
  return out;
}

describe('LogLevel', () => {
  it('DEBUG < INFO < WARN < ERROR < OFF', () => {
    expect(LogLevel.DEBUG).toBeLessThan(LogLevel.INFO);
    expect(LogLevel.INFO).toBeLessThan(LogLevel.WARN);
    expect(LogLevel.WARN).toBeLessThan(LogLevel.ERROR);
    expect(LogLevel.ERROR).toBeLessThan(LogLevel.OFF);
  });

  it('all levels are distinct', () => {
    const values = Object.values(LogLevel).filter(v => typeof v === 'number');
    expect(new Set(values).size).toBe(values.length);
  });
});

describe('ConsoleLogger', () => {
  it('writes info when minLevel=INFO', () => {
    const logger = new ConsoleLogger(LogLevel.INFO);
    const out = captureOutput(() => logger.info('store connected', { backend: 'pg' }));
    expect(out.log).toContain('store connected');
    expect(out.log).toContain('INFO');
  });

  it('suppresses debug when minLevel=INFO', () => {
    const logger = new ConsoleLogger(LogLevel.INFO);
    const out = captureOutput(() => logger.debug('verbose details'));
    expect(out.log).toBe('');
  });

  it('writes debug when minLevel=DEBUG', () => {
    const logger = new ConsoleLogger(LogLevel.DEBUG);
    const out = captureOutput(() => logger.debug('verbose details'));
    expect(out.log).toContain('verbose details');
  });

  it('writes warn to stderr', () => {
    const logger = new ConsoleLogger(LogLevel.WARN);
    const out = captureOutput(() => logger.warn('deprecated', { fn: 'old' }));
    expect(out.error).toContain('deprecated');
  });

  it('writes error to stderr', () => {
    const logger = new ConsoleLogger(LogLevel.ERROR);
    const out = captureOutput(() => logger.error('connection failed', { code: 500 }));
    expect(out.error).toContain('connection failed');
  });

  it('suppresses warn when minLevel=ERROR', () => {
    const logger = new ConsoleLogger(LogLevel.ERROR);
    const out = captureOutput(() => logger.warn('should not appear'));
    expect(out.error).toBe('');
  });

  it('suppresses all when minLevel=OFF', () => {
    const logger = new ConsoleLogger(LogLevel.OFF);
    const out = captureOutput(() => {
      logger.debug('a');
      logger.info('b');
      logger.warn('c');
      logger.error('d');
    });
    expect(out.log).toBe('');
    expect(out.error).toBe('');
  });

  it('JSON output is parseable', () => {
    const logger = new ConsoleLogger(LogLevel.INFO);
    const out = captureOutput(() => logger.info('test', { key: 'val', num: 42 }));
    const parsed = JSON.parse(out.log.trim());
    expect(parsed.level).toBe('INFO');
    expect(parsed.message).toBe('test');
    expect(parsed.data.key).toBe('val');
    expect(parsed.data.num).toBe(42);
    expect(parsed.timestamp).toBeDefined();
  });

  it('omits data field when no data provided', () => {
    const logger = new ConsoleLogger(LogLevel.INFO);
    const out = captureOutput(() => logger.info('no data'));
    const parsed = JSON.parse(out.log.trim());
    expect(parsed.data).toBeUndefined();
  });

  it('reports correct minLevel', () => {
    expect(new ConsoleLogger(LogLevel.DEBUG).minLevel).toBe(LogLevel.DEBUG);
    expect(new ConsoleLogger().minLevel).toBe(LogLevel.INFO);
  });
});

describe('NoopLogger', () => {
  it('all methods are no-ops', () => {
    const logger = new NoopLogger();
    const out = captureOutput(() => {
      logger.debug('a');
      logger.info('b');
      logger.warn('c');
      logger.error('d', { code: 500 });
    });
    expect(out.log).toBe('');
    expect(out.error).toBe('');
  });

  it('minLevel is OFF', () => {
    expect(new NoopLogger().minLevel).toBe(LogLevel.OFF);
  });

  it('implements Logger interface', () => {
    const logger: Logger = new NoopLogger();
    expect(logger.minLevel).toBe(LogLevel.OFF);
  });
});

describe('DI compatibility', () => {
  it('ConsoleLogger satisfies Logger interface', () => {
    const logger: Logger = new ConsoleLogger();
    expect(logger).toBeDefined();
    expect(typeof logger.debug).toBe('function');
    expect(typeof logger.info).toBe('function');
    expect(typeof logger.warn).toBe('function');
    expect(typeof logger.error).toBe('function');
    expect(typeof logger.minLevel).toBe('number');
  });
});
