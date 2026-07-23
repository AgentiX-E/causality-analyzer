/**
 * Structured Logger — DI-injectable logging abstraction.
 *
 * Design:
 * - Logger interface: 4 levels (debug/info/warn/error), each takes message + optional data
 * - ConsoleLogger: default production implementation, configurable minimum level
 * - NoopLogger: test/mock implementation, silently discards all messages
 * - DI-injectable via constructor options pattern
 *
 * Integration points:
 * - AuditLogger (observability.ts) — wraps Logger with timestamp + severity
 * - Store implementations — log connection/query/retry events
 * - Pipeline algorithms — log convergence/iteration/configuration
 *
 * @packageDocumentation
 */

// ── Log Level ────────────────────────────────────────────────────────────

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
  OFF = 4,
}

// ── Logger Interface ─────────────────────────────────────────────────────

/**
 * Structured logging interface.
 *
 * Implementations: ConsoleLogger (default), NoopLogger (test).
 * Consumers accept Logger via DI — never import a concrete implementation directly.
 */
export interface Logger {
  /** Debug-level diagnostic information */
  debug(message: string, data?: Record<string, unknown>): void;

  /** Info-level operational events */
  info(message: string, data?: Record<string, unknown>): void;

  /** Warning-level non-critical issues */
  warn(message: string, data?: Record<string, unknown>): void;

  /** Error-level failures requiring attention */
  error(message: string, data?: Record<string, unknown>): void;

  /** Minimum log level for this logger */
  readonly minLevel: LogLevel;
}

// ── ConsoleLogger ────────────────────────────────────────────────────────

/**
 * Default production logger — writes to console/stdout.
 *
 * @example
 * ```typescript
 * const log = new ConsoleLogger(LogLevel.INFO);
 * log.info('store connected', { backend: 'PostgreSQL', version: '14' });
 * ```
 */
export class ConsoleLogger implements Logger {
  readonly minLevel: LogLevel;

  constructor(minLevel: LogLevel = LogLevel.INFO) {
    this.minLevel = minLevel;
  }

  debug(message: string, data?: Record<string, unknown>): void {
    if (this.minLevel <= LogLevel.DEBUG) {
      this._write('DEBUG', message, data);
    }
  }

  info(message: string, data?: Record<string, unknown>): void {
    if (this.minLevel <= LogLevel.INFO) {
      this._write('INFO', message, data);
    }
  }

  warn(message: string, data?: Record<string, unknown>): void {
    if (this.minLevel <= LogLevel.WARN) {
      this._write('WARN', message, data);
    }
  }

  error(message: string, data?: Record<string, unknown>): void {
    if (this.minLevel <= LogLevel.ERROR) {
      this._write('ERROR', message, data);
    }
  }

  private _write(level: string, message: string, data?: Record<string, unknown>): void {
    const entry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      ...(data ? { data } : {}),
    };
    const stream = level === 'ERROR' || level === 'WARN' ? console.error : console.log;
    stream(JSON.stringify(entry));
  }
}

// ── NoopLogger ───────────────────────────────────────────────────────────

/**
 * Silent logger for testing and development.
 * All messages discarded — zero overhead.
 */
export class NoopLogger implements Logger {
  readonly minLevel = LogLevel.OFF;

  debug(_message: string, _data?: Record<string, unknown>): void {}
  info(_message: string, _data?: Record<string, unknown>): void {}
  warn(_message: string, _data?: Record<string, unknown>): void {}
  error(_message: string, _data?: Record<string, unknown>): void {}
}
