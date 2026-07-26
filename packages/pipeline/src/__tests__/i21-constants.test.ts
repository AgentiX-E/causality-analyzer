import { describe, it, expect } from 'vitest';
import {
  CONSTANTS,
  CausalityError, StoreError, ValidationError, ConfigError,
  NotFoundError, ConvergenceError, ErrorCode,
  clamp, safeDiv, safeLog,
} from '../constants.js';

describe('CONSTANTS', () => {
  it('all constants are finite numbers', () => {
    for (const [key, val] of Object.entries(CONSTANTS)) {
      expect(typeof val, `${key} should be number`).toBe('number');
      expect(Number.isFinite(val), `${key} should be finite`).toBe(true);
    }
  });

  it('key constants have valid ranges', () => {
    expect(CONSTANTS.PATH_LIKELIHOOD_CONNECTED).toBeGreaterThan(CONSTANTS.PATH_LIKELIHOOD_DISCONNECTED);
    expect(CONSTANTS.DA_PARENT_PENALTY).toBeGreaterThan(0);
    expect(CONSTANTS.DA_CHILD_BONUS).toBeGreaterThan(0);
    expect(CONSTANTS.FAIRNESS_DISPARITY_THRESHOLD).toBeGreaterThan(0);
    expect(CONSTANTS.FAIRNESS_DISPARITY_THRESHOLD).toBeLessThan(1);
  });
});

describe('CausalityError hierarchy (re-exported from core)', () => {
  it('CausalityError with code and message', () => {
    const e = new CausalityError(ErrorCode.INVALID_CONFIG, 'bad config');
    expect(e).toBeInstanceOf(Error);
    expect(e).toBeInstanceOf(CausalityError);
    expect(e.code).toBe(ErrorCode.INVALID_CONFIG);
    expect(e.message).toBe('bad config');
  });

  it('StoreError records store and operation', () => {
    const e = new StoreError(ErrorCode.CONNECTION_FAILED, 'pg down', {
      store: 'PostgreSQL', operation: 'connect',
    });
    expect(e).toBeInstanceOf(CausalityError);
    expect(e.store).toBe('PostgreSQL');
    expect(e.operation).toBe('connect');
  });

  it('ValidationError records field/expected/received', () => {
    const e = new ValidationError(ErrorCode.INVALID_CONFIG, 'out of range', {
      field: 'alpha', expected: '[0,1]', received: 2.5,
    });
    expect(e).toBeInstanceOf(CausalityError);
    expect(e.field).toBe('alpha');
    expect(e.received).toBe(2.5);
  });

  it('NotFoundError records resource and identifier', () => {
    const e = new NotFoundError(ErrorCode.NODE_NOT_FOUND, 'node not in graph', {
      resource: 'graph', identifier: 'X',
    });
    expect(e).toBeInstanceOf(CausalityError);
    expect(e.resource).toBe('graph');
    expect(e.identifier).toBe('X');
  });

  it('ConvergenceError records algorithm and iterations', () => {
    const e = new ConvergenceError(ErrorCode.NO_CONVERGENCE, 'did not converge', {
      algorithm: 'NOTEARS', iterations: 100, tolerance: 1e-4,
    });
    expect(e).toBeInstanceOf(CausalityError);
    expect(e.algorithm).toBe('NOTEARS');
    expect(e.iterations).toBe(100);
  });

  it('CausalityError.toJSON produces structured output', () => {
    const e = new CausalityError(ErrorCode.INTERNAL, 'boom', {
      context: { key: 'val' },
    });
    const json = e.toJSON();
    expect(json.code).toBe(ErrorCode.INTERNAL);
    expect(json.context).toEqual({ key: 'val' });
  });

  it('ErrorCode values are distinct', () => {
    const codes = new Set(Object.values(ErrorCode));
    expect(codes.size).toBe(Object.values(ErrorCode).length);
  });
});

describe('clamp / safeDiv / safeLog', () => {
  it('clamp within range returns value', () => { expect(clamp(5, 15)).toBe(5); });
  it('clamp above limit caps', () => { expect(clamp(100, 15)).toBe(15); });
  it('clamp below negative limit caps', () => { expect(clamp(-100, 15)).toBe(-15); });
  it('safeDiv with near-zero denominator returns 0', () => { expect(safeDiv(5, 1e-12)).toBe(0); });
  it('safeDiv with valid denominator returns ratio', () => { expect(safeDiv(6, 3)).toBe(2); });
  it('safeLog of 1 returns 0', () => { expect(safeLog(1)).toBe(0); });
  it('safeLog of 0 returns finite value (not -Infinity)', () => {
    expect(Number.isFinite(safeLog(0))).toBe(true);
  });
  it('safeLog of negative returns finite value', () => {
    expect(Number.isFinite(safeLog(-5))).toBe(true);
  });
});
