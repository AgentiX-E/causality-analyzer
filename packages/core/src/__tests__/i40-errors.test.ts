/**
 * I40: Error hierarchy tests.
 */
import { describe, it, expect } from 'vitest';
import {
  CausalityError,
  StoreError,
  ValidationError,
  ConfigError,
  NotFoundError,
  ConvergenceError,
  ErrorCode,
} from '../errors.js';

describe('CausalityError (base)', () => {
  it('constructs with code and message', () => {
    const e = new CausalityError(ErrorCode.INTERNAL, 'test error');
    expect(e).toBeInstanceOf(Error);
    expect(e).toBeInstanceOf(CausalityError);
    expect(e.code).toBe(ErrorCode.INTERNAL);
    expect(e.message).toBe('test error');
    expect(e.name).toBe('CausalityError');
  });

  it('preserves cause chain', () => {
    const cause = new Error('root cause');
    const e = new CausalityError(ErrorCode.INTERNAL, 'wrapper', { cause });
    expect(e.cause).toBe(cause);
  });

  it('stores structured context', () => {
    const e = new CausalityError(ErrorCode.INTERNAL, 'ctx test', {
      context: { key: 'value', num: 42 },
    });
    expect(e.context.key).toBe('value');
    expect(e.context.num).toBe(42);
  });

  it('toJSON includes all fields', () => {
    const cause = new Error('inner');
    const e = new CausalityError(ErrorCode.NO_CONVERGENCE, 'no converge', {
      cause,
      context: { iterations: 100 },
    });
    const json = e.toJSON();
    expect(json.code).toBe(ErrorCode.NO_CONVERGENCE);
    expect(json.message).toBe('no converge');
    expect(json.name).toBe('CausalityError');
    expect(json.cause).toBe('inner');
  });

  it('toJSON handles undefined cause', () => {
    const e = new CausalityError(ErrorCode.INTERNAL, 'no cause');
    const json = e.toJSON();
    expect(json.cause).toBe('');
  });

  it('instanceof checks work across hierarchy', () => {
    const se = new StoreError(ErrorCode.CONNECTION_FAILED, 'db down', {
      store: 'PG', operation: 'connect',
    });
    expect(se instanceof StoreError).toBe(true);
    expect(se instanceof CausalityError).toBe(true);
    expect(se instanceof Error).toBe(true);
  });

  it('prototype chain supports instanceof on subclasses', () => {
    const ve = new ValidationError(ErrorCode.INVALID_CONFIG, 'bad', {
      field: 'alpha',
    });
    expect(ve instanceof ValidationError).toBe(true);
    expect(ve instanceof CausalityError).toBe(true);
  });
});

describe('StoreError', () => {
  it('includes store and operation', () => {
    const e = new StoreError(ErrorCode.CONNECTION_FAILED, 'PG down', {
      store: 'PostgreSQL',
      operation: 'connect',
    });
    expect(e.store).toBe('PostgreSQL');
    expect(e.operation).toBe('connect');
    expect(e.code).toBe(ErrorCode.CONNECTION_FAILED);
  });

  it('chains with cause', () => {
    const pgError = new Error('ECONNREFUSED');
    const e = new StoreError(ErrorCode.QUERY_FAILED, 'query failed', {
      store: 'SQLite',
      operation: 'INSERT',
      cause: pgError,
      context: { sql: 'INSERT INTO...' },
    });
    expect(e.cause).toBe(pgError);
    expect(e.context.sql).toBe('INSERT INTO...');
    expect(e.context.store).toBe('SQLite');
    expect(e.context.operation).toBe('INSERT');
    expect(e).toBeInstanceOf(StoreError);
  });

  it('toJSON contains store info', () => {
    const e = new StoreError(ErrorCode.TRANSACTION_FAILED, 'txn rollback', {
      store: 'Neo4j', operation: 'commit',
    });
    const json = e.toJSON();
    expect(json.context).toEqual({ store: 'Neo4j', operation: 'commit' });
  });
});

describe('ValidationError', () => {
  it('captures field, expected, and received', () => {
    const e = new ValidationError(ErrorCode.INVALID_CONFIG, 'alpha out of range', {
      field: 'alpha',
      expected: '[0, 1]',
      received: 2.5,
    });
    expect(e.field).toBe('alpha');
    expect(e.expected).toBe('[0, 1]');
    expect(e.received).toBe(2.5);
  });

  it('works without optional fields', () => {
    const e = new ValidationError(ErrorCode.INVALID_DATA, 'bad data shape');
    expect(e.field).toBeUndefined();
    expect(e.expected).toBeUndefined();
    expect(e.received).toBeUndefined();
    expect(e).toBeInstanceOf(ValidationError);
  });

  it('includes context', () => {
    const e = new ValidationError(ErrorCode.SCHEMA_MISMATCH, 'type error', {
      field: 'name',
      context: { schema: 'string', actual: 'number' },
    });
    expect(e.context.schema).toBe('string');
    expect(e.context.actual).toBe('number');
  });
});

describe('ConfigError', () => {
  it('includes config name', () => {
    const e = new ConfigError(ErrorCode.MISSING_REQUIRED, 'missing required', {
      configName: 'storage',
    });
    expect(e.configName).toBe('storage');
    expect(e.code).toBe(ErrorCode.MISSING_REQUIRED);
    expect(e).toBeInstanceOf(ConfigError);
  });

  it('works without configName', () => {
    const e = new ConfigError(ErrorCode.CONFIG_CONFLICT, 'conflict');
    expect(e.configName).toBeUndefined();
    expect(e).toBeInstanceOf(ConfigError);
  });
});

describe('NotFoundError', () => {
  it('includes resource and identifier', () => {
    const e = new NotFoundError(ErrorCode.NODE_NOT_FOUND, 'node missing', {
      resource: 'node',
      identifier: 'Memory',
    });
    expect(e.resource).toBe('node');
    expect(e.identifier).toBe('Memory');
    expect(e.code).toBe(ErrorCode.NODE_NOT_FOUND);
    expect(e).toBeInstanceOf(NotFoundError);
  });

  it('works for graph not found', () => {
    const e = new NotFoundError(ErrorCode.GRAPH_NOT_FOUND, 'graph g1 missing', {
      resource: 'graph',
      identifier: 'g1',
      context: { namespace: 'prod' },
    });
    expect(e.context.namespace).toBe('prod');
  });
});

describe('ConvergenceError', () => {
  it('includes algorithm, iterations, tolerance', () => {
    const e = new ConvergenceError(ErrorCode.MAX_ITERATIONS, 'not converged', {
      algorithm: 'LoopyBP',
      iterations: 100,
      tolerance: 1e-6,
    });
    expect(e.algorithm).toBe('LoopyBP');
    expect(e.iterations).toBe(100);
    expect(e.tolerance).toBe(1e-6);
    expect(e).toBeInstanceOf(ConvergenceError);
  });

  it('works without iterations and tolerance', () => {
    const e = new ConvergenceError(ErrorCode.NO_CONVERGENCE, 'diverged', {
      algorithm: 'PC',
    });
    expect(e.algorithm).toBe('PC');
    expect(e.iterations).toBeUndefined();
    expect(e.tolerance).toBeUndefined();
  });
});

describe('ErrorCode constants', () => {
  it('all codes are unique strings', () => {
    const values = Object.values(ErrorCode);
    expect(new Set(values).size).toBe(values.length);
  });

  it('covers store codes', () => {
    expect(ErrorCode.CONNECTION_FAILED).toBe('CONNECTION_FAILED');
    expect(ErrorCode.QUERY_FAILED).toBe('QUERY_FAILED');
    expect(ErrorCode.TRANSACTION_FAILED).toBe('TRANSACTION_FAILED');
    expect(ErrorCode.STORE_CLOSED).toBe('STORE_CLOSED');
  });

  it('covers algorithm codes', () => {
    expect(ErrorCode.SINGULAR_MATRIX).toBe('SINGULAR_MATRIX');
    expect(ErrorCode.NUMERICAL_INSTABILITY).toBe('NUMERICAL_INSTABILITY');
  });
});

describe('Cross-class hierarchy', () => {
  it('all errors are CausalityError instances', () => {
    const errors = [
      new StoreError(ErrorCode.CONNECTION_FAILED, 'x', { store: 'S', operation: 'o' }),
      new ValidationError(ErrorCode.INVALID_CONFIG, 'x'),
      new ConfigError(ErrorCode.MISSING_REQUIRED, 'x'),
      new NotFoundError(ErrorCode.NODE_NOT_FOUND, 'x', { resource: 'r', identifier: 'i' }),
      new ConvergenceError(ErrorCode.MAX_ITERATIONS, 'x', { algorithm: 'A' }),
    ];
    for (const e of errors) {
      expect(e).toBeInstanceOf(CausalityError);
    }
  });

  it('error discrimination works via instanceof', () => {
    const e = new StoreError(ErrorCode.CONNECTION_FAILED, 'down', {
      store: 'PG', operation: 'connect',
    });
    expect(e instanceof StoreError).toBe(true);
    expect(e instanceof ValidationError).toBe(false);
    expect(e instanceof ConfigError).toBe(false);
    expect(e instanceof NotFoundError).toBe(false);
    expect(e instanceof ConvergenceError).toBe(false);
  });
});
