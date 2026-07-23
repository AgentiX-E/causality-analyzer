import { describe, it, expect } from 'vitest';
import {
  CONSTANTS,
  CausalityError, ConfigValidationError, NodeNotFoundError,
  SingularMatrixError, IdentificationError, ColumnNotFoundError,
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

describe('CausalityError hierarchy', () => {
  it('CausalityError is instanceof Error', () => {
    const e = new CausalityError('test');
    expect(e).toBeInstanceOf(Error);
    expect(e).toBeInstanceOf(CausalityError);
    expect(e.name).toBe('CausalityError');
    expect(e.message).toBe('test');
  });

  it('ConfigValidationError has path property', () => {
    const e = new ConfigValidationError('bad config', ['alpha', 'value']);
    expect(e).toBeInstanceOf(CausalityError);
    expect(e.name).toBe('ConfigValidationError');
    expect(e.path).toEqual(['alpha', 'value']);
  });

  it('NodeNotFoundError contains node name', () => {
    const e = new NodeNotFoundError('test-node');
    expect(e.message).toContain('test-node');
  });

  it('SingularMatrixError has context', () => {
    const e = new SingularMatrixError('OLS inversion');
    expect(e.message).toContain('OLS inversion');
    expect(e).toBeInstanceOf(CausalityError);
  });

  it('IdentificationError has reason', () => {
    const e = new IdentificationError('no backdoor set found');
    expect(e.message).toContain('no backdoor set found');
  });

  it('ColumnNotFoundError has column name', () => {
    const e = new ColumnNotFoundError('latency_p99');
    expect(e.message).toContain('latency_p99');
  });
});
