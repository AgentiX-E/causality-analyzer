import { describe, it, expect } from 'vitest';
import { RateLimiter } from '../rate-limiter.js';

describe('RateLimiter', () => {
  it('accepts points within buffer capacity', () => {
    const r = new RateLimiter({ maxBufferSize: 5 });
    for (let i = 0; i < 5; i++) {
      expect(r.push([i]).accepted).toBe(true);
    }
    expect(r.size).toBe(5);
  });

  it('drops oldest when full (default strategy)', () => {
    const r = new RateLimiter({ maxBufferSize: 3 });
    r.push([1]); r.push([2]); r.push([3]);
    const result = r.push([4]);
    expect(result.accepted).toBe(true);
    expect(result.dropped).toBe(1);
    expect(r.size).toBe(3);
  });

  it('drop_newest rejects overflow', () => {
    const r = new RateLimiter({ maxBufferSize: 2, strategy: 'drop_newest' });
    r.push([1]); r.push([2]);
    const result = r.push([3]);
    expect(result.accepted).toBe(false);
    expect(result.dropped).toBe(1);
  });

  it('block strategy rejects overflow without dropping', () => {
    const r = new RateLimiter({ maxBufferSize: 2, strategy: 'block' });
    r.push([1]); r.push([2]);
    const result = r.push([3]);
    expect(result.accepted).toBe(false);
    // block doesn't drop; buffer unchanged
    expect(r.size).toBe(2);
  });

  it('utilization tracks correctly', () => {
    const r = new RateLimiter({ maxBufferSize: 10 });
    expect(r.push([0]).utilization).toBe(0.1);
    for (let i = 0; i < 9; i++) r.push([i]);
    expect(r.push([9]).utilization).toBe(1);
  });

  it('drain empties buffer and returns all points', () => {
    const r = new RateLimiter({ maxBufferSize: 5 });
    r.push([1]); r.push([2]); r.push([3]);
    const drained = r.drain();
    expect(drained).toEqual([[1], [2], [3]]);
    expect(r.size).toBe(0);
  });

  it('reset clears state', () => {
    const r = new RateLimiter({ maxBufferSize: 5 });
    for (let i = 0; i < 5; i++) r.push([i]);
    r.reset();
    expect(r.size).toBe(0);
    expect(r.dropped).toBe(0);
  });

  it('sliding_window strategy accepts new and drops oldest', () => {
    const r = new RateLimiter({ maxBufferSize: 3, strategy: 'sliding_window' });
    r.push([1]); r.push([2]); r.push([3]);
    expect(r.size).toBe(3);
    const result = r.push([4]);
    expect(result.accepted).toBe(true);
    expect(r.size).toBe(3);
    expect(r.dropped).toBe(1);
  });
});
