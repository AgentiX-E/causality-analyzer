/**
 * L-BFGS Optimizer Tests.
 */
import { describe, it, expect } from 'vitest';
import { lbfgs, adam } from '@agentix-e/causality-analyzer-core';

describe('L-BFGS Optimizer', () => {
  it('minimizes quadratic f(x) = (x-3)²', () => {
    const f = (x: Float64Array): [number, Float64Array] => {
      const v = x[0]! - 3;
      return [v * v, new Float64Array([2 * v])];
    };
    const result = lbfgs(f, new Float64Array([0]));
    expect(result.converged).toBe(true);
    expect(Math.abs(result.x[0]! - 3)).toBeLessThan(1e-4);
    expect(result.value).toBeLessThan(1e-8);
  });

  it('minimizes 2D Rosenbrock-style function', () => {
    const f = (x: Float64Array): [number, Float64Array] => {
      const a = x[0]!, b = x[1]!;
      const v = (a - 1) * (a - 1) + 100 * (b - a * a) * (b - a * a);
      const grad = new Float64Array([
        2 * (a - 1) + 400 * a * (a * a - b),
        200 * (b - a * a),
      ]);
      return [v, grad];
    };
    const result = lbfgs(f, new Float64Array([0, 0]), { maxIter: 500 });
    expect(Math.abs(result.x[0]! - 1)).toBeLessThan(1e-3);
    expect(Math.abs(result.x[1]! - 1)).toBeLessThan(1e-3);
  });

  it('converges on simple quadratic in moderate iterations', () => {
    const f = (x: Float64Array): [number, Float64Array] => {
      const v = x[0]! * x[0]! + x[1]! * x[1]!;
      return [v, new Float64Array([2 * x[0]!, 2 * x[1]!])];
    };
    const result = lbfgs(f, new Float64Array([5, 5]));
    expect(result.converged).toBe(true);
    expect(result.iterations).toBeLessThan(50);
  });
});

describe('Adam Optimizer', () => {
  it('minimizes simple quadratic', () => {
    const f = (x: Float64Array): [number, Float64Array] => {
      const v = (x[0]! + 2) * (x[0]! + 2);
      return [v, new Float64Array([2 * (x[0]! + 2)])];
    };
    const result = adam(f, new Float64Array([10]), { lr: 0.1, maxIter: 500 });
    expect(Math.abs(result.x[0]! + 2)).toBeLessThan(5e-2);
  });
});
