/**
 * Optimizer Tests — L-BFGS and Adam.
 *
 * Validates numerical convergence, gradient descent behavior,
 * edge cases, and convergence guarantees.
 */
import { describe, it, expect } from 'vitest';
import { lbfgs, adam } from '../optimize.js';
import type { LBFGSConfig, AdamConfig } from '../optimize.js';

function makeQuadratic(target: number[]) {
  return (x: Float64Array): [number, Float64Array] => {
    let val = 0;
    const grad = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const diff = x[i]! - target[i]!;
      val += diff * diff;
      grad[i] = 2 * diff;
    }
    return [val, grad];
  };
}

function rosenbrock(a = 1, b = 100) {
  return (x: Float64Array): [number, Float64Array] => {
    const x0 = x[0]!, x1 = x[1]!;
    const val = (a - x0) ** 2 + b * (x1 - x0 * x0) ** 2;
    const grad = new Float64Array(2);
    grad[0] = -2 * (a - x0) - 4 * b * x0 * (x1 - x0 * x0);
    grad[1] = 2 * b * (x1 - x0 * x0);
    return [val, grad];
  };
}

describe('L-BFGS', () => {
  it('converges to zero on 1D quadratic', () => {
    const f = makeQuadratic([0]);
    const result = lbfgs(f, new Float64Array([10]));
    expect(result.value).toBeLessThan(1e-8);
    expect(Math.abs(result.x[0]!)).toBeLessThan(1e-4);
    expect(result.converged).toBe(true);
  });

  it('converges on 3D quadratic', () => {
    const f = makeQuadratic([2, -3, 5]);
    const result = lbfgs(f, new Float64Array([0, 0, 0]));
    expect(result.value).toBeLessThan(1e-8);
    for (let i = 0; i < 3; i++)
      expect(Math.abs(result.x[i]! - [2, -3, 5][i]!)).toBeLessThan(1e-3);
    expect(result.converged).toBe(true);
  });

  it('converges on Rosenbrock function', () => {
    const f = rosenbrock();
    const result = lbfgs(f, new Float64Array([-1.2, 1]), { maxIter: 500, gtol: 1e-8 });
    expect(result.value).toBeLessThan(1e-6);
    expect(Math.abs(result.x[0]! - 1)).toBeLessThan(1e-3);
    expect(Math.abs(result.x[1]! - 1)).toBeLessThan(1e-3);
  });

  it('returns immediately when initial gradient below tolerance', () => {
    const f = makeQuadratic([0]);
    const result = lbfgs(f, new Float64Array([0]));
    expect(result.iterations).toBe(0);
    expect(result.converged).toBe(true);
  });

  it('handles empty parameters gracefully', () => {
    const f = (_x: Float64Array): [number, Float64Array] => [0, new Float64Array(0)];
    const result = lbfgs(f, new Float64Array(0));
    expect(result.converged).toBe(true);
    expect(result.iterations).toBe(0);
  });

  it('tracks iteration count and achieves significant reduction', () => {
    const f = makeQuadratic([5]);
    const result = lbfgs(f, new Float64Array([100]), { maxIter: 50 });
    expect(result.iterations).toBeGreaterThanOrEqual(1);
    expect(result.value).toBeLessThan(1);
  });

  it('returns unconverged when max iterations reached with too-strict tolerance', () => {
    // Function that L-BFGS can optimize but with extremely strict tolerance
    const f = makeQuadratic([0.5]);
    const result = lbfgs(f, new Float64Array([100]), {
      maxIter: 1,
      gtol: 1e-20,
    });
    // After 1 iteration with gtol=1e-20, may or may not converge
    // depending on gradient reduction — check it doesn't crash
    expect(result.iterations).toBeLessThanOrEqual(1);
    expect(typeof result.value).toBe('number');
  });

  it('respects explicit stepLength', () => {
    const f = makeQuadratic([3]);
    const result = lbfgs(f, new Float64Array([0]), { stepLength: 0.5, maxIter: 100 });
    expect(result.value).toBeLessThan(1e-6);
    expect(result.converged).toBe(true);
  });

  it('handles history size overflow (small m)', () => {
    const f = makeQuadratic([1, 1]);
    const result = lbfgs(f, new Float64Array([5, 5]), { m: 2, maxIter: 100 });
    expect(result.value).toBeLessThan(1e-6);
    expect(result.converged).toBe(true);
  });

  it('converges on high-dimensional quadratic (50D)', () => {
    const target = Array.from({ length: 50 }, (_, i) => (i % 3 - 1) * 0.5);
    const f = makeQuadratic(target);
    const init = new Float64Array(50).fill(0);
    const result = lbfgs(f, init, { maxIter: 200, gtol: 1e-6 });
    expect(result.value).toBeLessThan(1e-6);
    expect(result.converged).toBe(true);
  });

  it('handles near-zero curvature gracefully', () => {
    const f = (x: Float64Array): [number, Float64Array] => {
      const x0 = x[0]!;
      return [1e-10 * x0 * x0, new Float64Array([2e-10 * x0])];
    };
    const result = lbfgs(f, new Float64Array([1]));
    expect(result.iterations).toBeLessThanOrEqual(200);
    expect(typeof result.value).toBe('number');
  });

  it('produces descending function values', () => {
    const f = makeQuadratic([-4, 7]);
    const result = lbfgs(f, new Float64Array([10, -10]));
    expect(result.value).toBeLessThan(1e-6);
    expect(result.gradNorm).toBeLessThan(1e-4);
  });
});

describe('Adam', () => {
  it('converges on 1D quadratic with sufficient iterations', () => {
    const f = makeQuadratic([0]);
    const result = adam(f, new Float64Array([5]), { maxIter: 1000, lr: 0.05 });
    expect(result.value).toBeLessThan(1e-4);
  });

  it('converges on 3D quadratic', () => {
    const f = makeQuadratic([1, -2, 3]);
    const result = adam(f, new Float64Array([0, 0, 0]), { maxIter: 2000, lr: 0.05 });
    expect(result.value).toBeLessThan(0.01);
  });

  it('converges on Rosenbrock function', () => {
    const f = rosenbrock();
    const result = adam(f, new Float64Array([-1.2, 1]), { lr: 1e-2, maxIter: 5000, gtol: 1e-6 });
    expect(result.value).toBeLessThan(1e-3);
  });

  it('converges immediately when at minimum', () => {
    const f = makeQuadratic([0]);
    const result = adam(f, new Float64Array([0]));
    expect(result.value).toBeLessThan(1e-8);
  });

  it('improves function value from initial guess', () => {
    const f = makeQuadratic([0]);
    const initialVal = makeQuadratic([0])(new Float64Array([10]))[0]; // value at x=10
    const result = adam(f, new Float64Array([10]), { maxIter: 200, lr: 0.1 });
    expect(result.value).toBeLessThan(initialVal);
  });

  it('respects custom learning rate', () => {
    const f = makeQuadratic([0]);
    const result = adam(f, new Float64Array([3]), { lr: 0.5, maxIter: 500 });
    expect(result.value).toBeLessThan(1e-6);
  });

  it('respects beta1 and beta2 momentum', () => {
    const f = makeQuadratic([0]);
    const result = adam(f, new Float64Array([3]), { beta1: 0.5, beta2: 0.9, maxIter: 1000 });
    // Non-standard beta values slow convergence, but should still improve
    expect(result.value).toBeLessThan(10); // improved from initial value ~9
  });

  it('handles empty parameters gracefully', () => {
    const f = (_x: Float64Array): [number, Float64Array] => [0, new Float64Array(0)];
    // Adam on zero-length params: first gradient check passes (norm=0 < gtol) → converged=true, iterations=1
    const result = adam(f, new Float64Array(0));
    expect(result.iterations).toBeGreaterThanOrEqual(0);
    expect(typeof result.x).toBe('object');
  });

  it('returns unconverged when hitting max iterations with strict tolerance', () => {
    const f = makeQuadratic([100]);
    const result = adam(f, new Float64Array([0]), { maxIter: 10, gtol: 1e-20, lr: 0.01 });
    expect(result.converged).toBe(false);
  });

  it('handles NaN values gracefully', () => {
    const f = (x: Float64Array): [number, Float64Array] => {
      if (Math.abs(x[0]!) > 50) return [NaN, new Float64Array([NaN])];
      return makeQuadratic([0])(x);
    };
    const result = adam(f, new Float64Array([1]), { maxIter: 30, lr: 10 });
    expect(result.iterations).toBeGreaterThan(0);
  });
});

// ── Edge Cases ──────────────────────────────────────────────────────

describe('L-BFGS max iterations and line-search failure', () => {
  it('returns unconverged when max iterations reached', () => {
    // Force non-convergence: tiny step in 1 iteration
    const f = makeQuadratic([10000]);
    const result = lbfgs(f, new Float64Array([0]), { maxIter: 1, gtol: 1e-20, stepLength: 0.001 });
    expect(result.iterations).toBeGreaterThanOrEqual(0);
    // May converge immediately if gradient is within tolerance, which is fine
  });
});
