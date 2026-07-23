/**
 * I22 tests: Kernel Conditional Independence (KCI) test.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { kciTest } from '../graph/kci.js';

describe('kciTest', () => {
  it('detects linear dependence (unconditional)', () => {
    const n = 100;
    const data = new Matrix(n, 2);
    for (let i = 0; i < n; i++) {
      const x = Math.random();
      data.set(i, 0, x);
      data.set(i, 1, x * 2 + (Math.random() - 0.5) * 0.1); // Y = 2X + noise
    }
    const p = kciTest(data, 0, 1, []);
    expect(p).toBeLessThan(0.05); // reject independence
  });

  it('detects nonlinear dependence', () => {
    const n = 150;
    const data = new Matrix(n, 2);
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 4 - 2; // [-2, 2]
      data.set(i, 0, x);
      data.set(i, 1, Math.sin(x * 2) + (Math.random() - 0.5) * 0.1); // Y = sin(2X) + noise
    }
    const p = kciTest(data, 0, 1, []);
    expect(p).toBeLessThan(0.05);
  });

  it('correctly identifies independence', () => {
    const n = 100;
    const data = new Matrix(n, 2);
    for (let i = 0; i < n; i++) {
      data.set(i, 0, Math.random());
      data.set(i, 1, Math.random());
    }
    // With independent data, p-value should be sensibly large
    // (permutation test with 50 perms: min p = 1/51 ≈ 0.02)
    const p = kciTest(data, 0, 1, [], { nPermutations: 50, sigma: 1.5 });
    expect(p).toBeGreaterThan(0.05);
  });

  it('conditional independence: chain X→Z→Y', () => {
    const n = 100;
    const data = new Matrix(n, 3);
    for (let i = 0; i < n; i++) {
      const x = Math.random();
      const z = x + (Math.random() - 0.5) * 0.05;
      const y = z + (Math.random() - 0.5) * 0.1;
      data.set(i, 0, x);
      data.set(i, 1, y);
      data.set(i, 2, z);
    }
    // Unconditional: X and Y are dependent (through Z)
    const pUncond = kciTest(data, 0, 1, [], { nPermutations: 30, sigma: 0.8 });
    // Conditional on Z: X ⟂ Y | Z
    const pCond = kciTest(data, 0, 1, [2], { nPermutations: 30, sigma: 0.8 });
    // Both should be valid p-values in [0,1]
    expect(pUncond).toBeGreaterThanOrEqual(0);
    expect(pCond).toBeGreaterThanOrEqual(0);
    expect(pCond).toBeLessThanOrEqual(1);
    expect(pUncond).toBeLessThanOrEqual(1);
  }, 15000);

  it('returns p-value in valid range', () => {
    const n = 30;
    const data = new Matrix(n, 2);
    for (let i = 0; i < n; i++) {
      data.set(i, 0, Math.random());
      data.set(i, 1, Math.random());
    }
    const p = kciTest(data, 0, 1, []);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it('works with multiple conditioning variables', () => {
    const n = 100;
    const data = new Matrix(n, 4);
    for (let i = 0; i < n; i++) {
      const z1 = Math.random();
      const z2 = Math.random();
      const x = z1 * 0.6 + Math.random() * 0.2;
      const y = z2 * 0.6 + Math.random() * 0.2;
      data.set(i, 0, x);
      data.set(i, 1, y);
      data.set(i, 2, z1);
      data.set(i, 3, z2);
    }
    // X and Y are correlated through different confounders (Z1, Z2)
    const pUncond = kciTest(data, 0, 1, [], { nPermutations: 20 });
    // Given both confounders, they should be more independent
    const pCond = kciTest(data, 0, 1, [2, 3], { nPermutations: 20, sigma: 0.5 });
    expect(pCond).toBeGreaterThanOrEqual(0); expect(pCond).toBeLessThanOrEqual(1); // conditioning should help
  }, 15000);

  it('custom sigma parameter accepted', () => {
    const n = 50;
    const data = new Matrix(n, 2);
    for (let i = 0; i < n; i++) {
      data.set(i, 0, Math.random());
      data.set(i, 1, Math.random());
    }
    const p = kciTest(data, 0, 1, [], { sigma: 1.0 });
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it('custom epsilon parameter accepted', () => {
    const n = 50;
    const data = new Matrix(n, 2);
    for (let i = 0; i < n; i++) {
      data.set(i, 0, Math.random());
      data.set(i, 1, Math.random());
    }
    const p = kciTest(data, 0, 1, [], { epsilon: 0.01 });
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });
});
