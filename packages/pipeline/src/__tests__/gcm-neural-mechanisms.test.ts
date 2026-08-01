/**
 * I4: Neural SCM mechanism tests.
 */
import { describe, it, expect } from 'vitest';
import { trainFFNMechanism, FFNMechanism } from '../gcm/neural-mechanisms.js';

describe('FFNMechanism', () => {
  it('trains on simple linear data', () => {
    // y = 2*x1 + 3*x2 + 0.1 (deterministic, no noise)
    // Seeded pattern guarantees reproducible training result
    const n = 200;
    const inputDim = 2;
    const X = new Float64Array(n * inputDim);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const x1 = (i % 20) / 20;          // deterministic [0, 0.95]
      const x2 = ((i * 7 + 3) % 23) / 23; // deterministic [0, ~0.96]
      X[i * inputDim] = x1;
      X[i * inputDim + 1] = x2;
      y[i] = 2 * x1 + 3 * x2 + 0.1;
    }

    const mech = trainFFNMechanism(X, y, n, inputDim, 'Y');
    expect(mech).toBeInstanceOf(FFNMechanism);
    expect(mech.noiseStd).toBeGreaterThan(0);

    // Forward should give reasonable predictions
    const pred = mech.forward([1, 1]);
    // 2*1 + 3*1 = 5. With deterministic training, should be close.
    expect(Math.abs(pred - 5)).toBeLessThan(2.0);
  });

  it('handles no-parents case', () => {
    const y = new Float64Array([1, 2, 3, 4, 5]);
    const mech = trainFFNMechanism(new Float64Array(0), y, 5, 0, 'Root');
    expect(mech.forward([])).toBeCloseTo(3, 0);
  });

  it('handles small sample size gracefully', () => {
    // 3 samples with 2 features → should use mean fallback
    const X = new Float64Array([1, 2, 3, 4, 5, 6]);
    const y = new Float64Array([10, 20, 30]);
    const mech = trainFFNMechanism(X, y, 3, 2, 'Small');
    expect(mech.forward([1, 2])).toBeCloseTo(20, 0);
  });

  it('invert recovers noise from observation', () => {
    const n = 100;
    const inputDim = 1;
    const X = new Float64Array(n);
    const y = new Float64Array(n);
    // Deterministic data: X = i/n, y = 5*X + 0.1
    for (let i = 0; i < n; i++) {
      X[i] = (i + 1) / n;
      y[i] = 5 * X[i]! + 0.1;
    }

    const mech = trainFFNMechanism(X, y, n, inputDim, 'Z');
    const noise = mech.invert(y[0]!, [X[0]!]);
    expect(Math.abs(noise - 0.1)).toBeLessThan(0.5);
  });

  describe('serialization', () => {
    it('round-trips via JSON', () => {
      const mech = trainFFNMechanism(
        new Float64Array([1, 2, 3, 4, 5, 6]),
        new Float64Array([10, 20, 30]),
        3, 2, 'Test',
      );
      const json = mech.toJSON();
      const restored = FFNMechanism.fromJSON(json);
      expect(restored.nodeName).toBe('Test');
      expect(restored.noiseStd).toBe(mech.noiseStd);
    });
  });
});
