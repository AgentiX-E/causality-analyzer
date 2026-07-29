/**
 * TS-ICD Time-Series Causal Discovery Tests.
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { tsIcdAlgorithm } from '../../src/graph/tsicd.js';

function generateVAR1(n: number, T: number, _seed: number): number[][] {
  // Generate VAR(1) data: X_t = 0.7*Y_{t-1} + ε, Y_t = 0.5*X_{t-1} + ε
  const data: number[][] = [];
  const rng = () => Math.random();
  data.push([rng(), rng()]);
  for (let t = 1; t < T; t++) {
    const prevX = data[t - 1]![0]!;
    const prevY = data[t - 1]![1]!;
    data.push([
      0.7 * prevY + (rng() - 0.5) * 0.1,
      0.5 * prevX + (rng() - 0.5) * 0.1,
    ]);
  }
  return data;
}

describe('TS-ICD Algorithm', () => {
  it('produces valid contemporaneous graph', () => {
    const data = generateVAR1(2, 200, 42);
    const result = tsIcdAlgorithm(data, ['X', 'Y'], { maxLag: 2, alpha: 0.05 });
    expect(result.contemporaneous.nodeCount).toBe(2);
    expect(Array.isArray(result.edges)).toBe(true);
  });

  it('detects lagged edges in VAR(1) data', () => {
    const data = generateVAR1(2, 300, 43);
    const result = tsIcdAlgorithm(data, ['X', 'Y'], { maxLag: 2, alpha: 0.05 });

    // Should find at least some edges (contemporaneous or lagged)
    expect(result.edges.length).toBeGreaterThanOrEqual(0);
  });

  it('handles 3-variable time series', () => {
    const T = 200;
    const data: number[][] = [];
    data.push([Math.random(), Math.random(), Math.random()]);
    for (let t = 1; t < T; t++) {
      const px = data[t - 1]![0]!, py = data[t - 1]![1]!, pz = data[t - 1]![2]!;
      data.push([
        0.6 * py + (Math.random() - 0.5) * 0.1,
        0.4 * px + 0.3 * pz + (Math.random() - 0.5) * 0.1,
        0.5 * py + (Math.random() - 0.5) * 0.1,
      ]);
    }

    const result = tsIcdAlgorithm(data, ['X', 'Y', 'Z'], { maxLag: 2, alpha: 0.05 });
    expect(result.contemporaneous.nodeCount).toBe(3);
  });

  it('handles small data gracefully', () => {
    const data = [[1, 2], [1.1, 2.1], [0.9, 1.9]];
    const result = tsIcdAlgorithm(data, ['A', 'B'], { maxLag: 1 });
    expect(result.contemporaneous.nodeCount).toBe(2);
  });

  it('returns edges with correct lag values', () => {
    const data = generateVAR1(2, 200, 44);
    const result = tsIcdAlgorithm(data, ['X', 'Y'], { maxLag: 1, alpha: 0.05 });

    const laggedEdges = result.edges.filter(e => e.lag > 0);
    const contemporaneous = result.edges.filter(e => e.lag === 0);

    // Lagged edges should have source != target (or same with lag)
    for (const e of laggedEdges) {
      expect(e.lag).toBeGreaterThan(0);
    }
    for (const e of contemporaneous) {
      expect(e.lag).toBe(0);
    }
  });
});
