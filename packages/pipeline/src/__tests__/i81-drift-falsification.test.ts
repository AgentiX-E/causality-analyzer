/**
 * I81: Drift Detection + Graph Falsification tests.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { computeSHD, detectCausalDrift, detectDriftFromGraphs } from '../../src/graph/drift-detection.js';
import { falsifyGraph } from '../../src/gcm/graph-falsification.js';

describe('computeSHD', () => {
  it('returns 0 for identical graphs', () => {
    const g1 = new CausalGraph(['X', 'Y', 'Z']);
    g1.addEdge('X', 'Y'); g1.addEdge('Y', 'Z');
    const g2 = new CausalGraph(['X', 'Y', 'Z']);
    g2.addEdge('X', 'Y'); g2.addEdge('Y', 'Z');
    const result = computeSHD(g1, g2);
    expect(result.shd).toBe(0);
    expect(result.normalizedSHD).toBe(0);
  });

  it('detects extra edges in g2', () => {
    const g1 = new CausalGraph(['X', 'Y']);
    const g2 = new CausalGraph(['X', 'Y']);
    g2.addEdge('X', 'Y');
    const result = computeSHD(g1, g2);
    expect(result.extraEdges).toBe(1);
    expect(result.shd).toBe(1);
  });

  it('detects missing edges in g2', () => {
    const g1 = new CausalGraph(['X', 'Y']);
    g1.addEdge('X', 'Y');
    const g2 = new CausalGraph(['X', 'Y']);
    const result = computeSHD(g1, g2);
    expect(result.missingEdges).toBe(1);
    expect(result.shd).toBe(1);
  });

  it('detects reversed edges', () => {
    const g1 = new CausalGraph(['X', 'Y']);
    g1.addEdge('X', 'Y');
    const g2 = new CausalGraph(['X', 'Y']);
    g2.addEdge('Y', 'X');
    const result = computeSHD(g1, g2);
    expect(result.reversedEdges).toBe(1);
    expect(result.shd).toBe(1);
  });

  it('normalizedSHD is between 0 and 1', () => {
    const g1 = new CausalGraph(['A', 'B', 'C']);
    g1.addEdge('A', 'B'); g1.addEdge('B', 'C');
    const g2 = new CausalGraph(['A', 'B', 'C']);
    g2.addEdge('A', 'C');
    const result = computeSHD(g1, g2);
    expect(result.normalizedSHD).toBeGreaterThanOrEqual(0);
    expect(result.normalizedSHD).toBeLessThanOrEqual(1);
  });

  it('handles empty graphs', () => {
    const g1 = new CausalGraph(['X', 'Y']);
    const g2 = new CausalGraph(['X', 'Y']);
    const result = computeSHD(g1, g2);
    expect(result.shd).toBe(0);
  });
});

describe('detectCausalDrift', () => {
  it('returns none for insufficient data', () => {
    const data = [[1, 2], [3, 4], [5, 6]];
    const discover = (d: number[][], names: string[]) =>
      new CausalGraph([...names]);
    const result = detectCausalDrift(discover, data, ['X', 'Y'], { windowSize: 5 });
    expect(result.drifted).toBe(false);
    expect(result.severity).toBe('none');
    expect(result.windows).toEqual([]);
  });

  it('detects no drift on stationary data', () => {
    const data: number[][] = [];
    for (let i = 0; i < 400; i++) data.push([Math.random(), Math.random()]);
    const discover = (d: number[][], names: string[]) => {
      const g = new CausalGraph([...names]);
      return g;
    };
    const result = detectCausalDrift(discover, data, ['X', 'Y'], {
      windowSize: 100, stepSize: 50, threshold: 0.3,
    });
    expect(result.windows.length).toBeGreaterThan(0);
    expect(typeof result.maxDrift).toBe('number');
    expect(typeof result.meanDrift).toBe('number');
  });

  it('produces valid severity classification', () => {
    const data: number[][] = [];
    for (let i = 0; i < 500; i++) data.push([i * 0.01, Math.random()]);
    const discover = (d: number[][], names: string[]) => {
      const g = new CausalGraph([...names]);
      g.addEdge(names[0]!, names[1]!);
      return g;
    };
    const result = detectCausalDrift(discover, data, ['X', 'Y'], {
      windowSize: 100, stepSize: 50, threshold: 0.2,
    });
    expect(['none', 'mild', 'moderate', 'severe']).toContain(result.severity);
    expect(result.windows.length).toBeGreaterThan(0);
  });
});

describe('detectDriftFromGraphs', () => {
  it('returns none for < 2 graphs', () => {
    const g = new CausalGraph(['X', 'Y']);
    const result = detectDriftFromGraphs([g]);
    expect(result.drifted).toBe(false);
    expect(result.severity).toBe('none');
  });

  it('detects severity levels', () => {
    const g1 = new CausalGraph(['A', 'B', 'C']);
    g1.addEdge('A', 'B'); g1.addEdge('B', 'C');
    const g2 = new CausalGraph(['A', 'B', 'C']);
    g2.addEdge('A', 'C'); g2.addEdge('C', 'B'); // different edges
    const g3 = new CausalGraph(['A', 'B', 'C']);
    g3.addEdge('A', 'B'); g3.addEdge('C', 'A'); // changed again

    const result = detectDriftFromGraphs([g1, g2, g3], { threshold: 0.1 });
    expect(typeof result.maxDrift).toBe('number');
    expect(typeof result.meanDrift).toBe('number');
    expect(result.windows.length).toBe(3);
  });

  it('returns no drift for identical graphs', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');
    const result = detectDriftFromGraphs([g, g], { threshold: 0.1 });
    expect(result.drifted).toBe(false);
    expect(result.maxDrift).toBe(0);
    expect(result.severity).toBe('none');
  });
});

describe('falsifyGraph', () => {
  it('returns validation result for DAG', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const data = Array.from({ length: 50 }, () => [Math.random(), Math.random(), Math.random()]);
    const result = falsifyGraph(g, new Matrix(data), ['X', 'Y', 'Z']);
    expect(result).toBeDefined();
    expect(typeof result.falsified).toBe('boolean');
  });

  it('handles empty graph', () => {
    const g = new CausalGraph(['X', 'Y']);
    const data = new Matrix([[1, 2], [3, 4]]);
    const result = falsifyGraph(g, data, ['X', 'Y']);
    expect(result).toBeDefined();
    expect(typeof result.falsified).toBe('boolean');
  });

  it('returns explanation string', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const data = new Matrix(Array.from({ length: 50 }, () => [Math.random(), Math.random()]));
    const result = falsifyGraph(g, data, ['A', 'B']);
    expect(typeof result.explanation).toBe('string');
    expect(result.explanation.length).toBeGreaterThan(0);
  });

  it('handles seed for reproducibility', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');
    const data = new Matrix(Array.from({ length: 30 }, () => [Math.random(), Math.random()]));
    const r1 = falsifyGraph(g, data, ['X', 'Y'], 0.05, 42);
    const r2 = falsifyGraph(g, data, ['X', 'Y'], 0.05, 42);
    expect(r1.falsified).toBe(r2.falsified);
  });
});
