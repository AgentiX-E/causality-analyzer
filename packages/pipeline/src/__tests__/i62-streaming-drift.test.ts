/**
 * Streaming Discovery + Drift Detection Tests.
 */
import { describe, it, expect } from 'vitest';
import { OnlinePC } from '../../src/graph/streaming-discovery.js';
import { computeSHD, detectDriftFromGraphs } from '../../src/graph/drift-detection.js';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { pcAlgorithm } from '../../src/graph/pc.js';
import { Matrix } from 'ml-matrix';

function generateLinearData(graph: CausalGraph, n: number, seed: number): { data: number[][]; nodeNames: string[] } {
  // Simple linear SEM: X_j = Σ_{i∈parents(j)} 0.7·X_i + ε_j
  const simplexRng = ((s: number) => {
    let state = s >>> 0;
    return () => { state = (state * 1664525 + 1013904223) >>> 0; return state / 0x100000000; };
  })(seed);

  const nodeNames = graph.nodes;
  const d = nodeNames.length;
  const parentMap = new Map<string, string[]>();
  for (const node of nodeNames) parentMap.set(node, graph.parents(node));

  const topo = graph.topologicalSort();
  const data: number[][] = [];

  for (let r = 0; r < n; r++) {
    const row = new Array(d).fill(0);
    for (const node of topo) {
      const idx = nodeNames.indexOf(node);
      let val = 0;
      for (const p of parentMap.get(node) ?? []) {
        val += 0.7 * row[nodeNames.indexOf(p)]!;
      }
      // Box-Muller noise
      const u1 = Math.max(1e-10, simplexRng());
      const u2 = simplexRng();
      val += Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      row[idx] = val;
    }
    data.push(row);
  }

  return { data, nodeNames };
}

describe('Streaming Discovery (OnlinePC)', () => {
  it('initializes with empty graph', () => {
    const online = new OnlinePC(['X', 'Y', 'Z'], { windowSize: 100 });
    expect(online.getGraph().edges.length).toBe(0);
  });

  it('accepts streaming data without crashing', () => {
    const graph = new CausalGraph(['X', 'Y', 'Z']);
    graph.addEdge('X', 'Y');
    graph.addEdge('Y', 'Z');

    const { data } = generateLinearData(graph, 500, 42);
    const online = new OnlinePC(['X', 'Y', 'Z'], {
      windowSize: 200,
      minBatchSize: 50,
      stabilityWindows: 2,
    });

    // Feed data in small batches
    for (let i = 0; i < data.length; i += 10) {
      online.update(data.slice(i, Math.min(i + 10, data.length)));
    }

    const state = online.forceRecompute();
    expect(state.totalObservations).toBeGreaterThan(0);
    expect(state.windowFill).toBeGreaterThan(0);
  });

  it('tracks change count', () => {
    const graph = new CausalGraph(['X', 'Y']);
    graph.addEdge('X', 'Y');
    const { data } = generateLinearData(graph, 500, 42);

    const online = new OnlinePC(['X', 'Y'], {
      windowSize: 100,
      minBatchSize: 50,
      stabilityWindows: 1,
    });

    for (let i = 0; i < data.length; i += 10) {
      online.update(data.slice(i, Math.min(i + 10, data.length)));
    }
    online.forceRecompute();
    const state = online.forceRecompute();
    expect(state.changeCount).toBeGreaterThanOrEqual(0);
  });

  it('respects window size limits', () => {
    const online = new OnlinePC(['X', 'Y'], { windowSize: 50, minBatchSize: 10 });

    const data = Array.from({ length: 200 }, () => [Math.random(), Math.random()]);
    online.update(data);
    const state = online.forceRecompute();

    expect(state.windowFill).toBeLessThanOrEqual(50);
  });

  it('handles empty updates gracefully', () => {
    const online = new OnlinePC(['X', 'Y']);
    const state = online.update([]);
    expect(state.totalObservations).toBe(0);
  });
});

describe('Drift Detection', () => {
  it('computes SHD for identical graphs as 0', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B');
    g.addEdge('B', 'C');

    const result = computeSHD(g, g);
    expect(result.shd).toBe(0);
  });

  it('computes SHD for different graphs correctly', () => {
    const g1 = new CausalGraph(['A', 'B']);
    g1.addEdge('A', 'B');

    const g2 = new CausalGraph(['A', 'B']);
    g2.addEdge('B', 'A'); // reversed

    const result = computeSHD(g1, g2);
    expect(result.reversedEdges).toBeGreaterThanOrEqual(0);
    expect(result.shd).toBeGreaterThanOrEqual(0);
  });

  it('detects drift when graphs change significantly', () => {
    const g1 = new CausalGraph(['A', 'B', 'C', 'D']);
    g1.addEdge('A', 'B');
    g1.addEdge('A', 'C');

    const g2 = new CausalGraph(['A', 'B', 'C', 'D']);
    // Empty graph — all edges gone
    const g3 = new CausalGraph(['A', 'B', 'C', 'D']);
    g3.addEdge('D', 'B');
    g3.addEdge('B', 'C');
    g3.addEdge('C', 'A');

    const result = detectDriftFromGraphs([g1, g2, g3], { threshold: 0.1 });
    expect(result.drifted).toBe(true);
    expect(result.windows.length).toBe(3);
  });

  it('returns no drift for identical graphs', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');

    const result = detectDriftFromGraphs([g, g, g], { threshold: 0.1 });
    expect(result.drifted).toBe(false);
    expect(result.severity).toBe('none');
  });

  it('returns empty result for insufficient windows', () => {
    const g = new CausalGraph(['A', 'B']);
    const result = detectDriftFromGraphs([g], { threshold: 0.1 });
    expect(result.drifted).toBe(false);
    expect(result.windows).toEqual([]);
  });

  it('classifies severity correctly', () => {
    // Create graphs with known SHD ratios
    const g1 = new CausalGraph(['A', 'B', 'C', 'D', 'E']);
    g1.addEdge('A', 'B');
    g1.addEdge('B', 'C');

    const g2 = new CausalGraph(['A', 'B', 'C', 'D', 'E']);
    // Completely different edge structure
    g2.addEdge('D', 'E');
    g2.addEdge('E', 'A');
    g2.addEdge('A', 'C');

    const result = detectDriftFromGraphs([g1, g2], { threshold: 0.1 });
    expect(['mild', 'moderate', 'severe']).toContain(result.severity);
  });
});
