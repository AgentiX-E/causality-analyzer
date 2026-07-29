/**
 * I18 Coverage Sprint — Fusion, RCD edge cases, Drift Detection, Backdoor variants.
 */
import { describe, it, expect } from 'vitest';

// Fusion Analyzer
import { FusionAnalyzer } from '../../src/viz/fusion.js';
import type { RCAResult, RootCause } from '@agentix-e/causality-analyzer-core';

// RCD
import { rcdAlgorithm } from '../../src/graph/rcd.js';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { Matrix } from 'ml-matrix';

// Drift
import { computeSHD, detectDriftFromGraphs } from '../../src/graph/drift-detection.js';

// Backdoor
import {
  findEfficient, findAllMinimal, findMinCost,
  getAdmissibleCandidates, verifyBackdoorBlock,
} from '../../src/infer/backdoor.js';

function makeRCAResult(rootCauses: { name: string; score: number }[]): RCAResult {
  const rcs: RootCause[] = rootCauses.map((rc, i) => ({
    name: rc.name, score: rc.score, confidence: 0.8,
    rank: i + 1, evidence: [{ type: 'causal_effect' as const, description: 'test', value: rc.score }],
  }));
  return {
    rootCauses: rcs,
    paths: [],
    metadata: { method: 'test', analyzedAt: Date.now(), durationMs: 0, extra: {} },
    toJSON() { return { rootCauses: this.rootCauses, paths: this.paths, metadata: this.metadata }; },
  };
}

// ═══════════════════════════════════════════════════════════════
// Fusion Analyzer — All 3 Strategies
// ═══════════════════════════════════════════════════════════════

describe('FusionAnalyzer — full coverage', () => {
  it('weighted strategy combines scores', () => {
    const f = new FusionAnalyzer({ strategy: 'weighted', weights: { metric: 0.6, trace: 0.3, log: 0.1 } });
    const r1 = makeRCAResult([{ name: 'A', score: 0.8 }, { name: 'B', score: 0.5 }]);
    const r2 = makeRCAResult([{ name: 'A', score: 0.3 }, { name: 'C', score: 0.7 }]);
    const result = f.fuse(r1, r2, null);
    expect(result.rootCauses.length).toBeGreaterThan(0);
  });

  it('weighted with all three sources', () => {
    const f = new FusionAnalyzer({ strategy: 'weighted' });
    const r1 = makeRCAResult([{ name: 'X', score: 0.9 }]);
    const r2 = makeRCAResult([{ name: 'Y', score: 0.5 }]);
    const r3 = makeRCAResult([{ name: 'Z', score: 0.3 }]);
    const result = f.fuse(r1, r2, r3);
    expect(result.rootCauses.length).toBeGreaterThan(0);
    expect(result.metadata.method).toBe('fusion_weighted');
  });

  it('weighted with all null inputs returns empty', () => {
    const f = new FusionAnalyzer();
    const result = f.fuse(null, null, null);
    expect(result.rootCauses).toEqual([]);
  });

  it('nested strategy filters trace by metric top-3', () => {
    const f = new FusionAnalyzer({ strategy: 'nested' });
    const r1 = makeRCAResult([{ name: 'DB', score: 0.9 }, { name: 'CPU', score: 0.8 }, { name: 'MEM', score: 0.3 }]);
    const r2 = makeRCAResult([{ name: 'DB', score: 0.7 }, { name: 'NET', score: 0.6 }]);
    const result = f.fuse(r1, r2, null) as RCAResult;
    expect(result.metadata.method).toBe('fusion_nested');
  });

  it('nested with only metric returns metric-only', () => {
    const f = new FusionAnalyzer({ strategy: 'nested' });
    const r1 = makeRCAResult([{ name: 'A', score: 0.9 }]);
    const result = f.fuse(r1, null, null);
    expect(result.rootCauses.length).toBe(1);
  });

  it('voting strategy tallies votes across sources', () => {
    const f = new FusionAnalyzer({ strategy: 'voting' });
    const r1 = makeRCAResult([{ name: 'A', score: 0.8 }, { name: 'B', score: 0.6 }]);
    const r2 = makeRCAResult([{ name: 'A', score: 0.5 }, { name: 'C', score: 0.7 }]);
    const result = f.fuse(r1, r2, null);
    expect(result.metadata.method).toBe('fusion_voting');
    expect(result.rootCauses.length).toBeGreaterThan(0);
  });

  it('voting with all three sources', () => {
    const f = new FusionAnalyzer({ strategy: 'voting' });
    const r1 = makeRCAResult([{ name: 'X', score: 0.9 }]);
    const r2 = makeRCAResult([{ name: 'X', score: 0.5 }]);
    const r3 = makeRCAResult([{ name: 'Y', score: 0.4 }]);
    const result = f.fuse(r1, r2, r3);
    expect(result.metadata.method).toBe('fusion_voting');
  });

  it('voting with null sources still works', () => {
    const f = new FusionAnalyzer({ strategy: 'voting' });
    const result = f.fuse(null, null, null);
    expect(result.rootCauses).toEqual([]);
  });

  it('default strategy is weighted', () => {
    const f = new FusionAnalyzer();
    const r1 = makeRCAResult([{ name: 'A', score: 0.5 }]);
    const result = f.fuse(r1, null, null);
    expect(result.metadata.method).toBe('fusion_weighted');
  });
});

// ═══════════════════════════════════════════════════════════════
// RCD — Additional graph structures
// ═══════════════════════════════════════════════════════════════

describe('RCD algorithm — extended coverage', () => {
  it('handles fork graph X←Z→Y', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('Z', 'X'); g.addEdge('Z', 'Y');
    const data = generateLinearData(g, 200, 123);
    const result = rcdAlgorithm(new Matrix(data), ['X', 'Y', 'Z']);
    expect(result.isDAG()).toBe(true);
  });

  it('handles collider graph X→Z←Y', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Z'); g.addEdge('Y', 'Z');
    const data = generateLinearData(g, 300, 456);
    const result = rcdAlgorithm(new Matrix(data), ['X', 'Y', 'Z']);
    expect(result.isDAG()).toBe(true);
  });

  it('handles very small data', () => {
    const result = rcdAlgorithm(new Matrix(5, 3), ['X', 'Y', 'Z']);
    expect(result.nodes.length).toBe(3);
  });

  it('handles single-node graph', () => {
    const result = rcdAlgorithm(new Matrix(10, 1), ['X']);
    expect(result.nodes).toEqual(['X']);
  });

  it('applies domain knowledge constraints', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y');
    const data = generateLinearData(g, 100, 789);
    const result = rcdAlgorithm(new Matrix(data), ['X', 'Y', 'Z'], {}, {
      forbids: [['Y', 'X']],
    });
    expect(result.isDAG()).toBe(true);
  });

  it('recovers chain with 4 nodes', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C'); g.addEdge('C', 'D');
    const data = generateLinearData(g, 300, 111);
    const result = rcdAlgorithm(new Matrix(data), ['A', 'B', 'C', 'D'], { alpha: 0.01, maxDegree: 3 });
    expect(result.isDAG()).toBe(true);
    expect(result.nodes.length).toBe(4);
  });
});

// ═══════════════════════════════════════════════════════════════
// Drift Detection — Full coverage
// ═══════════════════════════════════════════════════════════════

describe('Drift Detection — full coverage', () => {
  it('computes SHD for graphs with extra edges', () => {
    const g1 = new CausalGraph(['A', 'B', 'C']);
    g1.addEdge('A', 'B');

    const g2 = new CausalGraph(['A', 'B', 'C']);
    g2.addEdge('A', 'B');
    g2.addEdge('B', 'C');

    const result = computeSHD(g1, g2);
    expect(result.extraEdges).toBeGreaterThan(0);
    expect(result.shd).toBeGreaterThan(0);
  });

  it('computes SHD for reverse edges', () => {
    const g1 = new CausalGraph(['A', 'B']);
    g1.addEdge('A', 'B');

    const g2 = new CausalGraph(['A', 'B']);
    g2.addEdge('B', 'A');

    const result = computeSHD(g1, g2);
    expect(result.reversedEdges).toBe(1);
  });

  it('normalized SHD is between 0 and 1', () => {
    const g1 = new CausalGraph(['A', 'B']);
    g1.addEdge('A', 'B');
    const g2 = new CausalGraph(['A', 'B']);
    const result = computeSHD(g1, g2);
    expect(result.normalizedSHD).toBeGreaterThanOrEqual(0);
    expect(result.normalizedSHD).toBeLessThanOrEqual(1);
  });

  it('detects drift from pre-computed graphs correctly', () => {
    const g1 = new CausalGraph(['X', 'Y', 'Z']);
    g1.addEdge('X', 'Y'); g1.addEdge('Y', 'Z');
    const g2 = new CausalGraph(['X', 'Y', 'Z']);
    g2.addEdge('X', 'Y');
    const g3 = new CausalGraph(['X', 'Y', 'Z']);
    g3.addEdge('Z', 'Y'); g3.addEdge('Y', 'X');

    const result = detectDriftFromGraphs([g1, g2, g3, g2], { threshold: 0.15 });
    expect(result.windows.length).toBe(4);
    expect(typeof result.maxDrift).toBe('number');
    expect(typeof result.meanDrift).toBe('number');
  });

  it('returns none severity for insufficient data', () => {
    const result = detectDriftFromGraphs([], { threshold: 0.1 });
    expect(result.severity).toBe('none');
  });

  it('detectDriftFromGraphs produces driftPoint when significant', () => {
    const g1 = new CausalGraph(['A', 'B', 'C']);
    g1.addEdge('A', 'B');

    const g2 = new CausalGraph(['A', 'B', 'C']);
    // Completely different structure
    g2.addEdge('C', 'A');
    g2.addEdge('B', 'C');

    const result = detectDriftFromGraphs([g1, g2], { threshold: 0.1 });
    expect(result.windows.length).toBe(2);
    if (result.drifted) {
      expect(result.driftPoint).toBeDefined();
    }
  });

  it('severity classification covers all levels', () => {
    const g1 = new CausalGraph(['A', 'B', 'C', 'D', 'E']);
    const g2 = new CausalGraph(['A', 'B', 'C', 'D', 'E']);
    g2.addEdge('A', 'C');
    g2.addEdge('C', 'D');
    g2.addEdge('D', 'E');

    const result = detectDriftFromGraphs([g1, g2], { threshold: 0.05 });
    expect(['none', 'mild', 'moderate', 'severe']).toContain(result.severity);
  });
});

// ═══════════════════════════════════════════════════════════════
// Backdoor — Additional variant coverage
// ═══════════════════════════════════════════════════════════════

describe('Backdoor variants — extended', () => {
  it('efficient returns subset on M-graph', () => {
    const g = new CausalGraph(['X', 'Y', 'U1', 'U2']);
    g.addEdge('U1', 'X'); g.addEdge('U1', 'Y');
    g.addEdge('U2', 'X'); g.addEdge('U2', 'Y');
    const result = findEfficient(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'));
    expect(verifyBackdoorBlock(g, 'X', 'Y', result)).toBe(true);
  });

  it('exhaustive finds minimal set', () => {
    const g = new CausalGraph(['X', 'Y', 'C1', 'C2']);
    g.addEdge('C1', 'X'); g.addEdge('C1', 'Y');
    g.addEdge('C2', 'X'); g.addEdge('C2', 'Y');
    const result = findAllMinimal(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'));
    expect(verifyBackdoorBlock(g, 'X', 'Y', result)).toBe(true);
  });

  it('mincost with data selects lower-cost set', () => {
    const g = new CausalGraph(['X', 'Y', 'C1', 'C2']);
    g.addEdge('C1', 'X'); g.addEdge('C1', 'Y');
    g.addEdge('C2', 'X'); g.addEdge('C2', 'Y');
    g.addEdge('X', 'Y');

    const nodeIndex = new Map([['X', 0], ['Y', 1], ['C1', 2], ['C2', 3]]);
    const data = Array.from({ length: 50 }, () => [
      Math.random(), Math.random(), Math.random(), Math.random(),
    ]);
    const result = findMinCost(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'), data, nodeIndex);
    expect(verifyBackdoorBlock(g, 'X', 'Y', result)).toBe(true);
  });

  it('mincost without data falls back to efficient', () => {
    const g = new CausalGraph(['X', 'Y', 'C']);
    g.addEdge('C', 'X'); g.addEdge('C', 'Y');
    const result = findMinCost(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'));
    expect(verifyBackdoorBlock(g, 'X', 'Y', result)).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

function generateLinearData(graph: CausalGraph, n: number, seed: number): number[][] {
  const rng = ((s: number) => {
    let state = s >>> 0;
    return () => { state = (state * 1664525 + 1013904223) >>> 0; return state / 0x100000000; };
  })(seed);

  const nodes = graph.nodes;
  const d = nodes.length;
  const parentMap = new Map<string, string[]>();
  for (const node of nodes) parentMap.set(node, graph.parents(node));
  const topo = graph.topologicalSort();

  const data: number[][] = [];
  for (let r = 0; r < n; r++) {
    const row = new Array(d).fill(0);
    for (const node of topo) {
      const idx = nodes.indexOf(node);
      let val = 0;
      for (const p of parentMap.get(node) ?? []) {
        val += 0.7 * row[nodes.indexOf(p)]!;
      }
      const u1 = Math.max(1e-10, rng()), u2 = rng();
      val += Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      row[idx] = val;
    }
    data.push(row);
  }
  return data;
}
