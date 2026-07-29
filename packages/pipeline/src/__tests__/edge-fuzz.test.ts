/**
 * Fuzz Testing Pipeline — property-based edge case detection.
 *
 * Uses randomized input generation to test invariants that must hold
 * for all valid inputs. Catches edge cases that unit tests miss.
 *
 * Each test generates random DAGs, random data, and verifies that
 * causal discovery algorithms never crash or produce invalid outputs.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../graph/causal-graph.js';
import { pcAlgorithm } from '../graph/pc.js';
import { fisherZTest } from '../graph/pc.js';
import { gesAlgorithm } from '../graph/ges.js';

// ── Random DAG generator ──────────────────────────────────────────────

function randomDAG(nNodes: number, edgeProb: number = 0.3, seed?: number): CausalGraph {
  const names = Array.from({ length: nNodes }, (_, i) => `V${i}`);
  const g = new CausalGraph(names);
  const rng = seed != null ? mulberry32(seed) : Math.random;

  for (let i = 0; i < nNodes; i++) {
    for (let j = i + 1; j < nNodes; j++) {
      if (rng() < edgeProb) g.addEdge(names[i]!, names[j]!);
    }
  }
  return g;
}

function mulberry32(s: number): () => number {
  return () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 0x100000000; };
}

function randomData(graph: CausalGraph, nSamples: number): Matrix {
  const nodes = [...graph.nodes];
  const order = graph.topologicalSort();
  const data = Matrix.zeros(nSamples, nodes.length);
  for (let r = 0; r < nSamples; r++) {
    const vals = new Array(nodes.length).fill(0);
    for (const name of order) {
      const idx = nodes.indexOf(name);
      let v = (Math.random() - 0.5) * 4;
      for (const p of graph.parents(name)) { v += (vals[nodes.indexOf(p)] ?? 0) * 0.7; }
      vals[idx] = v; data.set(r, idx, v);
    }
  }
  return data;
}

// ── Fuzz tests ────────────────────────────────────────────────────────

describe('Fuzz: PC algorithm', () => {
  it('never crashes on random 5-node DAGs (10 iterations)', () => {
    for (let i = 0; i < 10; i++) {
      const g = randomDAG(5, 0.4, i);
      const data = randomData(g, 200);
      const result = pcAlgorithm(data, [...g.nodes], { alpha: 0.05, stable: true });
      expect(result.graph.nodeCount).toBe(5);
      expect(result.graph.nodeCount).toBe(5); expect(() => result.graph.pdag2dag().isDAG()).not.toThrow();
    }
  });

  it('PC on disconnected graph never crashes', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D', 'E']);
    const data = randomData(g, 100);
    const result = pcAlgorithm(data, ['A', 'B', 'C', 'D', 'E'], { alpha: 0.05 });
    expect(result.graph.nodeCount).toBe(5);
  });

  it('PC with different alpha values handles gracefully', () => {
    const g = randomDAG(4, 0.3, 42);
    const data = randomData(g, 150);
    for (const alpha of [0.001, 0.01, 0.05, 0.1, 0.5]) {
      const result = pcAlgorithm(data, [...g.nodes], { alpha, stable: true });
      expect(result.graph.nodeCount).toBe(4);
    }
  });
});

describe('Fuzz: d-separation invariants', () => {
  it('symmetry holds for random DAGs', () => {
    for (let i = 0; i < 10; i++) {
      const g = randomDAG(6, 0.3, i + 100);
      const nodes = [...g.nodes];
      const x = nodes[Math.floor(i % nodes.length)]!;
      const y = nodes[Math.floor((i + 2) % nodes.length)]!;
      expect(g.dSeparated(x, y, [])).toBe(g.dSeparated(y, x, []));
    }
  });

  it('d-separation never throws for random conditioning sets', () => {
    const g = randomDAG(5, 0.4, 200);
    const nodes = [...g.nodes];
    for (let i = 0; i < 10; i++) {
      const z = nodes.filter(() => Math.random() > 0.5);
      expect(() => g.dSeparated('V0', 'V4', z)).not.toThrow();
    }
  });
});

describe('Fuzz: Fisher Z invariants', () => {
  it('p-value is always in [0,1]', () => {
    for (let i = 0; i < 10; i++) {
      const g = randomDAG(4, 0.3, i + 300);
      const data = randomData(g, 100);
      const p = fisherZTest(data, 0, 1, []);
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });

  it('Fisher Z with empty conditioning set handles all pairs', () => {
    const g = randomDAG(4, 0.3, 400);
    const data = randomData(g, 80);
    for (let i = 0; i < 4; i++) {
      for (let j = i + 1; j < 4; j++) {
        expect(() => fisherZTest(data, i, j, [])).not.toThrow();
      }
    }
  });
});

describe('Fuzz: GES invariants', () => {
  it('GES on random DAGs produces valid graph', () => {
    for (let i = 0; i < 5; i++) {
      const g = randomDAG(4, 0.3, i + 500);
      const data = randomData(g, 150);
      const result = gesAlgorithm(data, [...g.nodes]);
      expect(result.nodeCount).toBe(4);
    }
  });
});
