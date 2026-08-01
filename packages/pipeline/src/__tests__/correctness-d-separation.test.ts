/**
 * d-Separation Correctness Validation.
 *
 * Validates the dSeparated() implementation against a brute-force
 * reference algorithm using 1000+ random DAG × (X,Y,Z) triples.
 * The reference computes the formal d-separation definition:
 *   X and Y are d-separated by Z iff NO trail between X and Y
 *   is d-connecting (all non-colliders NOT in Z, all colliders
 *   in Z or with descendant in Z).
 *
 * This is a self-contained correctness proof that requires no
 * external libraries — the reference algorithm is the mathematical
 * definition of d-separation (Pearl 2009, pp. 16–17).
 *
 * @packageDocumentation
 */

import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../graph/causal-graph.js';
import { createRNG } from '@agentix-e/causality-analyzer-core';

// ── Brute-Force d-Separation Reference ───────────────────────────────

/**
 * Compute all simple paths (undirected trails) between src and dst.
 * Uses DFS with visited-set to prevent cycles.
 */
function allSimplePaths(
  adj: boolean[][],
  src: number,
  dst: number,
): number[][] {
  const n = adj.length;
  const result: number[][] = [];

  function dfs(current: number, path: number[], visited: Set<number>): void {
    if (current === dst && path.length > 1) {
      result.push([...path]);
      return;
    }
    for (let next = 0; next < n; next++) {
      if (next === current) continue;
      if (!adj[current]![next]! && !adj[next]![current]!) continue;
      if (visited.has(next)) continue;
      visited.add(next);
      path.push(next);
      dfs(next, path, visited);
      path.pop();
      visited.delete(next);
    }
  }

  const v = new Set<number>([src]);
  dfs(src, [src], v);
  return result;
}

/**
 * Compute all ancestors of node (nodes FROM which there's a directed path TO node).
 * Follows REVERSE edges: if u→v exists, u is an ancestor of v.
 */
function ancestors(adj: boolean[][], node: number): Set<number> {
  const n = adj.length;
  const result = new Set<number>();
  const queue = [node];
  result.add(node);
  while (queue.length > 0) {
    const v = queue.shift()!;
    for (let u = 0; u < n; u++) {
      // u → v means u is an ancestor of v
      if (adj[u]![v]! && !result.has(u)) {
        result.add(u);
        queue.push(u);
      }
    }
  }
  return result;
}

/**
 * Brute-force reference d-separation test.
 *
 * Steps:
 * 1. Find ALL simple undirected paths between X and Y
 * 2. For each path, classify each interior node as collider/non-collider
 * 3. Check if the path is d-connecting given Z
 * 4. Return true (d-separated) only if NO path is d-connecting
 */
function bruteForceDSeparated(
  adj: boolean[][],
  x: number,
  y: number,
  z: Set<number>,
): boolean {
  if (x === y) return false;

  const n = adj.length;

  // Precompute: is node a collider given prev and next?
  // Collider: both edges point INTO the node (prev → node ← next)
  const isCollider = (prev: number, node: number, next: number): boolean => {
    return adj[prev]![node]! && adj[next]![node]!;
  };

  // Precompute Z-ancestors for descendant activation
  const zAncestors = new Set<number>();
  for (const zi of z) {
    const anc = ancestors(adj, zi);
    for (const a of anc) zAncestors.add(a);
  }

  const paths = allSimplePaths(adj, x, y);

  for (const path of paths) {
    let connecting = true;

    // Check each interior node on the path
    for (let i = 1; i < path.length - 1; i++) {
      const prev = path[i - 1]!;
      const node = path[i]!;
      const next = path[i + 1]!;

      if (isCollider(prev, node, next)) {
        // Collider: must be in Z OR have descendant in Z
        if (!z.has(node) && !zAncestors.has(node)) {
          connecting = false;
          break; // path blocked by inactive collider
        }
      } else {
        // Non-collider: must NOT be in Z
        if (z.has(node)) {
          connecting = false;
          break; // path blocked by conditioned non-collider
        }
      }
    }

    if (connecting) return false; // d-connecting path found → not d-separated
  }

  return true; // All paths blocked → d-separated
}

// ── Test Infrastructure ──────────────────────────────────────────────

interface RandomDAG {
  graph: CausalGraph;
  adj: boolean[][];
  nodes: string[];
}

function generateRandomDAG(nodes: number, density: number, seed: number): RandomDAG {
  const rng = createRNG(seed);
  const names = Array.from({ length: nodes }, (_, i) => `V${i}`);
  const graph = new CausalGraph(names);
  const adj: boolean[][] = Array.from({ length: nodes }, () =>
    Array.from({ length: nodes }, () => false),
  );

  for (let i = 0; i < nodes; i++) {
    for (let j = i + 1; j < nodes; j++) {
      if (rng() < density) {
        graph.addEdge(names[i]!, names[j]!);
        adj[i]![j] = true;
      }
    }
  }

  return { graph, adj, nodes: names };
}

/** Generate a random triple (x, y, z) for testing. */
function randomTriple(
  n: number,
  seed: number,
): { x: number; y: number; z: Set<number> } {
  const rng = createRNG(seed);
  const x = Math.floor(rng() * n);
  let y = Math.floor(rng() * n);
  while (y === x) y = Math.floor(rng() * n);

  const z = new Set<number>();
  const zSize = Math.floor(rng() * Math.min(3, n - 2));
  while (z.size < zSize) {
    const zi = Math.floor(rng() * n);
    if (zi !== x && zi !== y) z.add(zi);
  }

  return { x, y, z };
}

// ── Comprehensive Validation ─────────────────────────────────────────

describe('d-Separation Correctness Validation', () => {
  const NUM_DAGS = 200;
  const TESTS_PER_DAG = 5;
  const SEED = 20260101;
  let rngSeed = SEED;

  // Cache results for reporting
  const failures: Array<{
    dagDesc: string;
    triple: string;
    impl: boolean;
    ref: boolean;
  }> = [];
  let totalTests = 0;

  // Test on pre-baked canonical patterns first (guaranteed correctness)
  describe('canonical patterns (sanity check)', () => {
    it('chain X→Y→Z: blocked by Y, open without', () => {
      // Force exact chain: V0 → V1 → V2
      const g = new CausalGraph(['V0', 'V1', 'V2']);
      g.addEdge('V0', 'V1');
      g.addEdge('V1', 'V2');
      const adj: boolean[][] = [
        [false, true, false],
        [false, false, true],
        [false, false, false],
      ];

      const r1 = g.dSeparated('V0', 'V2', []);
      const r2 = g.dSeparated('V0', 'V2', ['V1']);
      const b1 = bruteForceDSeparated(adj, 0, 2, new Set());
      const b2 = bruteForceDSeparated(adj, 0, 2, new Set([1]));
      expect(r1).toBe(b1);
      expect(r2).toBe(b2);
      expect(r1).toBe(false); // d-connected (chain, no blocking)
      expect(r2).toBe(true);  // d-separated (blocked by Y, a non-collider)
    });

    it('fork X←Y→Z: blocked by Y', () => {
      const g = new CausalGraph(['X', 'Y', 'Z']);
      g.addEdge('Y', 'X');
      g.addEdge('Y', 'Z');
      const adj: boolean[][] = [[false, false, false], [true, false, true], [false, false, false]];

      expect(g.dSeparated('X', 'Z', [])).toBe(
        bruteForceDSeparated(adj, 0, 2, new Set()),
      );
      expect(g.dSeparated('X', 'Z', ['Y'])).toBe(
        bruteForceDSeparated(adj, 0, 2, new Set([1])),
      );
    });

    it('collider X→M←Z: blocked empty, opened by M', () => {
      const g = new CausalGraph(['X', 'M', 'Z']);
      g.addEdge('X', 'M');
      g.addEdge('Z', 'M');
      const adj: boolean[][] = [[false, true, false], [false, false, false], [false, true, false]];

      expect(g.dSeparated('X', 'Z', [])).toBe(
        bruteForceDSeparated(adj, 0, 2, new Set()),
      );
      expect(g.dSeparated('X', 'Z', ['M'])).toBe(
        bruteForceDSeparated(adj, 0, 2, new Set([1])),
      );
    });

    it('collider descendant activation: X→M←Z, M→W, Z={W}', () => {
      const g = new CausalGraph(['X', 'M', 'Z', 'W']);
      g.addEdge('X', 'M');
      g.addEdge('Z', 'M');
      g.addEdge('M', 'W');
      const adj: boolean[][] = [
        [false, true, false, false],
        [false, false, false, true],
        [false, true, false, false],
        [false, false, false, false],
      ];

      // W is descendant of collider M → should activate M
      expect(g.dSeparated('X', 'Z', ['W'])).toBe(
        bruteForceDSeparated(adj, 0, 2, new Set([3])),
      );
    });
  });

  // ── Random DAG fuzzing ─────────────────────────────────────────

  describe('random DAG fuzzing (200 DAGs × 5 triples = 1000 tests)', () => {
    // Pre-generate all test cases
    const testCases: Array<{ dag: RandomDAG; triple: { x: number; y: number; z: Set<number> } }> = [];

    for (let d = 0; d < NUM_DAGS; d++) {
      // Vary node count and density across tests
      const nodes = 4 + (d % 7); // 4-10 nodes
      const density = 0.2 + (d % 4) * 0.15; // 0.2, 0.35, 0.5, 0.65
      const dag = generateRandomDAG(nodes, density, rngSeed++);

      for (let t = 0; t < TESTS_PER_DAG; t++) {
        const triple = randomTriple(nodes, rngSeed++);
        testCases.push({ dag, triple });
      }
    }

    // Run each test case as an individual test
    testCases.forEach(({ dag, triple }, idx) => {
      const { x, y, z } = triple;
      const zNames = [...z].map(zi => dag.nodes[zi]!);
      const xName = dag.nodes[x]!;
      const yName = dag.nodes[y]!;

      it(`DAG #${Math.floor(idx / TESTS_PER_DAG)} triple ${idx % TESTS_PER_DAG}: ${xName}-${yName}|{${zNames.join(',')}}`, () => {
        const implResult = dag.graph.dSeparated(xName, yName, zNames);
        const refResult = bruteForceDSeparated(dag.adj, x, y, z);
        totalTests++;

        if (implResult !== refResult) {
          failures.push({
            dagDesc: `${dag.nodes.length}n d=${(dag.adj.flat().filter(Boolean).length / (dag.nodes.length * (dag.nodes.length - 1))).toFixed(2)}`,
            triple: `dSep(${xName}, ${yName} | {${zNames.join(',')}})`,
            impl: implResult,
            ref: refResult,
          });
        }

        expect(implResult).toBe(refResult);
      });
    });
  });

  // ── Summary check ──────────────────────────────────────────────

  it('achieves 100% agreement with brute-force reference', () => {
    if (failures.length > 0) {
      console.error(`\n${failures.length} d-separation mismatches found:`);
      for (const f of failures.slice(0, 10)) {
        console.error(`  ${f.dagDesc}: ${f.triple} → impl=${f.impl} ref=${f.ref}`);
      }
    }
    expect(failures.length).toBe(0);
    expect(totalTests).toBeGreaterThanOrEqual(NUM_DAGS * TESTS_PER_DAG);
  });
});

// ── d-Separation Invariants ──────────────────────────────────────────

describe('d-Separation Invariants', () => {
  it('symmetry: dSep(X,Y,Z) === dSep(Y,X,Z) for all random DAGs', () => {
    let seed = 9999;
    for (let i = 0; i < 50; i++) {
      const n = 4 + (i % 8);
      const { graph, nodes } = generateRandomDAG(n, 0.3, seed++);
      const t = randomTriple(n, seed++);

      const zNames = [...t.z].map(zi => nodes[zi]!);
      expect(graph.dSeparated(nodes[t.x]!, nodes[t.y]!, zNames)).toBe(
        graph.dSeparated(nodes[t.y]!, nodes[t.x]!, zNames),
      );
    }
  });

  it('never throws for random conditioning sets on random DAGs', () => {
    let seed = 8888;
    for (let i = 0; i < 50; i++) {
      const n = 3 + (i % 10);
      const { graph, nodes } = generateRandomDAG(n, 0.3, seed++);
      const zSize = Math.floor(Math.random() * (n + 1));
      const z = new Set<string>();
      while (z.size < zSize) {
        z.add(nodes[Math.floor(Math.random() * n)]!);
      }
      expect(() => graph.dSeparated(
        nodes[Math.floor(Math.random() * n)]!,
        nodes[Math.floor(Math.random() * n)]!,
        [...z],
      )).not.toThrow();
    }
  });

  it('self is never d-separated from itself (reflexivity)', () => {
    const { graph, nodes } = generateRandomDAG(5, 0.4, 7777);
    for (const node of nodes) {
      expect(graph.dSeparated(node, node, [])).toBe(false);
      // Even with conditioning on anything
      const other = nodes.filter(n => n !== node);
      expect(graph.dSeparated(node, node, other)).toBe(false);
    }
  });
});
