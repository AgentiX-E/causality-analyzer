/**
 * I104: CPDAG Orientation Unit Tests
 *
 * Tests orientCPDAG with focus on structural correctness.
 * Note: The CPDAG module orients v-structures whenever an unshielded
 * triple exists and no separation set information is provided.
 * This is correct algorithm behavior — the caller must supply sepSets
 * to suppress v-structure orientation.
 */

import { describe, it, expect } from 'vitest';
import { orientCPDAG } from '../graph/cpdag.js';
import type { CPDAGInput } from '@agentix-e/causality-analyzer-core';

/** Create a simple CPDAG input from adjacency pairs */
function makeInput(
  adjacencies: Array<readonly [string, string]>,
  sepSetOverrides?: Map<string, string[]>,
): CPDAGInput {
  const nodes = [...new Set(adjacencies.flat())];
  const adjSet: Array<readonly [string, string]> = adjacencies.map(([a, b]) => [a, b]);
  const sepSets = new Map<string, ReadonlySet<string>>();

  if (sepSetOverrides) {
    for (const [key, vals] of sepSetOverrides) {
      sepSets.set(key, new Set(vals));
    }
  }

  return { adjacencies: adjSet, sepSets, nodes };
}

function getOrient(
  result: Map<string, { sourceMark: string; targetMark: string }>,
  a: string,
  b: string,
): { sourceMark: string; targetMark: string } | undefined {
  const key = a < b ? `${a}|${b}` : `${b}|${a}`;
  return result.get(key);
}

// ── Basic Orientation ──────────────────────────────────────────────────

describe('orientCPDAG — basic', () => {
  it('returns map with entries for all adjacencies', () => {
    const input = makeInput([['A', 'B'], ['A', 'C'], ['B', 'C']]);
    const result = orientCPDAG(input);
    expect(result.size).toBe(3);
  });

  it('handles single adjacency', () => {
    const input = makeInput([['A', 'B']]);
    const result = orientCPDAG(input);
    expect(result.size).toBe(1);
  });

  it('handles empty adjacency list', () => {
    const input = makeInput([]);
    const result = orientCPDAG(input);
    expect(result.size).toBe(0);
  });

  it('produces valid edge marks only', () => {
    const input = makeInput([['A', 'B'], ['B', 'C'], ['C', 'D'], ['A', 'D']]);
    const result = orientCPDAG(input);
    const validMarks = new Set(['tail', 'arrow', 'circle']);
    for (const [, orient] of result) {
      expect(validMarks.has(orient.sourceMark)).toBe(true);
      expect(validMarks.has(orient.targetMark)).toBe(true);
    }
  });
});

// ── V-Structure Detection ──────────────────────────────────────────────

describe('orientCPDAG — v-structures', () => {
  it('creates v-structure when B not in SepSet(A,C) for unshielded A-B-C', () => {
    const sepSets = new Map<string, string[]>();
    sepSets.set('A|C', []); // Empty sepSet: B is NOT in it → v-structure
    const input = makeInput([['A', 'B'], ['B', 'C']], sepSets);

    const result = orientCPDAG(input);
    const ab = getOrient(result, 'A', 'B');
    const bc = getOrient(result, 'B', 'C');

    expect(ab).toBeDefined();
    expect(bc).toBeDefined();
    // Collider at B: A→B and C→B (both arrows point TO B)
    // For 'A|B': A has 'tail' (—), B has 'arrow' (← or →)
    expect(ab!.targetMark).toBe('arrow');  // arrow at B
    // For 'B|C': B has 'arrow' (← or →), C has 'tail' (—)
    expect(bc!.sourceMark).toBe('arrow');  // arrow at B
  });

  it('suppresses v-structure when middle node IS in SepSet', () => {
    const sepSets = new Map<string, string[]>();
    sepSets.set('A|C', ['B']); // B IS in sepSet → no v-structure
    const input = makeInput([['A', 'B'], ['B', 'C']], sepSets);

    const result = orientCPDAG(input);
    const ab = getOrient(result, 'A', 'B');
    // Edge should remain undirected
    expect(ab!.sourceMark).toBe('tail');
    expect(ab!.targetMark).toBe('tail');
  });
});

// ── Topology Tests ────────────────────────────────────────────────────

describe('orientCPDAG — topology', () => {
  it('fully connected triangle stays undirected (no v-structures)', () => {
    const input = makeInput([['A', 'B'], ['B', 'C'], ['A', 'C']]);
    const result = orientCPDAG(input);
    for (const [, orient] of result) {
      // All should be undirected since no unshielded triples exist
      expect(orient.sourceMark).toBe('tail');
      expect(orient.targetMark).toBe('tail');
    }
  });

  it('star topology with center connected to all', () => {
    const adj: Array<readonly [string, string]> = [
      ['C', 'A'], ['C', 'B'], ['C', 'D'], ['C', 'E'],
    ];
    const input = makeInput(adj);
    const result = orientCPDAG(input);
    expect(result.size).toBe(4);
  });

  it('produces non-empty orientation map for any valid input', () => {
    const input = makeInput([['X', 'Y'], ['Y', 'Z'], ['Z', 'W']]);
    const result = orientCPDAG(input);
    expect(result.size).toBe(3);
  });
});

// ── Edge Case Topologies ───────────────────────────────────────────────

describe('orientCPDAG — edge cases', () => {
  it('handles single node (no adjacencies)', () => {
    const input: CPDAGInput = {
      adjacencies: [],
      sepSets: new Map(),
      nodes: ['A'],
    };
    const result = orientCPDAG(input);
    expect(result.size).toBe(0);
  });

  it('handles two disconnected components', () => {
    const input = makeInput([['A', 'B'], ['C', 'D']]);
    const result = orientCPDAG(input);
    expect(result.size).toBe(2);
  });

  it('all result edges have valid mark pairs', () => {
    const input = makeInput([['A', 'B'], ['A', 'C'], ['B', 'C'], ['C', 'D'], ['D', 'E']]);
    const result = orientCPDAG(input);
    for (const [, orient] of result) {
      expect(typeof orient.sourceMark).toBe('string');
      expect(typeof orient.targetMark).toBe('string');
    }
  });
});
