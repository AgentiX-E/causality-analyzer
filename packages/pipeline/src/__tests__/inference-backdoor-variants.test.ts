/**
 * Backdoor Variants Test Suite.
 *
 * Validates all 5 backdoor set search methods:
 * - minimal (parents)
 * - maximal (all ancestors)
 * - efficient (greedy backward)
 * - exhaustive (all minimal sets)
 * - mincost (data-driven)
 */
import { describe, it, expect } from 'vitest';
import {
  findBackdoorAdjustmentSet,
  findMinimal,
  findMaximal,
  findEfficient,
  findAllMinimal,
  findMinCost,
  verifyBackdoorBlock,
  getAdmissibleCandidates,
} from '../../src/infer/backdoor.js';
import { CausalGraph } from '../../src/graph/causal-graph.js';

function makeChainGraph(): CausalGraph {
  const g = new CausalGraph(['X', 'M', 'Y']);
  g.addEdge('X', 'M');
  g.addEdge('M', 'Y');
  return g;
}

function makeConfoundedGraph(): CausalGraph {
  const g = new CausalGraph(['X', 'Y', 'C']);
  g.addEdge('C', 'X');
  g.addEdge('C', 'Y');
  return g;
}

function makeMultivarConfoundedGraph(): CausalGraph {
  const g = new CausalGraph(['X', 'Y', 'C1', 'C2', 'Z']);
  g.addEdge('C1', 'X'); g.addEdge('C1', 'Y');
  g.addEdge('C2', 'X'); g.addEdge('C2', 'Y');
  g.addEdge('X', 'Z');  // Z is a mediator, not confounder
  g.addEdge('Z', 'Y');
  return g;
}

function makeMGraph(): CausalGraph {
  const g = new CausalGraph(['X', 'Y', 'U1', 'U2']);
  g.addEdge('U1', 'X'); g.addEdge('U1', 'Y');
  g.addEdge('U2', 'X'); g.addEdge('U2', 'Y');
  g.addEdge('X', 'Y');
  return g;
}

describe('Backdoor Variants', () => {
  describe('Minimal (parents)', () => {
    it('returns parents for confounded graph', () => {
      const g = makeConfoundedGraph();
      const result = findMinimal(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'));
      expect(result).toEqual(['C']);
    });

    it('returns empty for chain (no confounders)', () => {
      const g = makeChainGraph();
      const result = findMinimal(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'));
      expect(result).toEqual([]);
    });

    it('returns both confounders', () => {
      const g = makeMultivarConfoundedGraph();
      const result = findMinimal(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'));
      expect(result).toContain('C1');
      expect(result).toContain('C2');
      expect(result).not.toContain('Z');
    });
  });

  describe('Maximal', () => {
    it('returns all admissible ancestors', () => {
      const g = makeMGraph();
      const result = findMaximal(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'));
      expect(result).toContain('U1');
      expect(result).toContain('U2');
      expect(result.length).toBe(2);
    });

    it('verifies maximal set blocks backdoor paths', () => {
      const g = makeMGraph();
      const result = findMaximal(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'));
      expect(verifyBackdoorBlock(g, 'X', 'Y', result)).toBe(true);
    });
  });

  describe('Efficient', () => {
    it('returns subset of admissible when some are redundant', () => {
      const g = makeMGraph();
      const result = findEfficient(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'));
      // Efficient should be ≤ maximal length
      const maximal = findMaximal(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'));
      expect(result.length).toBeLessThanOrEqual(maximal.length);
    });

    it('verifies efficient set blocks backdoor paths', () => {
      const g = makeMGraph();
      const result = findEfficient(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'));
      expect(verifyBackdoorBlock(g, 'X', 'Y', result)).toBe(true);
    });

    it('returns empty for no-confounding graph', () => {
      const g = makeChainGraph();
      const result = findEfficient(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'));
      expect(result).toEqual([]);
    });
  });

  describe('Exhaustive', () => {
    it('finds valid adjustment set for confounded graph', () => {
      const g = makeConfoundedGraph();
      const result = findAllMinimal(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'));
      expect(result).toContain('C');
    });

    it('verifies exhaustive result blocks backdoor paths', () => {
      const g = makeMGraph();
      const result = findAllMinimal(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'));
      expect(verifyBackdoorBlock(g, 'X', 'Y', result)).toBe(true);
    });
  });

  describe('MinCost (data-driven)', () => {
    it('falls back to efficient when no data provided', () => {
      const g = makeMGraph();
      const result = findMinCost(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'));
      expect(verifyBackdoorBlock(g, 'X', 'Y', result)).toBe(true);
    });

    it('uses data to select adjustment set', () => {
      const g = makeMGraph();
      const nodeIndex = new Map([['X', 0], ['Y', 1], ['U1', 2], ['U2', 3]]);
      const data = Array.from({ length: 50 }, () => [
        Math.random(), // X
        Math.random(), // Y
        Math.random(), // U1
        Math.random(), // U2
      ]);
      const result = findMinCost(g, 'X', 'Y', getAdmissibleCandidates(g, 'X', 'Y'), data, nodeIndex);
      expect(verifyBackdoorBlock(g, 'X', 'Y', result)).toBe(true);
    });
  });

  describe('findBackdoorAdjustmentSet with method parameter', () => {
    it('defaults to minimal', () => {
      const g = makeConfoundedGraph();
      const result = findBackdoorAdjustmentSet(g, 'X', 'Y');
      expect(result).toEqual(['C']);
    });

    it('accepts explicit method', () => {
      const g = makeConfoundedGraph();
      const minimal = findBackdoorAdjustmentSet(g, 'X', 'Y', 'minimal');
      const maximal = findBackdoorAdjustmentSet(g, 'X', 'Y', 'maximal');
      expect(minimal).toEqual(['C']);
      expect(maximal).toContain('C');
    });
  });
});
