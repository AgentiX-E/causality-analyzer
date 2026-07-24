/**
 * Tests for the unified backdoor module.
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { findBackdoorAdjustmentSet, verifyBackdoorBlock, findMediators } from '../../src/infer/backdoor.js';

describe('Unified Backdoor Module', () => {
  it('returns empty for graph with no parents of treatment', () => {
    const g = new CausalGraph(['X', 'Y']);
    const adj = findBackdoorAdjustmentSet(g, 'X', 'Y');
    expect(adj).toEqual([]);
  });

  it('returns parent for simple confounded graph', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('Z', 'X');
    g.addEdge('Z', 'Y');
    const adj = findBackdoorAdjustmentSet(g, 'X', 'Y');
    expect(adj).toContain('Z');
  });

  it('verifyBackdoorBlock returns true when d-separation holds', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('Z', 'X');
    g.addEdge('Z', 'Y');
    // Z blocks the backdoor path X ← Z → Y
    expect(verifyBackdoorBlock(g, 'X', 'Y', ['Z'])).toBe(true);
  });

  it('verifyBackdoorBlock returns false when conditioning set is insufficient', () => {
    const g = new CausalGraph(['X', 'Y', 'Z1', 'Z2']);
    g.addEdge('Z1', 'X');
    g.addEdge('Z1', 'Y');
    g.addEdge('Z2', 'X');
    g.addEdge('Z2', 'Y');
    // Z1 alone is not sufficient — Z2's backdoor path remains open
    expect(verifyBackdoorBlock(g, 'X', 'Y', ['Z1'])).toBe(false);
  });

  it('findMediators returns nodes on directed path from treatment to outcome', () => {
    const g = new CausalGraph(['X', 'M', 'Y']);
    g.addEdge('X', 'M');
    g.addEdge('M', 'Y');

    const meds = findMediators(g, 'X', 'Y');
    expect(meds).toContain('M');
    expect(meds.length).toBe(1);
  });

  it('findMediators returns empty when no mediators exist', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');

    const meds = findMediators(g, 'X', 'Y');
    expect(meds).toEqual([]);
  });
});
