import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../graph/causal-graph.js';
import {
  detectColliderBias, findColliders, isColliderBias, removeColliderBiasedAdjustments,
} from '../infer/collider-bias.js';

describe('findColliders', () => {
  it('finds collider in X→M←Z structure', () => {
    const g = new CausalGraph(['X', 'M', 'Z']);
    g.addEdge('X', 'M'); g.addEdge('Z', 'M');
    expect(findColliders(g)).toContain('M');
  });

  it('finds no colliders in chain X→Y→Z', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    expect(findColliders(g)).toEqual([]);
  });
});

describe('detectColliderBias', () => {
  it('detects collider conditioned directly', () => {
    const g = new CausalGraph(['X', 'M', 'Z']);
    g.addEdge('X', 'M'); g.addEdge('Z', 'M');
    const warnings = detectColliderBias(g, ['M']);
    expect(warnings.length).toBe(1);
    expect(warnings[0]!.collider).toBe('M');
    expect(warnings[0]!.parents).toEqual(expect.arrayContaining(['X', 'Z']));
  });

  it('detects descendant of collider conditioned', () => {
    const g = new CausalGraph(['X', 'M', 'Z', 'W']);
    g.addEdge('X', 'M'); g.addEdge('Z', 'M'); g.addEdge('M', 'W');
    const warnings = detectColliderBias(g, ['W']);
    expect(warnings.length).toBe(1);
    expect(warnings[0]!.collider).toBe('M');
    expect(warnings[0]!.descendantConditioned).toBe('W');
  });

  it('no warnings for non-collider conditioning', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    expect(detectColliderBias(g, ['Y']).length).toBe(0);
  });

  it('empty conditioning set yields no warnings', () => {
    const g = new CausalGraph(['X', 'M', 'Z']);
    g.addEdge('X', 'M'); g.addEdge('Z', 'M');
    expect(detectColliderBias(g, []).length).toBe(0);
  });

  it('multiple colliders detected independently', () => {
    const g = new CausalGraph(['X1', 'A', 'X2', 'B', 'Y']);
    g.addEdge('X1', 'A'); g.addEdge('X2', 'A');
    g.addEdge('X1', 'B'); g.addEdge('Y', 'B');
    const warnings = detectColliderBias(g, ['A', 'B']);
    expect(warnings.length).toBe(2);
  });
});

describe('isColliderBias', () => {
  it('returns true for conditioned collider', () => {
    const g = new CausalGraph(['X', 'M', 'Z']);
    g.addEdge('X', 'M'); g.addEdge('Z', 'M');
    expect(isColliderBias(g, 'M', ['M'])).toBe(true);
  });

  it('returns false for non-collider', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    expect(isColliderBias(g, 'Y', ['Y'])).toBe(false);
  });
});

describe('removeColliderBiasedAdjustments', () => {
  it('removes collider-biased variables from adjustment set', () => {
    const g = new CausalGraph(['X', 'M', 'Z', 'W']);
    g.addEdge('X', 'M'); g.addEdge('Z', 'M');
    g.addEdge('X', 'W'); g.addEdge('W', 'Z');
    const safe = removeColliderBiasedAdjustments(g, ['M', 'W']);
    // M is a collider → removed; W is safe → kept
    expect(safe).toContain('W');
    expect(safe).not.toContain('M');
  });
});
