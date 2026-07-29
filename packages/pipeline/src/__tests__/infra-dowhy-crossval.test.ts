/**
 * DoWhy Cross-validation Tests.
 *
 * Validates Causality Analyzer's backdoor criterion, frontdoor criterion,
 * and effect estimation against known-correct outputs from DoWhy.
 *
 * Test DAGs are constructed to match DoWhy's canonical examples.
 * Expected adjustment sets are verified against DoWhy's output.
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { findBackdoorAdjustmentSet } from '../../src/infer/backdoor.js';
import { identifyByDoCalculus } from '../../src/infer/do-calculus.js';
import { findBackdoorSet, adjustBackdoor } from '../../src/infer/effect-estimation.js';

describe('DoWhy Cross-validation: Backdoor Criterion', () => {
  // Canonical confounded graph: X ← C → Y  (DoWhy example 1)
  it('identifies C as confounder in X←C→Y (DoWhy backdoor canonical)', () => {
    const g = new CausalGraph(['X', 'Y', 'C']);
    g.addEdge('C', 'X');
    g.addEdge('C', 'Y');
    // DoWhy output: backdoor adjustment set = ['C']

    const adj = findBackdoorAdjustmentSet(g, 'X', 'Y');
    expect(adj).toEqual(['C']);
  });

  // Confounder + direct effect: X←C→Y, X→Y
  it('identifies C as confounder in X←C→Y, X→Y (with direct effect)', () => {
    const g = new CausalGraph(['X', 'Y', 'C']);
    g.addEdge('C', 'X');
    g.addEdge('C', 'Y');
    g.addEdge('X', 'Y');

    const adj = findBackdoorAdjustmentSet(g, 'X', 'Y');
    expect(adj).toEqual(['C']);
  });

  // Multiple confounders: X←C1→Y, X←C2→Y
  it('identifies multiple confounders', () => {
    const g = new CausalGraph(['X', 'Y', 'C1', 'C2']);
    g.addEdge('C1', 'X');
    g.addEdge('C1', 'Y');
    g.addEdge('C2', 'X');
    g.addEdge('C2', 'Y');

    const adj = findBackdoorAdjustmentSet(g, 'X', 'Y');
    expect(adj).toContain('C1');
    expect(adj).toContain('C2');
    expect(adj.length).toBe(2);
  });

  // Chain mediator: X → M → Y (no confounder)
  it('returns empty backdoor set for pure mediator (no confounder)', () => {
    const g = new CausalGraph(['X', 'M', 'Y']);
    g.addEdge('X', 'M');
    g.addEdge('M', 'Y');

    const adj = findBackdoorAdjustmentSet(g, 'X', 'Y');
    expect(adj).toEqual([]);
  });

  // Collider: X → Z ← Y (no confounding)
  it('returns empty backdoor set for collider (no confounding)', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Z');
    g.addEdge('Y', 'Z');

    const adj = findBackdoorAdjustmentSet(g, 'X', 'Y');
    expect(adj).toEqual([]);
  });
});

describe('DoWhy Cross-validation: Identifiability', () => {
  // Test that our do-calculus matches DoWhy's identification results

  it('identifies backdoor-identifiable DAG', () => {
    const g = new CausalGraph(['X', 'Y', 'C']);
    g.addEdge('C', 'X');
    g.addEdge('C', 'Y');

    const result = identifyByDoCalculus(g, 'X', 'Y');
    expect(result.identifiable).toBe(true);
    expect(result.expressionType).toBe('backdoor');
    expect(result.adjustmentSet).toContain('C');
  });

  it('identifies effect when treatment has no backdoor paths', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');

    const result = identifyByDoCalculus(g, 'X', 'Y');
    expect(result.identifiable).toBe(true);
    expect(result.adjustmentSet).toEqual([]);
  });

  it('correctly handles M-bias (collider blocks backdoor)', () => {
    const g = new CausalGraph(['X', 'Y', 'C1', 'C2', 'M']);
    g.addEdge('C1', 'X');
    g.addEdge('C1', 'M');
    g.addEdge('C2', 'M');
    g.addEdge('C2', 'Y');

    const result = identifyByDoCalculus(g, 'X', 'Y');
    expect(result.identifiable).toBe(true);
  });

  it('identifies frontdoor criterion graph', () => {
    const g = new CausalGraph(['X', 'M', 'Y', 'U']);
    g.addEdge('U', 'X');
    g.addEdge('U', 'Y');
    g.addEdge('X', 'M');
    g.addEdge('M', 'Y');

    const result = identifyByDoCalculus(g, 'X', 'Y');
    expect(result.identifiable).toBe(true);
  });
});

describe('DoWhy Cross-validation: Effect Estimation Numerical', () => {
  // Generate data with known causal effect structure and verify estimate

  it('estimates ATE ≈ 1.0 for X→Y with coefficient 1.0', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');

    const data: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const x = Math.random() * 2 - 1; // X ~ U(-1,1)
      const y = x + (Math.random() - 0.5) * 0.2; // Y = X + small noise
      data.push([x, y]);
    }

    const nodeIndex = new Map([['X', 0], ['Y', 1]]);
    const { ate } = adjustBackdoor(g, 'X', 'Y', data, nodeIndex);

    // ATE should be close to 1.0 (coeff of X in Y = X + noise)
    expect(ate).toBeGreaterThan(0.5);
    expect(ate).toBeLessThan(1.5);
  });

  it('estimates ATE ≈ 0 for X→Y with zero effect', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');

    const data: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const x = Math.random() * 2 - 1;
      const y = (Math.random() - 0.5) * 0.2; // Y = pure noise, no X effect
      data.push([x, y]);
    }

    const nodeIndex = new Map([['X', 0], ['Y', 1]]);
    const { ate, se } = adjustBackdoor(g, 'X', 'Y', data, nodeIndex);

    // ATE close to 0; SE captures uncertainty
    expect(Math.abs(ate)).toBeLessThan(0.3);
    expect(se).toBeGreaterThan(0);
  });

  it('estimates confounder-adjusted ATE correctly', () => {
    const g = new CausalGraph(['X', 'Y', 'C']);
    g.addEdge('C', 'X');
    g.addEdge('C', 'Y');
    g.addEdge('X', 'Y');

    const data: number[][] = [];
    for (let i = 0; i < 500; i++) {
      const c = Math.random() * 2 - 1;
      const x = 0.5 * c + (Math.random() - 0.5) * 0.2;
      const y = 0.7 * x + 0.3 * c + (Math.random() - 0.5) * 0.1;
      data.push([x, y, c]);
    }

    const nodeIndex = new Map([['X', 0], ['Y', 1], ['C', 2]]);
    const { ate } = adjustBackdoor(g, 'X', 'Y', data, nodeIndex);

    // True causal effect = 0.7 (coefficient of X in Y equation)
    expect(ate).toBeGreaterThan(0.4);
    expect(ate).toBeLessThan(1.0);
  });
});

describe('DoWhy Cross-validation: edge cases', () => {
  it('backdoor set is identical from both APIs', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('Z', 'X');
    g.addEdge('Z', 'Y');

    const s1 = findBackdoorSet(g, 'X', 'Y');
    const s2 = findBackdoorAdjustmentSet(g, 'X', 'Y');
    expect(s1.sort()).toEqual(s2.sort());
  });

  it('empty adjustment for randomized treatment (no parents)', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');

    const s = findBackdoorSet(g, 'X', 'Y');
    expect(s).toEqual([]);
  });
});
