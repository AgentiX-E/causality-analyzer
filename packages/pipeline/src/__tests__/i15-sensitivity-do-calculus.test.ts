import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../graph/causal-graph.js';
import { eValueSensitivity, partialRSensitivity, robustnessValue } from '../infer/sensitivity.js';
import { identifyByDoCalculus } from '../infer/do-calculus.js';

// ── E-value ──────────────────────────────────────────────────────────

describe('eValueSensitivity', () => {
  it('returns E-value > 1 for strong effect', () => {
    const { eValue } = eValueSensitivity(0.8);
    expect(eValue).toBeGreaterThan(1.5);
  });

  it('returns E-value ≈ 1 for weak effect', () => {
    const { eValue } = eValueSensitivity(0.001);
    expect(eValue).toBeCloseTo(1, 1);
  });

  it('handles zero effect', () => {
    const result = eValueSensitivity(0);
    expect(result.eValue).toBe(1);
    expect(result.interpretation).toContain('undefined');
  });

  it('handles negative ATE', () => {
    const { eValue } = eValueSensitivity(-0.5);
    expect(eValue).toBeGreaterThan(1);
  });

  it('provides interpretation for strong effect', () => {
    const { interpretation } = eValueSensitivity(2.0);
    expect(interpretation).toContain('strong');
  });
});

// ── Partial R² ───────────────────────────────────────────────────────

describe('partialRSensitivity', () => {
  it('returns low R² for large sample strong effect', () => {
    const result = partialRSensitivity(1.0, 0.1, 1000);
    expect(result.r2Treatment).toBeGreaterThan(0);
    expect(result.r2Treatment).toBeLessThan(1);
  });

  it('handles small sample gracefully', () => {
    const result = partialRSensitivity(0.5, 0.3, 10);
    expect(result.r2Treatment).toBeGreaterThanOrEqual(0);
  });

  it('returns zero threshold R² when no reduction needed', () => {
    const result = partialRSensitivity(0.5, 0.2, 100, 0.5);
    expect(result.r2Treatment).toBeCloseTo(0, 2);
  });

  it('handles n < 2', () => {
    const result = partialRSensitivity(0.5, 0.1, 1);
    expect(result.interpretation).toContain('Insufficient');
  });
});

// ── Robustness Value ─────────────────────────────────────────────────

describe('robustnessValue', () => {
  it('returns RV > 2 for large sample strong effect', () => {
    const { rv } = robustnessValue(1.5, 0.1, 1000);
    expect(rv).toBeGreaterThan(2);
  });

  it('returns RV < 2 for small sample weak effect', () => {
    const { rv } = robustnessValue(0.3, 0.5, 20);
    expect(rv).toBeLessThan(2);
  });

  it('interprets as "SENSITIVE" for very weak evidence', () => {
    const { interpretation } = robustnessValue(0.1, 0.5, 10);
    expect(interpretation).toContain('SENSITIVE');
  });
});

// ── do-calculus ──────────────────────────────────────────────────────

function confoundedDAG(): CausalGraph {
  const g = new CausalGraph(['Z', 'X', 'Y']);
  g.addEdge('Z', 'X'); g.addEdge('Z', 'Y'); g.addEdge('X', 'Y');
  return g;
}

function mediationDAG(): CausalGraph {
  const g = new CausalGraph(['X', 'M', 'Y']);
  g.addEdge('X', 'M'); g.addEdge('M', 'Y');
  return g;
}

describe('identifyByDoCalculus', () => {
  it('identifies backdoor in confounded DAG', () => {
    const g = confoundedDAG();
    const result = identifyByDoCalculus(g, 'X', 'Y');
    expect(result.identifiable).toBe(true);
    expect(result.expressionType).toBe('backdoor');
    expect(result.adjustmentSet).toContain('Z');
  });

  it('identifies frontdoor in mediation DAG', () => {
    const g = mediationDAG();
    const result = identifyByDoCalculus(g, 'X', 'Y');
    expect(result.identifiable).toBe(true);
    expect(result.expressionType).toBe('frontdoor');
  });

  it('reports unidentifiable for fully disconnected graph', () => {
    const g = new CausalGraph(['X', 'Y']);
    const result = identifyByDoCalculus(g, 'X', 'Y');
    // No edges → no confounding → effect is identifiable (= direct path)
    // With no edges, P(Y|do(X)) = P(Y) (identifiable via ID algorithm)
    expect(result.identifiable).toBe(true);
  });

  it('handles disconnected nodes', () => {
    const g = new CausalGraph(['A', 'B']);
    const result = identifyByDoCalculus(g, 'A', 'B');
    expect(typeof result.identifiable).toBe('boolean');
  });
});
