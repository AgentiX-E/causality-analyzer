/**
 * I39: NL Explainer tests.
 */
import { describe, it, expect } from 'vitest';
import {
  explainRCA,
  explainSensitivity,
  explainEstimate,
  explainDetection,
} from '../explainer.js';
import type { RCAResult, DetectionResult } from '@agentix-e/causality-analyzer-core';

function makeRCAResult(causes: Array<{ name: string; score: number }>): RCAResult {
  return {
    rootCauses: causes.map((c, i) => ({
      name: c.name,
      score: c.score,
      confidence: 0.8,
      rank: i + 1,
      evidence: [],
    })),
    paths: [],
    metadata: { method: 'test', analyzedAt: Date.now(), durationMs: 0, extra: {} },
    toJSON() { return { rootCauses: this.rootCauses, paths: this.paths, metadata: this.metadata }; },
  };
}

describe('explainRCA', () => {
  it('handles empty result', () => {
    const result = makeRCAResult([]);
    const explanation = explainRCA(result, 'VariableElimination');
    expect(explanation.summary).toContain('No root causes');
    expect(explanation.confidence).toBe('low');
    expect(explanation.caveats.length).toBeGreaterThan(0);
  });

  it('reports high confidence for strong root cause', () => {
    const result = makeRCAResult([{ name: 'Memory', score: 0.87 }]);
    const explanation = explainRCA(result, 'VE');
    expect(explanation.summary).toContain('Memory');
    expect(explanation.confidence).toBe('high');
  });

  it('reports medium confidence for moderate score', () => {
    const result = makeRCAResult([{ name: 'CPU', score: 0.5 }]);
    const explanation = explainRCA(result, 'HTRCA');
    expect(explanation.confidence).toBe('medium');
  });

  it('reports low confidence for weak score', () => {
    const result = makeRCAResult([{ name: 'Disk', score: 0.25 }]);
    const explanation = explainRCA(result, 'RandomWalk');
    expect(explanation.confidence).toBe('low');
  });

  it('explains gap between top candidates', () => {
    const result = makeRCAResult([
      { name: 'Memory', score: 0.85 },
      { name: 'CPU', score: 0.3 },
    ]);
    const explanation = explainRCA(result, 'VE');
    expect(explanation.reasoning).toContain('clear primary root cause');
  });

  it('explains close ranking', () => {
    const result = makeRCAResult([
      { name: 'Memory', score: 0.6 },
      { name: 'CPU', score: 0.55 },
    ]);
    const explanation = explainRCA(result, 'VE');
    expect(explanation.reasoning).toContain('marginally ahead');
  });

  it('includes node count when provided', () => {
    const result = makeRCAResult([{ name: 'A', score: 0.7 }]);
    const explanation = explainRCA(result, 'Test', 42);
    expect(explanation.reasoning).toContain('42 nodes');
  });

  it('ranks up to 5 candidates', () => {
    const result = makeRCAResult([
      { name: 'A', score: 0.9 },
      { name: 'B', score: 0.7 },
      { name: 'C', score: 0.5 },
      { name: 'D', score: 0.3 },
      { name: 'E', score: 0.2 },
      { name: 'F', score: 0.1 },
    ]);
    const explanation = explainRCA(result, 'Test');
    expect(explanation.ranking.length).toBe(5);
  });

  it('always includes caveats about model assumptions', () => {
    const result = makeRCAResult([{ name: 'X', score: 0.99 }]);
    const explanation = explainRCA(result, 'Perfect');
    const isHonest = explanation.caveats.some(c => c.includes('causal graph'));
    expect(isHonest).toBe(true);
  });
});

describe('explainSensitivity', () => {
  it('classifies as robust', () => {
    const explanation = explainSensitivity(3.0, 0.05, 0.05, 2.5);
    expect(explanation.summary).toContain('robust');
    expect(explanation.actionableAdvice).toContain('trustworthy');
  });

  it('classifies as moderate', () => {
    const explanation = explainSensitivity(2.0, 0.15, 0.1, 1.7);
    expect(explanation.summary).toContain('moderately robust');
    expect(explanation.actionableAdvice).toContain('caution');
  });

  it('classifies as sensitive', () => {
    const explanation = explainSensitivity(1.3, 0.3, 0.2, 1.0);
    expect(explanation.summary).toContain('sensitive');
    expect(explanation.actionableAdvice).toContain('NOT');
  });
});

describe('explainEstimate', () => {
  it('reports significant positive effect', () => {
    const explanation = explainEstimate(0.5, 0.1, 'Backdoor', 'Latency', 'Memory');
    expect(explanation.summary).toContain('increases');
    expect(explanation.confidenceStatement).toContain('High confidence');
  });

  it('reports non-significant effect', () => {
    const explanation = explainEstimate(0.05, 0.1, 'IV', 'Outcome', 'Treatment');
    expect(explanation.summary).toContain('statistically significant');
    expect(explanation.confidenceStatement).toContain('Insufficient');
  });

  it('reports CI with direction', () => {
    const explanation = explainEstimate(-0.3, 0.15, 'Frontdoor', 'Error Rate', 'Config');
    expect(explanation.summary).toContain('decreases');
  });
});

describe('explainDetection', () => {
  it('explains non-anomalous detection', () => {
    const result: DetectionResult = {
      isAnomalous: false,
      labels: new Float64Array([0]),
      scores: new Float64Array([0.1]),
      timestamp: Date.now(),
      metadata: { method: 'zscore' },
    };
    const explanation = explainDetection(result);
    expect(explanation).toContain('No anomaly');
  });

  it('explains anomalous detection', () => {
    const result: DetectionResult = {
      isAnomalous: true,
      labels: new Float64Array([1, 0, 1]),
      scores: new Float64Array([3.5, 1.2, 5.0]),
      timestamp: Date.now(),
      metadata: { method: 'spectral_residual' },
    };
    const explanation = explainDetection(result);
    expect(explanation).toContain('spectral_residual');
    expect(explanation).toContain('2 of 3');
  });
});
