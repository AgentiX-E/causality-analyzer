/**
 * Tests for LLM-Powered Explainer.
 *
 * Tests the fallback behavior when no API key is configured,
 * and verifies graceful degradation when LLM is unavailable.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { CausalGraph, HeuristicPathRCA } from '../../src/index.js';
import { explainRCAWithLLM } from '../../src/explain/llm-explainer.js';

describe('LLM Explainer — Fallback Behavior', () => {
  beforeAll(() => {
    // Ensure no API key is set for deterministic fallback testing
    delete process.env.DEEPSEEK_API_KEY;
  });

  it('falls back to template explainer when no API key', async () => {
    const graph = new CausalGraph(['A', 'B', 'C']);
    graph.addEdge('A', 'B');
    graph.addEdge('B', 'C');

    const rca = new HeuristicPathRCA();
    rca.train(graph, new Set(['C']), [
      [1, 2, 3],
      [1.1, 2.1, 2.9],
      [0.9, 1.9, 3.1],
    ]);
    const result = rca.findRootCauses(['C']);

    const explanation = await explainRCAWithLLM(result, 'HeuristicPathRCA', 3);
    expect(explanation.summary).toBeDefined();
    expect(explanation.reasoning).toBeDefined();
    expect(explanation.confidence).toBeDefined();
    expect(['high', 'medium', 'low']).toContain(explanation.confidence);
  });

  it('handles empty root cause results gracefully', async () => {
    const graph = new CausalGraph(['X', 'Y']);
    graph.addEdge('X', 'Y');

    const rca = new HeuristicPathRCA();
    rca.train(graph, new Set(['Y']), [
      [1, 2],
      [1, 3],
    ]);
    const result = rca.findRootCauses(['Y']);

    const explanation = await explainRCAWithLLM(result, 'HeuristicPathRCA', 2);
    expect(explanation).toBeDefined();
    expect(typeof explanation.summary).toBe('string');
  });
});
