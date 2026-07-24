/**
 * I45: Comprehensive LLM Explainer Coverage Tests.
 *
 * Tests both fallback paths (no API key) and mocked-API happy paths
 * to achieve high coverage on explainRCAWithLLM, explainSensitivityWithLLM,
 * explainEstimateWithLLM, and all internal helpers (getSensitivityInterpretation,
 * extractSection, estimateConfidence, buildRCAPrompt, buildSensitivityPrompt,
 * buildEstimatePrompt, callDeepSeek).
 */
import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach, vi } from 'vitest';
import type { RCAResult } from '@agentix-e/causality-analyzer-core';
import {
  explainRCAWithLLM,
  explainSensitivityWithLLM,
  explainEstimateWithLLM,
} from '../../src/explain/llm-explainer.js';

// ── Test Helpers ─────────────────────────────────────────────────────

const FAKE_API_KEY = 'sk-fake-test-key';

function makeRCAResult(
  causes: Array<{ name: string; score: number }>,
): RCAResult {
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
    toJSON() {
      return {
        rootCauses: this.rootCauses,
        paths: this.paths,
        metadata: this.metadata,
      };
    },
  };
}

/** Create a minimal mock fetch response for DeepSeek chat completions */
function mockFetchResponse(content: string, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () =>
      Promise.resolve({
        choices: [{ message: { content } }],
      }),
  } as Response;
}

/** Create a mock fetch that throws (simulating network error) */
function mockFetchError(): Response {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  return new Promise((_resolve, reject) => {
    reject(new Error('Network error'));
  }) as unknown as Response;
}

// ── Fallback Tests (No API Key) ──────────────────────────────────────

describe('LLM Explainer — Fallback Coverage', () => {
  beforeAll(() => {
    delete process.env.DEEPSEEK_API_KEY;
  });

  beforeEach(() => {
    delete process.env.DEEPSEEK_API_KEY;
  });

  // ── explainRCAWithLLM ──────────────────────────────────────────────

  describe('explainRCAWithLLM — fallback paths', () => {
    it('falls back to template explainer when no API key is set', async () => {
      const result = makeRCAResult([
        { name: 'Memory', score: 0.85 },
        { name: 'CPU', score: 0.45 },
      ]);

      const explanation = await explainRCAWithLLM(result, 'HeuristicPathRCA', 3);

      expect(explanation).toBeDefined();
      expect(explanation.summary).toBeDefined();
      expect(typeof explanation.summary).toBe('string');
      expect(explanation.reasoning).toBeDefined();
      expect(typeof explanation.reasoning).toBe('string');
      expect(explanation.confidence).toBeDefined();
      expect(['high', 'medium', 'low']).toContain(explanation.confidence);
      expect(Array.isArray(explanation.ranking)).toBe(true);
      expect(Array.isArray(explanation.caveats)).toBe(true);
    });

    it('falls back on empty root causes (also no API key)', async () => {
      const result = makeRCAResult([]);

      const explanation = await explainRCAWithLLM(result, 'VariableElimination', 5);

      expect(explanation).toBeDefined();
      expect(explanation.summary).toContain('No root causes');
      expect(explanation.confidence).toBe('low');
      expect(explanation.ranking).toEqual([]);
      expect(explanation.caveats.length).toBeGreaterThan(0);
    });

    it('handles multiple root causes with proper ranking', async () => {
      const result = makeRCAResult([
        { name: 'Database', score: 0.95 },
        { name: 'Cache', score: 0.72 },
        { name: 'Network', score: 0.58 },
        { name: 'DiskIO', score: 0.31 },
        { name: 'Auth', score: 0.15 },
      ]);

      const explanation = await explainRCAWithLLM(result, 'HeuristicPathRCA', 10);

      expect(explanation.ranking.length).toBeGreaterThan(0);
      expect(explanation.ranking.length).toBeLessThanOrEqual(5);
      for (const entry of explanation.ranking) {
        expect(entry.name).toBeDefined();
        expect(entry.rank).toBeGreaterThan(0);
        expect(entry.interpretation).toBeDefined();
      }
    });

    it('returns high confidence for strong root cause score', async () => {
      const result = makeRCAResult([{ name: 'ServiceX', score: 0.92 }]);

      const explanation = await explainRCAWithLLM(result, 'HTRCA');

      expect(explanation.confidence).toBe('high');
      expect(explanation.summary).toContain('ServiceX');
    });

    it('returns medium confidence for moderate score', async () => {
      const result = makeRCAResult([{ name: 'ServiceY', score: 0.55 }]);

      const explanation = await explainRCAWithLLM(result, 'RandomWalk');

      expect(explanation.confidence).toBe('medium');
    });

    it('returns low confidence for weak root cause score', async () => {
      const result = makeRCAResult([{ name: 'ServiceZ', score: 0.2 }]);

      const explanation = await explainRCAWithLLM(result, 'Circa');

      expect(explanation.confidence).toBe('low');
    });

    it('passes nodeCount through to template explainer', async () => {
      const result = makeRCAResult([{ name: 'NodeA', score: 0.7 }]);

      const explanation = await explainRCAWithLLM(result, 'GCM', 42);

      expect(explanation.reasoning).toContain('42');
      expect(explanation.reasoning).toContain('nodes');
    });

    it('works without nodeCount parameter', async () => {
      const result = makeRCAResult([{ name: 'NodeA', score: 0.7 }]);

      const explanation = await explainRCAWithLLM(result, 'GCM');

      expect(explanation.reasoning).toBeDefined();
    });

    it('always includes model caveats related to causal graph', async () => {
      const result = makeRCAResult([{ name: 'PerfectCause', score: 0.99 }]);

      const explanation = await explainRCAWithLLM(result, 'Perfect');

      const hasGraphCaveat = explanation.caveats.some(
        (c: string) => c.includes('causal graph'),
      );
      expect(hasGraphCaveat).toBe(true);
    });
  });

  // ── explainSensitivityWithLLM ──────────────────────────────────────

  describe('explainSensitivityWithLLM — fallback paths', () => {
    it('generates fallback for high e-value (robust)', async () => {
      const explanation = await explainSensitivityWithLLM(2.5, 0.1, 2.2, 0.5);

      expect(explanation.summary).toBeDefined();
      expect(explanation.interpretation).toBeDefined();
      expect(explanation.actionableAdvice).toBeDefined();

      expect(explanation.interpretation).toContain('Strong robustness');
      expect(explanation.actionableAdvice).toContain('reasonably robust');
    });

    it('generates fallback for moderate e-value', async () => {
      const explanation = await explainSensitivityWithLLM(1.7, 0.2, 1.6, 0.3);

      expect(explanation.interpretation).toContain('Moderate robustness');
      expect(explanation.actionableAdvice).toContain('reasonably robust');
    });

    it('generates fallback for low e-value (sensitive)', async () => {
      const explanation = await explainSensitivityWithLLM(1.2, 0.4, 1.1, 0.1);

      expect(explanation.interpretation).toContain('Low robustness');
      expect(explanation.actionableAdvice).toContain('collecting additional data');
    });

    it('generates fallback with e-value exactly at 2.0 boundary', async () => {
      const explanation = await explainSensitivityWithLLM(2.0, 0.05, 2.0, 0.8);

      expect(explanation.interpretation).toContain('Strong robustness');
    });

    it('generates fallback with e-value exactly at 1.5 boundary', async () => {
      const explanation = await explainSensitivityWithLLM(1.5, 0.3, 1.5, 0.4);

      expect(explanation.interpretation).toContain('Moderate robustness');
    });

    it('generates fallback for very low e-value', async () => {
      const explanation = await explainSensitivityWithLLM(1.01, 0.6, 1.0, 0.05);

      expect(explanation.interpretation).toContain('Low robustness');
      expect(explanation.actionableAdvice).toContain('collecting additional data');
    });

    it('summary includes e-value and robustness formatted values', async () => {
      const explanation = await explainSensitivityWithLLM(3.14159, 0.15, 3.0, 0.75);

      expect(explanation.summary).toContain('3.142');
      expect(explanation.summary).toContain('3.000');
    });

    it('interpretation includes robustness value', async () => {
      const explanation = await explainSensitivityWithLLM(2.5, 0.1, 2.345, 0.5);

      expect(explanation.interpretation).toContain('2.345');
    });
  });

  // ── explainEstimateWithLLM ─────────────────────────────────────────

  describe('explainEstimateWithLLM — fallback paths', () => {
    it('generates fallback for significant positive effect', async () => {
      const explanation = await explainEstimateWithLLM(
        'Backdoor', 0.5, 0.15, ['confounder1', 'confounder2'],
      );

      expect(explanation.summary).toBeDefined();
      expect(explanation.interpretation).toBeDefined();
      expect(explanation.confidenceStatement).toBeDefined();
      expect(explanation.summary).toContain('significant');
      expect(explanation.confidenceStatement).toContain('2 variable');
    });

    it('generates fallback for non-significant effect', async () => {
      const explanation = await explainEstimateWithLLM('IV', 0.05, 0.1, ['confounder1']);

      expect(explanation.summary).toContain('not significant');
      expect(explanation.interpretation).toContain('not significant');
      expect(explanation.confidenceStatement).toContain('1 variable');
    });

    it('generates fallback for negative significant effect', async () => {
      const explanation = await explainEstimateWithLLM(
        'Frontdoor', -0.8, 0.2, ['age', 'gender', 'income'],
      );

      expect(explanation.summary).toContain('significant');
      expect(explanation.summary).toContain('-0.800');
      expect(explanation.confidenceStatement).toContain('3 variable');
    });

    it('handles empty adjustment set', async () => {
      const explanation = await explainEstimateWithLLM('DirectAdjustment', 0.3, 0.12, []);

      expect(explanation.confidenceStatement).toContain('0 variable');
    });

    it('handles single adjustment variable', async () => {
      const explanation = await explainEstimateWithLLM(
        'PropensityScore', 0.25, 0.08, ['treatment_history'],
      );

      expect(explanation.confidenceStatement).toContain('1 variable');
    });

    it('handles multiple adjustment variables', async () => {
      const explanation = await explainEstimateWithLLM(
        'DoublyRobust', 0.4, 0.1, ['var_a', 'var_b', 'var_c', 'var_d', 'var_e'],
      );

      expect(explanation.confidenceStatement).toContain('5 variable');
    });

    it('formats ATE and SE to 4 decimal places in summary', async () => {
      const explanation = await explainEstimateWithLLM('Method', 0.123456, 0.0789, ['X']);

      expect(explanation.summary).toContain('0.1235');
      expect(explanation.summary).toContain('0.0789');
    });

    it('handles exactly-at-boundary significance (t = 1.96)', async () => {
      const explanation = await explainEstimateWithLLM('BoundaryMethod', 1.96, 1.0, ['covariate']);

      expect(explanation.summary).toContain('not significant');
    });

    it('handles just-above-boundary significance (t = 1.97)', async () => {
      const explanation = await explainEstimateWithLLM('BoundaryMethod', 1.97, 1.0, ['covariate']);

      expect(explanation.summary).toContain('significant');
    });

    it('handles very small standard error (large t-statistic)', async () => {
      const explanation = await explainEstimateWithLLM('PreciseMethod', 0.5, 0.0001, ['X']);

      expect(explanation.summary).toContain('significant');
    });

    it('interpretation mentions ATE/SE; confidenceStatement names method and adjustment count', async () => {
      const explanation = await explainEstimateWithLLM('CustomMethod', 0.3, 0.1, ['a', 'b']);

      expect(explanation.interpretation).toContain('0.3000');
      expect(explanation.interpretation).toContain('0.1000');
      expect(explanation.confidenceStatement).toContain('CustomMethod');
      expect(explanation.confidenceStatement).toContain('2 variable');
    });

    it('differs between significant and non-significant summaries', async () => {
      const sig = await explainEstimateWithLLM('M', 2.0, 0.5, ['X']);
      const notSig = await explainEstimateWithLLM('M', 0.3, 0.5, ['X']);

      expect(sig.summary).not.toBe(notSig.summary);
    });

    it('handles zero adjustment set with empty confounder statement', async () => {
      const explanation = await explainEstimateWithLLM('NoCovariates', 0.5, 0.1, []);

      expect(explanation.confidenceStatement).toContain('0 variable');
    });

    it('handles very large t-statistic (highly significant)', async () => {
      const explanation = await explainEstimateWithLLM('LargeEffect', 10.0, 0.1, ['W']);

      expect(explanation.summary).toContain('10.000');
      expect(explanation.summary).toContain('significant');
    });
  });
});

// ── Mocked API Tests (Happy Path + LLM Response Fallback) ────────────

describe('LLM Explainer — Mocked API', () => {
  let originalFetch: typeof global.fetch;

  beforeAll(() => {
    originalFetch = global.fetch;
  });

  afterAll(() => {
    global.fetch = originalFetch;
  });

  beforeEach(() => {
    process.env.DEEPSEEK_API_KEY = FAKE_API_KEY;
  });

  afterEach(() => {
    delete process.env.DEEPSEEK_API_KEY;
    vi.restoreAllMocks();
  });

  // ── explainRCAWithLLM ── with mocked API ──────────────────────────

  describe('explainRCAWithLLM — happy path', () => {
    it('returns LLM-generated explanation when API key is set', async () => {
      const mockContent = '1. Summary: Database latency is the primary root cause.\n2. Reasoning: The propagation path shows...\n3. High confidence based on strong evidence.';
      global.fetch = vi.fn().mockResolvedValue(mockFetchResponse(mockContent));

      const result = makeRCAResult([
        { name: 'Database', score: 0.95 },
        { name: 'Cache', score: 0.60 },
      ]);

      const explanation = await explainRCAWithLLM(result, 'HeuristicPathRCA', 5);

      expect(explanation.summary).toBeDefined();
      expect(explanation.ranking).toBeDefined();
      expect(explanation.caveats).toBeDefined();
      expect(explanation.confidence).toBeDefined();
      expect(explanation.ranking.length).toBe(2);
      expect(explanation.ranking[0]!.name).toBe('Database');
      expect(global.fetch).toHaveBeenCalledTimes(1);
    });

    it('detects high confidence from LLM response', async () => {
      const mockContent = 'Summary: Clear evidence.\nI have high confidence in this finding.';
      global.fetch = vi.fn().mockResolvedValue(mockFetchResponse(mockContent));

      const result = makeRCAResult([{ name: 'NodeA', score: 0.8 }]);
      const explanation = await explainRCAWithLLM(result, 'MethodX');

      expect(explanation.confidence).toBe('high');
    });

    it('detects low confidence from LLM response', async () => {
      const mockContent = 'Low confidence — results are uncertain due to limited data.';
      global.fetch = vi.fn().mockResolvedValue(mockFetchResponse(mockContent));

      const result = makeRCAResult([{ name: 'NodeB', score: 0.5 }]);
      const explanation = await explainRCAWithLLM(result, 'MethodY');

      expect(explanation.confidence).toBe('low');
    });

    it('detects medium confidence when no indicator in LLM response', async () => {
      const mockContent = 'The analysis shows mixed evidence for the root causes.';
      global.fetch = vi.fn().mockResolvedValue(mockFetchResponse(mockContent));

      const result = makeRCAResult([{ name: 'NodeC', score: 0.6 }]);
      const explanation = await explainRCAWithLLM(result, 'MethodZ');

      expect(explanation.confidence).toBe('medium');
    });

    it('extracts summary section from numbered LLM response', async () => {
      const mockContent = '1. The primary root cause is memory usage\n2. Evidence comes from propagation analysis\n3. Moderate confidence';
      global.fetch = vi.fn().mockResolvedValue(mockFetchResponse(mockContent));

      const result = makeRCAResult([{ name: 'Memory', score: 0.7 }]);
      const explanation = await explainRCAWithLLM(result, 'SectionTest');

      // extractSection should match against numbered lines or section names
      expect(explanation.summary).toBeDefined();
    });

    it('extracts summary section when named in LLM response', async () => {
      const mockContent = 'summary: Memory leak identified as the cause.\nreasoning: Memory showed anomalous patterns.\ncaveats: Limited data.';
      global.fetch = vi.fn().mockResolvedValue(mockFetchResponse(mockContent));

      const result = makeRCAResult([{ name: 'Memory', score: 0.75 }]);
      const explanation = await explainRCAWithLLM(result, 'NamedTest');

      expect(explanation.summary).toBeDefined();
    });

    it('falls back to template when LLM response is null', async () => {
      // API non-200 response on last retry → null
      global.fetch = vi.fn().mockResolvedValue(
        mockFetchResponse('', 500),
      );

      const result = makeRCAResult([{ name: 'ServiceA', score: 0.8 }]);
      const explanation = await explainRCAWithLLM(result, 'FailureTest');

      // Should fall back to template explainer
      expect(explanation.summary).toBeDefined();
      expect(explanation.confidence).toBeDefined();
      expect(explanation.caveats).toBeDefined();
      expect(global.fetch).toHaveBeenCalledTimes(3); // initial + 2 retries
    });

    it('falls back to template when LLM response has no content', async () => {
      global.fetch = vi.fn().mockResolvedValue(mockFetchResponse(''));

      const result = makeRCAResult([{ name: 'ServiceB', score: 0.7 }]);
      const explanation = await explainRCAWithLLM(result, 'EmptyTest');

      expect(explanation.summary).toBeDefined();
      expect(explanation.caveats).toBeDefined();
    });

    it('handles network error (retry + fallback)', async () => {
      global.fetch = vi.fn()
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Network error'));

      const result = makeRCAResult([{ name: 'ServiceC', score: 0.75 }]);
      const explanation = await explainRCAWithLLM(result, 'NetworkError');

      expect(explanation.summary).toBeDefined();
      expect(explanation.caveats).toBeDefined();
      expect(global.fetch).toHaveBeenCalledTimes(3);
    });
  });

  // ── explainSensitivityWithLLM ── with mocked API ──────────────────

  describe('explainSensitivityWithLLM — happy path', () => {
    it('returns LLM-generated explanation when API succeeds', async () => {
      const mockContent = 'The causal estimate shows strong robustness.\nE-value of 2.5 indicates the effect is not easily explained away.\nRecommendation: Proceed with confidence.';
      global.fetch = vi.fn().mockResolvedValue(mockFetchResponse(mockContent));

      const explanation = await explainSensitivityWithLLM(2.5, 0.1, 2.2, 0.5);

      expect(explanation.summary).toBeDefined();
      expect(explanation.interpretation).toBeDefined();
      expect(explanation.actionableAdvice).toBeDefined();
      expect(explanation.summary).toContain('robustness');
    });

    it('falls back to templated sensitivity when API returns error', async () => {
      global.fetch = vi.fn().mockResolvedValue(mockFetchResponse('', 500));

      const explanation = await explainSensitivityWithLLM(2.5, 0.1, 2.2, 0.5);

      // fallback path to getSensitivityInterpretation
      expect(explanation.summary).toContain('2.500');
      expect(explanation.summary).toContain('2.200');
      expect(explanation.interpretation).toContain('Strong robustness');
    });

    it('falls back with low E-value advice when API errors and eValue < 1.5', async () => {
      // Covers line 231 — LLM-null fallback with eValue < 1.5
      global.fetch = vi.fn().mockResolvedValue(mockFetchResponse('', 500));

      const explanation = await explainSensitivityWithLLM(1.2, 0.3, 1.1, 0.2);

      expect(explanation.actionableAdvice).toContain('Consider collecting additional data');
    });

    it('returns low E-value advice from LLM success path when eValue < 1.5', async () => {
      // Covers line 241 — LLM success path with eValue < 1.5
      const mockContent = 'Low robustness.\nRecommend collecting more confounder data.';
      global.fetch = vi.fn().mockResolvedValue(mockFetchResponse(mockContent));

      const explanation = await explainSensitivityWithLLM(1.3, 0.4, 1.2, 0.1);

      expect(explanation.actionableAdvice).toContain('Collect more confounder data');
      expect(explanation.actionableAdvice).toContain('E-value below threshold');
    });

    it('uses first line of LLM response as summary', async () => {
      const mockContent = 'First line summary.\nSecond line with more detail.\nThird line.';
      global.fetch = vi.fn().mockResolvedValue(mockFetchResponse(mockContent));

      const explanation = await explainSensitivityWithLLM(2.0, 0.1, 2.0, 0.5);

      expect(explanation.summary).toBe('First line summary.');
    });
  });

  // ── explainEstimateWithLLM ── with mocked API ─────────────────────

  describe('explainEstimateWithLLM — happy path', () => {
    it('returns LLM-generated explanation when API succeeds', async () => {
      const mockContent = 'The ATE of 0.5 indicates a moderate positive effect.\nStatistical significance: The effect is significant at the 95% level.\nConfidence: High.';
      global.fetch = vi.fn().mockResolvedValue(mockFetchResponse(mockContent));

      const explanation = await explainEstimateWithLLM('Backdoor', 0.5, 0.15, ['c1', 'c2']);

      expect(explanation.summary).toBeDefined();
      expect(explanation.interpretation).toBeDefined();
      expect(explanation.confidenceStatement).toBeDefined();
      expect(explanation.summary).toContain('0.5');
    });

    it('falls back to templated estimate when API returns error', async () => {
      global.fetch = vi.fn().mockResolvedValue(mockFetchResponse('', 503));

      const explanation = await explainEstimateWithLLM('Backdoor', 0.5, 0.15, ['c1']);

      // Should use fallback path
      expect(explanation.summary).toContain('0.500');
      expect(explanation.summary).toContain('significant');
      expect(explanation.confidenceStatement).toContain('Backdoor');
    });

    it('uses first line of LLM response as summary', async () => {
      const mockContent = 'Summary line from LLM.\nDetailed interpretation follows.\nConfidence statement.';
      global.fetch = vi.fn().mockResolvedValue(mockFetchResponse(mockContent));

      const explanation = await explainEstimateWithLLM('IV', 0.3, 0.1, []);

      expect(explanation.summary).toBe('Summary line from LLM.');
      expect(explanation.confidenceStatement).toContain('LLM');
    });
  });

  // ── Edge cases with mocked API ─────────────────────────────────────

  describe('API edge cases', () => {
    it('handles empty choices array in response', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ choices: [] }),
      } as Response);

      const result = makeRCAResult([{ name: 'X', score: 0.8 }]);
      const explanation = await explainRCAWithLLM(result, 'EmptyChoices');

      // Should fall back to template
      expect(explanation.summary).toBeDefined();
      expect(explanation.confidence).toBeDefined();
    });

    it('handles missing message field in choices', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ choices: [{ message: null }] }),
      } as Response);

      const result = makeRCAResult([{ name: 'Y', score: 0.7 }]);
      const explanation = await explainRCAWithLLM(result, 'NullMessage');

      // Should fall back to template
      expect(explanation.summary).toBeDefined();
    });

    it('recovers on first retry after initial failure', async () => {
      global.fetch = vi.fn()
        .mockRejectedValueOnce(new Error('Timeout'))
        .mockResolvedValue(mockFetchResponse('Recovered response.'));

      const result = makeRCAResult([{ name: 'Z', score: 0.85 }]);
      const explanation = await explainRCAWithLLM(result, 'RetrySuccess');

      expect(explanation.summary).toBeDefined();
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });

    // ── Edge cases for internal helper branches ──────────────────

    it('handles root cause with missing score (nullish coalesce)', async () => {
      // Covers rc.score?.toFixed(4) ?? 'N/A' in buildRCAPrompt (line 90)
      // and the same pattern in ranking (line 193/195)
      global.fetch = vi.fn().mockResolvedValue(
        mockFetchResponse('Root cause detected. High confidence in analysis.'),
      );

      const result = {
        rootCauses: [
          { name: 'MissingScore', score: undefined, confidence: 0.5, rank: 1, evidence: [] },
        ],
        paths: [],
        metadata: { method: 'test', analyzedAt: Date.now(), durationMs: 0, extra: {} },
        toJSON() { return { rootCauses: this.rootCauses, paths: this.paths, metadata: this.metadata }; },
      } as unknown as RCAResult;

      const explanation = await explainRCAWithLLM(result, 'NullScore');

      // Should not throw, and should use 'N/A' for the missing score
      expect(explanation.summary).toBeDefined();
      expect(explanation.ranking).toBeDefined();
      expect(explanation.ranking.length).toBe(1);
      // Ranking interpretation should use 'N/A' as score fallback
      expect(explanation.ranking[0]!.interpretation).toContain('N/A');
    });

    it('handles zero SE estimate (uses 1e-10 floor) in LLM-null fallback', async () => {
      // Covers Math.max(se, 1e-10) when se<<1e-10 → takes 1e-10 branch
      global.fetch = vi.fn()
        .mockRejectedValueOnce(new Error('Timeout'))
        .mockRejectedValueOnce(new Error('Timeout'))
        .mockRejectedValueOnce(new Error('Timeout'));

      const explanation = await explainEstimateWithLLM('Test', 1.0, 5e-11, ['conf']);

      expect(explanation.summary).toContain('significant');
      expect(explanation.confidenceStatement).toContain('Test');
      expect(global.fetch).toHaveBeenCalledTimes(3);
    });

    it('sensitivity retry succeeds after one failure', async () => {
      global.fetch = vi.fn()
        .mockRejectedValueOnce(new Error('Timeout'))
        .mockResolvedValue(mockFetchResponse('Strong result.\nInterpretation.'));
      const explanation = await explainSensitivityWithLLM(3.0, 0.05, 3.0, 0.8);
      expect(explanation.summary).toBe('Strong result.');
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });

    it('estimate retry succeeds after one failure', async () => {
      global.fetch = vi.fn()
        .mockRejectedValueOnce(new Error('Timeout'))
        .mockResolvedValue(mockFetchResponse('Estimate summary.\nDetails.'));
      const explanation = await explainEstimateWithLLM('Method', 0.5, 0.1, ['X']);
      expect(explanation.summary).toBe('Estimate summary.');
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });
  });
});
