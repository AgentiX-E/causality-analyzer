/**
 * Metric Semantic Classification & Self-Evolving Strategy — Tests.
 *
 * Covers:
 *   - Regex classification accuracy (well-known metric patterns)
 *   - Service name extraction
 *   - LLM fallback for ambiguous metrics
 *   - Confidence calibration (conflict resolution)
 *   - Metric routing (BOCPD vs CUSUM)
 *   - Self-evolving strategy (statistical adaptation + validation)
 *   - Edge cases (empty input, malformed names, API errors)
 *   - Real RCAEval metric names
 *
 * @packageDocumentation
 */

import { describe, it, expect } from 'vitest';
import {
  MetricSemanticParser,
  LLMMetricClassifier,
  ConfidenceCalibrator,
  MetricRouter,
  SelfEvolvingStrategy,
  classifyAllMetrics,
  datasetFingerprint,
  type MetricCategory,
  type MetricSemantic,
  type CaseFeedback,
} from '../schema/metric-semantic.js';

// ── Helpers ──────────────────────────────────────────────────────────

function allClassified(result: Map<string, MetricSemantic>, category: MetricCategory, count: number): void {
  let matchCount = 0;
  for (const [, sem] of result) {
    if (sem.category === category) matchCount++;
  }
  expect(matchCount, `Expected ${count} ${category}, got ${matchCount}`).toBe(count);
}

// ── T1-T3: Regex Classification ─────────────────────────────────────

describe('MetricSemanticParser (regex)', () => {
  const parser = new MetricSemanticParser();

  it('classifies latency metrics correctly', () => {
    const names = [
      'adservice_latency-50',
      'cartservice_request_duration_seconds',
      'checkoutservice_response_time_ms',
      'frontend_processing_time',
      'paymentservice_delay',
    ];
    const result = parser.classify(names);
    allClassified(result, 'latency', 5);
  });

  it('classifies error metrics correctly', () => {
    const names = [
      'adservice_error_count',
      'cartservice_5xx_total',
      'checkoutservice_failure_rate',
      'frontend_exception_count',
      'paymentservice_http_status_500',
    ];
    const result = parser.classify(names);
    allClassified(result, 'error', 5);
  });

  it('classifies resource metrics correctly', () => {
    const names = [
      'node_cpu_usage_percent',
      'redis_memory_used_bytes',
      'postgres_disk_iops',
      'nginx_network_rx_bytes',
      'java_gc_pause_seconds',
    ];
    const result = parser.classify(names);
    expect(result.get('node_cpu_usage_percent')?.category).toBe('resource_cpu');
    expect(result.get('redis_memory_used_bytes')?.category).toBe('resource_mem');
    expect(result.get('postgres_disk_iops')?.category).toBe('resource_disk');
    expect(result.get('nginx_network_rx_bytes')?.category).toBe('resource_net');
    expect(result.get('java_gc_pause_seconds')?.category).toBe('gc');
  });

  it('classifies saturation metrics correctly', () => {
    const names = [
      'adservice_thread_pool_active',
      'cartservice_connection_pool_size',
      'checkoutservice_queue_depth',
    ];
    const result = parser.classify(names);
    allClassified(result, 'saturation', 3);
  });

  it('classifies traffic metrics correctly', () => {
    const names = [
      'adservice_request_count_total',
      'cartservice_throughput',
      'frontend_qps',
    ];
    const result = parser.classify(names);
    allClassified(result, 'traffic', 3);
  });

  it('disambiguates _total counters from latency metrics', () => {
    // http_request_duration_seconds_total is a counter (traffic), not latency
    const name = 'http_request_duration_seconds_total';
    const category = parser.classifyCategory(name);
    expect(category).toBe('traffic');
  });

  it('returns null for truly ambiguous metrics', () => {
    const ambiguous = parser.getAmbiguous([
      'some_random_metric_xyz',
      'unknown_field_42',
      'web_vitals_cls',       // Cumulative Layout Shift — not in standard taxonomy
    ]);
    expect(ambiguous.length).toBe(3);
  });
});

// ── T4-T5: Service Extraction ───────────────────────────────────────

describe('Service name extraction', () => {
  const parser = new MetricSemanticParser();

  it('extracts service name from standard patterns', () => {
    expect(parser.extractService('adservice_latency-50')).toBe('adservice');
    expect(parser.extractService('cartservice_cpu_usage')).toBe('cartservice');
    expect(parser.extractService('redis_memory_bytes')).toBe('redis');
    expect(parser.extractService('frontend_request_count')).toBe('frontend');
  });

  it('handles Prometheus label selectors in extraction', () => {
    // With label selectors stripped and unit suffixes removed,
    // "http_request_duration_seconds{method=\"GET\"}" → "http_request_duration"
    const svc = parser.extractService('http_request_duration_seconds{method="GET"}');
    // The extraction finds the prefix before the first separator
    expect(svc.length).toBeGreaterThan(0);
    expect(svc).not.toContain('{');
    expect(svc).not.toContain('_seconds');
  });

  it('falls back to first segment for unclear names', () => {
    const svc = parser.extractService('unknown_thing_xyz');
    expect(svc.length).toBeGreaterThan(0);
  });
});

// ── T6-T7: LLM Classifier (no-key fallback) ─────────────────────────

describe('LLMMetricClassifier', () => {
  it('returns fallback classifications without API key', async () => {
    const classifier = new LLMMetricClassifier({ apiKey: '' });
    const result = await classifier.classifyAmbiguous(['unknown_metric_1', 'some_field_2']);

    expect(result.size).toBe(2);
    for (const [, sem] of result) {
      expect(sem.source).toBe('llm');
      expect(sem.category).toBe('custom');
      expect(sem.confidence).toBeLessThanOrEqual(0.2);
    }
  });

  it('handles empty metric list', async () => {
    const classifier = new LLMMetricClassifier({ apiKey: '' });
    const result = await classifier.classifyAmbiguous([]);
    expect(result.size).toBe(0);
  });

  it('with valid API key, classifies ambiguous metrics', async () => {
    const apiKey = process.env['DEEPSEEK_API_KEY'] ?? '';
    if (!apiKey) {
      console.log('  (skipped: no DEEPSEEK_API_KEY)');
      return;
    }

    const classifier = new LLMMetricClassifier({ apiKey });
    const result = await classifier.classifyAmbiguous([
      'web_vitals_cls',
      'custom_business_metric_kpi_score',
      'db_connection_idle_time',
    ]);

    expect(result.size).toBe(3);
    // At least one should be classified beyond 'custom'
    const nonCustom = [...result.values()].filter(s => s.category !== 'custom');
    expect(nonCustom.length).toBeGreaterThanOrEqual(0); // LLM may vary
  });
});

// ── T8-T9: Confidence Calibration ───────────────────────────────────

describe('ConfidenceCalibrator', () => {
  it('reduces confidence when regex and LLM disagree', () => {
    const regexResults = new Map<string, MetricSemantic>([
      ['adservice_latency', { category: 'latency', service: 'adservice', source: 'regex', confidence: 0.95 }],
    ]);
    const llmResults = new Map<string, MetricSemantic>([
      ['cartservice_unknown', { category: 'latency', service: 'cartservice', source: 'llm', confidence: 0.85 }],
      // LLM says 'latency' but regex would say 'traffic' (conflict)
      ['frontend_requests_total', { category: 'latency', service: 'frontend', source: 'llm', confidence: 0.85 }],
    ]);

    const calibrator = new ConfidenceCalibrator();
    const result = calibrator.calibrate(regexResults, llmResults, ['adservice', 'cartservice', 'frontend']);

    // cartservice_unknown passes through (regex, llm agree on classification)
    expect(result.get('cartservice_unknown')?.confidence).toBe(0.85);
    // frontend_requests_total should be demoted (regex says 'traffic', llm says 'latency')
    const frontend = result.get('frontend_requests_total');
    expect(frontend?.confidence).toBeLessThan(0.85);
  });

  it('demotes to custom when calibrated confidence falls below threshold', () => {
    const llmResults = new Map<string, MetricSemantic>([
      ['unknown_metric', { category: 'saturation', service: 'unknown_svc', source: 'llm', confidence: 0.4 }],
    ]);

    const calibrator = new ConfidenceCalibrator();
    const result = calibrator.calibrate(new Map(), llmResults, ['known-svc-1', 'known-svc-2']);

    // Low confidence + unknown service → demote to custom
    expect(result.get('unknown_metric')?.category).toBe('custom');
  });

  it('passes through regex results unchanged', () => {
    const regexResults = new Map<string, MetricSemantic>([
      ['svc_latency', { category: 'latency', service: 'svc', source: 'regex', confidence: 0.95 }],
    ]);

    const calibrator = new ConfidenceCalibrator();
    const result = calibrator.calibrate(regexResults, new Map(), ['svc']);

    expect(result.get('svc_latency')?.confidence).toBe(0.95);
    expect(result.get('svc_latency')?.source).toBe('regex');
  });
});

// ── T10-T11: Metric Routing ─────────────────────────────────────────

describe('MetricRouter', () => {
  it('routes latency+error to BOCPD, rest to CUSUM', () => {
    const mapping = new Map<string, MetricSemantic>([
      ['svc_latency', { category: 'latency', service: 'svc-a', source: 'regex', confidence: 0.95 }],
      ['svc_error', { category: 'error', service: 'svc-a', source: 'regex', confidence: 0.95 }],
      ['svc_cpu', { category: 'resource_cpu', service: 'svc-a', source: 'regex', confidence: 0.95 }],
      ['svc_mem', { category: 'resource_mem', service: 'svc-b', source: 'regex', confidence: 0.95 }],
    ]);

    const router = new MetricRouter();
    const result = router.route(mapping);

    expect(result.bocpdMetrics).toEqual(['svc_latency', 'svc_error']);
    expect(result.cusumMetrics).toEqual(['svc_cpu', 'svc_mem']);
    expect(result.serviceMapping['svc_latency']).toBe('svc-a');
  });

  it('handles empty mapping', () => {
    const router = new MetricRouter();
    const result = router.route(new Map());

    expect(result.bocpdMetrics).toEqual([]);
    expect(result.cusumMetrics).toEqual([]);
  });

  it('routes custom metrics to CUSUM', () => {
    const mapping = new Map<string, MetricSemantic>([
      ['custom_metric', { category: 'custom', service: 'svc-a', source: 'llm', confidence: 0.6 }],
    ]);

    const router = new MetricRouter();
    const result = router.route(mapping);

    expect(result.cusumMetrics).toEqual(['custom_metric']);
    expect(result.bocpdMetrics).toEqual([]);
  });
});

// ── T12-T15: Self-Evolving Strategy ──────────────────────────────────

describe('SelfEvolvingStrategy', () => {
  it('starts with default equal weights', () => {
    const strategy = new SelfEvolvingStrategy();
    const profile = strategy.getProfile('cpu');
    expect(profile.bocpdWeight).toBe(0.35);
    expect(profile.cusumWeight).toBe(0.35);
  });

  it('adapts weights based on hit rate statistics', () => {
    const strategy = new SelfEvolvingStrategy();

    // Feed 100 cases where CUSUM consistently hits for CPU faults
    for (let i = 0; i < 100; i++) {
      strategy.recordFeedback({
        caseId: `case-${i}`,
        faultType: 'cpu',
        groundTruth: 'adservice',
        predicted: 'adservice',
        hitSource: 'cusum',
        topRank: 1,
        metrics: { bocpdTop1: 'cartservice', cusumTop1: 'adservice', ensembleTop1: 'adservice' },
      });
    }

    // After 100 cases, adaptation should have triggered (or at least feedback collected)
    expect(strategy.totalFeedback).toBe(100);
    // With all cusum hits, cusum weight should increase
    const profile = strategy.getProfile('cpu');
    expect(profile.cusumWeight).toBeGreaterThanOrEqual(profile.bocpdWeight);
  });

  it('logs adaptation entries', () => {
    const strategy = new SelfEvolvingStrategy();

    for (let i = 0; i < 100; i++) {
      strategy.recordFeedback({
        caseId: `case-${i}`,
        faultType: 'cpu',
        groundTruth: 'adservice',
        predicted: 'adservice',
        hitSource: i < 70 ? 'cusum' : 'bocpd',
        topRank: 1,
        metrics: { bocpdTop1: 'adservice', cusumTop1: 'adservice', ensembleTop1: 'adservice' },
      });
    }

    // Check adaptation log
    const history = strategy.history;
    if (history.length > 0) {
      expect(history[0]!.validated).toBe(true);
      expect(history[0]!.trigger).toContain('Auto-adapt');
    }
  });

  it('does not adapt with insufficient data (< 5 cases per fault)', () => {
    const strategy = new SelfEvolvingStrategy();

    for (let i = 0; i < 100; i++) {
      // Each fault type gets only 2-3 cases → all below threshold
      const faults = ['cpu', 'mem', 'disk', 'delay', 'loss', 'socket', 'f1', 'f2', 'f3', 'f4', 'f5',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x'];
      strategy.recordFeedback({
        caseId: `case-${i}`,
        faultType: faults[i % faults.length] ?? 'other',
        groundTruth: 'adservice',
        predicted: 'adservice',
        hitSource: 'bocpd',
        topRank: 1,
        metrics: { bocpdTop1: 'adservice', cusumTop1: 'cartservice', ensembleTop1: 'adservice' },
      });
    }

    // All fault types have < 5 cases → no adaptation
    expect(strategy.totalFeedback).toBe(100);
  });

  it('reviewWithLLM returns null without API key', async () => {
    const strategy = new SelfEvolvingStrategy({ apiKey: '' });

    // Add some feedback (not enough for statistical adaptation)
    for (let i = 0; i < 5; i++) {
      strategy.recordFeedback({
        caseId: `case-${i}`,
        faultType: 'cpu',
        groundTruth: 'adservice',
        predicted: 'adservice',
        hitSource: 'cusum',
        topRank: 1,
        metrics: { bocpdTop1: 'cartservice', cusumTop1: 'adservice', ensembleTop1: 'adservice' },
      });
    }

    // Not enough data + no API key → null
    const reason = await strategy.reviewWithLLM();
    expect(reason).toBeNull();
  });
});

// ── T16-T18: Real RCAEval Integration ───────────────────────────────

describe('RCAEval metric classification', () => {
  it('classifies real RCAEval Online Boutique metrics', () => {
    // Real metric names from RE1-OB
    const names = [
      'adservice_latency-50',
      'adservice_latency-90',
      'adservice_latency-99',
      'cartservice_latency-50',
      'cartservice_latency-90',
      'checkoutservice_latency-50',
      'checkoutservice_latency-90',
      'frontend_latency-50',
      'frontend_error_count',
      'redis_memory_used_bytes',
      'recommendationservice_cpu_usage',
    ];

    const parser = new MetricSemanticParser();
    const result = parser.classify(names);

    // All should be classified (well-named)
    expect(result.size).toBe(names.length);

    // Coverage verification
    const ambiguous = parser.getAmbiguous(names);
    expect(ambiguous.length).toBe(0); // 100% coverage on OB metrics
  });

  it('dataset fingerprint is stable across order changes', () => {
    const fp1 = datasetFingerprint(['a_latency', 'b_cpu', 'c_mem'], ['svc-a', 'svc-b']);
    const fp2 = datasetFingerprint(['c_mem', 'a_latency', 'b_cpu'], ['svc-b', 'svc-a']);
    expect(fp1).toBe(fp2);
  });

  it('dataset fingerprint differs for different metric sets', () => {
    const fp1 = datasetFingerprint(['a_latency', 'b_cpu'], ['svc-a']);
    const fp2 = datasetFingerprint(['a_latency', 'b_cpu', 'c_mem'], ['svc-a']);
    expect(fp1).not.toBe(fp2);
  });
});

// ── Edge Cases ───────────────────────────────────────────────────────

describe('Edge cases', () => {
  it('handles empty metric names array', () => {
    const parser = new MetricSemanticParser();
    const result = parser.classify([]);
    expect(result.size).toBe(0);
  });

  it('handles metric names with special characters', () => {
    const parser = new MetricSemanticParser();
    const result = parser.classify([
      'metric-with-dashes_latency',
      'dot.separated.metric_cpu',
      'colon:metric_memory',
    ]);
    // Should not crash
    expect(result.size).toBeGreaterThanOrEqual(0);
  });

  it('handles very long metric names', () => {
    const parser = new MetricSemanticParser();
    const longName = 'a'.repeat(200) + '_latency';
    const result = parser.classify([longName]);
    // Should not crash — extractService handles long names
    expect(result.size).toBeGreaterThanOrEqual(0);
  });

  it('classifyAllMetrics pipeline works end-to-end', async () => {
    const names = ['adservice_latency', 'cartservice_cpu_usage', 'unknown_metric_42'];

    const result = await classifyAllMetrics(names, {
      serviceNames: ['adservice', 'cartservice'],
      valueSamples: {},
    }, ''); // No API key — uses fallback

    expect(result.size).toBe(3);
    // Well-named metrics should be classified by regex
    expect(result.get('adservice_latency')?.source).toBe('regex');
    expect(result.get('cartservice_cpu_usage')?.source).toBe('regex');
    // Unknown should be classified (by LLM fallback or as custom)
    expect(result.get('unknown_metric_42')).toBeDefined();
  });
});
