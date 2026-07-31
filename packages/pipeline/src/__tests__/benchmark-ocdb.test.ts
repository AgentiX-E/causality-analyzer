/**
 * OCDB Benchmark Tests — Lightweight configuration and formatting validation.
 *
 * Full benchmark execution is done via the CLI runner (benchmark:ocdb).
 * These tests validate configuration constants, formatting functions,
 * and edge cases without running the heavy benchmark.
 *
 * @packageDocumentation
 */

import { describe, it, expect } from 'vitest';
import {
  OCDB_GRAPH_SIZES,
  OCDB_SAMPLE_SIZES,
  OCDB_INSTANCES_PER_SIZE,
  OCDB_DENSITIES,
  formatOCDBMarkdown,
  formatOCDBJSON,
  type OCDBAggregateResult,
} from '../../benchmark/ocdb.js';

// ── Configuration Constants ─────────────────────────────────────

describe('OCDB Configuration', () => {
  it('defines correct graph size tiers', () => {
    expect(OCDB_GRAPH_SIZES.small).toEqual([8, 10]);
    expect(OCDB_GRAPH_SIZES.medium).toEqual([20, 50]);
    expect(OCDB_GRAPH_SIZES.large).toEqual([100]);
  });

  it('defines correct sample sizes', () => {
    expect(OCDB_SAMPLE_SIZES).toEqual([200, 500, 1000, 5000]);
  });

  it('defines balanced density settings in (0, 1)', () => {
    expect(OCDB_DENSITIES.low).toBeGreaterThan(0);
    expect(OCDB_DENSITIES.low).toBeLessThan(1);
    expect(OCDB_DENSITIES.medium).toBeGreaterThan(OCDB_DENSITIES.low);
    expect(OCDB_DENSITIES.medium).toBeLessThan(1);
    expect(OCDB_DENSITIES.high).toBeGreaterThan(OCDB_DENSITIES.medium);
    expect(OCDB_DENSITIES.high).toBeLessThan(1);
  });

  it('has positive instance count', () => {
    expect(OCDB_INSTANCES_PER_SIZE).toBeGreaterThan(0);
  });

  it('has consistent size ordering (small < medium < large)', () => {
    const maxSmall = Math.max(...OCDB_GRAPH_SIZES.small);
    const minMedium = Math.min(...OCDB_GRAPH_SIZES.medium);
    const maxMedium = Math.max(...OCDB_GRAPH_SIZES.medium);
    const minLarge = Math.min(...OCDB_GRAPH_SIZES.large);
    expect(maxSmall).toBeLessThan(minMedium);
    expect(maxMedium).toBeLessThan(minLarge);
  });
});

// ── Output Formatting ──────────────────────────────────────────

describe('OCDB Formatting', () => {
  const mockResults: OCDBAggregateResult[] = [{
    graphSize: 8,
    density: 0.15,
    sampleSize: 500,
    numInstances: 3,
    algorithms: [
      { algorithm: 'PC', avgShd: 2.5, stdShd: 0.5, avgTpr: 0.85, avgFpr: 0.1, avgF1: 0.875, avgTimeMs: 150 },
      { algorithm: 'GES', avgShd: 1.8, stdShd: 0.3, avgTpr: 0.92, avgFpr: 0.05, avgF1: 0.935, avgTimeMs: 200 },
    ],
  }];

  it('formatOCDBMarkdown produces valid markdown table', () => {
    const md = formatOCDBMarkdown(mockResults);
    expect(md).toContain('# OCDB Benchmark Results');
    expect(md).toContain('| Size | Density | Samples | Algorithm |');
    expect(md).toContain('|------|---------|---------|-----------|');
    expect(md).toContain('PC');
    expect(md).toContain('GES');
    expect(md).toContain('2.5±0.5');
    expect(md).toContain('0.875');
  });

  it('formatOCDBJSON produces valid parseable JSON', () => {
    const json = formatOCDBJSON(mockResults);
    const parsed = JSON.parse(json);
    expect(parsed.benchmark).toBe('OCDB');
    expect(parsed.timestamp).toBeDefined();
    expect(Array.isArray(parsed.results)).toBe(true);
    expect(parsed.results[0].algorithms[0].algorithm).toBe('PC');
  });

  it('formatOCDBJSON round-trips through parse', () => {
    const json = formatOCDBJSON(mockResults);
    const parsed = JSON.parse(json);
    expect(parsed.results[0].graphSize).toBe(8);
    expect(parsed.results[0].algorithms[0].avgF1).toBe(0.875);
  });

  it('formatOCDBMarkdown handles empty results gracefully', () => {
    const md = formatOCDBMarkdown([]);
    expect(md).toContain('# OCDB Benchmark Results');
    expect(md).toContain('| Size |');
    // No data rows
    const lines = md.split('\n');
    const dataLines = lines.filter(l => l.startsWith('| ') && !l.includes('---') && !l.includes('Size'));
    expect(dataLines.length).toBe(0);
  });
});

// ── Edge Cases ─────────────────────────────────────────────────

describe('OCDB Edge Cases', () => {
  it('handles single algorithm result', () => {
    const result: OCDBAggregateResult = {
      graphSize: 10, density: 0.25, sampleSize: 1000,
      numInstances: 1,
      algorithms: [
        { algorithm: 'PC', avgShd: 3.0, stdShd: 0, avgTpr: 0.8, avgFpr: 0.2, avgF1: 0.8, avgTimeMs: 100 },
      ],
    };
    const md = formatOCDBMarkdown([result]);
    expect(md).toContain('PC');
    expect(md).toContain('3±0');
  });

  it('handles zero SHD result', () => {
    const result: OCDBAggregateResult = {
      graphSize: 8, density: 0.15, sampleSize: 5000,
      numInstances: 1,
      algorithms: [
        { algorithm: 'PC', avgShd: 0, stdShd: 0, avgTpr: 1.0, avgFpr: 0, avgF1: 1.0, avgTimeMs: 50 },
      ],
    };
    const json = JSON.parse(formatOCDBJSON([result]));
    expect(json.results[0].algorithms[0].avgShd).toBe(0);
    expect(json.results[0].algorithms[0].avgF1).toBe(1);
  });

  it('handles large runtime values', () => {
    const result: OCDBAggregateResult = {
      graphSize: 100, density: 0.4, sampleSize: 5000,
      numInstances: 5,
      algorithms: [
        { algorithm: 'BOSS', avgShd: 15.0, stdShd: 2.0, avgTpr: 0.7, avgFpr: 0.3, avgF1: 0.7, avgTimeMs: 300000 },
      ],
    };
    const md = formatOCDBMarkdown([result]);
    expect(md).toContain('300000');
    expect(md).toContain('0.700');
  });
});
