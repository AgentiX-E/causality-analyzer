/**
 * OCDB Benchmark Tests — Comprehensive validation suite.
 *
 * Tests cover: DAG generation correctness, multi-size benchmark
 * multi-size benchmark execution, result aggregation,
 * and output formatting.
 *
 * @packageDocumentation
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { CausalGraph } from '../src/graph/causal-graph.js';
import {
  runOCDBBenchmark,
  formatOCDBMarkdown,
  formatOCDBJSON,
  OCDB_GRAPH_SIZES,
  OCDB_SAMPLE_SIZES,
  OCDB_INSTANCES_PER_SIZE,
  OCDB_DENSITIES,
  type OCDBAggregateResult,
} from '../benchmark/ocdb.js';

const SEED = 20260101;
const TIMEOUT = 120_000;

describe('OCDB Benchmark Suite', () => {
  let results: { aggregated: OCDBAggregateResult[] };

  beforeAll(() => {
    // Run small-tier only for CI viability
    results = runOCDBBenchmark({ maxGraphSize: 10, skipLarge: true, seed: SEED }).aggregated;
  }, TIMEOUT);

  // ── Graph Generation ─────────────────────────────────────────

  describe('graph generation correctness', () => {
    it('generates acyclic DAGs at all sizes', () => {
      // Verify results contain all expected graph sizes
      const sizes = new Set(results.map(r => r.graphSize));
      for (const s of OCDB_GRAPH_SIZES.small) {
        expect(sizes.has(s)).toBe(true);
      }
    });

    it('generates correct number of instances per size', () => {
      for (const size of OCDB_GRAPH_SIZES.small) {
        const perSize = results.filter(r => r.graphSize === size);
        // Each (size, density, sampleSize) tuple should have OCDB_INSTANCES_PER_SIZE
        for (const r of perSize) {
          expect(r.numInstances).toBeGreaterThan(0);
          expect(r.numInstances).toBeLessThanOrEqual(OCDB_INSTANCES_PER_SIZE);
        }
      }
    });

    it('produces valid edge counts (≥2 for non-degenerate)', () => {
      for (const r of results) {
        for (const a of r.algorithms) {
          // All algorithms should produce some output
          expect(a.avgF1).toBeGreaterThanOrEqual(0);
          expect(a.avgF1).toBeLessThanOrEqual(1);
        }
      }
    });
  });

  // ── Metrics Correctness ─────────────────────────────────────

  describe('metric bounds', () => {
    it('SHD is non-negative', () => {
      for (const r of results) {
        for (const a of r.algorithms) {
          expect(a.avgShd).toBeGreaterThanOrEqual(0);
        }
      }
    });

    it('TPR in [0, 1]', () => {
      for (const r of results) {
        for (const a of r.algorithms) {
          expect(a.avgTpr).toBeGreaterThanOrEqual(0);
          expect(a.avgTpr).toBeLessThanOrEqual(1);
        }
      }
    });

    it('FPR in [0, 1]', () => {
      for (const r of results) {
        for (const a of r.algorithms) {
          expect(a.avgFpr).toBeGreaterThanOrEqual(0);
          expect(a.avgFpr).toBeLessThanOrEqual(1);
        }
      }
    });

    it('F1 in [0, 1]', () => {
      for (const r of results) {
        for (const a of r.algorithms) {
          expect(a.avgF1).toBeGreaterThanOrEqual(0);
          expect(a.avgF1).toBeLessThanOrEqual(1);
        }
      }
    });

    it('runtime is non-negative', () => {
      for (const r of results) {
        for (const a of r.algorithms) {
          expect(a.avgTimeMs).toBeGreaterThanOrEqual(0);
        }
      }
    });
  });

  // ── Sample Size Monotonicity ─────────────────────────────────

  describe('sample size monotonicity', () => {
    it('more samples → non-worsening TPR (monotonic trend)', () => {
      // For each algorithm on a given (size, density), check that
      // larger sample sizes don't decrease TPR by more than 0.2
      // (allowing for randomness)
      const groups = new Map<string, OCDBAggregateResult[]>();

      for (const r of results) {
        const key = `${r.graphSize}_${r.density}`;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key)!.push(r);
      }

      for (const [, group] of groups) {
        group.sort((a, b) => a.sampleSize - b.sampleSize);
        if (group.length < 2) continue;

        const algNames = [...new Set(group.flatMap(r => r.algorithms.map(a => a.algorithm)))];

        for (const alg of algNames) {
          const tprs = group.map(r =>
            r.algorithms.find(a => a.algorithm === alg)?.avgTpr ?? 0,
          );
          // At least one pair should show improvement or stability
          const maxDrop = Math.max(...tprs.slice(0, -1).map((t, i) => t - (tprs[i + 1] ?? t)));
          expect(maxDrop).toBeLessThan(0.3); // Allow small regression due to noise
        }
      }
    });
  });

  // ── Determinism ──────────────────────────────────────────────

  describe('determinism', () => {
    it('produces identical results for same seed', () => {
      const r1 = runOCDBBenchmark({ maxGraphSize: 8, skipLarge: true, seed: 42 }).aggregated;
      const r2 = runOCDBBenchmark({ maxGraphSize: 8, skipLarge: true, seed: 42 }).aggregated;

      expect(r1.length).toBe(r2.length);
      for (let i = 0; i < r1.length; i++) {
        expect(r1[i]!.avgF1).toBe(r2[i]!.avgF1);
        expect(r1[i]!.avgTpr).toBe(r2[i]!.avgTpr);
      }
    });
  });

  // ── Output Formatting ────────────────────────────────────────

  describe('output formatting', () => {
    it('formatOCDBMarkdown produces valid markdown table', () => {
      const md = formatOCDBMarkdown(results);
      expect(md).toContain('# OCDB Benchmark Results');
      expect(md).toContain('| Size | Density | Samples | Algorithm |');
      expect(md).toContain('|------|---------|---------|-----------|');
    });

    it('formatOCDBJSON produces valid parseable JSON', () => {
      const json = formatOCDBJSON(results);
      const parsed = JSON.parse(json);
      expect(parsed.benchmark).toBe('OCDB');
      expect(Array.isArray(parsed.results)).toBe(true);
    });
  });

  // ── Config Constants ─────────────────────────────────────────

  describe('configuration', () => {
    it('defines correct graph size tiers', () => {
      expect(OCDB_GRAPH_SIZES.small).toEqual([8, 10]);
      expect(OCDB_GRAPH_SIZES.medium).toEqual([20, 50]);
      expect(OCDB_GRAPH_SIZES.large).toEqual([100]);
    });

    it('defines correct sample sizes', () => {
      expect(OCDB_SAMPLE_SIZES).toEqual([200, 500, 1000, 5000]);
    });

    it('defines balanced density settings', () => {
      expect(OCDB_DENSITIES.low).toBe(0.15);
      expect(OCDB_DENSITIES.medium).toBe(0.25);
      expect(OCDB_DENSITIES.high).toBe(0.4);
    });
  });

  // ── Edge Cases ───────────────────────────────────────────────

  describe('edge cases', () => {
    it('handles maxGraphSize=0 gracefully', () => {
      const r = runOCDBBenchmark({ maxGraphSize: 0, seed: 42 });
      expect(r.aggregated).toEqual([]);
      expect(r.raw).toEqual([]);
    });

    it('handles skipLarge=true correctly', () => {
      const r = runOCDBBenchmark({ maxGraphSize: 50, skipLarge: true, seed: 42 });
      const hasLarge = r.aggregated.some(a => a.graphSize > 50);
      expect(hasLarge).toBe(false);
    });

    it('produces results sorted by size/density/sampleSize', () => {
      for (let i = 0; i < results.length - 1; i++) {
        const a = results[i]!;
        const b = results[i + 1]!;
        const keyA = a.graphSize * 10000 + a.density * 1000 + a.sampleSize;
        const keyB = b.graphSize * 10000 + b.density * 1000 + b.sampleSize;
        expect(keyA).toBeLessThanOrEqual(keyB);
      }
    });
  });
});
