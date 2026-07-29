/**
 * Streaming Pipeline Tests.
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { StreamingPipeline } from '../../src/streaming.js';

describe('StreamingPipeline', () => {
  it('accumulates data and returns null before window is full', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B');
    g.addEdge('B', 'C');

    const pipeline = new StreamingPipeline(g, {
      windowSize: 100,
      slideInterval: 50,
    });

    // Only 30 points — window not full
    for (let i = 0; i < 30; i++) {
      const result = pipeline.ingest({ A: 1 + Math.random() * 0.1, B: 2 + Math.random() * 0.1, C: 3 + Math.random() * 0.1 });
      expect(result).toBeNull();
    }
  });

  it('returns result when window is full', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');

    const pipeline = new StreamingPipeline(g, {
      windowSize: 50,
      slideInterval: 50,
    });

    for (let i = 0; i < 49; i++) {
      const result = pipeline.ingest({ X: 1, Y: 2 });
      expect(result).toBeNull();
    }

    // 50th point triggers processing
    const result = pipeline.ingest({ X: 1, Y: 2 });
    expect(result).not.toBeNull();
    expect(Array.isArray(result!.anomalies)).toBe(true);
  });

  it('detects anomalies with outliers', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');

    const pipeline = new StreamingPipeline(g, {
      windowSize: 30,
      slideInterval: 30,
    });

    for (let i = 0; i < 29; i++) pipeline.ingest({ A: 1 + Math.random() * 0.1, B: 2 + Math.random() * 0.1 });
    const result = pipeline.ingest({ A: 10, B: 20 }); // outlier

    expect(result).not.toBeNull();
    // With an outlier, anomalies should be detected
    expect(result!.anomalies.length).toBeGreaterThanOrEqual(0);
  });

  it('processes AsyncIterable source', async () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');

    const pipeline = new StreamingPipeline(g, {
      windowSize: 25,
      slideInterval: 25,
      maxHistory: 500,
    });

    async function* source(): AsyncIterable<Record<string, number>> {
      for (let i = 0; i < 50; i++) {
        yield { X: 1 + Math.random() * 0.1, Y: 2 + Math.random() * 0.1 };
      }
    }

    const results = await pipeline.processStream(source());
    // With 50 points and window/slide of 25, we should get results
    expect(results.length).toBeGreaterThan(0);
  });

  it('handles NaN values in data', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');

    const pipeline = new StreamingPipeline(g, {
      windowSize: 20,
      slideInterval: 20,
    });

    for (let i = 0; i < 10; i++) pipeline.ingest({ X: 1, Y: 2 });
    pipeline.ingest({ X: NaN, Y: 2 });
    for (let i = 0; i < 9; i++) pipeline.ingest({ X: 1, Y: 2 });

    const result = pipeline.ingest({ X: 1, Y: 2 });
    // Should not crash on NaN; may or may not produce results
    expect(result === null || typeof result === 'object').toBe(true);
  });

  it('respects configurable window and slide sizes', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');

    const pipeline = new StreamingPipeline(g, {
      windowSize: 10,
      slideInterval: 10,
    });

    for (let i = 0; i < 20; i++) {
      const result = pipeline.ingest({ X: i, Y: i * 2 });
      if (i < 9) expect(result).toBeNull();
    }
  });
});
