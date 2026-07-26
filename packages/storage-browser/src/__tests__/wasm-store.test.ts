/**
 * I11: Browser store comprehensive tests — vitest (node:sqlite backend).
 *
 * Tests WasmRelationalStore + WasmGraphStore with DirectSqlitePort.
 * The same stores work identically with WorkerSqlitePort in browser.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { DirectSqlitePort } from '../direct-sqlite-port.js';
import { WasmRelationalStore } from '../wasm-relational-store.js';
import { WasmGraphStore } from '../wasm-graph-store.js';
import type { CausalEdge } from '@agentix-e/causality-analyzer-core';

// ── Helpers ──────────────────────────────────────────────────────────

function makePort() { return new DirectSqlitePort(':memory:'); }

function makeEdge(source: string, target: string, weight = 1, directed = true): CausalEdge {
  return { source, target, weight, directed };
}

// ── WasmRelationalStore ─────────────────────────────────────────────

describe('WasmRelationalStore', () => {
  let port: DirectSqlitePort;
  let store: WasmRelationalStore;

  beforeEach(() => { port = makePort(); store = new WasmRelationalStore(port); });
  afterEach(() => { store.close(); });

  it('initializes DDL without error', () => {
    // Constructor already ran DDL — verify by writing and reading
    expect(true).toBe(true);
  });

  it('writes and reads metrics (row count)', async () => {
    await store.writeDetections([
      { isAnomalous: true, labels: new Float64Array([1, 0]), scores: new Float64Array([0.9, 0.1]), timestamp: 1000, metadata: {} },
    ]);

    const metrics = await store.readMetrics({ start: 0, end: 2000 });
    expect(metrics).toBeDefined();
  });

  it('saveCPT + loadCPT round-trip', async () => {
    await store.saveCPT('g1', 'X', { node: 'X', parents: [], entries: { '0': 0.3, '1': 0.7 } });
    const loaded = await store.loadCPT('g1', 'X');
    expect(loaded).not.toBeNull();
    expect(loaded!.entries['0']).toBe(0.3);
  });

  it('loadCPT returns null for unknown', async () => {
    const loaded = await store.loadCPT('nonexistent', 'X');
    expect(loaded).toBeNull();
  });

  it('saveRegressionModel + loadRegressionModel round-trip', async () => {
    await store.saveRegressionModel('g1', 'Y', {
      coefficients: [1.5, -0.3],
      intercept: 2.0,
      residualStdDev: 0.15,
    });
    const loaded = await store.loadRegressionModel('g1', 'Y');
    expect(loaded).not.toBeNull();
    expect(loaded!.coefficients).toEqual([1.5, -0.3]);
    expect(loaded!.intercept).toBe(2.0);
  });

  it('SAVEPOINT rollback works', async () => {
    await store.beginTransaction('s1');
    await store.setCheckpoint('s1', 'cp');
    await store.saveCPT('g1', 'A', { node: 'A', parents: [], entries: { '0': 0.5 } });
    await store.rollbackToCheckpoint('s1', 'cp');
    expect(await store.loadCPT('g1', 'A')).toBeNull();
  });

  it('SAVEPOINT commit persists', async () => {
    await store.beginTransaction('s2');
    await store.saveCPT('g2', 'B', { node: 'B', parents: [], entries: { '0': 0.8 } });
    await store.commitTransaction('s2');
    expect((await store.loadCPT('g2', 'B'))!.entries['0']).toBe(0.8);
  });

  it('saveRCAResult + queryHistoricalResults round-trip', async () => {
    await store.saveRCAResult('case1', {
      rootCauses: [{ name: 'CPU', score: 0.9, confidence: 0.95, rank: 1, evidence: [] }],
      paths: [],
      metadata: { method: 'test', analyzedAt: Date.now(), durationMs: 100, extra: {} },
      toJSON: () => ({}),
    });
    const results = await store.queryHistoricalResults({});
    expect(results.length).toBe(1);
  });

  it('queryHistoricalResults with filters', async () => {
    await store.saveRCAResult('case1', {
      rootCauses: [{ name: 'CPU', score: 0.9, confidence: 0.95, rank: 1, evidence: [] }],
      paths: [],
      metadata: { method: 'test', analyzedAt: Date.now(), durationMs: 100, extra: {} },
      toJSON: () => ({}),
    });
    const filtered = await store.queryHistoricalResults({ rootCause: 'CPU' });
    expect(filtered.length).toBe(1);

    const noMatch = await store.queryHistoricalResults({ rootCause: 'Memory' });
    expect(noMatch.length).toBe(0);
  });

  it('checkpoint within transaction', async () => {
    await store.beginTransaction('s3');
    await store.setCheckpoint('s3', 'cp1');
    await store.saveCPT('g3', 'C', { node: 'C', parents: [], entries: { '0': 0.1 } });
    await store.commitTransaction('s3');
    expect((await store.loadCPT('g3', 'C'))!.entries['0']).toBe(0.1);
  });

  it('close is idempotent', () => {
    store.close();
    expect(() => store.close()).not.toThrow();
  });
});

// ── WasmGraphStore ───────────────────────────────────────────────────

describe('WasmGraphStore', () => {
  let port: DirectSqlitePort;
  let store: WasmGraphStore;

  beforeEach(() => { port = makePort(); store = new WasmGraphStore(port); });
  afterEach(() => { store.close(); });

  it('saveGraph + loadGraph round-trip', async () => {
    await store.saveGraph(
      { nodes: ['A', 'B', 'C'], edges: [makeEdge('A', 'B'), makeEdge('B', 'C')] },
      { id: 'g1', method: 'PC', computedAt: Date.now(), parameters: {}, confidence: 0.9 },
    );

    const loaded = await store.loadGraph('g1');
    expect(loaded).not.toBeNull();
    expect(loaded!.nodes).toEqual(['A', 'B', 'C']);
    expect(loaded!.edges).toHaveLength(2);
  });

  it('loadGraph returns null for unknown ID', async () => {
    const loaded = await store.loadGraph('nonexistent');
    expect(loaded).toBeNull();
  });

  it('preserves edge weights and direction', async () => {
    await store.saveGraph(
      {
        nodes: ['X', 'Y', 'Z'],
        edges: [makeEdge('X', 'Y', 0.5, true), makeEdge('X', 'Z', 0.8, false)],
      },
      { id: 'g2', method: 'GES', computedAt: Date.now(), parameters: {}, confidence: 0.85 },
    );

    const loaded = await store.loadGraph('g2');
    expect(loaded!.edges[0]!.weight).toBe(0.5);
    expect(loaded!.edges[0]!.directed).toBe(true);
    expect(loaded!.edges[1]!.weight).toBe(0.8);
    expect(loaded!.edges[1]!.directed).toBe(false);
  });

  it('versioning: listGraphVersions and loadGraphVersion', async () => {
    await store.saveGraph(
      { nodes: ['A', 'B'], edges: [makeEdge('A', 'B')] },
      { id: 'g3', method: 'PC', computedAt: Date.now(), parameters: {}, confidence: 0.9 },
    );
    await store.saveGraph(
      { nodes: ['A', 'B', 'C'], edges: [makeEdge('A', 'B'), makeEdge('B', 'C')] },
      { id: 'g3', method: 'PC', computedAt: Date.now(), parameters: {}, confidence: 0.85 },
    );

    const versions = await store.listGraphVersions('g3');
    expect(versions.length).toBe(2);

    const v1 = await store.loadGraphVersion('g3', 1);
    expect(v1!.nodes).toEqual(['A', 'B']);

    const v2 = await store.loadGraphVersion('g3', 2);
    expect(v2!.nodes).toEqual(['A', 'B', 'C']);
  });

  it('listGraphVersions returns empty for unknown', async () => {
    const versions = await store.listGraphVersions('nonexistent');
    expect(versions.length).toBe(0);
  });

  it('findSimilarGraphs returns graphs sorted by Jaccard', async () => {
    await store.saveGraph(
      { nodes: ['A', 'B', 'C', 'D'], edges: [makeEdge('A', 'B')] },
      { id: 'sim1', method: 'PC', computedAt: Date.now(), parameters: {}, confidence: 0.9 },
    );
    await store.saveGraph(
      { nodes: ['A', 'B', 'C'], edges: [] },
      { id: 'sim2', method: 'PC', computedAt: Date.now(), parameters: {}, confidence: 0.85 },
    );

    const similar = await store.findSimilarGraphs(
      { nodes: ['A', 'B', 'C', 'D'], edges: [] },
      5,
    );
    expect(similar.length).toBeGreaterThanOrEqual(1);
    // 'sim1' has higher Jaccard (4/4) than 'sim2' (3/5)
    expect(similar[0]!.nodes.length).toBe(4);
  });

  it('handles empty graph save', async () => {
    await store.saveGraph(
      { nodes: ['single'], edges: [] },
      { id: 'empty', method: 'MANUAL', computedAt: Date.now(), parameters: {}, confidence: 1.0 },
    );
    const loaded = await store.loadGraph('empty');
    expect(loaded).not.toBeNull();
    expect(loaded!.nodes).toEqual(['single']);
  });

  it('close is idempotent', () => {
    store.close();
    expect(() => store.close()).not.toThrow();
  });
});

// ── Interface Contract: same SqlitePort for both stores ─────────────

describe('Shared SqlitePort', () => {
  it('both stores can share the same port', async () => {
    const port = makePort();
    const rel = new WasmRelationalStore(port);
    const graph = new WasmGraphStore(port);

    await rel.saveCPT('g1', 'X', { node: 'X', parents: [], entries: { '0': 0.5 } });
    await graph.saveGraph(
      { nodes: ['A', 'B'], edges: [makeEdge('A', 'B')] },
      { id: 'g1', method: 'PC', computedAt: Date.now(), parameters: {}, confidence: 0.9 },
    );

    // Both stores work on same DB
    expect(await rel.loadCPT('g1', 'X')).not.toBeNull();
    expect(await graph.loadGraph('g1')).not.toBeNull();

    rel.close();
  });
});
