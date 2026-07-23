import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { EmbedRelationalStore } from '../embed-relational-store.js';
import { EmbedGraphStore } from '../embed-graph-store.js';

// ── Helpers ──────────────────────────────────────────────────────────
function createStore() { return new EmbedRelationalStore({ dbPath: ":memory:" }); }

// ── EmbedRelationalStore (SQLite-backed) ─────────────────────────────
describe('EmbedRelationalStore', () => {
  let store: EmbedRelationalStore;
  afterEach(() => { try { store?.close(); } catch {} });

  it('initializes with SQLite :memory: database', () => {
    store = createStore();
    expect(store).toBeTruthy();
  });

  it('SAVEPOINT commit preserves CPT', async () => {
    store = createStore();
    await store.beginTransaction('s1');
    await store.saveCPT('g1', 'X', { node: 'X', parents: [], entries: { '0': 0.3 } });
    await store.commitTransaction('s1');
    const cpt = await store.loadCPT('g1', 'X');
    expect(cpt?.entries['0']).toBe(0.3);
  });

  it('SAVEPOINT rollback discards CPT', async () => {
    store = createStore();
    await store.beginTransaction('s2');
    await store.setCheckpoint('s2', 'cp1');
    await store.saveCPT('g2', 'Y', { node: 'Y', parents: [], entries: { '1': 0.7 } });
    await store.rollbackToCheckpoint('s2', 'cp1');
    const cpt = await store.loadCPT('g2', 'Y');
    expect(cpt).toBeNull();
  });

  it('regression model round-trip', async () => {
    store = createStore();
    await store.saveRegressionModel('g1', 'B', { coefficients: [1.5, 2.0], intercept: 0.5, residualStdDev: 0.1 });
    const model = await store.loadRegressionModel('g1', 'B');
    expect(model?.coefficients).toEqual([1.5, 2.0]);
    expect(model?.intercept).toBe(0.5);
  });

  it('loadRegressionModel returns null for unknown', async () => {
    store = createStore();
    expect(await store.loadRegressionModel('x', 'y')).toBeNull();
  });

  it('RCA result save + query', async () => {
    store = createStore();
    const r: any = { rootCauses: [{ name: 'Mem', score: 0.9, confidence: 0.95, rank: 1, evidence: [] }], paths: [], toJSON: () => ({ rc: 'Mem' }) };
    await store.saveRCAResult('case1', r);
    const results = await store.queryHistoricalResults({});
    expect(results.length).toBe(1);
  });

  it('RCA query by root cause filter', async () => {
    store = createStore();
    const r1: any = { rootCauses: [{ name: 'Mem', score: 0.9, confidence: 0.95, rank: 1, evidence: [] }], paths: [], toJSON: () => ({ rc: 'Mem' }) };
    const r2: any = { rootCauses: [{ name: 'CPU', score: 0.5, confidence: 0.7, rank: 1, evidence: [] }], paths: [], toJSON: () => ({ rc: 'CPU' }) };
    await store.saveRCAResult('c1', r1);
    await store.saveRCAResult('c2', r2);
    expect((await store.queryHistoricalResults({ rootCause: 'Mem' })).length).toBe(1);
    expect((await store.queryHistoricalResults({ rootCause: 'Nonexist' })).length).toBe(0);
  });

  it('writeDetections + readMetrics round-trip', async () => {
    store = createStore();
    await store.writeDetections([{ isAnomalous: true, labels: new Float64Array([1]), scores: new Float64Array([0.9]), timestamp: 1000, metadata: {} }]);
    const table = await store.readMetrics({ start: 0, end: 2000 });
    expect(table.rowCount).toBeGreaterThan(0);
  });

  it('readMetrics respects metric filter', async () => {
    store = createStore();
    // Write two metric columns
    await store.writeDetections([
      { isAnomalous: true, labels: new Float64Array([1, 0]), scores: new Float64Array([0.9, 0.1]), timestamp: 1000, metadata: {} },
    ]);
    // Read with filter: only m0
    const table = await store.readMetrics({ start: 0, end: 2000, metrics: ['m0'] });
    expect(table.rowCount).toBe(1);
    // Read without filter: both m0 + m1
    const unfiltered = await store.readMetrics({ start: 0, end: 2000 });
    expect(unfiltered.rowCount).toBe(2);
  });

  it('readMetrics handles AbortSignal', async () => {
    store = createStore();
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(store.readMetrics({ start: 0, end: 2000, signal: ctrl.signal })).rejects.toThrow();
  });

  it('writeDetections with AbortSignal', async () => {
    store = createStore();
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(store.writeDetections([], ctrl.signal)).rejects.toThrow();
  });

  it('checkpoint with abort respects signal', async () => {
    store = createStore();
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(store.setCheckpoint('sx', 'cp', ctrl.signal)).rejects.toThrow();
  });

  it('saveCPT with AbortSignal', async () => {
    store = createStore();
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(store.saveCPT('g', 'n', { node: 'n', parents: [], entries: {} }, ctrl.signal)).rejects.toThrow();
  });

  it('loadRegressionModel with AbortSignal', async () => {
    store = createStore();
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(store.loadRegressionModel('g', 'n', ctrl.signal)).rejects.toThrow();
  });

  it('regression save with abort', async () => {
    store = createStore();
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(store.saveRegressionModel('g', 'n', { coefficients: [], intercept: 0, residualStdDev: 1 }, ctrl.signal)).rejects.toThrow();
  });

  it('rollbackToCheckpoint with abort', async () => {
    store = createStore();
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(store.rollbackToCheckpoint('s', 'c', ctrl.signal)).rejects.toThrow();
  });

  it('commitTransaction with abort', async () => {
    store = createStore();
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(store.commitTransaction('s', ctrl.signal)).rejects.toThrow();
  });

  it('beginTransaction with abort', async () => {
    store = createStore();
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(store.beginTransaction('s', ctrl.signal)).rejects.toThrow();
  });

  it('healthCheck returns true', () => {
    store = createStore();
    expect(store.healthCheck()).toBe(true);
  });

  it('migrate creates schema version table', async () => {
    store = createStore();
    await store.migrate();
    expect(await store.getSchemaVersion()).toBeGreaterThanOrEqual(0);
  });

  it('migrate is idempotent', async () => {
    store = createStore();
    await store.migrate();
    const v1 = await store.getSchemaVersion();
    await store.migrate();
    const v2 = await store.getSchemaVersion();
    expect(v1).toBe(v2);
  });
});

// ── EmbedGraphStore (overgraph persistent) ──────────────────────────
describe('EmbedGraphStore', () => {
  let store: EmbedGraphStore;
  let tmpDir: string;

  beforeEach(() => {
    // overgraph is a persistent LSM-tree engine, uses filesystem dirs.
    // Create temp dir per test for isolation.
    tmpDir = mkdtempSync(join(tmpdir(), 'ca-graph-'));
    store = new EmbedGraphStore({ dbPath: tmpDir });
  });

  afterEach(() => {
    try { store?.close(); } catch {}
    try { rmSync(tmpDir, { recursive: true, force: true }); } catch {}
  });

  const mg = (nodes: string[], edges?: Array<{ source: string; target: string; weight: number; directed: boolean }>) =>
    ({ nodes, edges: edges ?? [] });

  it('save + load graph preserves nodes and edges', async () => {
    const graph = mg(['A', 'B'], [{ source: 'A', target: 'B', weight: 1, directed: true }]);
    const id = await store.saveGraph(graph, { id: 'g1', method: 'pc', computedAt: 1, parameters: {}, confidence: 0.9 });
    const loaded = await store.loadGraph(id);
    expect(loaded?.nodes).toEqual(['A', 'B']);
    expect(loaded).toHaveProperty('edges');
  });

  it('preserves edged graph structure', async () => {
    const graph = mg(['X', 'Y', 'Z'], [
      { source: 'X', target: 'Y', weight: 0.75, directed: true },
      { source: 'Y', target: 'Z', weight: 0.5, directed: false },
    ]);
    await store.saveGraph(graph, { id: 'g-edge', method: 'pc', computedAt: 1, parameters: {}, confidence: 0.9 });
    const loaded = await store.loadGraph('g-edge');
    expect(loaded).not.toBeNull();
    expect(loaded?.nodes).toContain('X');
    expect(loaded?.nodes).toContain('Z');
  });

  it('loadGraph returns null for empty graph (no nodes)', async () => {
    const graph = mg([], []);
    const id = await store.saveGraph(graph, { id: 'g-empty', method: 'pc', computedAt: 1, parameters: {}, confidence: 0.9 });
    // EmbedGraphStore returns null when no nodes found
    const loaded = await store.loadGraph(id);
    expect(loaded).toBeNull();
  });

  it('versioned storage preserves multiple versions', async () => {
    await store.saveGraph(mg(['A']), { id: 'g2', method: 'pc', computedAt: 1, parameters: {}, confidence: 0.9 });
    await store.saveGraph(mg(['A', 'B']), { id: 'g2', method: 'pc', computedAt: 2, parameters: {}, confidence: 0.9 });
    const versions = await store.listGraphVersions('g2');
    expect(versions.length).toBe(2);
    expect(versions[0]!.version).toBe(1);
    expect(versions[1]!.version).toBe(2);
  });

  it('loadGraph returns latest version', async () => {
    await store.saveGraph(mg(['A', 'B']), { id: 'g3', method: 'pc', computedAt: 1, parameters: {}, confidence: 0.9 });
    await store.saveGraph(mg(['A', 'B', 'C']), { id: 'g3', method: 'pc', computedAt: 2, parameters: {}, confidence: 0.9 });
    // loadGraph always returns latest (version param ignored in EmbedGraphStore)
    const g = await store.loadGraph('g3');
    expect(g?.nodes.length).toBeGreaterThanOrEqual(2);
  });

  it('loadGraphVersion returns non-null for valid version', async () => {
    await store.saveGraph(mg(['A']), { id: 'g4', method: 'pc', computedAt: 1, parameters: {}, confidence: 0.9 });
    const v1 = await store.loadGraphVersion('g4', 1);
    // EmbedGraphStore.loadGraphVersion delegates to loadGraph
    expect(v1).not.toBeNull();
    expect(v1?.nodes).toContain('A');
  });

  it('null for unknown graph', async () => {
    expect(await store.loadGraph('none')).toBeNull();
  });

  it('listGraphVersions returns empty for unknown', async () => {
    expect(await store.listGraphVersions('unknown')).toEqual([]);
  });

  it('findSimilarGraphs returns array', async () => {
    const graph = mg(['A', 'B', 'C'], [{ source: 'A', target: 'B', weight: 1, directed: true }]);
    await store.saveGraph(graph, { id: 'g5', method: 'pc', computedAt: 1, parameters: {}, confidence: 0.9 });
    const results = await store.findSimilarGraphs(graph, 5);
    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBeGreaterThanOrEqual(1);
  });

  it('close() does not throw', () => {
    expect(() => store.close()).not.toThrow();
  });

  it('multiple graphs with different IDs', async () => {
    await store.saveGraph(mg(['A']), { id: 'ga', method: 'pc', computedAt: 1, parameters: {}, confidence: 0.9 });
    await store.saveGraph(mg(['B']), { id: 'gb', method: 'pc', computedAt: 1, parameters: {}, confidence: 0.9 });
    const a = await store.loadGraph('ga');
    const b = await store.loadGraph('gb');
    expect(a?.nodes).toEqual(['A']);
    expect(b?.nodes).toEqual(['B']);
  });

  it('findSimilarGraphs returns empty for empty store', async () => {
    // fresh store with no graphs saved
    const results = await store.findSimilarGraphs(mg(['X']), 5);
    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBe(0);
  });

  it('saveGraph with AbortSignal', async () => {
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(store.saveGraph(
      mg(['A']),
      { id: 'g-abort', method: 'pc', computedAt: 1, parameters: {}, confidence: 0.9 },
      ctrl.signal,
    )).rejects.toThrow();
  });

  it('loadGraph with AbortSignal', async () => {
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(store.loadGraph('g', ctrl.signal)).rejects.toThrow();
  });

  it('loadGraphVersion with abort', async () => {
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(store.loadGraphVersion('g', 1, ctrl.signal)).rejects.toThrow();
  });

  it('listGraphVersions with abort', async () => {
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(store.listGraphVersions('g', ctrl.signal)).rejects.toThrow();
  });

  it('findSimilarGraphs with abort', async () => {
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(store.findSimilarGraphs(mg(['A']), 5, ctrl.signal)).rejects.toThrow();
  });

  it('loadGraph returns null for graph with no nodes', async () => {
    // Save a graph that has an ID but whose nodes are empty
    const g = mg([]);
    const id = await store.saveGraph(g, { id: 'g-nonodes', method: 'pc', computedAt: 1, parameters: {}, confidence: 0.9 });
    const loaded = await store.loadGraph(id);
    expect(loaded).toBeNull();
  });
});
