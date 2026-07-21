import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { EmbedRelationalStore, SQL } from '../embed-relational-store.js';
import { EmbedGraphStore } from '../embed-graph-store.js';

// ── Helpers ──────────────────────────────────────────────────────────
function createStore() { return new EmbedRelationalStore(); }

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
});

// ── EmbedGraphStore ──────────────────────────────────────────────────
describe('EmbedGraphStore', () => {
  let store: EmbedGraphStore;
  beforeEach(() => { store = new EmbedGraphStore(); });

  it('save + load graph', async () => {
    const id = await store.saveGraph({ nodes: ['A', 'B'], edges: [{ source: 'A', target: 'B', weight: 1, directed: true }] }, { id: 'g1', method: 'pc', computedAt: 1, parameters: {}, confidence: 0.9 });
    const graph = await store.loadGraph(id);
    expect(graph?.nodes).toEqual(['A', 'B']);
  });

  it('versioned storage', async () => {
    const id = await store.saveGraph({ nodes: ['A'], edges: [] }, { id: 'g2', method: 'pc', computedAt: 1, parameters: {}, confidence: 0.9 });
    await store.saveGraph({ nodes: ['A', 'B'], edges: [] }, { id: 'g2', method: 'pc', computedAt: 2, parameters: {}, confidence: 0.9 });
    const versions = await store.listGraphVersions(id);
    expect(versions.length).toBe(2);
    expect((await store.loadGraphVersion(id, 1))?.nodes.length).toBe(1);
  });

  it('null for unknown graph', async () => {
    expect(await store.loadGraph('none')).toBeNull();
  });
});
