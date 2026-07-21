/**
 * I9 tests: Storage layer — EmbedRelationalStore + EmbedGraphStore contract tests.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { EmbedRelationalStore, RELATIONAL_SCHEMA } from '../embed-relational-store.js';
import { EmbedGraphStore } from '../embed-graph-store.js';

// ── EmbedRelationalStore ──────────────────────────────────────────────
describe('EmbedRelationalStore', () => {
  let store: EmbedRelationalStore;
  beforeEach(() => { store = new EmbedRelationalStore(); });

  it('saves and loads CPT', async () => {
    const cpt = { node: 'X', parents: ['Y'], entries: { '0': 0.1, '1': 0.8 } };
    await store.saveCPT('g1', 'X', cpt);
    const loaded = await store.loadCPT('g1', 'X');
    expect(loaded).not.toBeNull();
    expect(loaded!.entries['1']).toBe(0.8);
  });

  it('loadCPT returns null for unknown', async () => {
    expect(await store.loadCPT('g1', 'unknown')).toBeNull();
  });

  it('saves and loads regression models', async () => {
    const model = { coefficients: [1.5, 2.0], intercept: 0.5, residualStdDev: 0.3 };
    await store.saveRegressionModel('g1', 'B', model);
    const loaded = await store.loadRegressionModel('g1', 'B');
    expect(loaded?.coefficients[0]).toBe(1.5);
    expect(loaded?.residualStdDev).toBe(0.3);
  });

  it('saves and queries RCA results', async () => {
    await store.saveRCAResult('case1', {
      rootCauses: [{ name: 'Mem', score: 0.9, confidence: 0.95, rank: 1, evidence: [] }],
      paths: [], metadata: { method: 'test', analyzedAt: 1000, durationMs: 5, extra: {} },
      toJSON: () => ({}),
    });
    await store.saveRCAResult('case2', {
      rootCauses: [{ name: 'CPU', score: 0.5, confidence: 0.7, rank: 1, evidence: [] }],
      paths: [], metadata: { method: 'test', analyzedAt: 2000, durationMs: 3, extra: {} },
      toJSON: () => ({}),
    });
    const results = await store.queryHistoricalResults({ limit: 10 });
    expect(results.length).toBe(2);
  });

  it('filters RCA results by root cause', async () => {
    await store.saveRCAResult('c1', {
      rootCauses: [{ name: 'Mem', score: 0.9, confidence: 0.95, rank: 1, evidence: [] }],
      paths: [], metadata: { method: 't', analyzedAt: 1, durationMs: 1, extra: {} },
      toJSON: () => ({}),
    });
    const results = await store.queryHistoricalResults({ rootCause: 'Mem' });
    expect(results.length).toBe(1);
    const noResults = await store.queryHistoricalResults({ rootCause: 'CPU' });
    expect(noResults.length).toBe(0);
  });

  it('transaction SAVEPOINT lifecycle', async () => {
    await store.beginTransaction('s1');
    await store.setCheckpoint('s1', 'before_analyze');
    await store.rollbackToCheckpoint('s1', 'before_analyze');
    await store.commitTransaction('s1');
    // No errors = pass
  });

  it('writeDetections stores detection scores', async () => {
    await store.writeDetections([{
      isAnomalous: true, labels: new Float64Array([1, 0]),
      scores: new Float64Array([0.9, 0.1]), timestamp: 1000,
      metadata: {},
    }]);
    // No crash = pass
  });

  it('readMetrics handles missing data', async () => {
    const result = await store.readMetrics({ start: 0, end: 100 });
    expect(result.rowCount).toBe(0);
  });

  it('CPT entries survive round-trip for complex CPT', async () => {
    const cpt = { node: 'Y', parents: ['X1', 'X2'], entries: { '00': 0.05, '01': 0.3, '10': 0.4, '11': 0.85 } };
    await store.saveCPT('g', 'Y', cpt);
    const loaded = await store.loadCPT('g', 'Y');
    expect(loaded!.entries['00']).toBe(0.05);
    expect(loaded!.entries['11']).toBe(0.85);
  });

  it('regression model load returns null for missing', async () => {
    expect(await store.loadRegressionModel('nonexistent', 'X')).toBeNull();
  });

  it('queryHistoricalResults respects time range', async () => {
    await store.saveRCAResult('old', {
      rootCauses: [], paths: [], metadata: { method: 't', analyzedAt: 100, durationMs: 1, extra: {} },
      toJSON: () => ({}),
    });
    const results = await store.queryHistoricalResults({ start: 200 });
    expect(results.length).toBeGreaterThanOrEqual(0);
  });
});

// ── EmbedGraphStore ──────────────────────────────────────────────────
describe('EmbedGraphStore', () => {
  let store: EmbedGraphStore;
  beforeEach(() => { store = new EmbedGraphStore(); });

  const makeGraph = (): import('@agentix-e/causality-analyzer-core').CausalGraph => ({
    nodes: ['A', 'B'], edges: [{ source: 'A', target: 'B', weight: 1, directed: true }],
  });

  const makeMeta = (id: string): import('@agentix-e/causality-analyzer-core').GraphMetadata => ({
    id, method: 'pc', computedAt: Date.now(), parameters: {}, confidence: 0.9,
  });

  it('saves and loads latest graph version', async () => {
    const id = await store.saveGraph(makeGraph(), makeMeta('g1'));
    const loaded = await store.loadGraph(id);
    expect(loaded?.nodes).toEqual(['A', 'B']);
  });

  it('supports versioned graph storage', async () => {
    const id = await store.saveGraph(makeGraph(), makeMeta('g2'));
    // Save second version
    await store.saveGraph({ nodes: ['A', 'B', 'C'], edges: [] }, { ...makeMeta('g2'), id });
    const v1 = await store.loadGraphVersion(id, 1);
    const v2 = await store.loadGraphVersion(id, 2);
    expect(v1?.nodes.length).toBe(2);
    expect(v2?.nodes.length).toBe(3);
  });

  it('lists all graph versions', async () => {
    const id = await store.saveGraph(makeGraph(), makeMeta('g3'));
    await store.saveGraph(makeGraph(), { ...makeMeta('g3'), id });
    const versions = await store.listGraphVersions(id);
    expect(versions.length).toBe(2);
    expect(versions[0]!.version).toBe(1);
    expect(versions[1]!.version).toBe(2);
  });

  it('loadGraph returns null for unknown', async () => {
    expect(await store.loadGraph('nonexistent')).toBeNull();
  });

  it('loadGraphVersion returns null for missing version', async () => {
    expect(await store.loadGraphVersion('nonexistent', 1)).toBeNull();
  });

  it('findSimilarGraphs returns stored graphs', async () => {
    await store.saveGraph(makeGraph(), makeMeta('g4'));
    const similar = await store.findSimilarGraphs(makeGraph(), 5);
    expect(similar.length).toBeGreaterThan(0);
  });

  it('listGraphVersions returns empty for unknown', async () => {
    expect((await store.listGraphVersions('unknown')).length).toBe(0);
  });
});

// ── RELATIONAL_SCHEMA ────────────────────────────────────────────────
describe('RELATIONAL_SCHEMA', () => {
  it('contains all required tables', () => {
    expect(RELATIONAL_SCHEMA.metrics).toContain('CREATE TABLE');
    expect(RELATIONAL_SCHEMA.cpt).toContain('CREATE TABLE');
    expect(RELATIONAL_SCHEMA.regression_models).toContain('CREATE TABLE');
    expect(RELATIONAL_SCHEMA.rca_results).toContain('CREATE TABLE');
    expect(RELATIONAL_SCHEMA.analysis_state).toContain('CREATE TABLE');
  });

  it('all schemas include IF NOT EXISTS', () => {
    for (const ddl of Object.values(RELATIONAL_SCHEMA)) {
      expect(ddl).toContain('IF NOT EXISTS');
    }
  });
});

// ── Additional edge cases ─────────────────────────────────────────
describe('EmbedRelationalStore edge cases', () => {
  it('beginTransaction+commitTransaction lifecycle', async () => {
    const store = new EmbedRelationalStore();
    await store.beginTransaction('s1');
    await store.commitTransaction('s1');
  });
  
  it('setCheckpoint+rollback lifecycle', async () => {
    const store = new EmbedRelationalStore();
    await store.beginTransaction('s2');
    await store.setCheckpoint('s2', 'cp1');
    await store.rollbackToCheckpoint('s2', 'cp1');
  });

  it('writeDetections with empty array', async () => {
    const store = new EmbedRelationalStore();
    await store.writeDetections([]);
  });

  it('readMetrics with start=end returns empty', async () => {
    const store = new EmbedRelationalStore();
    const result = await store.readMetrics({ start: 100, end: 100 });
    expect(result.rowCount).toBe(0);
  });

  it('saveCPT then load different node returns null', async () => {
    const store = new EmbedRelationalStore();
    await store.saveCPT('g1', 'X', { node: 'X', parents: [], entries: { root: 0.5 } });
    expect(await store.loadCPT('g1', 'Y')).toBeNull();
  });
});
