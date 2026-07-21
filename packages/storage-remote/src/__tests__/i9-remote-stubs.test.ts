/**
 * Remote storage stubs — contract interface verification.
 */
import { describe, it, expect } from 'vitest';
import { RemoteRelationalStore } from '../remote-relational-store.js';
import { RemoteGraphStore } from '../remote-graph-store.js';

describe('RemoteRelationalStore', () => {
  it('implements IRelationalStore contract methods', async () => {
    const store = new RemoteRelationalStore();
    expect(await store.loadCPT('g', 'n')).toBeNull();
    expect(await store.loadRegressionModel('g', 'n')).toBeNull();
    expect((await store.queryHistoricalResults({})).length).toBe(0);
  });

  it('throws on readMetrics (stub)', async () => {
    const store = new RemoteRelationalStore();
    await expect(store.readMetrics({ start: 0, end: 10 })).rejects.toThrow('Not implemented');
  });
});

describe('RemoteGraphStore', () => {
  it('implements IGraphStore contract methods', async () => {
    const store = new RemoteGraphStore();
    expect(await store.loadGraph('x')).toBeNull();
    expect(await store.loadGraphVersion('x', 1)).toBeNull();
    expect((await store.listGraphVersions('x')).length).toBe(0);
    expect((await store.findSimilarGraphs({ nodes: [], edges: [] }, 5)).length).toBe(0);
  });

  it('throws on saveGraph (stub)', async () => {
    const store = new RemoteGraphStore();
    await expect(store.saveGraph({ nodes: [], edges: [] }, { id: 'x', method: 'pc', computedAt: 0, parameters: {}, confidence: 0 })).rejects.toThrow('Not implemented');
  });
});

describe('RemoteRelationalStore full contract', () => {
  let store: any;
  beforeEach(async () => {
    const { RemoteRelationalStore } = await import('../remote-relational-store.js');
    store = new RemoteRelationalStore();
  });

  it('writeDetections accepts empty array', async () => { await store.writeDetections([]); });
  it('saveCPT callable', async () => { await store.saveCPT('g', 'n', { node:'n', parents:[], entries:{} }); });
  it('saveRegressionModel callable', async () => { await store.saveRegressionModel('g', 'n', { coefficients:[], intercept:0, residualStdDev:0 }); });
  it('loadRegressionModel returns null', async () => { expect(await store.loadRegressionModel('g','n')).toBeNull(); });
  it('saveRCAResult callable', async () => { await store.saveRCAResult('c', { rootCauses:[], paths:[], metadata:{method:'x',analyzedAt:1,durationMs:1,extra:{}}, toJSON:()=>({}) }); });
  it('queryHistoricalResults returns empty', async () => { expect(await store.queryHistoricalResults({})).toEqual([]); });
  it('beginTransaction callable', async () => { await store.beginTransaction('s'); });
  it('commitTransaction callable', async () => { await store.commitTransaction('s'); });
  it('rollbackToCheckpoint callable', async () => { await store.rollbackToCheckpoint('s','c'); });
  it('setCheckpoint callable', async () => { await store.setCheckpoint('s','n'); });
});

describe('RemoteGraphStore full contract', () => {
  let store: any;
  beforeEach(async () => {
    const { RemoteGraphStore } = await import('../remote-graph-store.js');
    store = new RemoteGraphStore();
  });

  it('loadGraph returns null', async () => { expect(await store.loadGraph('x')).toBeNull(); });
  it('listGraphVersions returns empty', async () => { expect(await store.listGraphVersions('x')).toEqual([]); });
  it('findSimilarGraphs returns empty', async () => { expect(await store.findSimilarGraphs({nodes:[],edges:[]},5)).toEqual([]); });
});
