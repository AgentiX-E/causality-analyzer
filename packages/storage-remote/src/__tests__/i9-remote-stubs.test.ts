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
