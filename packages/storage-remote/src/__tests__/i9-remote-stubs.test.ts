import { describe, it, expect, beforeEach } from 'vitest';
import { RemoteRelationalStore } from '../remote-relational-store.js';
import { RemoteGraphStore } from '../remote-graph-store.js';

describe('RemoteRelationalStore with PGlite', () => {
  let store: RemoteRelationalStore;
  beforeEach(() => { store = new RemoteRelationalStore(); });

  it('CPT save + load', async () => {
    await store.saveCPT('g1', 'CPU', { node:'CPU', parents:[], entries:{'0':0.1,'1':0.75} });
    const cpt = await store.loadCPT('g1', 'CPU');
    expect(cpt?.entries['0']).toBe(0.1);
    expect(cpt?.entries['1']).toBe(0.75);
  });

  it('loadCPT returns null for unknown', async () => {
    expect(await store.loadCPT('x', 'y')).toBeNull();
  });

  it('SAVEPOINT lifecycle', async () => {
    await store.beginTransaction('s1');
    await store.saveCPT('g2','A',{node:'A',parents:[],entries:{'0':0.5}});
    await store.commitTransaction('s1');
    expect(await store.loadCPT('g2','A')).not.toBeNull();

    await store.beginTransaction('s2');
    await store.setCheckpoint('s2','cp');
    await store.saveCPT('g3','B',{node:'B',parents:[],entries:{'1':0.9}});
    await store.rollbackToCheckpoint('s2','cp');
    expect(await store.loadCPT('g3','B')).toBeNull();
  });

  it('regression model round-trip', async () => {
    await store.saveRegressionModel('g1','B',{coefficients:[1.5,2.0],intercept:0.5,residualStdDev:0.1});
    const m = await store.loadRegressionModel('g1','B');
    expect(m?.coefficients).toEqual([1.5,2.0]);
  });

  it('RCA save + query', async () => {
    await store.saveRCAResult('c1',{rootCauses:[],paths:[],metadata:{method:'x',analyzedAt:1,durationMs:1,extra:{}},toJSON:()=>({})});
    expect((await store.queryHistoricalResults({})).length).toBe(1);
  });
});

describe('RemoteGraphStore', () => {
  it('save + load graph', async () => {
    const store = new RemoteGraphStore();
    const id = await store.saveGraph({nodes:['A','B'],edges:[{source:'A',target:'B',weight:1,directed:true}]},{id:'g1',method:'pc',computedAt:1,parameters:{},confidence:0.9});
    expect((await store.loadGraph(id))?.nodes).toEqual(['A','B']);
  });

  it('versioned storage', async () => {
    const store = new RemoteGraphStore();
    const id = await store.saveGraph({nodes:['A'],edges:[]},{id:'g2',method:'pc',computedAt:1,parameters:{},confidence:0.9});
    await store.saveGraph({nodes:['A','B'],edges:[]},{id:'g2',method:'pc',computedAt:2,parameters:{},confidence:0.9});
    expect((await store.listGraphVersions(id)).length).toBe(2);
  });
});
