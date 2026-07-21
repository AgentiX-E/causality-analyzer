import { describe, it, expect } from 'vitest';
import { newDb } from 'pg-mem';
import { RemoteRelationalStore } from '../remote-relational-store.js';
import { RemoteGraphStore } from '../remote-graph-store.js';

describe('RemoteRelationalStore via pg-mem adapter', () => {
  it('CPT save + load', async () => {
    const db = newDb();
    const { Client } = db.adapters.createPg();
    const store = new RemoteRelationalStore({ _Client: Client as any });
    await store.saveCPT('g1', 'CPU', { node:'CPU', parents:[], entries:{'0':0.1,'1':0.75} });
    expect((await store.loadCPT('g1', 'CPU'))?.entries['0']).toBe(0.1);
    await store.close();
  });
});

describe('RemoteGraphStore', () => {
  it('save + load graph', async () => {
    const s = new RemoteGraphStore();
    const id = await s.saveGraph({nodes:['A','B'],edges:[{source:'A',target:'B',weight:1,directed:true}]},{id:'g1',method:'pc',computedAt:1,parameters:{},confidence:0.9});
    expect((await s.loadGraph(id))?.nodes).toEqual(['A','B']);
  });
});
