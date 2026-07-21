import { describe, it, expect } from 'vitest';
import { newDb } from 'pg-mem';
import { RemoteRelationalStore } from '../remote-relational-store.js';
import { RemoteGraphStore } from '../remote-graph-store.js';

// Test: inject pg-mem Client so RemoteRelationalStore connects to the in-memory PG server
describe('RemoteRelationalStore via pg-mem (PG-wire)', () => {
  it('CPT save + load through PG-wire client', async () => {
    const db = newDb();
    const { Client } = db.adapters.createPg();
    const store = new RemoteRelationalStore({ _Client: Client as any });
    await store.saveCPT('g1', 'CPU', { node:'CPU', parents:[], entries:{'0':0.1,'1':0.75} });
    const cpt = await store.loadCPT('g1', 'CPU');
    expect(cpt?.entries['0']).toBe(0.1);
    expect(cpt?.entries['1']).toBe(0.75);
    await store.close();
  });

  it('loadCPT returns null for unknown', async () => {
    const db = newDb();
    const { Client } = db.adapters.createPg();
    const store = new RemoteRelationalStore({ _Client: Client as any });
    expect(await store.loadCPT('x', 'y')).toBeNull();
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
