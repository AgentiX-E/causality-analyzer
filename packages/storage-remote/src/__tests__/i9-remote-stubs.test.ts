import { describe, it, expect, beforeEach } from 'vitest';
import { RemoteRelationalStore } from '../remote-relational-store.js';
import { RemoteGraphStore } from '../remote-graph-store.js';

describe('RemoteRelationalStore (PGlite)', () => {
  let store: RemoteRelationalStore;
  beforeEach(() => { store = new RemoteRelationalStore(); });

  it('CPT save + load', async () => {
    await store.saveCPT('g1', 'CPU', { node:'CPU', parents:[], entries:{'0':0.1,'1':0.75} });
    const cpt = await store.loadCPT('g1', 'CPU');
    expect(cpt?.entries['0']).toBe(0.1);
  });

  it('loadCPT returns null for unknown', async () => {
    expect(await store.loadCPT('x', 'y')).toBeNull();
  });

});

describe('RemoteGraphStore', () => {
  it('save + load graph', async () => {
    const s = new RemoteGraphStore();
    const id = await s.saveGraph({nodes:['A','B'],edges:[{source:'A',target:'B',weight:1,directed:true}]},{id:'g1',method:'pc',computedAt:1,parameters:{},confidence:0.9});
    expect((await s.loadGraph(id))?.nodes).toEqual(['A','B']);
  });

  it('versioned storage', async () => {
    const s = new RemoteGraphStore();
    const id = await s.saveGraph({nodes:['A'],edges:[]},{id:'g2',method:'pc',computedAt:1,parameters:{},confidence:0.9});
    await s.saveGraph({nodes:['A','B'],edges:[]},{id:'g2',method:'pc',computedAt:2,parameters:{},confidence:0.9});
    expect((await s.listGraphVersions(id)).length).toBe(2);
  });
});
