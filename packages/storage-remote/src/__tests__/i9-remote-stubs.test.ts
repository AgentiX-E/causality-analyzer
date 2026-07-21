import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { RemoteRelationalStore } from '../remote-relational-store.js';
import { RemoteGraphStore } from '../remote-graph-store.js';
import { startPgMemServer } from './pg-wire-server.js';

// Start a real PG-wire TCP server backed by pg-mem.
// RemoteRelationalStore connects via TCP → authentic PG-wire protocol.
let serverUrl: string;
let stopServer: () => void;

beforeAll(async () => {
  const srv = await startPgMemServer();
  serverUrl = srv.url;
  stopServer = srv.stop;
});

afterAll(() => {
  stopServer();
});

describe('RemoteRelationalStore via real PG-wire TCP', () => {
  it('connects via PG-wire and saves/loads CPT', async () => {
    const store = new RemoteRelationalStore({ connectionString: serverUrl });
    await store.saveCPT('g1', 'CPU', { node:'CPU', parents:[], entries:{'0':0.1,'1':0.75} });
    const cpt = await store.loadCPT('g1', 'CPU');
    expect(cpt?.entries['0']).toBe(0.1);
    await store.close();
  });

  it('SAVEPOINT commit + rollback', async () => {
    const store = new RemoteRelationalStore({ connectionString: serverUrl });
    await store.beginTransaction('s1');
    await store.saveCPT('g2', 'A', { node:'A', parents:[], entries:{'0':0.5} });
    await store.commitTransaction('s1');

    await store.beginTransaction('s2');
    await store.setCheckpoint('s2', 'cp');
    await store.saveCPT('g3', 'B', { node:'B', parents:[], entries:{'1':0.9} });
    await store.rollbackToCheckpoint('s2', 'cp');

    expect(await store.loadCPT('g2', 'A')).not.toBeNull();
    expect(await store.loadCPT('g3', 'B')).toBeNull();
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
