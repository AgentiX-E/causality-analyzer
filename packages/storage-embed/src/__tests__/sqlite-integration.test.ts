import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { EmbedRelationalStore } from '../embed-relational-store.js';
import fs from 'fs';
import os from 'os';
import path from 'path';

describe('EmbedRelationalStore persistent file', () => {
  let store: EmbedRelationalStore;
  let dbPath: string;

  beforeEach(() => {
    dbPath = path.join(os.tmpdir(), 'ca-file-' + Date.now() + '-' + Math.random().toString(36).slice(2) + '.db');
    store = new EmbedRelationalStore({ dbPath });
  });
  afterEach(() => { store?.close(); try { fs.unlinkSync(dbPath); } catch {} });

  it('creates file on disk', () => {
    expect(fs.statSync(dbPath).size).toBeGreaterThan(0);
  });

  it('SAVEPOINT rollback + release works with file persistence', async () => {
    await store.beginTransaction('s1');
    await store.setCheckpoint('s1', 'cp');
    await store.saveCPT('g1', 'X', { node: 'X', parents: [], entries: { '0': 0.3 } });
    await store.rollbackToCheckpoint('s1', 'cp');
    expect(await store.loadCPT('g1', 'X')).toBeNull();

    await store.beginTransaction('s2');
    await store.saveCPT('g2', 'Y', { node: 'Y', parents: [], entries: { '1': 0.7 } });
    await store.commitTransaction('s2');
    expect((await store.loadCPT('g2', 'Y'))?.entries['1']).toBe(0.7);
  });
});
