import { describe, it, expect } from 'vitest';
import { newDb } from 'pg-mem';
import { RemoteRelationalStore } from '../remote-relational-store.js';
import type { MtlsConfig } from '../types.js';

describe('RemoteRelationalStore via pg-mem adapter', () => {
  it('CPT save + load', async () => {
    const db = newDb();
    const { Client } = db.adapters.createPg();
    const store = new RemoteRelationalStore({ _Client: Client as any });
    await store.saveCPT('g1', 'CPU', { node:'CPU', parents:[], entries:{'0':0.1,'1':0.75} });
    expect((await store.loadCPT('g1', 'CPU'))?.entries['0']).toBe(0.1);
    await store.close();
  });

  it('creates with mtls config', async () => {
    const db = newDb();
    const { Client } = db.adapters.createPg();
    const mtls: MtlsConfig = {
      ca: '-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----',
      cert: '-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----',
      key: '-----BEGIN PRIVATE KEY-----\nMOCK\n-----END PRIVATE KEY-----',
      passphrase: 'test123',
    };
    const store = new RemoteRelationalStore({ _Client: Client as any, mtls });
    expect(store).toBeDefined();
    await store.close();
  });

  it('creates with ssl boolean', async () => {
    const db = newDb();
    const { Client } = db.adapters.createPg();
    const store = new RemoteRelationalStore({ _Client: Client as any, ssl: true });
    expect(store).toBeDefined();
    await store.close();
  });

  it('creates with ssl object overriding mtls', async () => {
    const db = newDb();
    const { Client } = db.adapters.createPg();
    const mtls: MtlsConfig = {
      cert: 'CERT',
      key: 'KEY',
    };
    const store = new RemoteRelationalStore({
      _Client: Client as any,
      mtls,
      ssl: { rejectUnauthorized: false, ciphers: 'AES256-GCM-SHA384' },
    });
    expect(store).toBeDefined();
    await store.close();
  });
});
