import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { newDb } from 'pg-mem';
import { RemoteRelationalStore, buildPgClientOpts } from '../remote-relational-store.js';
import type { MtlsConfig } from '../types.js';
import type { PgClientLike } from '../remote-relational-store.js';

// ── buildPgClientOpts (pure, no DB needed) ────────────────────────────

describe('buildPgClientOpts', () => {
  const cert = `-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----`;
  const key = `-----BEGIN PRIVATE KEY-----\nMOCK\n-----END PRIVATE KEY-----`;

  it('returns empty object with no config', () => {
    expect(buildPgClientOpts({})).toEqual({});
  });

  it('returns connectionString when provided', () => {
    const opts = buildPgClientOpts({ connectionString: 'postgresql://localhost/db' });
    expect(opts.connectionString).toBe('postgresql://localhost/db');
  });

  it('builds mTLS ssl object', () => {
    const opts = buildPgClientOpts({ mtls: { cert, key } });
    expect(opts.ssl).toEqual({ rejectUnauthorized: true, cert, key });
  });

  it('builds mTLS with CA', () => {
    const ca = '-----BEGIN CERTIFICATE-----\nCA\n-----END CERTIFICATE-----';
    const opts = buildPgClientOpts({ mtls: { cert, key, ca } });
    expect((opts.ssl as any).ca).toBe(ca);
  });

  it('builds mTLS with passphrase', () => {
    const opts = buildPgClientOpts({ mtls: { cert, key, passphrase: 'pw' } });
    expect((opts.ssl as any).passphrase).toBe('pw');
  });

  it('ssl override merges with mtls (ssl wins)', () => {
    const customCert = 'CUSTOM_CERT';
    const opts = buildPgClientOpts({
      mtls: { cert, key },
      ssl: { rejectUnauthorized: false, cert: customCert },
    });
    expect((opts.ssl as any).rejectUnauthorized).toBe(false);
    expect((opts.ssl as any).cert).toBe(customCert);
    expect((opts.ssl as any).key).toBe(key); // non-overlapping key preserved
  });

  it('ssl boolean passes through', () => {
    const opts = buildPgClientOpts({ ssl: true });
    expect(opts.ssl).toBe(true);
  });

  it('ssl object passes through when no mtls', () => {
    const opts = buildPgClientOpts({ ssl: { rejectUnauthorized: false, ciphers: 'X' } });
    expect(opts.ssl).toEqual({ rejectUnauthorized: false, ciphers: 'X' });
  });

  it('connectionString + ssl combined', () => {
    const opts = buildPgClientOpts({ connectionString: 'pg://h/db', ssl: true });
    expect(opts.connectionString).toBe('pg://h/db');
    expect(opts.ssl).toBe(true);
  });
});

// ── RemoteRelationalStore (client DI) ────────────────────────────────

describe('RemoteRelationalStore', () => {
  let client: PgClientLike;

  /** Create a pg-mem Client instance for injection via `client` config. */
  beforeEach(() => {
    const { Client } = newDb().adapters.createPg();
    client = new Client() as unknown as PgClientLike;
  });

  afterEach(async () => {
    try { await client.end(); } catch {}
  });

  it('CPT save + load', async () => {
    const store = new RemoteRelationalStore({ client });
    await store.saveCPT('g1', 'CPU', { node: 'CPU', parents: [], entries: { '0': 0.1, '1': 0.75 } });
    expect((await store.loadCPT('g1', 'CPU'))?.entries['0']).toBe(0.1);
    await store.close();
  });

  it('creates with mtls config (ignored when client is provided)', async () => {
    const mtls: MtlsConfig = {
      ca: '-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----',
      cert: '-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----',
      key: '-----BEGIN PRIVATE KEY-----\nMOCK\n-----END PRIVATE KEY-----',
      passphrase: 'test123',
    };
    const store = new RemoteRelationalStore({ client, mtls });
    expect(store).toBeDefined();
    await store.close();
  });

  it('creates with ssl boolean (ignored when client is provided)', async () => {
    const store = new RemoteRelationalStore({ client, ssl: true });
    expect(store).toBeDefined();
    await store.close();
  });

  it('creates with ssl object overriding mtls (all ignored when client is provided)', async () => {
    const mtls: MtlsConfig = { cert: 'CERT', key: 'KEY' };
    const store = new RemoteRelationalStore({
      client,
      mtls,
      ssl: { rejectUnauthorized: false, ciphers: 'AES256-GCM-SHA384' },
    });
    expect(store).toBeDefined();
    await store.close();
  });
});

describe('pool config', () => {
  it('creates with poolSize > 1 using pg-mem client', async () => {
    const { Client } = newDb().adapters.createPg();
    const client = new Client() as unknown as PgClientLike;
    const store = new RemoteRelationalStore({ client, poolSize: 4 });
    expect(store).toBeDefined();
    await store.close();
  });
});
