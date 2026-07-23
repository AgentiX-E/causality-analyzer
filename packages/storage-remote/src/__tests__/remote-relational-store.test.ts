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

  it('healthCheck returns boolean', async () => {
    const store = new RemoteRelationalStore({ client });
    const ok = await store.healthCheck();
    expect(typeof ok).toBe('boolean');
    await store.close();
  });

  it('gracefulShutdown closes cleanly', async () => {
    const store = new RemoteRelationalStore({ client });
    await store.gracefulShutdown(1000);
    expect(store).toBeDefined();
  });

  it('reads empty metrics successfully', async () => {
    const store = new RemoteRelationalStore({ client });
    const result = await store.readMetrics({ start: 0, end: 1 });
    expect(result).toBeDefined();
    await store.close();
  });

  it('loads null CPT for unknown graph', async () => {
    const store = new RemoteRelationalStore({ client });
    expect(await store.loadCPT('nonexistent', 'X')).toBeNull();
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

describe('writeDetections + readMetrics', () => {
  it('round-trips detection data', async () => {
    const { Client } = newDb().adapters.createPg();
    const client = new Client() as unknown as PgClientLike;
    const store = new RemoteRelationalStore({ client });

    const detections = [{
      isAnomalous: true,
      labels: new Float64Array([1, 0]),
      scores: new Float64Array([3.5, 1.2]),
      timestamp: 1000,
      metadata: {},
    } as any];
    await store.writeDetections(detections);

    const result = await store.readMetrics({ start: 999, end: 1001 });
    expect(result).toBeDefined();
    await store.close();
  });
});

describe('Regression model load (null path)', () => {
  it('loadRegressionModel returns null for unknown', async () => {
    const { Client } = newDb().adapters.createPg();
    const client = new Client() as unknown as PgClientLike;
    const store = new RemoteRelationalStore({ client });
    expect(await store.loadRegressionModel('unknown', 'X')).toBeNull();
    await store.close();
  });
});

describe('RCA query (empty)', () => {
  it('queryHistoricalResults with filters returns empty array', async () => {
    const { Client } = newDb().adapters.createPg();
    const client = new Client() as unknown as PgClientLike;
    const store = new RemoteRelationalStore({ client });

    const results = await store.queryHistoricalResults({
      start: 0,
      end: Date.now() + 1000,
      rootCause: 'Nonexistent',
      limit: 5,
    });
    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBe(0);
    await store.close();
  });
});

describe('healthCheck + gracefulShutdown', () => {
  it('healthCheck returns true for live connection', async () => {
    const { Client } = newDb().adapters.createPg();
    const client = new Client() as unknown as PgClientLike;
    const store = new RemoteRelationalStore({ client });
    expect(await store.healthCheck()).toBe(true);
    await store.close();
  });

  it('gracefulShutdown with timeout', async () => {
    const { Client } = newDb().adapters.createPg();
    const client = new Client() as unknown as PgClientLike;
    const store = new RemoteRelationalStore({ client });
    await store.gracefulShutdown(100);
    expect(store).toBeDefined();
  });
});
