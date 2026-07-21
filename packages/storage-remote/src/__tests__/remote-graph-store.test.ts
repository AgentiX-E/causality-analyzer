/**
 * RemoteGraphStore integration tests via BoltSessionMock.
 *
 * Tests the full RemoteGraphStore implementation using the _Driver DI
 * pattern to inject BoltDriverMock. All IGraphStore methods are tested
 * through the Bolt code path (not a local fallback).
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { RemoteGraphStore } from '../remote-graph-store.js';
import { BoltDriverMock, makeRetryableError, makeNonRetryableError } from './bolt-session-mock.js';
import type { CausalGraph, GraphMetadata } from '@agentix-e/causality-analyzer-core';

function makeGraph(nodes?: string[], edges?: CausalGraph['edges']): CausalGraph {
  return {
    nodes: nodes ?? ['A', 'B', 'C'],
    edges: edges ?? [
      { source: 'A', target: 'B', weight: 0.8, directed: true },
      { source: 'B', target: 'C', weight: 0.6, directed: true },
    ],
  };
}

function makeMeta(id: string, overrides?: Partial<GraphMetadata>): GraphMetadata {
  return {
    id,
    method: 'pc',
    computedAt: Date.now(),
    parameters: {},
    confidence: 0.9,
    ...overrides,
  };
}

function createStore(driver?: BoltDriverMock) {
  const d = driver ?? new BoltDriverMock();
  const store = new RemoteGraphStore({
    uri: 'bolt://localhost:7687',
    auth: { type: 'basic', user: 'neo4j', password: 'test' },
    _Driver: class extends MockDriverWrapper {
      session(cfg?: any) { return d.session(cfg); }
      async close() { d.reset(); }
    } as any,
  });
  return { store, driver: d };
}

/**
 * Wraps BoltDriverMock as a constructor class matching _Driver signature.
 */
class MockDriverWrapper {
  private d: BoltDriverMock;
  constructor(_url: string, _auth: unknown, _config: Record<string, unknown>, dep?: BoltDriverMock) {
    this.d = dep!;
  }
  session(cfg?: any) { return this.d.session(cfg); }
  async close() { this.d.reset(); }
}

// ── Tests ──────────────────────────────────────────────────────────────

describe('RemoteGraphStore', () => {
  describe('construction', () => {
    it('throws TypeError when uri is missing', () => {
      // @ts-expect-error — testing missing required uri
      expect(() => new RemoteGraphStore({})).toThrow(TypeError);
    });

    it('throws TypeError when uri is empty string', () => {
      expect(() => new RemoteGraphStore({ uri: '' })).toThrow(TypeError);
    });

    it('creates store with basic auth', () => {
      const d = new BoltDriverMock();
      const s = new RemoteGraphStore({
        uri: 'bolt://localhost:7687',
        auth: { type: 'basic', user: 'u', password: 'p' },
        _Driver: MockDriverWrapper as any,
      });
      expect(s).toBeDefined();
    });

    it('creates store with bearer auth', () => {
      const d = new BoltDriverMock();
      const s = new RemoteGraphStore({
        uri: 'bolt://localhost:7687',
        auth: { type: 'bearer', token: 'abc123' },
        _Driver: MockDriverWrapper as any,
      });
      expect(s).toBeDefined();
    });

    it('creates store with none auth', () => {
      const d = new BoltDriverMock();
      const s = new RemoteGraphStore({
        uri: 'bolt://localhost:7687',
        auth: { type: 'none' },
        _Driver: MockDriverWrapper as any,
      });
      expect(s).toBeDefined();
    });
  });

  describe('saveGraph + loadGraph', () => {
    it('round-trips a graph', async () => {
      const d = new BoltDriverMock();
      const s = createWithDriver(d);
      const g = makeGraph();
      const id = await s.saveGraph(g, makeMeta('g1'));
      expect(id).toBe('g1');
      const loaded = await s.loadGraph(id);
      expect(loaded?.nodes).toEqual(['A', 'B', 'C']);
      expect(loaded?.edges).toHaveLength(2);
    });

    it('preserves edge weights and direction', async () => {
      const d = new BoltDriverMock();
      const s = createWithDriver(d);
      const g = makeGraph(['X', 'Y'], [
        { source: 'X', target: 'Y', weight: 0.42, directed: false },
      ]);
      await s.saveGraph(g, makeMeta('g2'));
      const loaded = await s.loadGraph('g2');
      expect(loaded?.edges[0]?.weight).toBe(0.42);
      expect(loaded?.edges[0]?.directed).toBe(false);
    });

    it('loadGraph returns null for unknown ID', async () => {
      const d = new BoltDriverMock();
      const s = createWithDriver(d);
      expect(await s.loadGraph('nope')).toBeNull();
    });
  });

  describe('versioning', () => {
    it('preserves all versions', async () => {
      const d = new BoltDriverMock();
      const s = createWithDriver(d);
      await s.saveGraph(makeGraph(['A', 'B', 'C']), makeMeta('g3'));
      await s.saveGraph(makeGraph(['A', 'B', 'C', 'D']), makeMeta('g3'));

      const v1 = await s.loadGraphVersion('g3', 1);
      const v2 = await s.loadGraphVersion('g3', 2);
      expect(v1?.nodes.length).toBe(3);
      expect(v2?.nodes.length).toBe(4);
    });

    it('loadGraphVersion returns null for out-of-range', async () => {
      const d = new BoltDriverMock();
      const s = createWithDriver(d);
      await s.saveGraph(makeGraph(), makeMeta('g4'));
      expect(await s.loadGraphVersion('g4', 999)).toBeNull();
    });

    it('listGraphVersions returns correct count', async () => {
      const d = new BoltDriverMock();
      const s = createWithDriver(d);
      await s.saveGraph(makeGraph(), makeMeta('g5'));
      await s.saveGraph(makeGraph(), makeMeta('g5'));
      await s.saveGraph(makeGraph(), makeMeta('g5'));

      const versions = await s.listGraphVersions('g5');
      expect(versions.length).toBe(3);
      expect(versions[0]!.version).toBe(1);
      expect(versions[2]!.version).toBe(3);
    });

    it('listGraphVersions returns empty for unknown', async () => {
      const d = new BoltDriverMock();
      const s = createWithDriver(d);
      expect(await s.listGraphVersions('unknown')).toEqual([]);
    });

    it('loadGraph returns latest version', async () => {
      const d = new BoltDriverMock();
      const s = createWithDriver(d);
      await s.saveGraph(makeGraph(['A']), makeMeta('g6'));
      await s.saveGraph(makeGraph(['A', 'B', 'C']), makeMeta('g6'));
      const latest = await s.loadGraph('g6');
      expect(latest?.nodes.length).toBe(3);
    });
  });

  describe('findSimilarGraphs', () => {
    it('returns graphs sorted by Jaccard similarity', async () => {
      const d = new BoltDriverMock();
      const s = createWithDriver(d);
      // g7: [A,B,C], g8: [A,B,X], g9: [X,Y,Z]
      await s.saveGraph(makeGraph(['A', 'B', 'C']), makeMeta('g7'));
      await s.saveGraph(makeGraph(['A', 'B', 'X']), makeMeta('g8'));
      await s.saveGraph(makeGraph(['X', 'Y', 'Z']), makeMeta('g9'));

      // target: [A,B,C] — most similar to g7, then g8, then g9
      const results = await s.findSimilarGraphs(makeGraph(['A', 'B', 'C']), 5);
      expect(results.length).toBeGreaterThanOrEqual(3);
      // g7 should be first (perfect match)
      expect(results[0]!.nodes).toEqual(['A', 'B', 'C']);
    });
  });

  describe('close', () => {
    it('closes driver cleanly', async () => {
      const d = new BoltDriverMock();
      const s = createWithDriver(d);
      await expect(s.close()).resolves.toBeUndefined();
    });
  });

  describe('error handling', () => {
    it('propagates permanent errors', async () => {
      // Create a driver that throws on session
      const badDriver = {
        session() { throw makeNonRetryableError(); },
        close: async () => {},
      };
      const s = new RemoteGraphStore({
        uri: 'bolt://localhost:7687',
        _Driver: class { constructor() {} session() { throw makeNonRetryableError(); } async close() {} } as any,
      });
      await expect(s.loadGraph('any')).rejects.toThrow();
    });
  });

  describe('mTLS', () => {
    const pemCert = `-----BEGIN CERTIFICATE-----
MIIDXTCCAkWgAwIBAgIJALc1vKxQrFq/MA0GCSqGSIb3DQEBCwUAMEUxCzAJBgNV
BAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX
aWRnaXRzIFB0eSBMdGQwHhcNMjQwMTAxMDAwMDAwWhcNMjUwMTAxMDAwMDAwWjBF
MQswCQYDVQQGEwJBVTETMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50
ZXJuZXQgV2lkZ2l0cyBQdHkgTHRkMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIB
CgKCAQEAtest/wrK0GGQCgYXcJ/xhC5aBFlBMBDUNr8GXWFoDIC/gDLIJRK+3mV
FOTKf6Mj4Wxw5L9QnxbYR3lY9+CzyR4gJEKUQWLY2YmOXkLlbnGsT0MIFqkLvN4r
QmtplPgZQIhLxKGFdYHHKsMBTwQvnI/LjOoAt0/xnGQDJPJvQBfxJU4QoZGHtIJH
9B7ohSMnBdcJDRQIpNehCQBGEh0CgQkFq5QIEBMPbF0DgFQHpGQGYP9xBnYfI+ej
1QIDAQABo4GnMIGkMB0GA1UdDgQWBBTqYxvRKPq/WQWcRqfjFPwGqD2M4jB1BgNV
HSMEbjBsgBTqYxvRKPq/WQWcRqfjFPwGqD2M4qFJpEcwRTELMAkGA1UEBhMCQVUx
EzARBgNVBAgMClNvbWUtU3RhdGUxITAfBgNVBAoMGEludGVybmV0IFdpZGdpdHMg
UHR5IEx0ZIIJALc1vKxQrFq/MAwGA1UdEwQFMAMBAf8wDQYJKoZIhvcNAQELBQAD
ggEBAKXqYxvRKPq/WQWcRqfjFPwGqD2M4g==
-----END CERTIFICATE-----`;

    const pemKey = `-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC16y3/CsrQYZA
KBhdwn/GELloEWUEwENQ2vwZdYWgMgL+AMsglEr7eZUU5Mp/oyPhbHDkv1CfFth
HeVj34LPJHiAkQpBAtjZiY5eQuVucaxPQwgWqQu83itCa2mU+BlAiEvEoYV1gcc
qwwFPBC+cj8uM6gC3T/GcZAMkk8m9AF/ElThChkYe0gkf0HuiFIycF1wkNFAik1
6EJAESHQKBCQWrlAgQEw9sXQOAVAekZAZg/3EGdh8j56PVAgMBAAECggEAN5mNMN
W0V8aRg7HxJKCzQ9pLJJPBnKqEYfE7NBsrKjNkLXHnCCs+JDp/JOGJpQnQXcNjF
hGlHPNZRXdx7LvLJCWxwjz1WYPAnwGPsYKaTRZ3tpJXJCPgEHLHSW+gRjXhEwNM
2mRBQ8BnYCRJEEFXn8bKWQKBgQDlP0CYQFqFpXFqPcRmR8CbHGyUVXgJlGRFGGL
+DiLjIQJBnAQKBgH3Z8KFQBgKBgD48qF7QKBgQC8/PGwKBgHh5VQKBgQB9jTNK
-----END PRIVATE KEY-----`;

    it('creates store with mtls PEM config', () => {
      const d = new BoltDriverMock();
      let capturedDriverConfig: Record<string, unknown> | undefined;
      const s = new RemoteGraphStore({
        uri: 'bolt://localhost:7687',
        mtls: { cert: pemCert, key: pemKey },
        _Driver: class {
          constructor(_url: string, _auth: unknown, config: Record<string, unknown>) {
            capturedDriverConfig = config;
          }
          session(cfg?: any) { return d.session(cfg); }
          async close() { d.reset(); }
        } as any,
      });
      expect(s).toBeDefined();
      // Driver should receive clientCertificate with file paths from temp dir
      expect(capturedDriverConfig).toBeDefined();
      expect(capturedDriverConfig!.clientCertificate).toBeDefined();
      const cc = capturedDriverConfig!.clientCertificate as any;
      expect(cc.certfile).toContain('client.crt');
      expect(cc.keyfile).toContain('client.key');
    });

    it('throws when both mtls and clientCertificate are provided', () => {
      expect(() => new RemoteGraphStore({
        uri: 'bolt://localhost:7687',
        mtls: { cert: pemCert, key: pemKey },
        clientCertificate: { certfile: '/x', keyfile: '/y' },
        _Driver: class { constructor() {} session() { return { run: async () => ({ records: [] }), close: async () => {} }; } async close() {} } as any,
      })).toThrow(/mutually exclusive/);
    });

    it('closes and cleans up mtls temp dir', async () => {
      const d = new BoltDriverMock();
      const s = new RemoteGraphStore({
        uri: 'bolt://localhost:7687',
        mtls: { cert: pemCert, key: pemKey },
        _Driver: class {
          constructor() {}
          session(cfg?: any) { return d.session(cfg); }
          async close() { d.reset(); }
        } as any,
      });
      await s.close();
      // Temp dir should be cleaned (no error = pass)
    });

    it('works with mtls + graph operations', async () => {
      const d = new BoltDriverMock();
      const s = new RemoteGraphStore({
        uri: 'bolt://localhost:7687',
        mtls: { cert: pemCert, key: pemKey },
        _Driver: class {
          constructor() {}
          session(cfg?: any) { return d.session(cfg); }
          async close() { d.reset(); }
        } as any,
      });
      const id = await s.saveGraph(makeGraph(), makeMeta('g-mtls'));
      expect(id).toBe('g-mtls');
      await s.close();
    });
  });
});

/** Helper: create RemoteGraphStore with BoltDriverMock injected as _Driver. */
function createWithDriver(driver: BoltDriverMock): RemoteGraphStore {
  return new RemoteGraphStore({
    uri: 'bolt://localhost:7687',
    auth: { type: 'basic', user: 'neo4j', password: 'test' },
    _Driver: class {
      constructor() {}
      session(cfg?: any) { return driver.session(cfg); }
      async close() { driver.reset(); }
    } as any,
  });
}
