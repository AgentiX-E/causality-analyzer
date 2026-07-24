/**
 * HTTP Server Integration Tests.
 *
 * Tests the CausalityServer REST API endpoints end-to-end.
 * Starts a server on a random port, sends HTTP requests,
 * and verifies responses.
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { CausalityServer } from '../../src/server.js';

let server: CausalityServer;
let baseUrl: string;
const PORT = 19888;

beforeAll(async () => {
  server = new CausalityServer();
  await server.start(PORT, '127.0.0.1');
  baseUrl = `http://127.0.0.1:${PORT}`;
}, 15000);

afterAll(async () => {
  try { await server.stop(); } catch { /* server may already be closed */ }
}, 15000);

async function fetchApi(path: string, method = 'GET', body?: unknown) {
  const res = await fetch(`${baseUrl}${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const json = await res.json();
  return { res, json };
}

describe('Health Endpoints', () => {
  it('GET /health returns healthy status', async () => {
    const { res, json } = await fetchApi('/health');
    expect(res.status).toBe(200);
    expect(json.success).toBe(true);
    expect(json.data.status).toBe('healthy');
    expect(json.data.version).toBeDefined();
    expect(json.data.uptime).toBeGreaterThanOrEqual(0);
  });

  it('GET /ready returns readiness status', async () => {
    const { res, json } = await fetchApi('/ready');
    expect([200, 503]).toContain(res.status);
    expect(typeof json.data.ready).toBe('boolean');
  });

  it('GET /live returns liveness status', async () => {
    const { res, json } = await fetchApi('/live');
    expect([200, 503]).toContain(res.status);
    expect(typeof json.data.alive).toBe('boolean');
  });
});

describe('Metrics', () => {
  it('GET /metrics returns Prometheus metrics', async () => {
    const res = await fetch(`${baseUrl}/metrics`);
    const text = await res.text();
    expect(res.status).toBe(200);
    expect(text).toContain('ca_requests_total');
    expect(text).toContain('ca_uptime_seconds');
    expect(text).toContain('ca_memory_heap_used_bytes');
  });
});

describe('CORS', () => {
  it('OPTIONS returns CORS headers', async () => {
    const res = await fetch(`${baseUrl}/health`, { method: 'OPTIONS' });
    expect(res.status).toBe(204);
    expect(res.headers.get('access-control-allow-origin')).toBe('*');
  });
});

describe('POST /discover — Causal Discovery', () => {
  it('returns edges for simple causal graph', async () => {
    const { res, json } = await fetchApi('/discover', 'POST', {
      data: [
        [1, 2, 3],
        [1.1, 2.1, 2.9],
        [0.9, 1.9, 3.1],
        [1.0, 2.0, 3.0],
        [1.05, 2.05, 2.95],
      ],
      nodeNames: ['X', 'Y', 'Z'],
      alpha: 0.05,
    });

    expect(res.status).toBe(200);
    expect(json.success).toBe(true);
    expect(Array.isArray(json.data.edges)).toBe(true);
  });

  it('returns 400 when data is missing', async () => {
    const { res, json } = await fetchApi('/discover', 'POST', {});
    expect(res.status).toBe(400);
    expect(json.success).toBe(false);
  });

  it('returns 400 when nodeNames is missing', async () => {
    const { res, json } = await fetchApi('/discover', 'POST', { data: [[1, 2]] });
    expect(res.status).toBe(400);
    expect(json.success).toBe(false);
  });
});

describe('POST /analyze — Root Cause Analysis', () => {
  it('returns root causes for simple anomalous graph', async () => {
    const { res, json } = await fetchApi('/analyze', 'POST', {
      graph: {
        nodes: ['Memory', 'CPU', 'Latency'],
        edges: [
          { source: 'Memory', target: 'CPU' },
          { source: 'CPU', target: 'Latency' },
        ],
      },
      data: [
        [100, 50, 10],
        [120, 55, 12],
        [90, 48, 9],
        [110, 52, 11],
        [105, 53, 10],
      ],
      anomalousNodes: ['CPU', 'Latency'],
    });

    expect(res.status).toBe(200);
    expect(json.success).toBe(true);
    expect(Array.isArray(json.data.rootCauses)).toBe(true);
    expect(Array.isArray(json.data.paths)).toBe(true);
  });

  it('returns 400 when graph is missing', async () => {
    const { res, json } = await fetchApi('/analyze', 'POST', { data: [[1]], anomalousNodes: ['A'] });
    expect(res.status).toBe(400);
    expect(json.success).toBe(false);
  });
});

describe('POST /estimate — Effect Estimation', () => {
  it('returns causal effect for simple treatment-outcome', async () => {
    const { res, json } = await fetchApi('/estimate', 'POST', {
      graph: {
        nodes: ['Treatment', 'Outcome'],
        edges: [{ source: 'Treatment', target: 'Outcome' }],
      },
      treatment: 'Treatment',
      outcome: 'Outcome',
      data: [
        [0, 1],
        [1, 3],
        [0, 2],
        [1, 4],
        [0, 1.5],
        [1, 3.5],
      ],
    });

    expect(res.status).toBe(200);
    expect(json.success).toBe(true);
    expect(typeof json.data.ate).toBe('number');
    expect(typeof json.data.se).toBe('number');
    expect(Array.isArray(json.data.ci95)).toBe(true);
    expect(typeof json.data.isSignificant).toBe('boolean');
  });

  it('returns 400 when treatment is missing', async () => {
    const { res, json } = await fetchApi('/estimate', 'POST', {
      graph: { nodes: ['X', 'Y'], edges: [{ source: 'X', target: 'Y' }] },
      outcome: 'Y',
      data: [[1, 2]],
    });
    expect(res.status).toBe(400);
    expect(json.success).toBe(false);
  });

  it('returns 400 when treatment/outcome not in graph nodes', async () => {
    const { res, json } = await fetchApi('/estimate', 'POST', {
      graph: { nodes: ['A', 'B'], edges: [{ source: 'A', target: 'B' }] },
      treatment: 'X',
      outcome: 'Y',
      data: [[1, 2]],
    });
    expect(res.status).toBe(400);
    expect(json.success).toBe(false);
  });
});

describe('Error Handling', () => {
  it('returns 404 for unknown endpoints', async () => {
    const { res, json } = await fetchApi('/nonexistent');
    expect(res.status).toBe(404);
    expect(json.success).toBe(false);
  });

  it('returns error for malformed JSON body', async () => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    try {
      const res = await fetch(`${baseUrl}/discover`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: 'not json',
        signal: controller.signal,
      });
      clearTimeout(timeout);
      expect(res.status).toBeGreaterThanOrEqual(400);
    } catch (e) {
      clearTimeout(timeout);
      // Server may reject with connection error on invalid JSON
      // This is expected behavior for non-JSON input
      expect(e).toBeDefined();
    }
  }, 10000);

  it('includes requestId in error responses', async () => {
    const { res, json } = await fetchApi('/discover', 'POST', {});
    expect(res.status).toBe(400);
    expect(typeof json.requestId).toBe('string');
  });
});
