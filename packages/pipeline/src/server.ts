/**
 * Causality Analyzer HTTP/REST Server — v1 API.
 *
 * Provides a lightweight, production-ready HTTP API for causal analysis.
 * Supports both HTTP and HTTPS with optional mTLS client certificate auth.
 *
 * Endpoints (all under /v1/):
 *   GET  /health         — combined health + liveness + readiness
 *   GET  /ready          — readiness probe
 *   GET  /live           — liveness probe
 *   GET  /metrics        — Prometheus-compatible metrics
 *   POST /v1/discover    — run causal discovery
 *   POST /v1/analyze     — run causal analysis pipeline
 *   POST /v1/estimate    — run effect estimation
 *   GET  /v1/openapi.json — OpenAPI 3.1 specification
 *
 * Authentication (layered):
 *   1. mTLS — when `tls.requestCert: true`, client must present valid certificate
 *      signed by the configured CA. Rejected at TLS handshake level.
 *   2. Bearer Token — when `apiToken` is set, all /v1/* endpoints require
 *      `Authorization: Bearer <token>`. Health endpoints remain public.
 *
 * @packageDocumentation
 */
import { createServer, type Server } from 'node:http';
import { createServer as createSecureServer } from 'node:https';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { ServerOptions as TlsServerOptions } from 'node:https';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './graph/causal-graph.js';
import { pcAlgorithm } from './graph/pc.js';
import { identifyBackdoor } from './infer/causal-inference.js';
import { adjustBackdoor } from './infer/effect-estimation.js';
import { findBackdoorAdjustmentSet } from './infer/backdoor.js';
import { HeuristicPathRCA } from './analyze/rca.js';
import { HealthChecker, type HealthStatus } from './health.js';
import { RateLimiter } from './rate-limiter.js';
import { ConsoleLogger, type Logger } from '@agentix-e/causality-analyzer-core';

// ── Types ────────────────────────────────────────────────────────────

interface ApiResponse {
  success: boolean;
  data?: unknown;
  error?: string | undefined;
  requestId?: string | undefined;
}

interface DiscoverRequest {
  data: number[][];
  nodeNames: string[];
  alpha?: number;
}

interface AnalyzeRequest {
  graph: { nodes: string[]; edges: Array<{ source: string; target: string }> };
  data: number[][];
  anomalousNodes: string[];
}

interface EstimateRequest {
  graph: { nodes: string[]; edges: Array<{ source: string; target: string }> };
  treatment: string;
  outcome: string;
  data: number[][];
}

/** TLS/mTLS configuration for HTTPS server */
export interface CausalityServerTlsConfig {
  /** PEM-encoded server certificate */
  cert: string;
  /** PEM-encoded server private key */
  key: string;
  /** PEM-encoded CA certificate(s) for client verification */
  ca?: string;
  /** Whether to request client certificate (default: false) */
  requestCert?: boolean;
  /** Whether to reject connections without valid client cert (default: false) */
  rejectUnauthorized?: boolean;
  /** Optional passphrase for encrypted private key */
  passphrase?: string;
}

// ── Auth ─────────────────────────────────────────────────────────────

function extractBearerToken(req: IncomingMessage): string | null {
  const header = req.headers['authorization'];
  if (!header) return null;
  const match = /^[Bb]earer\s+(.+)$/.exec(header);
  return match ? (match[1] ?? null) : null;
}

// ── Server ───────────────────────────────────────────────────────────

export class CausalityServer {
  private server: Server | null = null;
  private healthChecker: HealthChecker;
  private startTime: number = 0;
  private requestCount = 0;
  private rateLimiter: RateLimiter;
  private maxBodySize: number;
  private logger: Logger;
  private serverTimeout: number;
  private apiToken: string | null;
  private port: number = 3000;
  private tlsConfig: CausalityServerTlsConfig | null = null;

  constructor(opts?: {
    maxBodySize?: number;
    rateLimitMax?: number;
    rateLimitWindowMs?: number;
    logger?: Logger;
    timeout?: number;
    /** Bearer token for API authentication. Set via CAUSALITY_API_TOKEN env var. */
    apiToken?: string;
    /** TLS/mTLS configuration for HTTPS server. When set, uses HTTPS instead of HTTP. */
    tls?: CausalityServerTlsConfig;
  }) {
    this.healthChecker = new HealthChecker();
    this.maxBodySize = opts?.maxBodySize ?? 10 * 1024 * 1024;
    this.rateLimiter = new RateLimiter({
      maxRequests: opts?.rateLimitMax ?? 100,
      windowMs: opts?.rateLimitWindowMs ?? 60000,
    });
    this.logger = opts?.logger ?? new ConsoleLogger();
    this.serverTimeout = opts?.timeout ?? 30000;
    this.apiToken = opts?.apiToken ?? null;
    this.tlsConfig = opts?.tls ?? null;
  }

  start(port: number = 3000, host: string = '0.0.0.0'): Promise<void> {
    this.port = port;
    return new Promise((resolve, reject) => {
      const handler = (req: IncomingMessage, res: ServerResponse) => this.handleRequest(req, res);

      if (this.tlsConfig) {
        const tlsOpts: TlsServerOptions = {
          cert: this.tlsConfig.cert,
          key: this.tlsConfig.key,
        };
        if (this.tlsConfig.ca) tlsOpts.ca = this.tlsConfig.ca;
        if (this.tlsConfig.passphrase) tlsOpts.passphrase = this.tlsConfig.passphrase;
        if (this.tlsConfig.requestCert) tlsOpts.requestCert = true;
        if (this.tlsConfig.rejectUnauthorized) tlsOpts.rejectUnauthorized = true;

// eslint-disable-next-line @typescript-eslint/no-misused-promises
        this.server = createSecureServer(tlsOpts, handler);
        if (this.logger.info) this.logger.info('mTLS enabled: client certificate auth active');
      } else {
// eslint-disable-next-line @typescript-eslint/no-misused-promises
        this.server = createServer(handler);
      }

      this.server.timeout = this.serverTimeout;
      this.startTime = Date.now();
      this.server.listen(port, host, () => {
        this.healthChecker.markReady();
        const proto = this.tlsConfig ? 'https' : 'http';
        const mtls = this.tlsConfig?.requestCert ? ' (mTLS)' : '';
        if (this.logger.info) this.logger.info(`Causality Analyzer API v1.0.0 started on ${proto}://${host}:${port}${mtls}`);
        resolve();
      });
      this.server.on('error', reject);
    });
  }

  stop(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.server) { resolve(); return; }
      this.server.close(err => err ? reject(err) : resolve());
      this.server = null;
    });
  }

  // ── Request Routing ───────────────────────────────────────────────

  private async handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
    this.requestCount++;
    const requestId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    // Rate limiting
    const clientIp = req.socket.remoteAddress ?? 'unknown';
    const limitResult = this.rateLimiter.check(clientIp);
    if (!limitResult.allowed) {
      res.setHeader('Retry-After', `${Math.ceil(limitResult.resetInMs / 1000)}`);
      this.sendJson(res, 429, { success: false, error: 'Too many requests', requestId });
      return;
    }

    try {
      // CORS headers
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
      res.setHeader('Access-Control-Max-Age', '86400');

      if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
      }

      const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
      const method = req.method ?? 'GET';

      // ── Public endpoints (no auth required) ──────────────────────
      if (method === 'GET' && url.pathname === '/health') return this.handleHealth(res, requestId);
      if (method === 'GET' && url.pathname === '/ready') return this.handleReady(res, requestId);
      if (method === 'GET' && url.pathname === '/live') return this.handleLiveness(res, requestId);
      if (method === 'GET' && url.pathname === '/metrics') return this.handleMetrics(res, requestId);
      if (method === 'GET' && url.pathname === '/v1/openapi.json') return this.handleOpenApi(res, requestId);

      // ── Auth check for v1 business endpoints ────────────────────
      if (this.apiToken) {
        const token = extractBearerToken(req);
        if (token !== this.apiToken) {
          res.setHeader('WWW-Authenticate', 'Bearer');
          this.sendJson(res, 401, { success: false, error: 'Unauthorized — provide valid Bearer token', requestId });
          return;
        }
      }

      // ── v1 API endpoints ────────────────────────────────────────
      if (method === 'POST' && url.pathname === '/v1/discover') return this.handleDiscover(req, res, requestId);
      if (method === 'POST' && url.pathname === '/v1/analyze') return this.handleAnalyze(req, res, requestId);
      if (method === 'POST' && url.pathname === '/v1/estimate') return this.handleEstimate(req, res, requestId);

      // 404
      this.sendJson(res, 404, { success: false, error: 'Not found', requestId });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Internal server error';
      if (message === 'Payload Too Large') {
        this.sendJson(res, 413, { success: false, error: message, requestId });
      } else if (message.startsWith('Unsupported Media Type')) {
        this.sendJson(res, 415, { success: false, error: message, requestId });
      } else {
        this.sendJson(res, 500, { success: false, error: message, requestId });
      }
    }
  }

  // ── Health Endpoints ──────────────────────────────────────────────

  private handleHealth(res: ServerResponse, requestId: string): void {
    const heapMB = Math.round(process.memoryUsage().heapUsed / 1024 / 1024);
    const status: HealthStatus = {
      status: 'healthy',
      uptime: Date.now() - this.startTime,
      version: '1.0.0',
      checks: {
        memory: { status: 'ok', detail: `${heapMB}MB` },
        ready: { status: this.healthChecker.isReady() ? 'ok' : 'error', detail: '' },
        alive: { status: this.healthChecker.isAlive() ? 'ok' : 'error', detail: '' },
        tls: { status: this.tlsConfig ? 'ok' : 'ok', detail: this.tlsConfig?.requestCert ? 'mTLS' : this.tlsConfig ? 'TLS' : 'none' },
      },
    };
    this.sendJson(res, 200, { success: true, data: status, requestId });
  }

  private handleReady(res: ServerResponse, requestId: string): void {
    const ready = this.healthChecker.isReady();
    this.sendJson(res, ready ? 200 : 503, {
      success: ready, data: { ready },
      error: ready ? undefined : 'Service not ready',
      requestId,
    });
  }

  private handleLiveness(res: ServerResponse, requestId: string): void {
    const alive = this.healthChecker.isAlive();
    this.sendJson(res, alive ? 200 : 503, {
      success: alive, data: { alive },
      error: alive ? undefined : 'Service not alive',
      requestId,
    });
  }

  // ── OpenAPI Spec ──────────────────────────────────────────────────

  private handleOpenApi(res: ServerResponse, _requestId: string): void {
    const scheme = this.tlsConfig ? 'https' : 'http';
    const spec = {
      openapi: '3.1.0',
      info: {
        title: 'Causality Analyzer API',
        version: '1.0.0',
        description: 'Enterprise-grade causal AI REST API — discovery, root cause analysis, and effect estimation.',
        license: { name: 'MIT' },
      },
      servers: [{ url: `${scheme}://localhost:${this.port}/v1`, description: 'Local development' }],
      paths: {
        '/discover': {
          post: {
            operationId: 'discoverGraph',
            summary: 'Run causal discovery',
            tags: ['Causal Discovery'],
            security: this.getSecurity(),
            requestBody: { required: true, content: { 'application/json': { schema: { $ref: '#/components/schemas/DiscoverRequest' } } } },
            responses: {
              '200': { description: 'Causal graph discovered', content: { 'application/json': { schema: { $ref: '#/components/schemas/DiscoverResponse' } } } },
              '400': { $ref: '#/components/responses/BadRequest' },
              '401': { $ref: '#/components/responses/Unauthorized' },
              '429': { $ref: '#/components/responses/RateLimited' },
            },
          },
        },
        '/analyze': {
          post: {
            operationId: 'analyzeRootCauses',
            summary: 'Run root cause analysis',
            tags: ['Root Cause Analysis'],
            security: this.getSecurity(),
            requestBody: { required: true, content: { 'application/json': { schema: { $ref: '#/components/schemas/AnalyzeRequest' } } } },
            responses: {
              '200': { description: 'Root causes identified', content: { 'application/json': { schema: { $ref: '#/components/schemas/AnalyzeResponse' } } } },
              '400': { $ref: '#/components/responses/BadRequest' },
              '401': { $ref: '#/components/responses/Unauthorized' },
              '429': { $ref: '#/components/responses/RateLimited' },
            },
          },
        },
        '/estimate': {
          post: {
            operationId: 'estimateEffect',
            summary: 'Run effect estimation',
            tags: ['Causal Inference'],
            security: this.getSecurity(),
            requestBody: { required: true, content: { 'application/json': { schema: { $ref: '#/components/schemas/EstimateRequest' } } } },
            responses: {
              '200': { description: 'Effect estimated', content: { 'application/json': { schema: { $ref: '#/components/schemas/EstimateResponse' } } } },
              '400': { $ref: '#/components/responses/BadRequest' },
              '401': { $ref: '#/components/responses/Unauthorized' },
              '429': { $ref: '#/components/responses/RateLimited' },
            },
          },
        },
      },
      components: {
        securitySchemes: this.getSecuritySchemes(),
        schemas: this.getSchemas(),
        responses: {
          BadRequest: { description: 'Bad request', content: { 'application/json': { schema: { $ref: '#/components/schemas/ErrorResponse' } } } },
          Unauthorized: { description: 'Unauthorized', headers: { 'WWW-Authenticate': { schema: { type: 'string' } } }, content: { 'application/json': { schema: { $ref: '#/components/schemas/ErrorResponse' } } } },
          RateLimited: { description: 'Too many requests', headers: { 'Retry-After': { schema: { type: 'integer' } } }, content: { 'application/json': { schema: { $ref: '#/components/schemas/ErrorResponse' } } } },
        },
      },
      tags: [
        { name: 'Causal Discovery' },
        { name: 'Root Cause Analysis' },
        { name: 'Causal Inference' },
      ],
    };

    res.setHeader('Content-Type', 'application/json');
    res.writeHead(200);
    res.end(JSON.stringify(spec, null, 2));
  }

  private getSecurity(): Array<Record<string, string[]>> {
    const schemes: Array<Record<string, string[]>> = [];
    if (this.tlsConfig?.requestCert) schemes.push({ mTLS: [] });
    if (this.apiToken) schemes.push({ bearerAuth: [] });
    return schemes.length > 0 ? schemes : [];
  }

  private getSecuritySchemes(): Record<string, unknown> {
    const schemes: Record<string, unknown> = {};
    if (this.tlsConfig?.requestCert) {
      schemes.mTLS = { type: 'mutualTLS', description: 'Client certificate authentication via TLS handshake' };
    }
    if (this.apiToken) {
      schemes.bearerAuth = { type: 'http', scheme: 'bearer', description: 'API token from CAUSALITY_API_TOKEN' };
    }
    return schemes;
  }

  private getSchemas(): Record<string, unknown> {
    return {
      DiscoverRequest: {
        type: 'object', required: ['data', 'nodeNames'],
        properties: {
          data: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
          nodeNames: { type: 'array', items: { type: 'string' } },
          alpha: { type: 'number', default: 0.05 },
        },
      },
      DiscoverResponse: {
        type: 'object',
        properties: {
          success: { type: 'boolean' },
          data: { type: 'object', properties: { edges: { type: 'array', items: { $ref: '#/components/schemas/CausalEdge' } }, adjustmentSets: { type: 'object' } } },
          requestId: { type: 'string' },
        },
      },
      AnalyzeRequest: {
        type: 'object', required: ['graph', 'data', 'anomalousNodes'],
        properties: {
          graph: { $ref: '#/components/schemas/GraphInput' },
          data: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
          anomalousNodes: { type: 'array', items: { type: 'string' } },
        },
      },
      AnalyzeResponse: {
        type: 'object',
        properties: {
          success: { type: 'boolean' },
          data: { type: 'object', properties: { rootCauses: { type: 'array', items: { $ref: '#/components/schemas/RootCauseEntry' } }, paths: { type: 'array', items: { $ref: '#/components/schemas/RootCausePath' } } } },
          requestId: { type: 'string' },
        },
      },
      EstimateRequest: {
        type: 'object', required: ['graph', 'treatment', 'outcome', 'data'],
        properties: {
          graph: { $ref: '#/components/schemas/GraphInput' },
          treatment: { type: 'string' }, outcome: { type: 'string' },
          data: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
        },
      },
      EstimateResponse: {
        type: 'object',
        properties: {
          success: { type: 'boolean' },
          data: { type: 'object', properties: { ate: { type: 'number' }, se: { type: 'number' }, ci95: { type: 'array', items: { type: 'number' } }, adjustmentSet: { type: 'array', items: { type: 'string' } }, isSignificant: { type: 'boolean' } } },
          requestId: { type: 'string' },
        },
      },
      GraphInput: { type: 'object', required: ['nodes', 'edges'], properties: { nodes: { type: 'array', items: { type: 'string' } }, edges: { type: 'array', items: { type: 'object', properties: { source: { type: 'string' }, target: { type: 'string' } } } } } },
      CausalEdge: { type: 'object', properties: { source: { type: 'string' }, target: { type: 'string' }, weight: { type: 'number' }, directed: { type: 'boolean' } } },
      RootCauseEntry: { type: 'object', properties: { name: { type: 'string' }, score: { type: 'number' }, evidence: { type: 'array' } } },
      RootCausePath: { type: 'object', properties: { nodes: { type: 'array', items: { type: 'string' } }, score: { type: 'number' }, direction: { type: 'string' } } },
      ErrorResponse: { type: 'object', properties: { success: { type: 'boolean' }, error: { type: 'string' }, requestId: { type: 'string' } } },
    };
  }

  // ── Metrics ───────────────────────────────────────────────────────

  private handleMetrics(res: ServerResponse, _requestId: string): void {
    const metrics = [
      '# HELP ca_requests_total Total HTTP requests',
      '# TYPE ca_requests_total counter',
      `ca_requests_total ${this.requestCount}`,
      '# HELP ca_uptime_seconds Server uptime in seconds',
      '# TYPE ca_uptime_seconds gauge',
      `ca_uptime_seconds ${((Date.now() - this.startTime) / 1000).toFixed(1)}`,
      '# HELP ca_memory_heap_used_bytes Memory heap used',
      '# TYPE ca_memory_heap_used_bytes gauge',
      `ca_memory_heap_used_bytes ${process.memoryUsage().heapUsed}`,
    ].join('\n');
    res.setHeader('Content-Type', 'text/plain');
    res.writeHead(200);
    res.end(metrics);
  }

  // ── Business Endpoints ────────────────────────────────────────────

  private async handleDiscover(req: IncomingMessage, res: ServerResponse, requestId: string): Promise<void> {
    const body = await this.parseBody<DiscoverRequest>(req);
    if (!body.data || !body.nodeNames || !Array.isArray(body.nodeNames)) {
      this.sendJson(res, 400, { success: false, error: 'Missing data or nodeNames', requestId });
      return;
    }
    const matrix = new Matrix(body.data);
    const { graph, sepSet } = pcAlgorithm(matrix, body.nodeNames, { alpha: body.alpha ?? 0.05, maxDegree: -1, stable: true });
    this.sendJson(res, 200, { success: true, data: { edges: graph.edges, adjustmentSets: Object.fromEntries(sepSet) }, requestId });
  }

  private async handleAnalyze(req: IncomingMessage, res: ServerResponse, requestId: string): Promise<void> {
    const body = await this.parseBody<AnalyzeRequest>(req);
    if (!body.graph || !body.data || !body.anomalousNodes) {
      this.sendJson(res, 400, { success: false, error: 'Missing graph, data, or anomalousNodes', requestId });
      return;
    }
    const graph = new CausalGraph(body.graph.nodes);
    for (const e of body.graph.edges) graph.addEdge(e.source, e.target);
    const rca = new HeuristicPathRCA();
    rca.train(graph, new Set(body.anomalousNodes), new Matrix(body.data));
    const result = rca.findRootCauses(body.anomalousNodes);
    const adjustmentInfo: Record<string, string[]> = {};
    for (const anom of body.anomalousNodes) {
      for (const rc of result.rootCauses) {
        adjustmentInfo[`${rc.name}→${anom}`] = findBackdoorAdjustmentSet(graph, rc.name, anom);
      }
    }
    this.sendJson(res, 200, { success: true, data: { rootCauses: result.rootCauses.map(rc => ({ name: rc.name, score: rc.score, evidence: rc.evidence })), paths: result.paths, adjustmentInfo }, requestId });
  }

  private async handleEstimate(req: IncomingMessage, res: ServerResponse, requestId: string): Promise<void> {
    const body = await this.parseBody<EstimateRequest>(req);
    if (!body.graph || !body.treatment || !body.outcome || !body.data) {
      this.sendJson(res, 400, { success: false, error: 'Missing graph, treatment, outcome, or data', requestId });
      return;
    }
    const graph = new CausalGraph(body.graph.nodes);
    for (const e of body.graph.edges) graph.addEdge(e.source, e.target);
    const nodeIndex = new Map(body.graph.nodes.map((n, i) => [n, i]));
    if (nodeIndex.get(body.treatment) === undefined || nodeIndex.get(body.outcome) === undefined) {
      this.sendJson(res, 400, { success: false, error: 'Treatment or outcome not found', requestId });
      return;
    }
    const { ate, se } = adjustBackdoor(graph, body.treatment, body.outcome, body.data, nodeIndex);
    const estimand = identifyBackdoor(graph, body.treatment, body.outcome);
    this.sendJson(res, 200, { success: true, data: { ate, se, ci95: [ate - 1.96 * se, ate + 1.96 * se], adjustmentSet: estimand.backdoorVariables.backdoor ?? [], isSignificant: Math.abs(ate / Math.max(se, 1e-10)) > 1.96 }, requestId });
  }

  // ── Helpers ───────────────────────────────────────────────────────

  private sendJson(res: ServerResponse, statusCode: number, body: ApiResponse): void {
    res.setHeader('Content-Type', 'application/json');
    res.writeHead(statusCode);
    res.end(JSON.stringify(body));
  }

  private parseBody<T>(req: IncomingMessage): Promise<T> {
    return new Promise((resolve, reject) => {
      const contentType = req.headers['content-type'] ?? '';
      if (!contentType.includes('application/json')) { reject(new Error('Unsupported Media Type')); return; }
      const contentLength = parseInt(req.headers['content-length'] ?? '0', 10);
      if (contentLength > this.maxBodySize) { reject(new Error('Payload Too Large')); return; }
      let raw = '', size = 0;
      req.on('data', (chunk: Buffer) => { size += chunk.length; if (size > this.maxBodySize) { req.destroy(); reject(new Error('Payload Too Large')); return; } raw += chunk.toString(); });
      req.on('end', () => { try { resolve(JSON.parse(raw || '{}') as T); } catch { reject(new Error('Bad Request: invalid JSON')); } });
      req.on('error', err => reject(err));
    });
  }
}
