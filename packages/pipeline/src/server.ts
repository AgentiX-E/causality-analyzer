/**
 * Causality Analyzer HTTP/REST Server.
 *
 * Provides a lightweight, production-ready HTTP API for causal analysis.
 * Uses built-in Node.js `http` module — zero additional dependencies.
 *
 * Endpoints:
 *   GET  /health     — health check (liveness + readiness)
 *   GET  /metrics    — Prometheus-compatible metrics
 *   POST /analyze    — run causal analysis pipeline
 *   POST /discover   — run causal discovery
 *   POST /estimate   — run effect estimation
 *
 * @packageDocumentation
 */
import { createServer, type IncomingMessage, type ServerResponse, type Server } from 'node:http';
import { CausalGraph } from './graph/causal-graph.js';
import { pcAlgorithm } from './graph/pc.js';
import { identifyBackdoor } from './infer/causal-inference.js';
import { adjustBackdoor } from './infer/effect-estimation.js';
import { findBackdoorAdjustmentSet } from './infer/backdoor.js';
import { HeuristicPathRCA } from './analyze/rca.js';
import { HealthChecker, type HealthStatus } from './health.js';

// ─��� Types ────────────────────────────────────────────────────────────

interface ApiResponse {
  success: boolean;
  data?: unknown;
  error?: string;
  requestId?: string;
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

// ── Server ───────────────────────────────────────────────────────────

export class CausalityServer {
  private server: Server | null = null;
  private healthChecker: HealthChecker;
  private startTime: number = 0;
  private requestCount = 0;

  constructor() {
    this.healthChecker = new HealthChecker();
  }

  start(port: number = 3000, host: string = '0.0.0.0'): Promise<void> {
    return new Promise((resolve, reject) => {
      this.server = createServer((req, res) => this.handleRequest(req, res));
      this.startTime = Date.now();
      this.server.listen(port, host, () => resolve());
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

    try {
      // CORS headers
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

      if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
      }

      const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);

      switch (`${req.method} ${url.pathname}`) {
        case 'GET /health': return this.handleHealth(res, requestId);
        case 'GET /ready': return this.handleReady(res, requestId);
        case 'GET /live': return this.handleLiveness(res, requestId);
        case 'GET /metrics': return this.handleMetrics(res, requestId);
        case 'POST /discover': return this.handleDiscover(req, res, requestId);
        case 'POST /analyze': return this.handleAnalyze(req, res, requestId);
        case 'POST /estimate': return this.handleEstimate(req, res, requestId);
        default:
          this.sendJson(res, 404, { success: false, error: 'Not found', requestId });
      }
    } catch (err) {
      this.sendJson(res, 500, {
        success: false,
        error: err instanceof Error ? err.message : 'Internal server error',
        requestId,
      });
    }
  }

  // ── Health Endpoints ──────────────────────────────────────────────

  private handleHealth(res: ServerResponse, requestId: string): void {
    const status: HealthStatus = {
      status: 'healthy',
      uptime: Date.now() - this.startTime,
      version: '2.0.0',
      checks: {
        memory: {
          status: 'ok',
          detail: `${Math.round(process.memoryUsage().heapUsed / 1024 / 1024)}MB`,
        },
      },
    };
    this.sendJson(res, 200, { success: true, data: status, requestId });
  }

  private handleReady(res: ServerResponse, requestId: string): void {
    const ready = this.healthChecker.isReady();
    this.sendJson(res, ready ? 200 : 503, {
      success: ready,
      data: { ready },
      error: ready ? undefined : 'Service not ready',
      requestId,
    });
  }

  private handleLiveness(res: ServerResponse, requestId: string): void {
    const alive = this.healthChecker.isAlive();
    this.sendJson(res, alive ? 200 : 503, {
      success: alive,
      data: { alive },
      error: alive ? undefined : 'Service not alive',
      requestId,
    });
  }

  // ── Metrics ───────────────────────────────────────────────────────

  private handleMetrics(res: ServerResponse, requestId: string): void {
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

    const matrix = this.toMatrix(body.data);
    const { graph, sepSet } = pcAlgorithm(matrix, body.nodeNames, {
      alpha: body.alpha ?? 0.05,
      maxDegree: -1,
      stable: true,
    });

    this.sendJson(res, 200, {
      success: true,
      data: {
        edges: graph.edges,
        adjustmentSets: Object.fromEntries(sepSet),
      },
      requestId,
    });
  }

  private async handleAnalyze(req: IncomingMessage, res: ServerResponse, requestId: string): Promise<void> {
    const body = await this.parseBody<AnalyzeRequest>(req);
    if (!body.graph || !body.data || !body.anomalousNodes) {
      this.sendJson(res, 400, { success: false, error: 'Missing graph, data, or anomalousNodes', requestId });
      return;
    }

    const graph = new CausalGraph(body.graph.nodes);
    for (const e of body.graph.edges) graph.addEdge(e.source, e.target);

    const data = body.data;
    const rca = new HeuristicPathRCA();
    rca.train(graph, new Set(body.anomalousNodes), this.toMatrix(data));
    const result = rca.findRootCauses(body.anomalousNodes);

    // Compute backdoor sets for each anomalous node
    const adjustmentInfo: Record<string, string[]> = {};
    for (const anom of body.anomalousNodes) {
      for (const rc of result.rootCauses) {
        const key = `${rc.name}→${anom}`;
        adjustmentInfo[key] = findBackdoorAdjustmentSet(graph, rc.name, anom);
      }
    }

    this.sendJson(res, 200, {
      success: true,
      data: {
        rootCauses: result.rootCauses.map(rc => ({ name: rc.name, score: rc.score, evidence: rc.evidence })),
        paths: result.paths,
        adjustmentInfo,
      },
      requestId,
    });
  }

  private async handleEstimate(req: IncomingMessage, res: ServerResponse, requestId: string): Promise<void> {
    const body = await this.parseBody<EstimateRequest>(req);
    if (!body.graph || !body.treatment || !body.outcome || !body.data) {
      this.sendJson(res, 400, { success: false, error: 'Missing graph, treatment, outcome, or data', requestId });
      return;
    }

    const graph = new CausalGraph(body.graph.nodes);
    for (const e of body.graph.edges) graph.addEdge(e.source, e.target);

    const data = body.data;
    const nodeIndex = new Map(body.graph.nodes.map((n, i) => [n, i]));
    const treatmentIdx = nodeIndex.get(body.treatment);
    const outcomeIdx = nodeIndex.get(body.outcome);

    if (treatmentIdx === undefined || outcomeIdx === undefined) {
      this.sendJson(res, 400, { success: false, error: 'Treatment or outcome not found in graph nodes', requestId });
      return;
    }

    const { ate, se } = adjustBackdoor(graph, body.treatment, body.outcome, data, nodeIndex);
    const estimand = identifyBackdoor(graph, body.treatment, body.outcome);

    this.sendJson(res, 200, {
      success: true,
      data: {
        ate,
        se,
        ci95: [ate - 1.96 * se, ate + 1.96 * se],
        adjustmentSet: estimand.backdoorVariables.backdoor ?? [],
        isSignificant: Math.abs(ate / Math.max(se, 1e-10)) > 1.96,
      },
      requestId,
    });
  }

  // ── Helpers ───────────────────────────────────────────────────────

  private sendJson(res: ServerResponse, statusCode: number, body: ApiResponse): void {
    res.setHeader('Content-Type', 'application/json');
    res.writeHead(statusCode);
    res.end(JSON.stringify(body));
  }

  private parseBody<T>(req: IncomingMessage): Promise<T> {
    return new Promise((resolve) => {
      let raw = '';
      req.on('data', chunk => { raw += chunk; });
      req.on('end', () => {
        try { resolve(JSON.parse(raw || '{}')); }
        catch { resolve({} as T); } // return empty object — validation will catch missing fields
      });
      req.on('error', () => resolve({} as T));
    });
  }

  /** Convert number[][] to ml-matrix Matrix */
  private toMatrix(data: number[][]): import('ml-matrix').Matrix {
    const { Matrix } = require('ml-matrix');
    return new Matrix(data);
  }
}
