#!/usr/bin/env node
/**
 * Causality Analyzer CLI — command-line interface for causal analysis.
 *
 * Usage:
 *   causal-analyzer discover <data.json> --nodes CPU,Memory,Latency
 *   causal-analyzer analyze <anomaly.json> --slis CPU,Latency
 *   causal-analyzer serve --port 3000
 *   causal-analyzer version
 *
 * Environment variables:
 *   CAUSALITY_API_TOKEN — if set, all /v1/* endpoints require Bearer token auth.
 *
 * @packageDocumentation
 */
import { readFileSync } from 'fs';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './graph/causal-graph.js';
import { pcAlgorithm } from './graph/pc.js';
import { CIRCAPipeline } from './analyze/circa.js';
import { CausalityServer } from './server.js';

interface CliArgs {
  file?: string;
  nodes?: string[];
  slis?: string[];
  port?: number;
}

function parseArgs(argv: string[]): { cmd: string; args: CliArgs } {
  const cmd = argv[0] ?? '';
  const args: CliArgs = {};
  let i = 1;
  while (i < argv.length) {
    const a = argv[i];
    if (a === '--nodes' && i + 1 < argv.length) { args.nodes = argv[++i].split(','); }
    else if (a === '--slis' && i + 1 < argv.length) { args.slis = argv[++i].split(','); }
    else if (a === '--port' && i + 1 < argv.length) { args.port = parseInt(argv[++i], 10); }
    else if (!a.startsWith('--')) { args.file = a; }
    i++;
  }
  return { cmd, args };
}

function main(): void {
  const argv = process.argv.slice(2);
  if (argv.length === 0) {
    console.log(`Causality Analyzer CLI v1.0.0

Commands:
  discover <file.json> --nodes A,B,C    PC causal discovery
  analyze  <file.json> --slis CPU,Lat   CIRCA root cause analysis
  serve    --port 3000                  REST API server (with Auth)
  version                              Show version

Environment:
  CAUSALITY_API_TOKEN    Bearer token for API authentication (optional)
`);
    process.exit(0);
  }

  const { cmd, args } = parseArgs(argv);
  switch (cmd) {
    case 'discover': execDiscover(args); break;
    case 'analyze': execAnalyze(args); break;
    case 'serve': execServe(args); break;
    case 'version': console.log('v1.0.0'); break;
    default: console.error(`Unknown command: ${cmd}`); process.exit(1);
  }
}

function execDiscover(args: CliArgs): void {
  if (!args.file || !args.nodes || args.nodes.length === 0) {
    console.error('Usage: causal-analyzer discover <file.json> --nodes A,B,C');
    process.exit(1);
  }
  const raw: unknown = JSON.parse(readFileSync(args.file, 'utf-8'));
  const rawObj = raw as Record<string, unknown>;
  const data = new Matrix((rawObj.data ?? rawObj) as ArrayLike<ArrayLike<number>>);
  const result = pcAlgorithm(data, args.nodes, { alpha: 0.05, stable: true });
  console.log(JSON.stringify({
    edges: result.graph.edges.map((e: { source: string; target: string; weight: number }) =>
      ({ from: e.source, to: e.target, weight: e.weight })),
    isDAG: result.graph.isDAG(),
  }, null, 2));
}

function execAnalyze(args: CliArgs): void {
  if (!args.file || !args.slis || args.slis.length === 0) {
    console.error('Usage: causal-analyzer analyze <file.json> --slis CPU,Latency');
    process.exit(1);
  }
  const raw: unknown = JSON.parse(readFileSync(args.file, 'utf-8'));
  const rawObj = raw as Record<string, unknown>;
  const data = rawObj.data as Record<string, unknown> ?? rawObj;
  const graphData = (rawObj.graph as Record<string, unknown>) ?? { nodes: [] as string[], edges: [] as Array<{ from: string; to: string }> };

  const g = new CausalGraph(graphData.nodes as string[]);
  const edges = graphData.edges as Array<{ from: string; to: string }>;
  for (const e of edges) { g.addEdge(e.from, e.to); }

  const pipeline = new CIRCAPipeline();
  const normalData = data.normal as number[][] ?? (data as unknown as number[][]);
  pipeline.train(g, normalData);
  const anomalyData = data.anomaly as number[][] ?? (data as unknown as number[][]);
  const result = pipeline.analyze(anomalyData, args.slis);

  console.log(JSON.stringify({
    rootCauses: result.rootCauses.map(
      (r: { name: string; score: number; rank: number }) =>
        ({ name: r.name, score: r.score, rank: r.rank })),
  }, null, 2));
}

function execServe(args: CliArgs): void {
  const port = args.port ?? 3000;
  const apiToken = process.env['CAUSALITY_API_TOKEN'] ?? undefined;

  // TLS/mTLS configuration from environment variables
  const tlsCert = process.env['CAUSALITY_TLS_CERT'];
  const tlsKey = process.env['CAUSALITY_TLS_KEY'];
  const tlsCa = process.env['CAUSALITY_TLS_CA'];
  const tlsRequestCert = process.env['CAUSALITY_MTLS_REQUEST_CERT'] === 'true';
  const tlsRejectUnauthorized = process.env['CAUSALITY_MTLS_REJECT_UNAUTHORIZED'] === 'true';
  const tlsPassphrase = process.env['CAUSALITY_TLS_PASSPHRASE'] ?? undefined;

  const serverOpts: { apiToken?: string; tls?: import('./server.js').CausalityServerTlsConfig } = {};
  if (apiToken) serverOpts.apiToken = apiToken;

  if (tlsCert && tlsKey) {
    const tlsConfig: import('./server.js').CausalityServerTlsConfig = {
      cert: tlsCert,
      key: tlsKey,
      requestCert: tlsRequestCert,
      rejectUnauthorized: tlsRejectUnauthorized,
    };
    if (tlsCa) tlsConfig.ca = tlsCa;
    if (tlsPassphrase) tlsConfig.passphrase = tlsPassphrase;
    serverOpts.tls = tlsConfig;
  }

  const proto = serverOpts.tls ? 'https' : 'http';
  const authFlags: string[] = [];
  if (apiToken) authFlags.push('Bearer');
  if (serverOpts.tls?.requestCert) authFlags.push('mTLS');
  const authNote = authFlags.length > 0 ? ` (auth: ${authFlags.join(' + ')})` : ' (no auth)';

  const server = new CausalityServer(serverOpts);
  server.start(port).then(() => {
    console.log(`Causality Analyzer API v1.0.0 listening on ${proto}://localhost:${port}${authNote}`);
    console.log(`Endpoints:`);
    console.log(`  GET  /health       — combined health check`);
    console.log(`  GET  /ready        — readiness probe`);
    console.log(`  GET  /live         — liveness probe`);
    console.log(`  GET  /metrics      — Prometheus metrics`);
    console.log(`  GET  /v1/openapi.json — OpenAPI 3.1 spec`);
    console.log(`  POST /v1/discover  — causal discovery`);
    console.log(`  POST /v1/analyze   — root cause analysis`);
    console.log(`  POST /v1/estimate  — effect estimation`);
  });
}

main();
