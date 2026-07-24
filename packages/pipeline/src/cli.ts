#!/usr/bin/env node
/**
 * Causality Analyzer CLI — causal analysis for SRE and AIOps workflows.
 *
 * Commands:
 *   discover <data.json> --nodes A,B,C [--alpha 0.05] [--method pc|ges|fci|notears|golem]
 *   analyze  <data.json> --slis CPU,Lat [--graph graph.json] [--method circa|bayesian|ht]
 *   benchmark --size 1000
 *   serve    --port 3000 [--metrics]
 *   version
 *
 * Config file: --config causality-analyzer.config.json (overrides env/CLI args)
 * Env vars: CA_ALPHA=0.01, CA_PORT=3000
 *
 * @packageDocumentation
 */
import { readFileSync, existsSync } from 'fs';
import { createServer, IncomingMessage, ServerResponse } from 'http';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './graph/causal-graph.js';
import { pcAlgorithm } from './graph/pc.js';
import { fciAlgorithm } from './graph/advanced-discovery.js';
import { gesAlgorithm } from './graph/ges.js';
import { notearsAlgorithm, golemAlgorithm } from './graph/notears.js';
import { CIRCAPipeline } from './analyze/circa.js';
import { kciTest } from './graph/kci.js';
import { BayesianRCA } from './analyze/bayesian-rca.js';
import { StatsDetector } from './detect/stats-detector.js';
import { SPOTDetector } from './detect/spot.js';
import { MetricsRegistry } from './observability.js';

// ── Types ──────────────────────────────────────────────────────────
interface CliArgs {
  file?: string;
  config?: string;
  nodes?: string[];
  slis?: string[];
  port?: number;
  alpha?: number;
  method?: string;
  graph?: string;
  output?: string;
  size?: number;
  metrics?: boolean;
}

interface CliConfig {
  alpha: number;
  port: number;
  method: string;
}

// ── Config loading ──────────────────────────────────────────────────
function loadConfig(args: CliArgs): CliConfig {
  const cfg: CliConfig = { alpha: 0.05, port: 3000, method: 'pc' };

  // 1. File config
  if (args.config && existsSync(args.config)) {
    const raw = JSON.parse(readFileSync(args.config, 'utf-8'));
    if (raw.alpha != null) cfg.alpha = raw.alpha;
    if (raw.port != null) cfg.port = raw.port;
    if (raw.method) cfg.method = raw.method;
  }

  // 2. Env vars
  if (process.env.CA_ALPHA) cfg.alpha = Number(process.env.CA_ALPHA) || cfg.alpha;
  if (process.env.CA_PORT) cfg.port = Number(process.env.CA_PORT) || cfg.port;
  if (process.env.CA_METHOD) cfg.method = process.env.CA_METHOD;

  // 3. CLI args (highest priority)
  if (args.alpha != null) cfg.alpha = args.alpha;
  if (args.port != null) cfg.port = args.port;
  if (args.method) cfg.method = args.method;

  return cfg;
}

function parseArgs(argv: string[]): { cmd: string; args: CliArgs } {
  const cmd = argv[0] ?? '';
  const args: CliArgs = {};
  let i = 1;
  const boolFlags = new Set(['--metrics']);
  while (i < argv.length) {
    const a = argv[i]!;
    if (boolFlags.has(a)) { (args as Record<string, unknown>)[a.slice(2)] = true; }
    else if (i + 1 < argv.length) {
      if (a === '--nodes') args.nodes = argv[++i]!.split(',');
      else if (a === '--slis') args.slis = argv[++i]!.split(',');
      else if (a === '--port') args.port = parseInt(argv[++i]!, 10);
      else if (a === '--alpha') args.alpha = parseFloat(argv[++i]!);
      else if (a === '--method') args.method = argv[++i];
      else if (a === '--graph') args.graph = argv[++i];
      else if (a === '--config') args.config = argv[++i];
      else if (a === '--output') args.output = argv[++i];
      else if (a === '--size') args.size = parseInt(argv[++i]!, 10);
      else if (!a.startsWith('--')) args.file = a;
    } else if (!a.startsWith('--')) args.file = a;
    i++;
  }
  return { cmd, args };
}

// ── Main ────────────────────────────────────────────────────────────
function main(): void {
  const argv = process.argv.slice(2);
  if (argv.length === 0 || argv[0] === 'help' || argv[0] === '--help') {
    console.log(`Causality Analyzer CLI v1.0.0

Commands:
  discover <file.json>  Causal discovery (PC/FCI/GES/NOTEARS/GOLEM)
    --nodes A,B,C        Node names (required)
    --method pc          Algorithm: pc|fci|ges|notears|golem
    --alpha 0.05         Significance level
    --config cfg.json    Config file path
    --output out.json    Output file path

  analyze  <file.json>  Root cause analysis (CIRCA/Bayesian/HT)
    --slis CPU,Lat       Anomalous nodes list (required)
    --graph graph.json   Causal graph file (optional, use discovered)
    --method circa       RCA method: circa|bayesian|ht
    --output out.json    Output file path

  benchmark             Performance benchmark suite
    --size 200           Data size per test

  serve                 REST API + optional Prometheus metrics
    --port 3000          Listen port
    --metrics            Enable /metrics endpoint

  version               Show version

Config priority: CLI args > env vars > config file
Env vars: CA_ALPHA, CA_PORT, CA_METHOD
`);
    process.exit(0);
  }

  const { cmd, args } = parseArgs(argv);
  const config = loadConfig(args);

  switch (cmd) {
    case 'discover': execDiscover(args, config); break;
    case 'analyze': execAnalyze(args, config); break;
    case 'benchmark': execBenchmark(args); break;
    case 'serve': execServe(args, config); break;
    case 'version': console.log('v1.0.0'); break;
    default:
      console.error(`Unknown command: ${cmd}. Use --help for usage.`);
      process.exit(1);
  }
}

// ── Discover ────────────────────────────────────────────────────────
function execDiscover(args: CliArgs, cfg: CliConfig): void {
  if (!args.file || !args.nodes || args.nodes.length === 0) {
    console.error('Usage: causal-analyzer discover <file.json> --nodes A,B,C [--method pc]');
    process.exit(2);
  }

  try {
    const raw = JSON.parse(readFileSync(args.file, 'utf-8'));
    const data = new Matrix(raw.data ?? raw);
    const method = cfg.method;

    let result: { graph: CausalGraph; sepSet?: Map<string, Set<string>>; pagEdges?: Map<string, string> };
    const t0 = performance.now();

    switch (method) {
      case 'fci':
        result = fciAlgorithm(data, args.nodes, { alpha: cfg.alpha });
        break;
      case 'ges':
        result = { graph: gesAlgorithm(data, args.nodes) };
        break;
      case 'notears':
        result = notearsAlgorithm(data, args.nodes, { lambda1: 0.05 });
        break;
      case 'golem':
        result = golemAlgorithm(data, args.nodes, { maxIter: 200 });
        break;
      case 'pc':
      default:
        result = pcAlgorithm(data, args.nodes, { alpha: cfg.alpha, stable: true });
    }

    const ms = Math.round(performance.now() - t0);
    const output = {
      algorithm: method,
      durationMs: ms,
      nodes: args.nodes.length,
      edges: result.graph.edges.length,
      isDAG: result.graph.isDAG(),
      graph: {
        edges: result.graph.edges.map(e => ({ from: e.source, to: e.target, weight: e.weight, directed: e.directed })),
      },
    };

    if (args.output) {
      const { writeFileSync } = require('fs');
      writeFileSync(args.output, JSON.stringify(output, null, 2));
    }
    console.log(JSON.stringify(output, null, 2));
  } catch (e: any) {
    console.error(`Error: ${e.message}`);
    process.exit(3);
  }
}

// ── Analyze ─────────────────────────────────────────────────────────
function execAnalyze(args: CliArgs, cfg: CliConfig): void {
  if (!args.file || !args.slis || args.slis.length === 0) {
    console.error('Usage: causal-analyzer analyze <file.json> --slis CPU,Lat [--method circa]');
    process.exit(2);
  }

  try {
    const raw = JSON.parse(readFileSync(args.file, 'utf-8'));
    const data = Array.isArray(raw.data) ? raw.data : (raw.normal ?? raw);
    const graphData = raw.graph ?? args.graph ? JSON.parse(args.graph ? readFileSync(args.graph, 'utf-8') : '{"nodes":[],"edges":[]}') : null;

    const g = new CausalGraph(graphData?.nodes ?? []);
    if (graphData?.edges) for (const e of graphData.edges) g.addEdge(e.from ?? e.source, e.to ?? e.target);

    const t0 = performance.now();

    let result: any;
    switch (cfg.method) {
      case 'bayesian': {
        const rca = new BayesianRCA({ engine: 'variable_elimination' });
        rca.train(g, data);
        result = rca.findRootCauses(args.slis);
        break;
      }
      case 'ht': {
        const { HTRCA } = require('./analyze/rca.js') as typeof import('./analyze/rca.js');
        const ht = new HTRCA();
        ht.train(g, new Matrix(data));
        result = ht.findRootCauses(args.slis, new Matrix(data));
        break;
      }
      case 'circa':
      default: {
        const pipeline = new CIRCAPipeline();
        pipeline.train(g, data);
        result = pipeline.analyze(data, args.slis);
      }
    }

    const ms = Math.round(performance.now() - t0);
    const output = {
      method: cfg.method,
      durationMs: ms,
      rootCauses: (result.rootCauses ?? []).map((r: any) => ({
        name: r.name, score: Math.round(r.score * 1000) / 1000,
        confidence: Math.round(r.confidence * 1000) / 1000, rank: r.rank,
      })),
    };

    if (args.output) {
      require('fs').writeFileSync(args.output, JSON.stringify(output, null, 2));
    }
    console.log(JSON.stringify(output, null, 2));
  } catch (e: any) {
    console.error(`Error: ${e.message}`);
    process.exit(3);
  }
}

// ── Benchmark ───────────────────────────────────────────────────────
function execBenchmark(args: CliArgs): void {
  const n = args.size ?? 200;
  console.log(`Causality Analyzer Benchmark Suite (n=${n})\n`);
  const results: Record<string, { ms: number; ops: string }> = {};

  // 1. PC algorithm
  { const nodes = ['A','B','C','D'], data = new Matrix(n, nodes.length);
    for (let r = 0; r < n; r++) for (let c = 0; c < nodes.length; c++) data.set(r, c, Math.random());
    const t0 = performance.now();
    pcAlgorithm(data, nodes, { alpha: 0.05 });
    results['pc_4node'] = { ms: Math.round(performance.now() - t0), ops: `${n} samples × 4 nodes` };
  }

  // 2. StatsDetector batch
  { const data = Array.from({ length: n }, () => [Math.random() * 10]);
    const t0 = performance.now();
    const d = new StatsDetector({ minSamples: 10 }); d.train(data);
    results['stats_detector'] = { ms: Math.round(performance.now() - t0), ops: `${n} samples` };
  }

  // 3. SPOT calibration
  { const t0 = performance.now();
    const s = new SPOTDetector({ initSize: Math.min(50, n - 1), q: 0.95 });
    for (let i = 0; i < n; i++) s.update(1 + Math.random() * 0.2);
    results['spot_calibration'] = { ms: Math.round(performance.now() - t0), ops: `${n} updates` };
  }

  // 4. d-separation
  { const g = new CausalGraph(['A','B','C','D','E','F','G','H','I','J']);
    for (let i = 0; i < 9; i++) g.addEdge(g.nodes[i]!, g.nodes[i+1]!);
    const t0 = performance.now();
    for (let i = 0; i < 100; i++) g.dSeparated('A', 'J', ['E']);
    results['dsep_10node'] = { ms: Math.round((performance.now() - t0) / 100 * 100) / 100, ops: '100 queries' };
  }

  // 5. solveLinear
  { const A = Array.from({ length: 50 }, () => Array.from({ length: 50 }, () => Math.random()));
    const b = Array.from({ length: 50 }, () => Math.random());
    const t0 = performance.now();
    for (let i = 0; i < 100; i++) {
      const { solveLinear } = require('@agentix-e/causality-analyzer-core');
      solveLinear(A.map(r => [...r]), [...b]);
    }
    results['solve_linear'] = { ms: Math.round((performance.now() - t0) / 100 * 100) / 100, ops: '100× 50D solves' };
  }

  console.log(JSON.stringify(results, null, 2));
}

// ── Serve ───────────────────────────────────────────────────────────
const metrics = new MetricsRegistry();

function execServe(args: CliArgs, cfg: CliConfig): void {
  const port = cfg.port;
  const enableMetrics = args.metrics === true;

  const server = createServer((req: IncomingMessage, res: ServerResponse) => {
    const url = req.url ?? '/';

    if (enableMetrics && url === '/metrics') {
      metrics.setGauge('ca_server_uptime_seconds', process.uptime());
      res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end(metrics.toPrometheus());
      return;
    }

    if (url === '/health' || url === '/') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        status: 'ok',
        version: '1.0.0',
        uptime: Math.round(process.uptime()),
        metrics: enableMetrics ? '/metrics' : undefined,
      }));
      return;
    }

    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found', endpoints: ['/health', ...(enableMetrics ? ['/metrics'] : [])] }));
  });

  server.on('error', (e: Error) => {
    console.error(`Server error: ${e.message}`);
    process.exit(4);
  });

  server.listen(port, () => {
    metrics.inc('ca_server_started_total');
    console.log(`Causality Analyzer API on http://localhost:${port}`);
    if (enableMetrics) console.log(`Metrics: http://localhost:${port}/metrics`);
  });

  process.on('SIGTERM', () => { server.close(); process.exit(0); });
  process.on('SIGINT', () => { server.close(); process.exit(0); });
}

main();
