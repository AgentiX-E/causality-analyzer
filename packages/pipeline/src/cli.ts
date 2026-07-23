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
 * @packageDocumentation
 */
import { readFileSync } from 'fs';
import { createServer, IncomingMessage, ServerResponse } from 'http';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './graph/causal-graph.js';
import { pcAlgorithm } from './graph/pc.js';
import { CIRCAPipeline } from './analyze/circa.js';

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
    const a = argv[i]!;
    if (a === '--nodes' && i + 1 < argv.length) { args.nodes = argv[++i]!.split(','); }
    else if (a === '--slis' && i + 1 < argv.length) { args.slis = argv[++i]!.split(','); }
    else if (a === '--port' && i + 1 < argv.length) { args.port = parseInt(argv[++i]!, 10); }
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
  serve    --port 3000                  REST API server (health only)
  version                              Show version
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
  const raw = JSON.parse(readFileSync(args.file, 'utf-8'));
  const data = new Matrix(raw.data ?? raw);
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
  const raw = JSON.parse(readFileSync(args.file, 'utf-8'));
  const data = raw.data ?? raw;
  const graphData = raw.graph ?? { nodes: [], edges: [] };

  const g = new CausalGraph(graphData.nodes);
  for (const e of graphData.edges) { g.addEdge(e.from, e.to); }

  const pipeline = new CIRCAPipeline();
  pipeline.train(g, data.normal ?? data);
  const result = pipeline.analyze(data.anomaly ?? data, args.slis);

  console.log(JSON.stringify({
    rootCauses: result.rootCauses.map(
      (r: { name: string; score: number; rank: number }) =>
        ({ name: r.name, score: r.score, rank: r.rank })),
  }, null, 2));
}

function execServe(args: CliArgs): void {
  const port = args.port ?? 3000;
  const server = createServer((_req: IncomingMessage, res: ServerResponse) => {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', version: '1.0.0' }));
  });
  server.listen(port, () => {
    console.log(`Causality Analyzer REST API listening on http://localhost:${port}`);
  });
}

main();
