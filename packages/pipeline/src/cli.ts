#!/usr/bin/env node
/**
 * Causality Analyzer CLI — command-line interface for causal analysis.
 *
 * Usage:
 *   npx causality-analyzer discover <data.json> --nodes CPU,Memory,Latency
 *   npx causality-analyzer analyze <anomaly.json> --slis CPU,Latency
 *   npx causality-analyzer serve --port 3000
 *
 * @packageDocumentation
 */
import { readFileSync } from 'fs';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../graph/causal-graph.js';
import { pcAlgorithm } from '../graph/pc.js';
import { CIRCAPipeline } from '../analyze/circa.js';

function main(): void {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.log(`Causality Analyzer CLI v1.0.0

Commands:
  discover <file> --nodes <names>  Run PC causal discovery
  analyze  <file> --slis <names>   Run CIRCA root cause analysis
  serve    --port <N>              Start REST API server
  version                         Show version
`);
    process.exit(0);
  }

  const cmd = args[0];
  switch (cmd) {
    case 'discover': runDiscover(args.slice(1)); break;
    case 'analyze': runAnalyze(args.slice(1)); break;
    case 'serve': runServe(args.slice(1)); break;
    case 'version': console.log('v1.0.0'); break;
    default: console.error(`Unknown command: ${cmd}`); process.exit(1);
  }
}

function runDiscover(args: string[]): void {
  const file = args[0];
  const nodes = parseFlag(args, '--nodes')?.split(',') ?? [];
  if (!file || nodes.length === 0) {
    console.error('Usage: discover <file.json> --nodes A,B,C');
    process.exit(1);
  }

  const raw = JSON.parse(readFileSync(file, 'utf-8'));
  const data = new Matrix(raw.data ?? raw);
  const result = pcAlgorithm(data, nodes, { alpha: 0.05, stable: true });
  console.log(JSON.stringify({
    edges: result.graph.edges.map(e => ({ from: e.source, to: e.target, weight: e.weight })),
    isDAG: result.graph.isDAG(),
  }, null, 2));
}

function runAnalyze(args: string[]): void {
  const file = args[0];
  const slis = parseFlag(args, '--slis')?.split(',') ?? [];
  if (!file || slis.length === 0) {
    console.error('Usage: analyze <file.json> --slis CPU,Latency');
    process.exit(1);
  }

  const raw = JSON.parse(readFileSync(file, 'utf-8'));
  const data = raw.data ?? raw;
  const graphData = raw.graph ?? { nodes: [], edges: [] };

  const g = new CausalGraph(graphData.nodes);
  for (const e of graphData.edges) g.addEdge(e.from, e.to);

  const pipeline = new CIRCAPipeline();
  pipeline.train(g, data.normal ?? data);
  const result = pipeline.analyze(data.anomaly ?? data, slis);

  console.log(JSON.stringify({
    rootCauses: result.rootCauses.map(r => ({ name: r.name, score: r.score, rank: r.rank })),
  }, null, 2));
}

function runServe(args: string[]): void {
  const port = parseInt(parseFlag(args, '--port') ?? '3000', 10);

  // Lightweight HTTP server using Node.js built-in http module
  const http = require('http');
  const server = http.createServer((_req: any, res: any) => {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', version: '1.0.0' }));
  });

  server.listen(port, () => {
    console.log(`Causality Analyzer REST API listening on http://localhost:${port}`);
  });
}

function parseFlag(args: string[], flag: string): string | undefined {
  const idx = args.indexOf(flag);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : undefined;
}

main();
