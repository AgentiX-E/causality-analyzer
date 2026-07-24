/**
 * Parallel Execution — Worker Threads for CPU-bound causal inference.
 *
 * Node.js is single-threaded by default. Worker Threads enable true
 * parallel execution for bootstrap resampling, permutation tests,
 * and chunked causal discovery on multi-core machines.
 *
 * Two modes:
 * 1. Direct parallel: partition work, run in workers, merge results
 * 2. Chunked PC: split data into overlapping chunks, run PC on each, merge skeletons
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { pcAlgorithm } from './graph/pc.js';
import type { CausalEdge } from '@agentix-e/causality-analyzer-core';

// ── Parallel Bootstrap ──────────────────────────────────────────────
export interface ParallelBootstrapConfig {
  nBootstrap?: number;
  nWorkers?: number;
  seed?: number;
}

/**
 * Run a CPU-bound task in parallel across Worker Threads.
 *
 * The `task` function must be self-contained (no closure dependencies)
 * because Workers have isolated memory.
 *
 * @param taskFn — function body as string (runs in worker context)
 * @param args — per-worker arguments
 * @param combine — reducer to merge worker results
 * @param nWorkers — number of workers (default: cpuCount)
 */
export async function parallelMap<T, W, R = W>(
  args: T[],
  taskScript: string,
  combine: (results: W[]) => R,
  nWorkers?: number,
): Promise<R> {
  const workerCount = nWorkers ?? Math.max(1, Math.ceil(require('os').cpus().length / 2));
  if (workerCount <= 1 || args.length <= 1) {
    // eslint-disable-next-line no-eval
    const taskFn = eval(`(async (data) => { ${taskScript} })`) as (data: T) => Promise<W>;
    const results: W[] = [];
    for (const arg of args) results.push(await taskFn(arg));
    return combine(results);
  }

  const chunkSize = Math.ceil(args.length / workerCount);
  const workers: Promise<W>[] = [];

  for (let w = 0; w < workerCount; w++) {
    const chunk = args.slice(w * chunkSize, (w + 1) * chunkSize);
    if (chunk.length === 0) break;
    workers.push(runInWorker<T, W>(chunk, taskScript));
  }

  const results = await Promise.all(workers);
  return combine(results);
}

function runInWorker<T, R>(data: T[], taskScript: string): Promise<R> {
  const { Worker } = require('worker_threads') as typeof import('worker_threads');
  return new Promise((resolve, reject) => {
    const workerCode = `
      const { parentPort } = require('worker_threads');
      const { Matrix } = require('ml-matrix');
      parentPort.on('message', async (payload) => {
        try {
          const task = async (data) => { ${taskScript} };
          const results = [];
          for (const item of payload) {
            results.push(await task(item));
          }
          parentPort.postMessage({ results });
        } catch (e) {
          parentPort.postMessage({ error: e.message });
        }
      });
    `;
    const worker = new Worker(workerCode, { eval: true });
    worker.on('message', (msg: { results?: R; error?: string }) => {
      if (msg.error) reject(new Error(msg.error));
      else resolve(msg.results!);
    });
    worker.on('error', reject);
    worker.postMessage(data);
  });
}

// ── Parallel Permutation Test ───────────────────────────────────────
export interface ParallelPermutationConfig {
  nPermutations?: number;
  nWorkers?: number;
  seed?: number;
}

/**
 * Run permutation test in parallel across workers.
 * Each worker processes a subset of permutations.
 *
 * Returns p-value: (countGreater + 1) / (nPermutations + 1)
 */
export async function parallelPermutationTest(
  observedStat: number,
  data: Matrix,
  xIdx: number,
  yIdx: number,
  zIndices: number[],
  nPermutations: number = 200,
  nWorkers?: number,
): Promise<number> {
  const taskFn = `
    const { kciTest } = require('./graph/kci.js');
    let count = 0;
    for (let p = 0; p < data.nPerms; p++) {
      // Shuffle Y column
      const yCol = new Float64Array(data.yData);
      for (let i = yCol.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [yCol[i], yCol[j]] = [yCol[j], yCol[i]];
      }
      // Rebuild data matrix with shuffled Y
      const permData = new Matrix(data.nRows, data.nCols);
      for (let r = 0; r < data.nRows; r++) {
        for (let c = 0; c < data.nCols; c++) {
          permData.set(r, c, c === data.yIdx ? yCol[r] : data.matrix[r * data.nCols + c]);
        }
      }
      const pval = kciTest(permData, data.xIdx, data.yIdx, data.zIndices, { nPermutations: 0 });
      if (pval <= data.observedStat) count++;
    }
    return { count, total: data.nPerms };
  `;

  const nWorkersActual = nWorkers ?? Math.ceil(require('os').cpus().length / 2);
  const perWorker = Math.ceil(nPermutations / nWorkersActual);
  const workerArgs: Array<{
    nPerms: number;
    matrix: Float64Array;
    nRows: number;
    nCols: number;
    xIdx: number;
    yIdx: number;
    yData: number[];
    zIndices: number[];
    observedStat: number;
  }> = [];

  const flat = new Float64Array(data.rows * data.columns);
  for (let r = 0; r < data.rows; r++) for (let c = 0; c < data.columns; c++) flat[r * data.columns + c] = data.get(r, c);

  const yCol: number[] = [];
  for (let r = 0; r < data.rows; r++) yCol.push(data.get(r, yIdx));

  for (let w = 0; w < nWorkersActual; w++) {
    const count = Math.min(perWorker, nPermutations - w * perWorker);
    if (count <= 0) break;
    workerArgs.push({
      nPerms: count,
      matrix: flat,
      nRows: data.rows,
      nCols: data.columns,
      xIdx, yIdx,
      yData: [...yCol],
      zIndices,
      observedStat,
    });
  }

  const combine = (results: Array<{ count: number; total: number }>) => {
    const sum = results.reduce((s, r) => s + r.count, 0);
    const total = results.reduce((s, r) => s + r.total, 0);
    return (sum + 1) / (total + 1);
  };

  return parallelMap<typeof workerArgs[0], { count: number; total: number }, number>(
    workerArgs, taskFn,
    (results) => {
      const sum = results.reduce((s, r) => s + r.count, 0);
      const total = results.reduce((s, r) => s + r.total, 0);
      return (sum + 1) / (total + 1);
    },
    nWorkersActual,
  );
}

// ── Chunked PC Algorithm ────────────────────────────────────────────
/**
 * Chunked PC: split large dataset into overlapping chunks, run PC on each,
 * and merge skeleton results by voting on edge presence.
 *
 * For datasets with N > 10,000 rows, naive PC becomes O(N·d²·2^k) which
 * is prohibitive. Chunking reduces per-chunk N while preserving CI test
 * power through overlap and voting.
 *
 * @param data — full data matrix
 * @param nodeNames — variable names
 * @param chunkSize — rows per chunk (default 2000)
 * @param overlap — overlap between chunks (default 500)
 * @param alpha — CI test significance level
 * @returns merged causal graph with edge confidence
 */
export async function chunkedPC(
  data: Matrix,
  nodeNames: string[],
  chunkSize: number = 2000,
  overlap: number = 500,
  alpha: number = 0.05,
): Promise<{ graph: { nodes: string[]; edges: Array<CausalEdge & { voteCount: number }> }; convergence: number }> {
  const N = data.rows;
  const d = nodeNames.length;
  if (N <= chunkSize) {
    const result = pcAlgorithm(data, nodeNames, { alpha });
    return {
      graph: {
        nodes: [...result.graph.nodes],
        edges: result.graph.edges.map(e => ({ ...e, voteCount: 1 })),
      },
      convergence: 1,
    };
  }

  // Create overlapping chunks
  const chunks: Matrix[] = [];
  for (let start = 0; start < N - overlap; start += chunkSize - overlap) {
    const end = Math.min(start + chunkSize, N);
    const chunk = new Matrix(end - start, d);
    for (let r = start; r < end; r++) {
      for (let c = 0; c < d; c++) chunk.set(r - start, c, data.get(r, c));
    }
    chunks.push(chunk);
    if (end >= N) break;
  }

  // Run PC on each chunk
  const edgeVotes = new Map<string, number>();
  let totalVotes = 0;

  for (const chunk of chunks) {
    totalVotes++;
    const result = pcAlgorithm(chunk, nodeNames, { alpha });
    const seen = new Set<string>();
    for (const edge of result.graph.edges) {
      const key = `${edge.source}→${edge.target}`;
      seen.add(key);
      edgeVotes.set(key, (edgeVotes.get(key) ?? 0) + 1);
    }
  }

  // Consolidate edges with vote threshold (> 50%)
  const threshold = Math.ceil(totalVotes * 0.5);
  const edges: Array<CausalEdge & { voteCount: number }> = [];
  for (const [key, votes] of edgeVotes) {
    if (votes >= threshold) {
      const [source, target] = key.split('→');
      edges.push({ source: source!, target: target!, weight: votes / totalVotes, directed: true, voteCount: votes });
    }
  }

  return {
    graph: { nodes: [...nodeNames], edges },
    convergence: chunks.length > 0 ? totalVotes / chunks.length : 0,
  };
}
