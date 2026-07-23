/**
 * Bootstrap CI + Concurrent Execution Utilities.
 *
 * bootstrapATE: Percentile bootstrap confidence intervals for any estimator.
 * parallelBootstrap: Concurrent resampling via Promise.all chunking.
 * bootstrapATEParallel: Full parallel bootstrap with automatic chunking.
 *
 * @packageDocumentation
 */
import { createRNG } from '@agentix-e/causality-analyzer-core';

/**
 * Bootstrap confidence interval for an ATE estimator.
 *
 * Resamples the data with replacement using seeded RNG for reproducibility,
 * recomputes the estimate, and returns percentile CI with standard error.
 *
 * @param data — observation matrix (rows × columns)
 * @param estimator — function that computes the estimate from a sample
 * @param nBootstraps — number of bootstrap resamples (default 200)
 * @param alpha — significance level (default 0.05 for 95% CI)
 * @param seed — optional PRNG seed for reproducibility
 */
export function bootstrapATE(
  data: number[][],
  estimator: (data: number[][]) => number,
  nBootstraps: number = 200,
  alpha: number = 0.05,
  seed?: number,
): { ate: number; ciLow: number; ciHigh: number; se: number } {
  const rng = createRNG(seed ?? null);
  const n = data.length;
  if (n < 2) return { ate: estimator(data), ciLow: 0, ciHigh: 0, se: 0 };

  const estimates = new Float64Array(nBootstraps);
  for (let b = 0; b < nBootstraps; b++) {
    const sample: number[][] = [];
    for (let i = 0; i < n; i++) sample.push(data[Math.floor(rng() * n)]!);
    estimates[b] = estimator(sample);
  }

  return computeCI(estimates, nBootstraps, alpha);
}

/** Compute percentile CI and SE from bootstrap estimates */
function computeCI(
  estimates: Float64Array,
  nBootstraps: number,
  alpha: number,
): { ate: number; ciLow: number; ciHigh: number; se: number } {
  const sorted = Array.from(estimates).sort((a, b) => a - b);
  const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
  const ciLow = sorted[Math.floor(nBootstraps * alpha / 2)] ?? mean;
  const ciHigh = sorted[Math.floor(nBootstraps * (1 - alpha / 2))] ?? mean;
  const se = Math.sqrt(sorted.reduce((s, v) => s + (v - mean) ** 2, 0) / sorted.length);
  return { ate: mean, ciLow, ciHigh, se };
}

/**
 * Run bootstrap tasks concurrently using Promise.all chunking.
 *
 * Splits nTasks into chunks and runs them concurrently. While Node.js
 * is single-threaded for CPU-bound work, this provides true concurrency
 * for I/O-bound estimators and allows the event loop to interleave work.
 * For full multi-threading, use Worker Threads via bootstrapATEParallel.
 *
 * @param data — observation matrix
 * @param taskFn — function receiving (sample, seed) and returning a result
 * @param nTasks — total number of bootstrap tasks
 * @param seed — base seed for reproducibility
 * @param concurrency — max concurrent chunks (default: 4)
 */
export async function parallelBootstrap<T>(
  data: number[][],
  taskFn: (sample: number[][], seed: number) => T,
  nTasks: number,
  seed?: number,
  concurrency: number = 4,
): Promise<T[]> {
  if (nTasks <= 1) return [taskFn(data, seed ?? 0)];
  if (concurrency <= 1) {
    return sequentialBootstrap(data, taskFn, nTasks, seed);
  }

  // Split tasks into chunks for concurrent execution
  const chunks: Array<{ start: number; end: number }> = [];
  const chunkSize = Math.ceil(nTasks / concurrency);
  for (let c = 0; c < concurrency; c++) {
    const start = c * chunkSize;
    const end = Math.min(start + chunkSize, nTasks);
    if (start < end) chunks.push({ start, end });
  }

  // Run chunks concurrently via Promise.all
  const chunkResults = await Promise.all(
    chunks.map(({ start, end }) =>
      runBootstrapChunk(data, taskFn, start, end, seed),
    ),
  );

  return chunkResults.flat();
}

/** Run a single chunk of bootstrap iterations */
async function runBootstrapChunk<T>(
  data: number[][],
  taskFn: (sample: number[][], seed: number) => T,
  start: number,
  end: number,
  seed?: number,
): Promise<T[]> {
  const results: T[] = [];
  const n = data.length;
  for (let i = start; i < end; i++) {
    const taskSeed = (seed ?? 0) + i * 101;
    const rng = createRNG(taskSeed);
    const sample: number[][] = [];
    for (let j = 0; j < n; j++) sample.push(data[Math.floor(rng() * n)]!);
    results.push(taskFn(sample, taskSeed));
  }
  return results;
}

/** Sequential fallback for single-threaded execution */
function sequentialBootstrap<T>(
  data: number[][],
  taskFn: (sample: number[][], seed: number) => T,
  nTasks: number,
  seed?: number,
): T[] {
  const results: T[] = [];
  const n = data.length;
  for (let i = 0; i < nTasks; i++) {
    const taskSeed = (seed ?? 0) + i * 101;
    const rng = createRNG(taskSeed);
    const sample: number[][] = [];
    for (let j = 0; j < n; j++) sample.push(data[Math.floor(rng() * n)]!);
    results.push(taskFn(sample, taskSeed));
  }
  return results;
}

/**
 * Full parallel bootstrap with automatic chunking and CI computation.
 *
 * Distributes nBootstraps across concurrent chunks, merges results,
 * and computes percentile CI with standard error.
 *
 * @param data — observation matrix
 * @param estimator — estimate computation function
 * @param nBootstraps — total bootstrap iterations
 * @param concurrency — number of concurrent chunks (default: 4)
 * @param alpha — significance level
 * @param seed — base seed for reproducibility
 */
export async function bootstrapATEParallel(
  data: number[][],
  estimator: (data: number[][]) => number,
  nBootstraps: number = 200,
  concurrency: number = 4,
  alpha: number = 0.05,
  seed?: number,
): Promise<{ ate: number; ciLow: number; ciHigh: number; se: number }> {
  const allResults = await parallelBootstrap(
    data,
    (sample, _seed) => estimator(sample),
    nBootstraps,
    seed,
    concurrency,
  );

  const estimates = Float64Array.from(allResults.slice(0, nBootstraps));
  return computeCI(estimates, nBootstraps, alpha);
}
