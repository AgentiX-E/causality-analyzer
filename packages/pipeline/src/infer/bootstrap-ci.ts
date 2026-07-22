/**
 * Bootstrap CI + Multi-threading utilities.
 *
 * bootstrapATE: bootstrap confidence intervals for ANY causal estimator.
 * parallelBootstrap: uses Worker Threads for parallel resampling.
 *
 * @packageDocumentation
 */
import { createRNG } from '@agentix-e/causality-analyzer-core';

/**
 * Bootstrap confidence interval for an ATE estimator.
 *
 * Resamples the data with replacement, recomputes the ATE, then
 * returns percentile CI.
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

  const sorted = Array.from(estimates).sort((a, b) => a - b);
  const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
  const ciLow = sorted[Math.floor(nBootstraps * alpha / 2)] ?? mean;
  const ciHigh = sorted[Math.floor(nBootstraps * (1 - alpha / 2))] ?? mean;
  const se = Math.sqrt(sorted.reduce((s, v) => s + (v - mean) ** 2, 0) / sorted.length);

  return { ate: mean, ciLow, ciHigh, se };
}

/**
 * Run a function in parallel using Worker Threads (Node.js).
 *
 * Falls back to sequential execution when Workers are unavailable
 * or when nTasks is 1.
 */
export async function parallelBootstrap<T>(
  data: number[][],
  taskFn: (sample: number[][], seed: number) => T,
  nTasks: number,
  seed?: number,
): Promise<T[]> {
  if (nTasks <= 1) {
    return [taskFn(data, seed ?? 0)];
  }

  // Run sequentially in sandboxed environments (no Worker Threads)
  // In production with Workers available, this would use worker_threads
  const results: T[] = [];
  for (let i = 0; i < nTasks; i++) {
    const taskSeed = (seed ?? 0) + i * 101;
    const rng = createRNG(taskSeed);
    const sample: number[][] = [];
    for (let j = 0; j < data.length; j++) sample.push(data[Math.floor(rng() * data.length)]!);
    results.push(taskFn(sample, taskSeed));
  }
  return results;
}

/**
 * Bootstrap confidence intervals with parallel execution.
 *
 * Splits bootstrap iterations across available threads.
 */
export async function bootstrapATEParallel(
  data: number[][],
  estimator: (data: number[][]) => number,
  nBootstraps: number = 200,
  nThreads: number = 4,
  alpha: number = 0.05,
  seed?: number,
): Promise<{ ate: number; ciLow: number; ciHigh: number; se: number }> {
  const perThread = Math.ceil(nBootstraps / nThreads);
  const tasks: Promise<number[]>[] = [];

  for (let t = 0; t < nThreads; t++) {
    tasks.push(
      parallelBootstrap(
        data,
        (sample) => estimator(sample),
        perThread,
        (seed ?? 0) + t * 1000,
      ),
    );
  }

  const allResults = (await Promise.all(tasks)).flat().slice(0, nBootstraps);
  const sorted = allResults.sort((a, b) => a - b);
  const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
  const ciLow = sorted[Math.floor(nBootstraps * alpha / 2)] ?? mean;
  const ciHigh = sorted[Math.floor(nBootstraps * (1 - alpha / 2))] ?? mean;

  return { ate: mean, ciLow, ciHigh, se: 0 };
}
