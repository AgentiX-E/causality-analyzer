/**
 * Parallel algorithm runner for multi-restart discovery.
 *
 * Uses Promise.all for concurrent execution of independent restarts
 * in BOSS, GRaSP, and other multi-start algorithms. Node.js's
 * single-threaded nature means this provides throughput benefits
 * for I/O-heavy or subprocess-based discovery, not CPU parallelism.
 *
 * For true CPU parallelism, use WorkerPool from parallel/worker-pool.js.
 *
 * @packageDocumentation
 */

/**
 * Run a discovery function with multiple random seeds concurrently.
 * Selects the best result by minimizing a score function.
 *
 * @param runner — function (seed) → { result, score }
 * @param numRuns — number of parallel runs
 * @param seedBase — base seed (runs use seedBase + 0, 1, ..., numRuns-1)
 * @returns best result
 */
export async function parallelRestart<T>(
  runner: (seed: number) => Promise<{ result: T; score: number }>,
  numRuns: number,
  seedBase: number = 42,
): Promise<T> {
  const tasks = Array.from({ length: numRuns }, (_, i) =>
    runner(seedBase + i),
  );
  const results = await Promise.all(tasks);
  let best = results[0];
  for (const r of results) {
    if (r.score < best.score) best = r;
  }
  return best.result;
}

/**
 * Synchronous version for CPU-bound algorithms.
 */
export function sequentialRestart<T>(
  runner: (seed: number) => { result: T; score: number },
  numRuns: number,
  seedBase: number = 42,
): T {
  let bestScore = Infinity;
  let bestResult: T | null = null;

  for (let i = 0; i < numRuns; i++) {
    const { result, score } = runner(seedBase + i);
    if (score < bestScore) {
      bestScore = score;
      bestResult = result;
    }
  }

  return bestResult!;
}
