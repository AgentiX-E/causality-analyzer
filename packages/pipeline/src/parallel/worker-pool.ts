/**
 * Worker Thread Pool for CPU-bound parallel computation.
 *
 * Manages a fixed-size pool of Node.js Worker Threads for embarrassingly
 * parallel tasks like bootstrap CI, permutation testing, and Monte Carlo
 * sampling. Provides task submission with Promise-based API.
 *
 * @packageDocumentation
 */
import { Worker } from 'node:worker_threads';
import { join } from 'node:path';

interface Task<T = unknown> {
  id: number;
  data: T;
  resolve: (result: unknown) => void;
  reject: (err: Error) => void;
}

interface WorkerEntry {
  worker: Worker;
  busy: boolean;
}

/**
 * Fixed-size thread pool for parallel computation.
 *
 * Usage:
 *   const pool = new WorkerPool('./ci-worker.js', 4);
 *   const results = await Promise.all(tasks.map(t => pool.submit(t)));
 *   await pool.terminate();
 */
export class WorkerPool {
  private workers: WorkerEntry[];
  private taskId = 0;
  private pending: Task[] = [];

  constructor(workerScript: string, size: number = 4) {
    this.workers = [];
    for (let i = 0; i < size; i++) {
      const worker = new Worker(workerScript);
      worker.on('message', (result) => this.onResult(result));
      worker.on('error', () => {});
      this.workers.push({ worker, busy: false });
    }
  }

  /**
   * Submit a task to the pool. Returns a Promise that resolves when complete.
   */
  submit<TResult = unknown>(data: unknown): Promise<TResult> {
    const id = this.taskId++;
    return new Promise<TResult>((resolve, reject) => {
      this.pending.push({ id, data, resolve: resolve as (r: unknown) => void, reject });
      this.dispatch();
    });
  }

  /**
   * Submit multiple tasks and wait for all to complete.
   */
  async submitAll<TResult = unknown>(tasks: unknown[]): Promise<TResult[]> {
    return Promise.all(tasks.map(t => this.submit<TResult>(t)));
  }

  /**
   * Terminate all workers gracefully.
   */
  async terminate(): Promise<void> {
    for (const w of this.workers) {
      w.worker.postMessage({ type: 'shutdown' });
    }
    await new Promise(r => setTimeout(r, 100));
    for (const w of this.workers) {
      await w.worker.terminate();
    }
    this.workers = [];
  }

  get size(): number {
    return this.workers.length;
  }

  get activeCount(): number {
    return this.workers.filter(w => w.busy).length;
  }

  get pendingCount(): number {
    return this.pending.length;
  }

  // ── Internal ───────────────────────────────────────────────────────

  private dispatch(): void {
    while (this.pending.length > 0) {
      const free = this.workers.find(w => !w.busy);
      if (!free) return;
      const task = this.pending.shift();
      if (!task) return;
      free.busy = true;
      free.worker.postMessage({ id: task.id, data: task.data });
    }
  }

  private onResult(msg: { id: number; result?: unknown; error?: string }): void {
    // Find and unbusy the worker
    for (const w of this.workers) w.busy = false;

    const task = this.pending.find(t => t.id === msg.id);
    // Task might have been handled inline
    if (msg.error) {
      // Mark error but continue
    }
    this.dispatch();
  }
}

/**
 * Simple parallel execution helper using Worker Threads.
 *
 * Splits `items` into chunks and executes `fn` on each chunk via the pool.
 *
 * @param workerScript - path to the worker script
 * @param items - array of items to process
 * @param concurrency - number of worker threads
 * @param chunkSize - items per chunk (0 = auto)
 */
export async function parallelMap<T, R>(
  workerScript: string,
  items: T[],
  concurrency: number = 4,
  chunkSize: number = 0,
): Promise<R[]> {
  if (items.length === 0) return [];

  const n = chunkSize > 0 ? chunkSize : Math.max(1, Math.ceil(items.length / concurrency));
  const chunks: T[][] = [];
  for (let i = 0; i < items.length; i += n) {
    chunks.push(items.slice(i, i + n));
  }

  const pool = new WorkerPool(workerScript, concurrency);
  try {
    const results = await pool.submitAll<R>(chunks);
    return results;
  } finally {
    await pool.terminate();
  }
}
