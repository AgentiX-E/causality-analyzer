/**
 * Worker Thread for Bootstrap CI computation.
 *
 * Loaded by WorkerPool to compute bootstrap confidence intervals
 * in parallel across multiple CPU cores.
 *
 * Message format:
 *   in:  { data: { samples: number[][], nBootstrap: number } }
 *   out: { id: number, result: { mean: number, ci: [number, number] } }
 */
import { parentPort } from 'node:worker_threads';

function bootstrapMean(data: number[][], nBootstrap: number): { mean: number; ci: [number, number] } {
  const n = data.length;
  const p = data[0]?.length ?? 0;
  const estimates: number[] = [];

  for (let b = 0; b < nBootstrap; b++) {
    // Resample with replacement
    const sample: number[][] = [];
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(Math.random() * n);
      sample.push([...(data[idx] ?? [])]);
    }

    // Compute mean per column, then average
    let sum = 0;
    for (let j = 0; j < p; j++) {
      let col = 0;
      for (let i = 0; i < n; i++) col += sample[i]?.[j] ?? 0;
      sum += col / n;
    }
    estimates.push(sum / p);
  }

  estimates.sort((a, b) => a - b);
  const mean = estimates.reduce((s, v) => s + v, 0) / estimates.length;
  const ciLow = estimates[Math.floor(estimates.length * 0.025)] ?? mean;
  const ciHigh = estimates[Math.floor(estimates.length * 0.975)] ?? mean;

  return { mean, ci: [ciLow, ciHigh] };
}

parentPort?.on('message', (msg: { id: number; data?: { samples: number[][]; nBootstrap: number } }) => {
  if (!msg.data) {
    parentPort?.postMessage({ id: msg.id, result: { mean: 0, ci: [0, 0] } });
    return;
  }
  const result = bootstrapMean(msg.data.samples, msg.data.nBootstrap);
  parentPort?.postMessage({ id: msg.id, result });
});
