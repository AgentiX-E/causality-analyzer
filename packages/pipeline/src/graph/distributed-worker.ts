/**
 * Stateless Distributed Worker — executes CI test batches against a
 * SQL cluster without any local persistent state.
 *
 * Workers are pure computation units. They receive CI task batches from
 * the Coordinator, fetch the required data columns from the SQL cluster
 * (via the IRelationalStore interface), compute local CI test statistics,
 * and return results. No graph writing — that's the Coordinator's job.
 *
 * All state lives in external Raft clusters:
 *   - SQL cluster: time-series metrics, regression models, CPTs
 *   - Graph cluster: causal graphs, versions, similarity search
 *
 * @packageDocumentation
 */

import { fisherZTest, type CITestResult } from '@agentix-e/causality-analyzer-core';
import type {
  DistributedCITask,
  DistributedCIResult,
  DistributedCITaskBatch,
  DistributedCIBatchResult,
  VectorClock,
} from '@agentix-e/causality-analyzer-core';
import { ciTest } from './ci-backend.js';

/** Configuration for a stateless distributed worker */
export interface WorkerConfig {
  /** Unique worker identifier */
  readonly workerId: string;
  /** CI test backend to use */
  readonly ciBackend: 'parcorr' | 'cmiknn' | 'gsquared';
}

/**
 * Stateless distributed worker for parallel CI test execution.
 *
 * Has zero local persistent state. All data access goes through the
 * provided data matrix (loaded from SQL cluster by the caller).
 */
export class StatelessDistributedWorker {
  readonly workerId: string;
  private readonly ciBackend: WorkerConfig['ciBackend'];
  private currentClock: Record<string, number>;

  constructor(config: WorkerConfig) {
    this.workerId = config.workerId;
    this.ciBackend = config.ciBackend;
    this.currentClock = {};
  }

  /**
   * Execute a batch of CI test tasks.
   *
   * @param batch — task batch with required column indices
   * @param data — data matrix fetched from SQL cluster (full or column subset)
   * @returns batch result with per-task CI results and updated vector clock
   */
  executeBatch(
    batch: DistributedCITaskBatch,
    data: number[][],
  ): DistributedCIBatchResult {
    const startTime = Date.now();
    const results: DistributedCIResult[] = [];

    for (const task of batch.tasks) {
      const start = Date.now();
      let result: CITestResult;

      if (task.condSet.length === 0 && this.ciBackend === 'parcorr' && task.lag === 0) {
        // Fast path: Fisher Z for unconditional contemporaneous tests
        const pValue = fisherZTest(data, task.source, task.target, []);
        result = { pValue, testStatistic: 0 };
      } else {
        result = ciTest(
          data,
          task.source,
          task.target,
          [...task.condSet],
          this.ciBackend,
          { knnK: 5, nPermutations: 100 },
        );
      }

      results.push({
        taskId: task.taskId,
        workerId: this.workerId,
        source: task.source,
        target: task.target,
        lag: task.lag,
        condSet: task.condSet,
        pValue: result.pValue,
        testStatistic: result.testStatistic,
        sampleSize: data.length,
        runtimeMs: Date.now() - start,
      });
    }

    // Advance vector clock
    this.currentClock[this.workerId] = (this.currentClock[this.workerId] ?? 0) + 1;

    return {
      batchId: batch.batchId,
      workerId: this.workerId,
      results,
      vectorClock: { ...this.currentClock },
      batchRuntimeMs: Date.now() - startTime,
    };
  }

  /**
   * Get the current vector clock for this worker.
   */
  getClock(): VectorClock {
    return { ...this.currentClock };
  }

  /**
   * Synchronize the worker's clock from the Coordinator.
   * Used after the Coordinator has merged results.
   */
  syncClock(clock: VectorClock): void {
    const merged: Record<string, number> = {};
    for (const wid of new Set([...Object.keys(this.currentClock), ...Object.keys(clock)])) {
      merged[wid] = Math.max(this.currentClock[wid] ?? 0, clock[wid] ?? 0);
    }
    this.currentClock = merged;
  }
}

/**
 * Run a single CI test task to compute a p-value.
 * Used standalone for testing/validation.
 *
 * @param task — CI test task
 * @param data — data matrix
 * @param workerId — worker identifier for result provenance
 */
export function executeCITask(
  task: DistributedCITask,
  data: number[][],
  workerId: string,
): DistributedCIResult {
  const start = Date.now();
  const result = ciTest(
    data,
    task.source,
    task.target,
    [...task.condSet],
    task.ciBackend,
    { knnK: 5, nPermutations: 100 },
  );

  return {
    taskId: task.taskId,
    workerId,
    source: task.source,
    target: task.target,
    lag: task.lag,
    condSet: task.condSet,
    pValue: result.pValue,
    testStatistic: result.testStatistic,
    sampleSize: data.length,
    runtimeMs: Date.now() - start,
  };
}
