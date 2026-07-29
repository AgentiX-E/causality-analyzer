/**
 * Distributed Coordinator — orchestrates stateless workers for parallel
 * causal discovery, merges CI test results, and resolves graph conflicts.
 *
 * The Coordinator is the single point of coordination for distributed
 * causal discovery. It:
 *   1. Partitions the CI test matrix into batches
 *   2. Dispatches batches to stateless workers
 *   3. Collects results and merges p-values via Fisher's method
 *   4. Resolves graph conflicts using vector clocks
 *   5. Writes the final global causal graph to the Graph cluster
 *
 * All persistent state lives in external Raft clusters:
 *   - SQL cluster: data for CI tests (fetched by workers)
 *   - Graph cluster: causal graphs and versions (written by coordinator)
 *
 * @packageDocumentation
 */

import type {
  DistributedCITask,
  DistributedCIResult,
  DistributedCITaskBatch,
  DistributedCIBatchResult,
  DistributedDiscoveryConfig,
  DistributedGraphVersion,
  VectorClock,
  CIBackend,
} from '@agentix-e/causality-analyzer-core';
import { compareClocks, mergeClocks, incrementClock } from '@agentix-e/causality-analyzer-core';
import { partitionCITasks, mergeDistributedCIResults } from './distributed-ci.js';
import { StatelessDistributedWorker } from './distributed-worker.js';

/** Per-worker task queue with results */
interface WorkerState {
  worker: StatelessDistributedWorker;
  pendingBatches: DistributedCITaskBatch[];
  completedResults: DistributedCIBatchResult[];
}

/** Edge decision after merging worker results */
interface MergedEdge {
  source: number;
  target: number;
  lag: number;
  pValue: number;
  strength: number;
  consensus: number;
  keepEdge: boolean;
}

/**
 * Distributed Coordinator for parallel causal discovery.
 *
 * Orchestrates a pool of stateless workers, partitions CI tasks,
 * collects and merges results, and produces the final causal graph.
 */
export class DistributedCoordinator {
  private readonly config: DistributedDiscoveryConfig;
  private readonly workers: Map<string, StatelessDistributedWorker> = new Map();
  private coordinatorClock: Record<string, number> = {};

  constructor(config: DistributedDiscoveryConfig) {
    this.config = config;

    // Initialize stateless workers
    for (let i = 0; i < config.workers.count; i++) {
      const workerId = `worker-${i}`;
      const worker = new StatelessDistributedWorker({
        workerId,
        ciBackend: config.workers.ciBackend,
      });
      this.workers.set(workerId, worker);
    }
  }

  /**
   * Run a full distributed causal discovery cycle.
   *
   * @param nodeCount — number of variables
   * @param tauMax — maximum time lag (0 for IID data)
   * @param dataMatrix — data loaded from SQL cluster (row-major: [sample][variable])
   * @returns merged edge decisions ready for graph construction
   */
  runDiscovery(
    nodeCount: number,
    tauMax: number,
    dataMatrix: number[][],
  ): MergedEdge[] {
    // Step 1: Partition CI test matrix into batches
    const batches = partitionCITasks(nodeCount, tauMax, this.config);

    // Step 2: Dispatch batches to workers (parallel execution)
    const batchResults = this.dispatchBatches(batches, dataMatrix);

    // Step 3: Group results by (source, target, lag) pair
    const grouped = this.groupResultsByEdge(batchResults);

    // Step 4: Merge results for each edge using configured strategy
    const mergedEdges: MergedEdge[] = [];
    for (const [key, results] of grouped) {
      const [source, target, lag] = key.split('|').map(Number) as [number, number, number];
      const merged = mergeDistributedCIResults(
        results,
        this.config.alpha,
        this.config.coordinator.mergeStrategy,
      );

      if (merged.keepEdge) {
        mergedEdges.push({
          source: source!,
          target: target!,
          lag: lag!,
          pValue: merged.pValue,
          strength: merged.testStatistic,
          consensus: merged.consensus,
          keepEdge: true,
        });
      }
    }

    // Step 5: Advance coordinator clock
    this.coordinatorClock = incrementClock(this.coordinatorClock, 'coordinator');

    return mergedEdges;
  }

  /**
   * Dispatch task batches to workers and collect results.
   *
   * @internal
   */
  private dispatchBatches(
    batches: DistributedCITaskBatch[],
    data: number[][],
  ): DistributedCIBatchResult[] {
    const results: DistributedCIBatchResult[] = [];

    // Execute batches sequentially for deterministic testing;
    // in production, these would be dispatched via Worker Threads or HTTP.
    for (const batch of batches) {
      const worker = this.workers.get(batch.workerId);
      if (!worker) continue;

      const result = worker.executeBatch(batch, data);
      results.push(result);
    }

    return results;
  }

  /**
   * Group CI results by (source, target, lag) for merging.
   *
   * @internal
   */
  private groupResultsByEdge(
    results: DistributedCIBatchResult[],
  ): Map<string, DistributedCIResult[]> {
    const map = new Map<string, DistributedCIResult[]>();

    for (const batch of results) {
      for (const r of batch.results) {
        const key = `${r.source}|${r.target}|${r.lag}`;
        if (!map.has(key)) map.set(key, []);
        map.get(key)!.push(r);
      }
    }

    return map;
  }

  /**
   * Get the coordinator's current vector clock.
   */
  getClock(): VectorClock {
    return { ...this.coordinatorClock };
  }

  /**
   * Get the number of active workers.
   */
  get workerCount(): number {
    return this.workers.size;
  }

  /**
   * Resolve a conflict between two graph versions using vector clocks.
   *
   * If version A happened-before B, return B (most recent).
   * If versions are concurrent, return null to signal manual resolution needed.
   *
   * @returns the winning version, or null if manual resolution required
   */
  resolveGraphConflict(
    versionA: DistributedGraphVersion,
    versionB: DistributedGraphVersion,
  ): DistributedGraphVersion | null {
    const order = compareClocks(versionA.vectorClock, versionB.vectorClock);

    switch (order) {
      case 'before':
        return versionB; // B happened after A
      case 'after':
        return versionA; // A happened after B
      case 'equal':
        return versionA; // Same version, either is fine
      case 'concurrent':
        return null; // Manual resolution needed
    }
  }
}
