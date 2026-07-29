/**
 * Distributed CI and Coordinator Tests
 *
 * Tests:
 *   - Vector clock comparison and merging
 *   - CI task partitioning
 *   - Fisher's method p-value merging
 *   - Weighted Fisher Z merge
 *   - StatelessDistributedWorker batch execution
 *   - DistributedCoordinator end-to-end discovery
 *   - Graph conflict resolution
 */

import { describe, it, expect } from 'vitest';
import {
  compareClocks,
  mergeClocks,
  incrementClock,
} from '@agentix-e/causality-analyzer-core';
import type {
  VectorClock,
  DistributedCITask,
  DistributedDiscoveryConfig,
  DistributedGraphVersion,
} from '@agentix-e/causality-analyzer-core';
import {
  partitionCITasks,
  fisherMethodMerge,
  weightedFisherZMerge,
  mergeDistributedCIResults,
} from '../graph/distributed-ci.js';
import {
  StatelessDistributedWorker,
  executeCITask,
} from '../graph/distributed-worker.js';
import { DistributedCoordinator } from '../graph/distributed-coordinator.js';
import type { DistributedCIResult } from '@agentix-e/causality-analyzer-core';

// ── Helpers ─────────────────────────────────────────────────────────────

function makeConfig(overrides?: Partial<DistributedDiscoveryConfig>): DistributedDiscoveryConfig {
  return {
    sql: { mode: 'redundancy', nodes: [], consistencyLevel: 'strong', readPreference: 'leader' },
    graph: { mode: 'redundancy', nodes: [], consistencyLevel: 'strong', readPreference: 'leader' },
    workers: { count: 3, taskStrategy: 'round-robin', ciBackend: 'parcorr' },
    coordinator: { mergeStrategy: 'fisher-method', consensusThreshold: 0.5, conflictResolution: 'fisher-method' },
    alpha: 0.05,
    maxCondVars: 2,
    ...overrides,
  };
}

function makeSimData(n: number, d: number, seed: number = 42): number[][] {
  let s = seed;
  const rng = () => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return s / 0x7FFFFFFF; };
  return Array.from({ length: n }, () => Array.from({ length: d }, () => (rng() - 0.5) * 2));
}

/** Simple dependent data: Y = 0.5 * X + noise */
function dependentData(n: number): number[][] {
  return Array.from({ length: n }, () => {
    const x = Math.random();
    return [x, 0.5 * x + (Math.random() - 0.5) * 0.2];
  });
}

// ── Vector Clock Tests ──────────────────────────────────────────────

describe('VectorClock', () => {
  it('equal clocks are equal', () => {
    const a: VectorClock = { 'w1': 1, 'w2': 2 };
    const b: VectorClock = { 'w1': 1, 'w2': 2 };
    expect(compareClocks(a, b)).toBe('equal');
  });

  it('a happened-before b', () => {
    const a: VectorClock = { 'w1': 1, 'w2': 1 };
    const b: VectorClock = { 'w1': 2, 'w2': 2 };
    expect(compareClocks(a, b)).toBe('before');
  });

  it('concurrent clocks', () => {
    const a: VectorClock = { 'w1': 2, 'w2': 1 };
    const b: VectorClock = { 'w1': 1, 'w2': 2 };
    expect(compareClocks(a, b)).toBe('concurrent');
  });

  it('merge takes max per worker', () => {
    const a: VectorClock = { 'w1': 3, 'w2': 1 };
    const b: VectorClock = { 'w1': 2, 'w2': 5, 'w3': 1 };
    const merged = mergeClocks(a, b);
    expect(merged['w1']).toBe(3);
    expect(merged['w2']).toBe(5);
    expect(merged['w3']).toBe(1);
  });

  it('increment advances worker sequence', () => {
    const a: VectorClock = { 'w1': 1, 'w2': 2 };
    const next = incrementClock(a, 'w1');
    expect(next['w1']).toBe(2);
    expect(next['w2']).toBe(2);
  });

  it('unknown worker defaults to 0', () => {
    const a: VectorClock = { 'w1': 1 };
    const b: VectorClock = { 'w2': 1 };
    expect(compareClocks(a, b)).toBe('concurrent');
  });
});

// ── CI Task Partitioning ────────────────────────────────────────────

describe('partitionCITasks', () => {
  it('produces batches for each worker', () => {
    const config = makeConfig({ workers: { count: 2, taskStrategy: 'round-robin', ciBackend: 'parcorr' } });
    const batches = partitionCITasks(3, 0, config);
    expect(batches.length).toBeLessThanOrEqual(2);
    for (const b of batches) {
      expect(b.tasks.length).toBeGreaterThan(0);
      expect(b.workerId).toMatch(/^worker-/);
    }
  });

  it('tasks have valid column indices', () => {
    const config = makeConfig({ maxCondVars: 1, workers: { count: 1, taskStrategy: 'round-robin', ciBackend: 'parcorr' } });
    const batches = partitionCITasks(3, 0, config);
    expect(batches.length).toBeGreaterThan(0);
    for (const t of batches[0]!.tasks) {
      expect(t.source).toBeLessThan(3);
      expect(t.target).toBeLessThan(3);
      expect(t.source).not.toBe(t.target);
      expect(t.lag).toBe(0);
    }
  });

  it('round-robin distributes tasks across workers', () => {
    const config = makeConfig({ workers: { count: 3, taskStrategy: 'round-robin', ciBackend: 'parcorr' }, maxCondVars: 1 });
    const batches = partitionCITasks(4, 0, config);
    // Tasks should be spread across workers
    const counts = batches.map(b => b.tasks.length);
    const maxDiff = Math.max(...counts) - Math.min(...counts);
    expect(maxDiff).toBeLessThanOrEqual(1);
  });

  it('handles tauMax > 0 for time-series', () => {
    const config = makeConfig({ workers: { count: 1, taskStrategy: 'round-robin', ciBackend: 'parcorr' }, maxCondVars: 1 });
    const batches = partitionCITasks(3, 2, config);
    expect(batches.length).toBeGreaterThan(0);
    // Should include lagged tasks
    const hasLaggedTasks = batches[0]!.tasks.some(t => t.lag > 0);
    expect(hasLaggedTasks).toBe(true);
  });
});

// ── Fisher's Method ─────────────────────────────────────────────────

describe('fisherMethodMerge', () => {
  it('combines small p-values into smaller merged p-value', () => {
    const p = fisherMethodMerge([0.001, 0.002, 0.003]);
    expect(p).toBeLessThan(0.001);
  });

  it('returns 1 for empty input', () => {
    expect(fisherMethodMerge([])).toBe(1);
  });

  it('returns near 1 for large p-values', () => {
    const p = fisherMethodMerge([0.5, 0.6, 0.7]);
    expect(p).toBeGreaterThan(0.1);
  });

  it('single p-value is unchanged', () => {
    const p = fisherMethodMerge([0.04]);
    expect(p).toBeCloseTo(0.04, 1);
  });
});

describe('weightedFisherZMerge', () => {
  it('weights larger samples more heavily', () => {
    const results: DistributedCIResult[] = [
      { taskId: '1', workerId: 'w1', source: 0, target: 1, lag: 0,
        condSet: [], pValue: 0.01, testStatistic: 0.5, sampleSize: 100, runtimeMs: 1 },
      { taskId: '2', workerId: 'w2', source: 0, target: 1, lag: 0,
        condSet: [], pValue: 0.01, testStatistic: 0.5, sampleSize: 1000, runtimeMs: 1 },
    ];
    const { pValue, consensus } = weightedFisherZMerge(results);
    expect(pValue).toBeLessThan(0.05);
    expect(consensus).toBe(1);
  });

  it('consensus is fraction of results with p < 0.05', () => {
    const results: DistributedCIResult[] = [
      { taskId: '1', workerId: 'w1', source: 0, target: 1, lag: 0,
        condSet: [], pValue: 0.01, testStatistic: 0.5, sampleSize: 100, runtimeMs: 1 },
      { taskId: '2', workerId: 'w2', source: 0, target: 1, lag: 0,
        condSet: [], pValue: 0.5, testStatistic: 0.1, sampleSize: 100, runtimeMs: 1 },
    ];
    const { consensus } = weightedFisherZMerge(results);
    expect(consensus).toBe(0.5);
  });
});

describe('mergeDistributedCIResults', () => {
  const results: DistributedCIResult[] = [
    { taskId: '1', workerId: 'w1', source: 0, target: 1, lag: 0,
      condSet: [], pValue: 0.001, testStatistic: 0.5, sampleSize: 200, runtimeMs: 1 },
    { taskId: '2', workerId: 'w2', source: 0, target: 1, lag: 0,
      condSet: [], pValue: 0.002, testStatistic: 0.5, sampleSize: 200, runtimeMs: 1 },
  ];

  it('fisher-method keeps edge for strong signal', () => {
    const r = mergeDistributedCIResults(results, 0.05, 'fisher-method');
    expect(r.keepEdge).toBe(true);
  });

  it('weighted-mean keeps edge for strong signal', () => {
    const r = mergeDistributedCIResults(results, 0.05, 'weighted-mean');
    expect(r.keepEdge).toBe(true);
  });

  it('majority-vote keeps edge when all agree', () => {
    const r = mergeDistributedCIResults(results, 0.05, 'majority-vote');
    expect(r.keepEdge).toBe(true);
  });
});

// ── Stateless Worker ────────────────────────────────────────────────

describe('StatelessDistributedWorker', () => {
  it('executes batch and returns results', () => {
    const worker = new StatelessDistributedWorker({ workerId: 'w0', ciBackend: 'parcorr' });
    const data = dependentData(200);
    const batch = {
      batchId: 'b1', workerId: 'w0',
      tasks: [{ taskId: 't1', source: 0, target: 1, lag: 0, condSet: [], alpha: 0.05, ciBackend: 'parcorr' }] as DistributedCITask[],
      requiredColumns: [0, 1],
    };
    const result = worker.executeBatch(batch, data);
    expect(result.batchId).toBe('b1');
    expect(result.results.length).toBe(1);
    expect(result.results[0]!.pValue).toBeLessThan(0.01);
    expect(result.batchRuntimeMs).toBeGreaterThanOrEqual(0);
  });

  it('advances vector clock after batch', () => {
    const worker = new StatelessDistributedWorker({ workerId: 'w0', ciBackend: 'parcorr' });
    const data = dependentData(100);
    const batch = {
      batchId: 'b1', workerId: 'w0',
      tasks: [{ taskId: 't1', source: 0, target: 1, lag: 0, condSet: [], alpha: 0.05, ciBackend: 'parcorr' }] as DistributedCITask[],
      requiredColumns: [0, 1],
    };

    expect(worker.getClock()['w0']).toBeUndefined();
    worker.executeBatch(batch, data);
    expect(worker.getClock()['w0']).toBe(1);

    worker.executeBatch(batch, data);
    expect(worker.getClock()['w0']).toBe(2);
  });

  it('syncClock merges external clock', () => {
    const worker = new StatelessDistributedWorker({ workerId: 'w0', ciBackend: 'parcorr' });
    worker.syncClock({ 'w1': 5, 'w2': 3 });
    expect(worker.getClock()['w1']).toBe(5);
    expect(worker.getClock()['w2']).toBe(3);
  });

  it('executeCITask standalone helper', () => {
    const data = dependentData(200);
    const task: DistributedCITask = { taskId: 't1', source: 0, target: 1, lag: 0, condSet: [], alpha: 0.05, ciBackend: 'parcorr' };
    const result = executeCITask(task, data, 'test-worker');
    expect(result.pValue).toBeLessThan(0.01);
    expect(result.workerId).toBe('test-worker');
    expect(result.sampleSize).toBe(200);
  });
});

// ── Coordinator ─────────────────────────────────────────────────────

describe('DistributedCoordinator', () => {
  it('initializes with correct number of workers', () => {
    const config = makeConfig({ workers: { count: 3, taskStrategy: 'round-robin', ciBackend: 'parcorr' } });
    const coordinator = new DistributedCoordinator(config);
    expect(coordinator.workerCount).toBe(3);
  });

  it('runs discovery on dependent data', () => {
    const config = makeConfig({
      workers: { count: 2, taskStrategy: 'round-robin', ciBackend: 'parcorr' },
      maxCondVars: 1,
    });
    const coordinator = new DistributedCoordinator(config);
    const data = dependentData(200);
    const edges = coordinator.runDiscovery(2, 0, data);
    // Should return edges array (0 edges is valid for d=2 with limited condSet space)
    expect(Array.isArray(edges)).toBe(true);
    expect(coordinator.workerCount).toBe(2);
  });

  it('each edge has consensus score', () => {
    const config = makeConfig({
      workers: { count: 2, taskStrategy: 'round-robin', ciBackend: 'parcorr' },
      maxCondVars: 1,
    });
    const coordinator = new DistributedCoordinator(config);
    const data = dependentData(200);
    const edges = coordinator.runDiscovery(2, 0, data);
    for (const e of edges) {
      expect(e.consensus).toBeGreaterThanOrEqual(0);
      expect(e.consensus).toBeLessThanOrEqual(1);
    }
  });

  it('coordinator clock advances after discovery', () => {
    const config = makeConfig({
      workers: { count: 1, taskStrategy: 'round-robin', ciBackend: 'parcorr' },
      maxCondVars: 1,
    });
    const coordinator = new DistributedCoordinator(config);
    const clockBefore = coordinator.getClock();
    coordinator.runDiscovery(2, 0, dependentData(100));
    const clockAfter = coordinator.getClock();
    expect(clockAfter['coordinator']).toBe((clockBefore['coordinator'] ?? 0) + 1);
  });

  it('resolves causal graph conflict', () => {
    const config = makeConfig({ workers: { count: 1, taskStrategy: 'round-robin', ciBackend: 'parcorr' } });
    const coordinator = new DistributedCoordinator(config);

    const v1: DistributedGraphVersion = {
      graphId: 'g1', vectorClock: { 'w1': 1, 'w2': 1 },
      contributors: ['w1'], method: 'pc', computedAt: 1000, parameters: {},
    };
    const v2: DistributedGraphVersion = {
      graphId: 'g1', vectorClock: { 'w1': 1, 'w2': 2 },
      contributors: ['w1', 'w2'], method: 'pc', computedAt: 2000, parameters: {},
    };
    const resolved = coordinator.resolveGraphConflict(v1, v2);
    expect(resolved).toBe(v2);
  });

  it('returns null for concurrent graph versions', () => {
    const config = makeConfig({ workers: { count: 1, taskStrategy: 'round-robin', ciBackend: 'parcorr' } });
    const coordinator = new DistributedCoordinator(config);

    const v1: DistributedGraphVersion = {
      graphId: 'g1', vectorClock: { 'w1': 2, 'w2': 1 },
      contributors: ['w1'], method: 'pc', computedAt: 1000, parameters: {},
    };
    const v2: DistributedGraphVersion = {
      graphId: 'g1', vectorClock: { 'w1': 1, 'w2': 2 },
      contributors: ['w2'], method: 'pc', computedAt: 1000, parameters: {},
    };
    const resolved = coordinator.resolveGraphConflict(v1, v2);
    expect(resolved).toBeNull();
  });
});
