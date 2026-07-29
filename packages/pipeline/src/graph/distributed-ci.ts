/**
 * Distributed CI — Conditional Independence test parallelization and
 * Fisher's method result merging for stateless causal discovery.
 *
 * Partitions the CI test matrix for constraint-based algorithms (PC, FCI,
 * PCMCI+) into independent tasks that can be dispatched to stateless
 * Workers. Workers fetch data columns from the SQL cluster, compute local
 * CI test statistics, and return p-values. The Coordinator merges results
 * using Fisher's method meta-analysis.
 *
 * Key design constraints:
 *   - Workers are stateless: no local durable storage
 *   - All data lives in SQL cluster (accessed via IRelationalStore)
 *   - Cross-shard CI tests are coordinated application-side (no DB JOIN)
 *   - No vendor lock-in: works with any pg-wire compatible SQL cluster
 *
 * @packageDocumentation
 */

import { combinations } from '@agentix-e/causality-analyzer-core';
import type {
  DistributedCITask,
  DistributedCIResult,
  DistributedCITaskBatch,
  DistributedDiscoveryConfig,
  CITestObserver,
} from '@agentix-e/causality-analyzer-core';

// ── Task Partitioning ───────────────────────────────────────────────────

/**
 * Partition the CI test matrix for constraint-based causal discovery
 * into independent tasks suitable for distributed execution.
 *
 * For PC algorithm with d variables, maxCondVars condition set size,
 * and tauMax time lags (for PCMCI+), the total CI test count is:
 *   O(d² × (d)^maxCondVars × tauMax)
 *
 * Tasks are grouped by variable pair to minimize redundant data fetching
 * from the SQL cluster.
 *
 * @param nodeCount - number of variables (d)
 * @param tauMax - maximum time lag (0 for IID, >0 for time series)
 * @param config - distributed discovery configuration
 * @returns array of task batches, one per worker
 */
export function partitionCITasks(
  nodeCount: number,
  tauMax: number,
  config: DistributedDiscoveryConfig,
): DistributedCITaskBatch[] {
  const { workers, alpha, maxCondVars } = config;
  const tasks: DistributedCITask[] = [];
  const maxLag = tauMax > 0 ? tauMax : 0;
  let taskId = 0;

  // For each target variable j
  for (let j = 0; j < nodeCount; j++) {
    // For each source variable i and lag combination
    for (let i = 0; i < nodeCount; i++) {
      const lagRange = tauMax > 0 ? createLagRange(maxLag, i === j) : [0];
      for (const lag of lagRange) {
        if (lag === 0 && i === j) continue;

        // Generate conditioning sets of increasing size
        for (let condSize = 0; condSize <= maxCondVars; condSize++) {
          // All possible variables that could be in the conditioning set
          const candidates: number[] = [];
          for (let c = 0; c < nodeCount; c++) {
            if (c !== i && c !== j) candidates.push(c);
          }

          if (condSize > candidates.length) break;

          // Generate all combinations of size condSize
          const combos = combinations([...candidates], condSize);
          for (const comboIndices of combos) {
            const condSet = (comboIndices as number[]).sort((a, b) => a - b);
            tasks.push({
              taskId: `ci-${taskId++}`,
              source: i,
              target: j,
              lag,
              condSet,
              alpha,
              ciBackend: workers.ciBackend,
            });
          }
        }
      }
    }
  }

  // Partition tasks across workers using the configured strategy
  return partitionToBatches(tasks, workers.count, workers.taskStrategy);
}

/**
 * Create the valid lag range for a variable pair.
 * For tauMax > 0: lags from 0 to tauMax.
 * For tauMax = 0 (IID): only lag 0 (contemporaneous).
 *
 * @internal
 */
function createLagRange(tauMax: number, isSelfPair: boolean): number[] {
  const lags: number[] = [];
  for (let lag = 0; lag <= tauMax; lag++) {
    lags.push(lag);
  }
  return lags;
}

/**
 * Distribute tasks across workers using the specified strategy.
 *
 * @internal
 */
function partitionToBatches(
  tasks: DistributedCITask[],
  workerCount: number,
  strategy: DistributedDiscoveryConfig['workers']['taskStrategy'],
): DistributedCITaskBatch[] {
  const batches: DistributedCITaskBatch[] = [];
  const workerQueues: DistributedCITask[][] = Array.from(
    { length: workerCount },
    () => [],
  );

  switch (strategy) {
    case 'round-robin':
      for (let i = 0; i < tasks.length; i++) {
        workerQueues[i % workerCount]!.push(tasks[i]!);
      }
      break;

    case 'least-loaded':
      // Greedy: assign each task to the worker with fewest tasks
      const counts = new Array(workerCount).fill(0);
      for (const task of tasks) {
        const minIdx = counts.indexOf(Math.min(...counts));
        workerQueues[minIdx]!.push(task);
        counts[minIdx]++;
      }
      break;

    case 'partition-aware':
      // Group tasks by source variable to minimize cross-shard reads
      // Within each partition, distribute round-robin
      const bySource = new Map<number, DistributedCITask[]>();
      for (const task of tasks) {
        if (!bySource.has(task.source)) bySource.set(task.source, []);
        bySource.get(task.source)!.push(task);
      }
      let wi = 0;
      for (const [, sourceTasks] of bySource) {
        for (const task of sourceTasks) {
          workerQueues[wi % workerCount]!.push(task);
          wi++;
        }
      }
      break;
  }

  for (let w = 0; w < workerCount; w++) {
    const batchTasks = workerQueues[w]!;
    if (batchTasks.length === 0) continue;
    const requiredCols = uniqueColumns(batchTasks);
    batches.push({
      batchId: `batch-${w}-${Date.now()}`,
      tasks: batchTasks,
      workerId: `worker-${w}`,
      requiredColumns: requiredCols,
    });
  }

  return batches;
}

// ── Fisher's Method Merge ───────────────────────────────────────────────

/**
 * Merge p-values from multiple CI tests using Fisher's method.
 *
 * Fisher's method combines K independent p-values into a single χ² statistic:
 *   χ²_{2K} = -2 Σ_{k=1}^{K} ln(p_k)
 *
 * Under H₀ (all null hypotheses true), this follows χ² with 2K degrees
 * of freedom. A small merged p-value indicates that at least one of the
 * individual tests rejects H₀.
 *
 * Reference: Fisher, R. A. (1925). *Statistical Methods for Research Workers*.
 *
 * @param pValues - array of p-values from independent CI tests
 * @returns merged p-value
 */
export function fisherMethodMerge(pValues: number[]): number {
  const valid = pValues.filter(p => p > 0 && p < 1);
  if (valid.length === 0) return 1;

  let chi2 = 0;
  for (const p of valid) {
    chi2 += -2 * Math.log(p);
  }

  const df = 2 * valid.length;
  // Chi-squared survival function via regularized gamma
  return chiSquareSurvival(chi2, df);
}

// ── Weighted Merging (for Fisher Z / ParCorr) ───────────────────────────

/**
 * Merge partial correlation results using weighted Fisher Z transform.
 *
 * For Fisher Z partial correlation tests, the sample size n determines
 * the precision: Var(Z) ≈ 1/(n-3). Weighted averaging gives more
 * weight to results from larger samples.
 *
 *   Z̄ = Σ (n_k - 3) · Z_k / Σ (n_k - 3)
 *   ρ̄ = tanh(Z̄)
 *   p̄ = 2 · Φ(-|Z̄| · √(Σ (n_k - 3)))
 *
 * @param results - CI results with sample sizes and test statistics
 * @returns merged p-value and weighted correlation
 */
export function weightedFisherZMerge(
  results: DistributedCIResult[],
): { pValue: number; correlation: number; consensus: number } {
  let weightedZSum = 0;
  let totalWeight = 0;
  let agreementCount = 0;

  for (const r of results) {
    const weight = Math.max(1, r.sampleSize - 3);
    // Inverse Fisher Z: ρ = tanh(Z)
    const rho = Math.tanh(r.testStatistic);
    const rhoClamped = Math.max(-0.9999, Math.min(0.9999, rho));
    const z = 0.5 * Math.log((1 + rhoClamped) / (1 - rhoClamped));
    weightedZSum += weight * z;
    totalWeight += weight;
    if (r.pValue < 0.05) agreementCount++;
  }

  if (totalWeight === 0) return { pValue: 1, correlation: 0, consensus: 0 };

  const mergedZ = weightedZSum / totalWeight;
  const mergedRho = Math.tanh(mergedZ);
  const se = 1 / Math.sqrt(totalWeight);

  // Two-tailed p-value from normal approximation
  const zScore = Math.abs(mergedZ) / se;
  const pValue = 2 * (1 - normalCDF(zScore));

  return {
    pValue,
    correlation: mergedRho,
    consensus: agreementCount / results.length,
  };
}

/**
 * Merge CI results from multiple workers into a single edge decision.
 *
 * Route to the appropriate merge strategy based on the CI backend.
 *
 * @param results - results from all workers for the same (source, target, lag)
 * @param alpha - significance threshold
 * @param strategy - merge strategy
 * @returns merged result and whether the edge should be kept
 */
export function mergeDistributedCIResults(
  results: DistributedCIResult[],
  alpha: number,
  strategy: DistributedDiscoveryConfig['coordinator']['mergeStrategy'] = 'fisher-method',
): {
  pValue: number;
  testStatistic: number;
  consensus: number;
  keepEdge: boolean;
} {
  if (results.length === 0) return { pValue: 1, testStatistic: 0, consensus: 0, keepEdge: false };

  switch (strategy) {
    case 'fisher-method': {
      const pValues = results.map(r => r.pValue);
      const mergedP = fisherMethodMerge(pValues);
      return {
        pValue: mergedP,
        testStatistic: 0,
        consensus: results.filter(r => r.pValue < alpha).length / results.length,
        keepEdge: mergedP < alpha,
      };
    }

    case 'weighted-mean': {
      const { pValue, correlation, consensus } = weightedFisherZMerge(results);
      return {
        pValue,
        testStatistic: Math.abs(correlation),
        consensus,
        keepEdge: pValue < alpha,
      };
    }

    case 'majority-vote': {
      const keepCount = results.filter(r => r.pValue < alpha).length;
      const majorityThreshold = Math.ceil(results.length / 2);
      return {
        pValue: keepCount / results.length,
        testStatistic: 0,
        consensus: keepCount / results.length,
        keepEdge: keepCount >= majorityThreshold,
      };
    }

    default:
      return { pValue: 1, testStatistic: 0, consensus: 0, keepEdge: false };
  }
}

// ── Helpers ─────────────────────────────────────────────────────────────

function uniqueColumns(tasks: DistributedCITask[]): number[] {
  const cols = new Set<number>();
  for (const t of tasks) {
    cols.add(t.source);
    cols.add(t.target);
    for (const c of t.condSet) cols.add(c);
  }
  return [...cols].sort((a, b) => a - b);
}

function extractIndices(comboSet: Set<number> | number[], candidates: number[]): number[] {
  if (comboSet instanceof Set) {
    return [...comboSet].map(c => candidates[c]!).sort((a, b) => a - b);
  }
  // comboSet is actually an array of indices from combinations()
  const arr = comboSet as unknown as number[];
  return arr.map(c => candidates[c]!).sort((a, b) => a - b);
}

/**
 * Standard normal cumulative distribution function.
 * Abramowitz & Stegun 7.1.26 approximation (max error < 1.5×10⁻⁷).
 *
 * @internal
 */
function normalCDF(z: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp(-z * z / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - p : p;
}

/**
 * Chi-squared survival function P(χ² ≥ x | df) via regularized gamma.
 * Uses the series expansion for small x and continued fraction for large x.
 *
 * @internal
 */
function chiSquareSurvival(x: number, df: number): number {
  if (x <= 0) return 1;
  if (df <= 0) return 1;

  const a = df / 2;
  const y = x / 2;

  // Series expansion for small y
  if (y < a + 1) {
    let sum = 1 / a;
    let term = sum;
    for (let n = 1; n <= 200; n++) {
      term *= y / (a + n);
      sum += term;
      if (Math.abs(term) < Math.abs(sum) * 1e-15) break;
    }
    const lowerGamma = sum * Math.exp(-y + a * Math.log(y) - logGammaStirling(a));
    return 1 - lowerGamma;
  }

  // Continued fraction for large y
  return gammaQ_CF(a, y);
}

function gammaQ_CF(a: number, x: number): number {
  const fpMin = 1e-30;
  let f = fpMin;
  let c = fpMin;
  let d = 0;
  let b = x + 1 - a;

  for (let i = 1; i <= 200; i++) {
    const an = (i % 2 === 1) ? i * (a - (i + 1) / 2) : (i / 2) * (i / 2);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < fpMin) d = fpMin;
    c = an / c + b;
    if (Math.abs(c) < fpMin) c = fpMin;
    d = 1 / d;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-12) break;
  }

  if (f === 0) return 1;
  return Math.exp(-x + a * Math.log(x) - logGammaStirling(a)) * f;
}

function logGammaStirling(z: number): number {
  if (z <= 0) return Infinity;
  const g = 7;
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  let base = z + g + 0.5;
  let s = c[0]!;
  for (let i = 1; i < c.length; i++) s += c[i]! / (z + i - 1);
  return Math.log(Math.sqrt(2 * Math.PI)) + (z + 0.5) * Math.log(base) - base + Math.log(s);
}
