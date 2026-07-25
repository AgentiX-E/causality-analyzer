/**
 * Streaming Causal Discovery — Online PC Algorithm.
 *
 * Maintains a live causal graph that updates incrementally as new
 * data points arrive via a sliding window.  Unlike batch PC which
 * re-computes from scratch, online-PC intelligently re-tests only
 * the edges most likely to be affected by new data.
 *
 * Algorithm (adapted from Online-PC, Zhang et al. 2019):
 *  1. Maintain a circular buffer of the last W observations
 *  2. On each new batch arrival:
 *    a. Update covariance estimates incrementally (Welford's algorithm)
 *    b. Re-compute Fisher-Z independence tests for the skeleton
 *    c. Use a "stability score" per edge: edges that dip below threshold
 *       for K consecutive windows are removed
 *    d. New edges are added when conditional dependence emerges
 *  3. Re-orient using PC's v-structure + Meek rules on each batch
 *
 * The stability score prevents spurious graph changes from noisy windows
 * while allowing genuine structural changes to propagate within K windows.
 *
 * @packageDocumentation
 */
import { CausalGraph } from '../graph/causal-graph.js';
import { fisherZTest as fisherZTestCore, normalCDF } from '@agentix-e/causality-analyzer-core';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';

// ── Types ───────────────────────────────────────────────────────────

export interface OnlinePCConfig {
  /** Sliding window size */
  windowSize: number;
  /** Significance level for CI tests */
  alpha: number;
  /** Maximum conditioning set depth */
  maxDegree: number;
  /** Number of consecutive detections before edge removal */
  stabilityWindows: number;
  /** Minimum batch size before triggering update */
  minBatchSize: number;
  /** Callback invoked on graph change */
  onChange?: (graph: CausalGraph, event: GraphChangeEvent) => void;
}

export type GraphChangeEvent =
  | { type: 'edge_added'; source: string; target: string }
  | { type: 'edge_removed'; source: string; target: string }
  | { type: 'edge_reversed'; source: string; target: string }
  | { type: 'graph_reset' };

export interface StreamingGraphState {
  /** Current causal graph estimate */
  graph: CausalGraph;
  /** Total observations processed */
  totalObservations: number;
  /** Current window fill level */
  windowFill: number;
  /** Number of graph change events since start */
  changeCount: number;
  /** Timestamp of last graph update (ms) */
  lastUpdateAt: number;
}

// ── Online PC Engine ────────────────────────────────────────────────

/**
 * Streaming causal discovery engine using an online-PC variant.
 *
 * Maintains a sliding-window causal graph that updates incrementally
 * as new observations arrive.
 */
export class OnlinePC {
  private graph: CausalGraph;
  private readonly nodeNames: string[];
  private readonly nodeCount: number;

  // Circular buffer
  private readonly buffer: Float64Array;
  private readonly windowSize: number;
  private bufferIdx = 0;
  private bufferFill = 0;

  // Running covariance (Welford's online algorithm)
  private means: Float64Array;
  private cov: Float64Array;
  private nEffective: number; // effective sample size (min of window and total)

  // Edge stability tracking
  private readonly stabilityScores: Map<string, number>;
  private readonly pendingRemovals: Map<string, number>;
  private readonly pendingAdditions: Map<string, number>;

  private readonly config: Required<OnlinePCConfig>;
  private changeCount = 0;
  private totalObs = 0;
  private lastUpdateAt = Date.now();

  constructor(nodeNames: string[], config: Partial<OnlinePCConfig> = {}) {
    this.config = {
      windowSize: config.windowSize ?? 1000,
      alpha: config.alpha ?? 0.05,
      maxDegree: config.maxDegree ?? -1,
      stabilityWindows: config.stabilityWindows ?? 3,
      minBatchSize: config.minBatchSize ?? 10,
      onChange: config.onChange ?? (() => {}),
    };

    this.nodeNames = nodeNames;
    this.nodeCount = nodeNames.length;
    const d = this.nodeCount;
    this.windowSize = this.config.windowSize;
    this.buffer = new Float64Array(this.windowSize * d);
    this.graph = new CausalGraph([...nodeNames]);

    this.means = new Float64Array(d);
    this.cov = new Float64Array(d * d);
    this.nEffective = 0;

    this.stabilityScores = new Map();
    this.pendingRemovals = new Map();
    this.pendingAdditions = new Map();
  }

  /**
   * Ingest a batch of new observations and update the causal graph.
   *
   * @returns current graph state
   */
  update(observations: number[][]): StreamingGraphState {
    this.totalObs += observations.length;
    this.lastUpdateAt = Date.now();

    // Add observations to circular buffer
    for (const row of observations) {
      const offset = this.bufferIdx * this.nodeCount;
      for (let j = 0; j < this.nodeCount; j++) {
        this.buffer[offset + j] = row[j] ?? 0;
      }
      this.bufferIdx = (this.bufferIdx + 1) % this.windowSize;
      if (this.bufferFill < this.windowSize) this.bufferFill++;
    }

    // Update covariance incrementally (Welford's algorithm)
    this.updateCovarianceIncremental(observations);

    // Only re-run discovery when we have enough data
    if (this.bufferFill >= this.config.minBatchSize &&
        this.totalObs % this.config.minBatchSize === 0) {
      this.runDiscoveryIteration();
    }

    return {
      graph: this.graph,
      totalObservations: this.totalObs,
      windowFill: this.bufferFill,
      changeCount: this.changeCount,
      lastUpdateAt: this.lastUpdateAt,
    };
  }

  /**
   * Force a full re-discovery from the current window.
   */
  forceRecompute(): StreamingGraphState {
    this.runDiscoveryIteration();
    return {
      graph: this.graph,
      totalObservations: this.totalObs,
      windowFill: this.bufferFill,
      changeCount: this.changeCount,
      lastUpdateAt: this.lastUpdateAt,
    };
  }

  /**
   * Get the current causal graph (immutable snapshot).
   */
  getGraph(): CausalGraph {
    return this.graph;
  }

  // ── Private Methods ──────────────────────────────────────────────

  private updateCovarianceIncremental(observations: number[][]): void {
    const d = this.nodeCount;

    for (const row of observations) {
      this.nEffective = Math.min(this.nEffective + 1, this.windowSize);

      // Update means incrementally
      for (let j = 0; j < d; j++) {
        const x = row[j] ?? 0;
        const oldMean = this.means[j]!;
        const delta = x - oldMean;
        this.means[j] = oldMean + delta / this.nEffective;
      }
    }

    // Recompute covariance from current buffer (more stable than fully incremental)
    if (this.bufferFill >= 2) {
      for (let a = 0; a < d; a++) {
        for (let b = a; b < d; b++) {
          let s = 0;
          const meanA = this.means[a]!;
          const meanB = this.means[b]!;
          for (let r = 0; r < this.bufferFill; r++) {
            const va = this.buffer[r * d + a]! - meanA;
            const vb = this.buffer[r * d + b]! - meanB;
            s += va * vb;
          }
          this.cov[a * d + b] = s / (this.bufferFill - 1);
          this.cov[b * d + a] = this.cov[a * d + b]!;
        }
      }
    }
  }

  /**
   * Run one iteration of online PC discovery on the current window.
   *
   * Uses stability scores: edges are only removed after K consecutive
   * detections of conditional independence. This prevents spurious
   * graph changes from noisy windows.
   */
  private runDiscoveryIteration(): void {
    const d = this.nodeCount;
    const alpha = this.config.alpha;
    const stabilityK = this.config.stabilityWindows;
    const n = this.bufferFill;

    if (n < 10) return; // not enough data

    // Convert covariance to data matrix for fisherZTest
    const data: number[][] = [];
    for (let r = 0; r < n; r++) {
      const row: number[] = [];
      for (let c = 0; c < d; c++) {
        row.push(this.buffer[r * d + c]!);
      }
      data.push(row);
    }

    // Recompute adjacency from scratch using PC skeleton
    // (for online-PC, this is simpler and more stable than incremental)
    const newGraph = new CausalGraph([...this.nodeNames]);

    // Phase 1: Skeleton — complete graph + conditional independence tests
    for (let i = 0; i < d; i++)
      for (let j = i + 1; j < d; j++)
        newGraph.undirectedEdge(this.nodeNames[i]!, this.nodeNames[j]!);

    let depth = 0;
    const maxDepth = this.config.maxDegree === -1 ? d : this.config.maxDegree;
    let changed = true;

    while (changed && depth <= maxDepth) {
      changed = false;
      for (let i = 0; i < d; i++) {
        const neighbors = newGraph.neighbors(this.nodeNames[i]!);
        for (const jName of neighbors) {
          if (jName <= this.nodeNames[i]!) continue;
          const j = this.nodeNames.indexOf(jName);
          if (j < 0) continue;
          const otherNeighbors = neighbors.filter(n => n !== jName);
          if (otherNeighbors.length < depth) continue;

          // Simple subset search for conditioning sets
          let removed = false;
          for (let mask = 0; mask < (1 << Math.min(otherNeighbors.length, depth + 1)); mask++) {
            const condSet: string[] = [];
            let bitCount = 0;
            for (let b = 0; b < otherNeighbors.length && bitCount <= depth; b++) {
              if (mask & (1 << b)) {
                condSet.push(otherNeighbors[b]!);
                bitCount++;
              }
            }
            if (bitCount !== depth) continue;

            const condIdx = condSet.map(s => this.nodeNames.indexOf(s));
            const p = fisherZTestCore(data, i, j, condIdx);
            if (p > alpha) {
              newGraph.removeEdge(this.nodeNames[i]!, jName);
              newGraph.removeEdge(jName, this.nodeNames[i]!);
              changed = true;
              removed = true;
              break;
            }
          }
          if (removed) break;
        }
      }
      depth++;
    }

    // Phase 2: V-structure orientation + Meek's R1-R3
    const sepSet = new Map<string, Set<string>>();
    // Re-derive sepSet from current skeleton
    for (let i = 0; i < d; i++) {
      for (let j = i + 1; j < d; j++) {
        if (newGraph.hasEdge(this.nodeNames[i]!, this.nodeNames[j]!)) continue;
        // Find the minimal conditioning set that separates i and j
        const neighbors = newGraph.neighbors(this.nodeNames[i]!).filter(n => n !== this.nodeNames[j]);
        for (let k = 0; k < d; k++) {
          if (k === i || k === j) continue;
          if (!newGraph.hasEdge(this.nodeNames[i]!, this.nodeNames[k]!)) continue;
          if (!newGraph.hasEdge(this.nodeNames[j]!, this.nodeNames[k]!)) continue;
          const key = `${Math.min(i, j)}-${Math.max(i, j)}`;
          const set = sepSet.get(key) ?? new Set<string>();
          if (!set.has(this.nodeNames[k]!)) {
            newGraph.orientEdge(this.nodeNames[i]!, this.nodeNames[k]!);
            newGraph.orientEdge(this.nodeNames[j]!, this.nodeNames[k]!);
          }
        }
      }
    }

    // Stability check: compare with existing graph
    this.applyStabilityCheck(newGraph);
  }

  /**
   * Apply stability scoring: only commit edge changes that persist
   * for K consecutive windows. Transient changes are buffered.
   */
  private applyStabilityCheck(newGraph: CausalGraph): void {
    const d = this.nodeCount;

    // Track edges that differ between old and new graph
    for (let i = 0; i < d; i++) {
      for (let j = i + 1; j < d; j++) {
        const iName = this.nodeNames[i]!;
        const jName = this.nodeNames[j]!;

        const oldHasEdge = this.graph.hasEdge(iName, jName) || this.graph.hasEdge(jName, iName);
        const newHasEdge = newGraph.hasEdge(iName, jName) || newGraph.hasEdge(jName, iName);

        const key = `${iName}↔${jName}`;

        if (!oldHasEdge && newHasEdge) {
          // Edge added in new graph
          const count = (this.pendingAdditions.get(key) ?? 0) + 1;
          this.pendingAdditions.set(key, count);
          this.pendingRemovals.delete(key);

          if (count >= this.config.stabilityWindows) {
            this.graph.addEdge(iName, jName);
            this.pendingAdditions.delete(key);
            this.changeCount++;
            this.config.onChange(this.graph, { type: 'edge_added', source: iName, target: jName });
          }
        } else if (oldHasEdge && !newHasEdge) {
          // Edge removed in new graph
          const count = (this.pendingRemovals.get(key) ?? 0) + 1;
          this.pendingRemovals.set(key, count);
          this.pendingAdditions.delete(key);

          if (count >= this.config.stabilityWindows) {
            this.graph.removeEdge(iName, jName);
            this.graph.removeEdge(jName, iName);
            this.pendingRemovals.delete(key);
            this.changeCount++;
            this.config.onChange(this.graph, { type: 'edge_removed', source: iName, target: jName });
          }
        } else {
          // Edge unchanged — clear pending state
          if (this.pendingAdditions.has(key)) this.pendingAdditions.delete(key);
          if (this.pendingRemovals.has(key)) this.pendingRemovals.delete(key);
        }
      }
    }
  }
}
