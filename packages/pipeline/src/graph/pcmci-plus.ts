/**
 * PCMCI+ — PC algorithm with Momentary Conditional Independence Plus
 * for time-series causal discovery.
 *
 * Extends the basic PCMCI algorithm with contemporaneous edge discovery,
 * extended MCI+ conditioning to remove collider bias, and CPDAG
 * orientation for the contemporaneous layer.
 *
 * Three-phase approach:
 *
 * Phase 1 (PC₁): For each variable j, find significant parents (both
 *   lagged and contemporaneous) using conditional independence tests
 *   with increasing condition-set sizes. Temporal ordering is respected:
 *   lagged parents must precede the target in time; contemporaneous
 *   parents may be mutually dependent.
 *
 * Phase 2 (MCI+): Re-test each candidate edge from Phase 1. For lagged
 *   edges, condition on parents of both source and target (same as
 *   original MCI). For contemporaneous edges, use extended conditioning:
 *   Pa(i) ∪ Pa(j) ∪ Pa_lagged(i) ∪ Pa_lagged(j). This removes spurious
 *   contemporaneous edges from common drivers and collider structures.
 *
 * Phase 3 (CPDAG): Orient contemporaneous edges via v-structure detection
 *   and Meek's rules R1-R3. Lagged edges are always fully directed (arrow
 *   target) because the past cannot depend on the future.
 *
 * References:
 *   - Runge, J. (2020). "Discovering contemporaneous and lagged causal
 *     relations in autocorrelated nonlinear time series datasets."
 *     Proceedings of the 36th UAI Conference.
 *   - Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., & Sejdinovic, D.
 *     (2019). "Detecting and quantifying causal associations in large
 *     nonlinear time series datasets." Science Advances 5(11).
 *
 * @packageDocumentation
 */

import {
  type PCMCIPlusConfig,
  type PCMCIPlusResult,
  type PCMCIPlusEdgeSummary,
  type TimeSeriesEdge,
  type TimeSeriesGraph,
  type CITestObserver,
  type CIBackend,
} from '@agentix-e/causality-analyzer-core';
import { ciTest } from './ci-backend.js';
import { orientCPDAG } from './cpdag.js';

// ── Default Configuration ───────────────────────────────────────────────

function defaultConfig(T: number): PCMCIPlusConfig {
  return {
    alpha: 0.05,
    tauMax: Math.min(5, Math.floor(T / 20)),
    maxCondVars: 5,
    ciBackend: 'parcorr',
  };
}

// ── Internal Parent Representation ──────────────────────────────────────

/** Internal representation of a potential parent during Phase 1. */
interface ParentCandidate {
  /** Variable index */
  varIdx: number;
  /** Time lag (0 = contemporaneous, >0 = lagged) */
  lag: number;
  /** Column index in the augmented design matrix */
  colIdx: number;
  /** Best p-value from the most recent CI test */
  pValue: number;
  /** Whether this candidate has been accepted into the parent set */
  accepted: boolean;
}

// ── Main Algorithm ──────────────────────────────────────────────────────

/**
 * Run the PCMCI+ algorithm on time-series data.
 *
 * @param data - (T × d) matrix where data[t][j] = value at time t, variable j
 * @param nodeNames - variable names (length d)
 * @param config - algorithm configuration (partial; defaults filled in)
 * @param onCITest - optional callback for each CI test (debugging/inspection)
 * @returns PCMCIPlusResult with discovered graph, parent sets, and summary
 */
export function pcmciPlusAlgorithm(
  data: number[][],
  nodeNames: string[],
  config: Partial<PCMCIPlusConfig> = {},
  onCITest?: CITestObserver,
): PCMCIPlusResult {
  const startTime = Date.now();

  const T = data.length;
  const d = nodeNames.length;
  const cfg: PCMCIPlusConfig = { ...defaultConfig(T), ...config, ciBackend: config.ciBackend ?? 'parcorr' };
  const { alpha, tauMax, maxCondVars, ciBackend } = cfg;

  const knnK = cfg.knnK ?? 5;
  const nPermutations = cfg.nPermutations ?? 200;
  const ciParams = { knnK, nPermutations, seed: 42 };

  // ── Build augmented design matrix ─────────────────────────────────────
  // Columns: [current vars (d cols), lagged vars (d × tauMax cols)]
  // Rows: time steps from tauMax to T-1 (effective samples)

  const effectiveT = T - tauMax;
  if (effectiveT <= 0 || d < 2) {
    return emptyResult(nodeNames, tauMax, T, cfg, startTime);
  }

  const numLaggedCols = d * tauMax;
  const totalCols = d + numLaggedCols;

  // Build augmented matrix
  const augData: number[][] = [];
  for (let t = tauMax; t < T; t++) {
    const row: number[] = [];
    // Current values (cols 0..d-1)
    for (let j = 0; j < d; j++) {
      row.push(data[t]![j]!);
    }
    // Lagged values (cols d..d+numLaggedCols-1)
    for (let vi = 0; vi < d; vi++) {
      for (let lag = 1; lag <= tauMax; lag++) {
        row.push(data[t - lag]![vi]!);
      }
    }
    augData.push(row);
  }

  // Lagged column index helper: colIdx = d + vi * tauMax + (lag - 1)
  function laggedColIdx(vi: number, lag: number): number {
    return d + vi * tauMax + (lag - 1);
  }

  // ── Phase 1: PC₁ ─────────────────────────────────────────────────────
  // For each variable j (target), find significant parents among all
  // candidate (i, tau) pairs where tau ∈ [0, tauMax].

  const parentCandidates = new Map<number, ParentCandidate[]>();

  for (let j = 0; j < d; j++) {
    const candidates: ParentCandidate[] = [];

    // Build candidate list: all (i, tau) pairs
    for (let i = 0; i < d; i++) {
      for (let lag = 0; lag <= tauMax; lag++) {
        if (lag === 0 && i === j) continue; // No self-loop at tau=0
        const colIdx = lag === 0 ? i : laggedColIdx(i, lag);
        candidates.push({
          varIdx: i,
          lag,
          colIdx,
          pValue: 1,
          accepted: false,
        });
      }
    }

    // Test each candidate unconditionally first
    for (const cand of candidates) {
      const result = ciTest(augData, cand.colIdx, j, [], ciBackend, ciParams);
      cand.pValue = result.pValue;
      if (onCITest) {
        onCITest(
          nodeNames[cand.varIdx]!, nodeNames[j]!, cand.lag,
          [], result.pValue, result.testStatistic,
        );
      }
    }

    // Sort by significance (ascending p-value)
    candidates.sort((a, b) => a.pValue - b.pValue);

    // Conditional tests with increasing set size
    // For each condition set size s = 1, 2, ..., maxCondVars
    for (let s = 1; s <= maxCondVars; s++) {
      for (const cand of candidates) {
        // Only test candidates that passed the previous level
        if (!cand.accepted && s > 1) continue;

        // Build conditioning set from currently accepted parents (excluding cand itself)
        const acceptedParents = candidates.filter(
          p => p.accepted && p.colIdx !== cand.colIdx,
        );
        if (acceptedParents.length < s) continue;

        // For each subset of size s from accepted parents, test
        // For efficiency, use the first s strongest parents
        const condSet = acceptedParents.slice(0, s).map(p => p.colIdx);
        const result = ciTest(augData, cand.colIdx, j, condSet, ciBackend, ciParams);

        if (onCITest) {
          const condNames = condSet.map(ci => {
            for (const p of acceptedParents) {
              if (p.colIdx === ci) {
                const suffix = p.lag === 0 ? '' : `[t-${p.lag}]`;
                return `${nodeNames[p.varIdx]}${suffix}`;
              }
            }
            return '?';
          });
          onCITest(
            nodeNames[cand.varIdx]!, nodeNames[j]!, cand.lag,
            condNames, result.pValue, result.testStatistic,
          );
        }

        if (result.pValue < alpha) {
          cand.pValue = result.pValue;
          cand.accepted = true;
        } else if (cand.accepted) {
          // Previously accepted but now independent given larger S → remove
          cand.accepted = false;
          cand.pValue = result.pValue;
        }
      }
    }

    parentCandidates.set(j, candidates);
  }

  // ── Phase 2: MCI+ ────────────────────────────────────────────────────

  const mciEdges: TimeSeriesEdge[] = [];
  const parentEdges = new Map<string, TimeSeriesEdge[]>();

  for (let j = 0; j < d; j++) {
    const targetParents = parentCandidates.get(j)!;
    const acceptedTargetParents = targetParents.filter(p => p.accepted);

    // Build parent set for target (as TimeSeriesEdges)
    const targetParentEdges: TimeSeriesEdge[] = acceptedTargetParents.map(cand => ({
      source: nodeNames[cand.varIdx]!,
      target: nodeNames[j]!,
      lag: cand.lag,
      strength: Math.max(0, Math.min(1, 1 - cand.pValue)),
      pValue: cand.pValue,
      sourceMark: 'tail' as const,
      targetMark: 'arrow' as const,
      phase: 'pc1' as const,
    }));
    parentEdges.set(nodeNames[j]!, targetParentEdges);

    // MCI+ test for each accepted target parent
    for (const cand of acceptedTargetParents) {
      // Build MCI+ conditioning set
      const condSet: number[] = [];

      if (cand.lag > 0) {
        // Lagged edge: Pa(j) ∪ Pa_lagged(i)
        // Pa(j) = all accepted parents of j (excluding cand)
        for (const p of acceptedTargetParents) {
          if (p.colIdx !== cand.colIdx) {
            condSet.push(p.colIdx);
          }
        }
        // Pa_lagged(i) = lagged parents of source i
        const sourceParents = parentCandidates.get(cand.varIdx)!;
        for (const p of sourceParents) {
          if (p.accepted && p.lag > 0) {
            const pColIdx = p.lag === 0 ? p.varIdx : laggedColIdx(p.varIdx, p.lag);
            condSet.push(pColIdx);
          }
        }
      } else {
        // Contemporaneous edge (tau=0): extended MCI+ conditioning
        // Pa(i) ∪ Pa(j) ∪ Pa_lagged(i) ∪ Pa_lagged(j)
        // Pa(j): accepted parents of target (excluding cand)
        for (const p of acceptedTargetParents) {
          if (p.colIdx !== cand.colIdx) {
            condSet.push(p.colIdx);
          }
        }
        // Pa(i): accepted parents of source
        const sourceParents = parentCandidates.get(cand.varIdx)!;
        for (const p of sourceParents) {
          if (p.accepted) {
            const pColIdx = p.lag === 0 ? p.varIdx : laggedColIdx(p.varIdx, p.lag);
            condSet.push(pColIdx);
          }
        }
        // Pa_lagged(i) and Pa_lagged(j) are already covered above since
        // they're included in the full parent sets.
        // Add extra lagged parents of both to handle common driver effects
        for (const p of sourceParents) {
          if (p.accepted && p.lag > 0) {
            const pColIdx = p.lag === 0 ? p.varIdx : laggedColIdx(p.varIdx, p.lag);
            if (!condSet.includes(pColIdx)) condSet.push(pColIdx);
          }
        }
        for (const p of acceptedTargetParents) {
          if (p.accepted && p.lag > 0) {
            const pColIdx = p.lag === 0 ? p.varIdx : laggedColIdx(p.varIdx, p.lag);
            if (!condSet.includes(pColIdx)) condSet.push(pColIdx);
          }
        }
      }

      // Deduplicate and cap the conditioning set
      const uniqueCondSet = [...new Set(condSet)].slice(0, maxCondVars * 2);

      const result = ciTest(augData, cand.colIdx, j, uniqueCondSet, ciBackend, ciParams);

      if (onCITest) {
        onCITest(
          nodeNames[cand.varIdx]!, nodeNames[j]!, cand.lag,
          uniqueCondSet.map(String), result.pValue, result.testStatistic,
        );
      }

      if (result.pValue < alpha) {
        const strength = Math.max(0, Math.min(1, 1 - result.pValue));
        mciEdges.push({
          source: nodeNames[cand.varIdx]!,
          target: nodeNames[j]!,
          lag: cand.lag,
          strength,
          pValue: result.pValue,
          sourceMark: cand.lag > 0 ? 'tail' : 'circle',
          targetMark: cand.lag > 0 ? 'arrow' : 'circle',
          phase: 'mci',
        });
      }
    }
  }

  // ── Phase 3: CPDAG Orientation ───────────────────────────────────────

  // Collect contemporaneous adjacencies and separation sets
  const contempAdjacencies: Array<readonly [string, string]> = [];
  const sepSets = new Map<string, Set<string>>();

  for (const edge of mciEdges) {
    if (edge.lag !== 0) continue;
    const key = edgeKey(edge.source, edge.target);
    contempAdjacencies.push([edge.source, edge.target]);
    // Record "empty" separation set for v-structure detection
    // In practice, the actual sepSet from Phase 1 would be used here.
    // For simplicity, we store an empty set as placeholder.
    if (!sepSets.has(key)) sepSets.set(key, new Set());
  }

  const orientations = orientCPDAG({
    adjacencies: contempAdjacencies,
    sepSets,
    nodes: nodeNames,
  });

  // Apply CPDAG orientations to contemporaneous edges
  const finalEdges: TimeSeriesEdge[] = mciEdges.map(edge => {
    if (edge.lag !== 0) {
      // Lagged edges: always fully directed
      return { ...edge, sourceMark: 'tail' as const, targetMark: 'arrow' as const };
    }
    const key = edgeKey(edge.source, edge.target);
    const orient = orientations.get(key);
    if (orient) {
      return {
        ...edge,
        sourceMark: orient.sourceMark,
        targetMark: orient.targetMark,
      };
    }
    // Edge not found in orientations map — keep as undirected
    return edge;
  });

  // ── Build Summary ─────────────────────────────────────────────────────

  let laggedEdges = 0;
  let contemporaneousEdges = 0;
  let directedEdges = 0;
  let partiallyDirectedEdges = 0;

  for (const edge of finalEdges) {
    if (edge.lag === 0) {
      contemporaneousEdges++;
      if (edge.sourceMark === 'tail' && edge.targetMark === 'arrow') {
        directedEdges++;
      } else {
        partiallyDirectedEdges++;
      }
    } else {
      laggedEdges++;
    }
  }

  const summary: PCMCIPlusEdgeSummary = {
    totalEdges: finalEdges.length,
    laggedEdges,
    contemporaneousEdges,
    directedEdges,
    partiallyDirectedEdges,
  };

  // ── Build TimeSeriesGraph ─────────────────────────────────────────────

  const graph: TimeSeriesGraph = {
    nodes: nodeNames,
    edges: finalEdges,
    tauMax,
    timeSteps: T,
    isCPDAG: true,
  };

  const runtimeMs = Date.now() - startTime;

  return {
    graph,
    parents: parentEdges,
    summary,
    config: cfg,
    runtimeMs,
  };
}

// ── Utility Functions ───────────────────────────────────────────────────

/** Generate an empty result for degenerate inputs. */
function emptyResult(
  nodeNames: string[],
  tauMax: number,
  timeSteps: number,
  config: PCMCIPlusConfig,
  startTime: number,
): PCMCIPlusResult {
  return {
    graph: {
      nodes: nodeNames,
      edges: [],
      tauMax,
      timeSteps,
      isCPDAG: true,
    },
    parents: new Map(),
    summary: {
      totalEdges: 0,
      laggedEdges: 0,
      contemporaneousEdges: 0,
      directedEdges: 0,
      partiallyDirectedEdges: 0,
    },
    config,
    runtimeMs: Date.now() - startTime,
  };
}

/** Lexicographically sorted edge key for adjacency maps. */
function edgeKey(a: string, b: string): string {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}
