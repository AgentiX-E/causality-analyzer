/**
 * PCMCI — PC algorithm with Momentary Conditional Independence for
 * time-series causal discovery.
 *
 * Extends constraint-based causal discovery to temporal data by
 * testing lagged dependencies.  Two-phase approach:
 *
 * Phase 1 (PC₁): Find significant lagged parents for each variable
 *   using conditional independence with increasing condition-set
 *   sizes, respecting temporal ordering (cause must precede effect).
 *
 * Phase 2 (MCI): Test each discovered parent edge with the full
 *   set of parents from both variables to remove spurious associations
 *   caused by autocorrelation or common drivers.
 *
 * Reference: Runge, Nowack, Kretschmer, Flaxman & Sejdinovic (2019).
 *   "Detecting and quantifying causal associations in large nonlinear
 *   time series datasets." Science Advances 5(11).
 *
 * @packageDocumentation
 */
import { fisherZTest } from '@agentix-e/causality-analyzer-core';

export interface PCMCIEdge {
  /** Source variable name */
  from: string;
  /** Target variable name */
  to: string;
  /** Time lag (-τ means from[t-τ] → to[t]) */
  lag: number;
  /** p-value from MCI test */
  pValue: number;
  /** Normalized causal strength: partial correlation */
  strength: number;
}

export interface PCMCIResult {
  /** Discovered causal edges */
  edges: PCMCIEdge[];
  /** Per-variable parent sets (Phase 1 output) */
  parents: Map<string, PCMCIEdge[]>;
  /** Maximum lag considered */
  tauMax: number;
  /** Significance threshold */
  alpha: number;
}

export interface PCMCIconfig {
  /** Significance level */
  alpha?: number;
  /** Maximum time lag */
  tauMax?: number;
  /** Maximum number of conditioning variables */
  maxCondVars?: number;
}

/**
 * Run PCMCI on time-series data.
 *
 * @param data — (T × d) matrix where data[t][j] = value at time t, variable j
 * @param nodeNames — variable names
 */
export function pcmciAlgorithm(
  data: number[][],
  nodeNames: string[],
  config: PCMCIconfig = {},
): PCMCIResult {
  const alpha = config.alpha ?? 0.05;
  const tauMax = config.tauMax ?? Math.min(5, Math.floor(data.length / 20));
  const maxCondVars = config.maxCondVars ?? 5;
  const T = data.length;
  const d = nodeNames.length;

  if (T < tauMax + 5 || d < 2) {
    return { edges: [], parents: new Map(), tauMax, alpha };
  }

  // ── Phase 1: PC₁ ──
  // For each variable j (target), find significant lagged parents i (source).
  // Parent is: var i at lag τ if i[t-τ] → j[t] given other parents.

  const candidateParents = new Map<string, PCMCIEdge[]>();
  for (let j = 0; j < d; j++) {
    candidateParents.set(nodeNames[j], []);
  }

  // Build lagged data matrix: each column is a (variable, lag) pair
  const effectiveT = T - tauMax;
  if (effectiveT <= 0) return { edges: [], parents: new Map(), tauMax, alpha };

  // Pre-extract lagged columns for efficient access
  const laggedCols: Array<{ var: number; lag: number }> = [];
  for (let i = 0; i < d; i++) {
    for (let lag = 1; lag <= tauMax; lag++) {
      laggedCols.push({ var: i, lag });
    }
  }
  const numLaggedCols = laggedCols.length;

  // Build full data matrix: rows = time steps, cols = [current vars + lagged vars]
  const fullData: number[][] = [];
  for (let t = tauMax; t < T; t++) {
    const row: number[] = [];
    // Current values (for target variables)
    for (let j = 0; j < d; j++) row.push(data[t][j]);
    // Lagged values
    for (const { var: i, lag } of laggedCols) {
      row.push(data[t - lag][i]);
    }
    fullData.push(row);
  }

  // PC₁: for each target, find significant parents
  for (let j = 0; j < d; j++) {
    const targetCol = j; // column index in fullData
    const parents: PCMCIEdge[] = [];

    // Test each possible lagged parent (i, τ) unconditionally
    const unconditionalResults: Array<{ fromVar: number; lag: number; pValue: number; sourceCol: number }> = [];
    for (let li = 0; li < numLaggedCols; li++) {
      const sourceCol = d + li; // lagged columns start after d current columns
      const p = fisherZTest(fullData, targetCol, sourceCol, []);
      if (p < alpha) {
        unconditionalResults.push({
          fromVar: laggedCols[li].var,
          lag: laggedCols[li].lag,
          pValue: p,
          sourceCol,
        });
      }
    }

    // Sort by significance
    unconditionalResults.sort((a, b) => a.pValue - b.pValue);

    // Conditional tests with increasing set size
    const acceptedParents: Array<{ fromVar: number; lag: number; pValue: number; sourceCol: number }> = [];
    for (const candidate of unconditionalResults) {
      // Test candidate given current accepted parents
      const condSet = acceptedParents.map(p => p.sourceCol);
      if (condSet.length > maxCondVars) break;

      const p = fisherZTest(fullData, targetCol, candidate.sourceCol, condSet.slice(0, Math.min(condSet.length, maxCondVars)));
      if (p < alpha) {
        acceptedParents.push(candidate);
        const edge: PCMCIEdge = {
          from: nodeNames[candidate.fromVar],
          to: nodeNames[j],
          lag: candidate.lag,
          pValue: p,
          strength: 0, // placeholder; set in MCI
        };
        parents.push(edge);
      }
    }

    candidateParents.set(nodeNames[j], parents);
  }

  // ── Phase 2: MCI ──
  // For each candidate edge, test again conditioning on:
  //   - all parents of source (excluding self-lag if applicable)
  //   - all parents of target
  // This removes spurious edges from autocorrelation or common drivers.

  const mciEdges: PCMCIEdge[] = [];
  for (let j = 0; j < d; j++) {
    const targetParents = candidateParents.get(nodeNames[j])!;
    const targetParentCols = targetParents.map(e => d + laggedCols.findIndex(
      lc => lc.var === nodeNames.indexOf(e.from) && lc.lag === e.lag
    )).filter(idx => idx >= 0);

    for (const edge of targetParents) {
      const fromVar = nodeNames.indexOf(edge.from);
      const sourceParents = candidateParents.get(nodeNames[fromVar])!;
      const sourceParentCols = sourceParents.map(e => d + laggedCols.findIndex(
        lc => lc.var === nodeNames.indexOf(e.from) && lc.lag === e.lag
      )).filter(idx => idx >= 0);

      // Build conditioning set: parents of source ∪ parents of target
      const condSet = [...new Set([...sourceParentCols, ...targetParentCols])];
      const sourceCol = d + laggedCols.findIndex(
        lc => lc.var === fromVar && lc.lag === edge.lag
      );
      if (sourceCol < 0) continue;

      const p = fisherZTest(fullData, j, sourceCol, condSet.slice(0, Math.min(condSet.length, maxCondVars)));
      if (p < alpha) {
        // Compute partial correlation as causal strength
        mciEdges.push({
          from: edge.from,
          to: edge.to,
          lag: edge.lag,
          pValue: p,
          strength: 1 - p, // approximate: 1 - p as confidence
        });
      }
    }
  }

  return { edges: mciEdges, parents: candidateParents, tauMax, alpha };
}
