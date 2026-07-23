/**
 * Mediation Analysis — natural direct/indirect effects and arrow strength.
 *
 * Critical for AIOps RCA: understanding not just WHICH service is the root
 * cause, but HOW the failure propagates — is it through a direct link or
 * through intermediate services?
 *
 * Methods:
 *   naturalDirectEffect — NDE = E[Y(x, M(x*)) - Y(x*, M(x*))]
 *   naturalIndirectEffect — NIE = E[Y(x, M(x)) - Y(x, M(x*))]
 *   arrowStrength — quantifies causal flow along each edge
 *
 * @packageDocumentation
 */
import { CausalGraph } from '../graph/causal-graph.js';
import { solveLinear } from '@agentix-e/causality-analyzer-core';

/**
 * Result of mediation analysis.
 */
export interface MediationResult {
  /** Natural Direct Effect */
  nde: number;
  /** Natural Indirect Effect */
  nie: number;
  /** Total Effect = NDE + NIE */
  totalEffect: number;
  /** Proportion mediated = NIE / TE */
  proportionMediated: number;
  /** Mediator variables */
  mediators: string[];
  /** Human-readable explanation */
  explanation: string;
}

/**
 * Compute Natural Direct and Indirect Effects via regression-based
 * Baron-Kenny approach (linear mediation).
 *
 * Step 1: Y ~ X (total effect c)
 * Step 2: M ~ X (path a)
 * Step 3: Y ~ X + M (direct c', indirect a*b)
 *
 * NDE = c' (direct path)
 * NIE = a * b (indirect path through M)
 */
export function naturalDirectEffect(
  data: number[][],
  treatmentIdx: number,
  outcomeIdx: number,
  mediatorIdx: number,
): MediationResult {
  const n = data.length;
  if (n < 3) return { nde: 0, nie: 0, totalEffect: 0, proportionMediated: 0, mediators: [], explanation: 'Insufficient data' };

  // Step 1: Y ~ X (total effect)
  const c = simpleRegression(data, treatmentIdx, outcomeIdx);

  // Step 2: M ~ X (treatment → mediator)
  const a = simpleRegression(data, treatmentIdx, mediatorIdx);

  // Step 3: Y ~ X + M
  const { coefs } = multipleRegression(data, outcomeIdx, [treatmentIdx, mediatorIdx]);
  const cPrime = coefs[0] ?? 0;
  const b = coefs[1] ?? 0;

  const nie = a * b;
  const totalEffect = cPrime + nie;

  return {
    nde: cPrime,
    nie,
    totalEffect,
    proportionMediated: Math.abs(totalEffect) > 1e-10 ? nie / totalEffect : 0,
    mediators: [],
    explanation: `NDE=${cPrime.toFixed(3)}, NIE=${nie.toFixed(3)} (${(Math.abs(nie / Math.max(1e-10, totalEffect)) * 100).toFixed(0)}% mediated)`,
  };
}

/**
 * Compute arrow strength for each edge in a causal graph.
 *
 * Arrow strength = |β_edge| / Σ|β_parents| — the proportion of
 * the outcome variance explained by a specific causal path.
 */
export function arrowStrength(
  graph: CausalGraph,
  data: number[][],
  nodeNames: string[],
): Map<string, number> {
  const strengths = new Map<string, number>();

  for (const edge of graph.edges) {
    const parentIdx = nodeNames.indexOf(edge.source);
    const childIdx = nodeNames.indexOf(edge.target);
    if (parentIdx === -1 || childIdx === -1) continue;

    // Simple regression: child ~ parent
    const beta = simpleRegression(data, parentIdx, childIdx);
    strengths.set(`${edge.source}→${edge.target}`, Math.abs(beta));
  }

  // Normalize by total incoming strength per node
  const normalized = new Map<string, number>();
  const nodeTotals = new Map<string, number>();
  for (const [key, val] of strengths) {
    const target = key.split('→')[1]!;
    nodeTotals.set(target, (nodeTotals.get(target) ?? 0) + val);
  }
  for (const [key, val] of strengths) {
    const target = key.split('→')[1]!;
    const total = nodeTotals.get(target) ?? 1;
    normalized.set(key, total > 0 ? val / total : 0);
  }

  return normalized;
}

// ── Helpers ──────────────────────────────────────────────────────────

function simpleRegression(data: number[][], xIdx: number, yIdx: number): number {
  const n = data.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let r = 0; r < n; r++) {
    const x = data[r]![xIdx] ?? 0;
    const y = data[r]![yIdx] ?? 0;
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  return (n * sxy - sx * sy) / Math.max(1e-10, n * sxx - sx * sx);
}

function multipleRegression(
  data: number[][], yIdx: number, xIndices: number[],
): { coefs: number[]; intercept: number } {
  const n = data.length;
  const k = xIndices.length;
  const XtX = Array.from({ length: k }, () => Array(k).fill(0) as number[]);
  const Xty = Array(k).fill(0) as number[];
  let ySum = 0;

  for (let r = 0; r < n; r++) {
    const y = data[r]![yIdx] ?? 0;
    ySum += y;
    for (let i = 0; i < k; i++) {
      const xi = data[r]![xIndices[i]!] ?? 0;
      Xty[i] += xi * y;
      for (let j = 0; j < k; j++) XtX[i]![j] += xi * (data[r]![xIndices[j]!] ?? 0);
    }
  }

  const coefs = solveLinear(XtX, Xty);
  const yMean = ySum / n;
  const intercept = yMean - coefs.reduce((s, c, i) => s + c * colMean(data, xIndices[i]!), 0);
  return { coefs, intercept };
}

function colMean(data: number[][], col: number): number {
  let sum = 0, n = 0;
  for (const row of data) { const v = row[col]; if (v != null && !Number.isNaN(v)) { sum += v; n++; } }
  return n > 0 ? sum / n : 0;
}
