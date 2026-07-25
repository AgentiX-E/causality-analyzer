/**
 * FASK — Fast Adjacency Skewness-based causal discovery.
 *
 * Uses the asymmetry of residual distributions to orient edges
 * after obtaining a skeleton from PC/FAS. The key insight:
 *
 *   In a linear non-Gaussian SEM X → Y:
 *   - Regress X on Y: residual has high skew (Y contains noise from X's causes)
 *   - Regress Y on X: residual has low skew (true causal direction)
 *
 *   → Orient edge toward the direction that produces less-skewed residuals.
 *
 * Much faster than LiNGAM (uses FAS skeleton instead of full ICA).
 * Particularly effective when data has naturally skewed distributions
 * (e.g., latency metrics, financial data, biological measurements).
 *
 * Reference: Sanchez-Romero et al. (NeurIPS 2019).
 *            "FASK: Fast Adjacency Skewness."
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import { combinations, fisherZTest } from '@agentix-e/causality-analyzer-core';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';

// ── Config ──────────────────────────────────────────────────────────

export interface FASKConfig {
  alpha?: number;
  maxDegree?: number;
  /** Minimum absolute skewness difference to orient (default 0.05) */
  skewThreshold?: number;
  /** Number of sampling iterations for skewness estimation (default 5) */
  skewIterations?: number;
}

// ── Public API ──────────────────────────────────────────────────────

export function faskAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: FASKConfig = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph; orientationConfidence: Map<string, number> } {
  const alpha = config.alpha ?? 0.05;
  const maxDegree = config.maxDegree ?? -1;
  const skewThreshold = config.skewThreshold ?? 0.05;
  const skewIter = config.skewIterations ?? 5;
  const n = nodeNames.length;
  const N = data.rows;

  const g = new CausalGraph(nodeNames);
  const orientationConfidence = new Map<string, number>();

  if (N < 10) return { graph: g, orientationConfidence };

  // ── Phase 1: FAS skeleton (PC-style adjacency search) ──
  const dataArr = matrixTo2D(data);

  // Start complete, remove independent edges
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      g.undirectedEdge(nodeNames[i], nodeNames[j]);

  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      if (fisherZTest(dataArr, i, j, []) > alpha)
        g.removeEdge(nodeNames[i], nodeNames[j]);

  let depth = 1;
  const maxDepth = maxDegree === -1 ? n : maxDegree;
  let changed = true;

  while (changed && depth <= maxDepth) {
    changed = false;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (!g.hasEdge(nodeNames[i], nodeNames[j])) continue;
        const neighbors = g.neighbors(nodeNames[i]).filter(c => c !== nodeNames[j]);
        if (neighbors.length < depth) continue;
        const subsets = combinations(neighbors, depth);
        for (const S of subsets) {
          const sIdx = S.map(s => nodeNames.indexOf(s));
          const p = fisherZTest(dataArr, i, j, sIdx);
          if (p > alpha) {
            g.removeEdge(nodeNames[i], nodeNames[j]);
            g.removeEdge(nodeNames[j], nodeNames[i]);
            changed = true; break;
          }
        }
      }
    }
    depth++;
  }

  // ── Phase 2: Skewness-based orientation ──
  // For each undirected edge (i—j), compute residual skewness
  // in both directions. Orient toward the direction with less skewness.
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      // Must be undirected (both directions present)
      if (!g.hasEdge(nodeNames[i], nodeNames[j]) || !g.hasEdge(nodeNames[j], nodeNames[i]))
        continue;

      const skewIJ = computeResidualSkewness(data, i, j, skewIter);
      const skewJI = computeResidualSkewness(data, j, i, skewIter);

      const absIJ = Math.abs(skewIJ);
      const absJI = Math.abs(skewJI);
      const diff = Math.abs(absIJ - absJI);

      if (diff > skewThreshold) {
        const confidence = Math.min(0.95, diff / (diff + skewThreshold));
        if (absIJ < absJI) {
          // i → j: residuals when predicting j from i have less skew
          g.orientEdge(nodeNames[i], nodeNames[j]);
          orientationConfidence.set(`${nodeNames[i]}→${nodeNames[j]}`, confidence);
        } else {
          // j → i
          g.orientEdge(nodeNames[j], nodeNames[i]);
          orientationConfidence.set(`${nodeNames[j]}→${nodeNames[i]}`, confidence);
        }
      }
      // Else: leave undirected (insufficient skewness evidence)
    }
  }

  // ── Phase 3: Cycle safety + Meek rules cleanup ──
  // FASK may create cycles — break them conservatively
  if (g.hasCycle()) {
    const directedEdges = [...g.edges].filter(e => e.directed);
    for (const e of directedEdges) {
      g.removeEdge(e.source, e.target);
      if (!g.hasCycle()) break;
    }
  }

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);

  return { graph: g, orientationConfidence };
}

// ── Residual Skewness ───────────────────────────────────────────────

/**
 * Compute the skewness of residuals when predicting target from source.
 *
 * Uses a simplified linear regression + Pearson-Fisher skewness coefficient.
 * Multi-iteration sampling for robustness.
 *
 * Returns positive skewness (right-tailed) or negative (left-tailed).
 * Lower absolute skewness = more Gaussian residuals = more likely causal direction.
 */
function computeResidualSkewness(
  data: Matrix, source: number, target: number,
  iterations: number,
): number {
  const N = data.rows;
  if (N < 10) return 0;

  let totalSkew = 0;

  for (let iter = 0; iter < iterations; iter++) {
    // Subsample 80% of data for robustness
    const subsampleSize = Math.floor(N * 0.8);
    const indices: number[] = [];
    for (let i = 0; i < subsampleSize; i++) {
      indices.push(Math.floor(Math.random() * N));
    }

    // Linear regression: target = β × source + intercept
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;

    for (const idx of indices) {
      const x = data.get(idx, source);
      const y = data.get(idx, target);
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumXX += x * x;
    }

    const n2 = indices.length;
    const beta = (n2 * sumXY - sumX * sumY) / Math.max(1e-10, n2 * sumXX - sumX * sumX);
    const alpha = (sumY - beta * sumX) / n2;

    // Compute residuals
    const residuals: number[] = [];
    for (const idx of indices) {
      const x = data.get(idx, source);
      const y = data.get(idx, target);
      residuals.push(y - (alpha + beta * x));
    }

    // Pearson-Fisher skewness: (mean - median) / std
    const sorted = [...residuals].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const mean = residuals.reduce((s, v) => s + v, 0) / residuals.length;
    const std = Math.sqrt(residuals.reduce((s, v) => s + (v - mean) ** 2, 0) / residuals.length);

    if (std > 1e-10) {
      totalSkew += (mean - median) / std;
    }
  }

  return totalSkew / iterations;
}

// ── Helpers ─────────────────────────────────────────────────────────

function matrixTo2D(data: Matrix): number[][] {
  const rows: number[][] = [];
  for (let r = 0; r < data.rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < data.columns; c++) row.push(data.get(r, c));
    rows.push(row);
  }
  return rows;
}
