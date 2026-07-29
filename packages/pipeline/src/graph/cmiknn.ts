/**
 * CMIknn — Conditional Mutual Information via k-Nearest Neighbors.
 *
 * Implements the Kraskov-Stögbauer-Grassberger (KSG) estimator for
 * conditional mutual information I(X;Y|Z). Uses permutation testing to
 * build a null distribution and compute p-values.
 *
 * The KSG estimator is distribution-free: it can detect both linear and
 * nonlinear dependencies without assuming any parametric form.
 *
 * Complexity: O(n²·d) per CI test (k-NN search is quadratic without
 * spatial indexing). Use ParCorr for large datasets.
 *
 * Reference:
 *   Runge, J. (2018). "Conditional independence testing based on a
 *   nearest-neighbor estimator of conditional mutual information."
 *   AISTATS 2018.
 *
 * @packageDocumentation
 */

import { digamma, createRNG, type CITestResult } from '@agentix-e/causality-analyzer-core';

/** Default number of nearest neighbors for KSG estimator */
const DEFAULT_K = 5;

/** Default number of permutations for null distribution */
const DEFAULT_N_PERMUTATIONS = 200;

/**
 * Configuration for the CMIknn CI test.
 */
export interface CMIknnConfig {
  /** Number of nearest neighbors (default: 5) */
  k?: number;
  /** Number of permutations for p-value estimation (default: 200) */
  nPermutations?: number;
  /** Random seed for permutation test reproducibility */
  seed?: number;
}

/**
 * Run a conditional mutual information independence test.
 *
 * Tests H₀: X ⟂ Y | Z using the KSG nearest-neighbor estimator of
 * conditional mutual information, with permutation-based p-values.
 *
 * @param data - (n × totalCols) design matrix
 * @param xCol - column index for X
 * @param yCol - column index for Y
 * @param condCols - column indices for Z
 * @param config - k and permutation settings
 * @returns CITestResult with p-value and estimated CMI
 */
export function cmiknnTest(
  data: number[][],
  xCol: number,
  yCol: number,
  condCols: number[],
  config: CMIknnConfig = {},
): CITestResult {
  const k = config.k ?? DEFAULT_K;
  const nPermutations = config.nPermutations ?? DEFAULT_N_PERMUTATIONS;
  const n = data.length;

  if (n < k + 2) {
    return { pValue: 1, testStatistic: 0 };
  }

  // Observed CMI
  const observedCMI = estimateCMI(data, xCol, yCol, condCols, k);

  // Permutation test: shuffle Y and recompute CMI
  const rng = createRNG(config.seed ?? 42);
  const yValues = data.map(row => row[yCol]!);
  let countGreater = 0;

  for (let p = 0; p < nPermutations; p++) {
    // Fisher-Yates shuffle of Y (in-place on copy)
    const shuffled = [...yValues];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j]!, shuffled[i]!];
    }

    // Reconstruct data with shuffled Y
    const permData = data.map((row, idx) => {
      const newRow = [...row];
      newRow[yCol] = shuffled[idx]!;
      return newRow;
    });

    const permCMI = estimateCMI(permData, xCol, yCol, condCols, k);
    if (permCMI >= observedCMI) countGreater++;
  }

  // p-value = (countGreater + 1) / (nPermutations + 1) — avoids zero p-value
  const pValue = (countGreater + 1) / (nPermutations + 1);

  return { pValue, testStatistic: observedCMI };
}

/**
 * Estimate I(X;Y|Z) using the KSG (Kraskov-Stögbauer-Grassberger) estimator.
 *
 * The KSG estimator for conditional mutual information:
 *   I(X;Y|Z) = ψ(k) + ⟨ ψ(n_z+1) - ψ(n_{xz}+1) - ψ(n_{yz}+1) ⟩
 *
 * where ⟨·⟩ denotes averaging over data points, and n_z, n_{xz}, n_{yz}
 * are the numbers of points within the distance to the k-th neighbor in
 * the joint (X,Y,Z) space for each marginal subspace.
 *
 * @internal
 */
function estimateCMI(
  data: number[][],
  xCol: number,
  yCol: number,
  condCols: number[],
  k: number,
): number {
  const n = data.length;
  if (n === 0) return 0;

  // Build data arrays for X, Y, Z (Z is combined into a single vector per row)
  const xArr = data.map(row => row[xCol]!);
  const yArr = data.map(row => row[yCol]!);

  // For each point, find distances to k-th neighbor in joint (X,Y,Z) space
  let sum = 0;
  let validPoints = 0;

  for (let i = 0; i < n; i++) {
    // Compute distances from point i to all other points in joint space
    const distances: Array<{ idx: number; dist: number }> = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      let d2 = (xArr[i]! - xArr[j]!) ** 2 + (yArr[i]! - yArr[j]!) ** 2;
      for (const c of condCols) {
        d2 += (data[i]![c]! - data[j]![c]!) ** 2;
      }
      distances.push({ idx: j, dist: Math.sqrt(d2) });
    }

    // Sort and find k-th nearest neighbor distance
    distances.sort((a, b) => a.dist - b.dist);
    if (distances.length < k) continue;
    const rho = distances[k - 1]!.dist;
    if (rho === 0) continue; // Skip if k-th neighbor at zero distance

    // Count points within rho in each marginal space
    let n_xz = 0, n_yz = 0, n_z = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;

      let d2_xz = (xArr[i]! - xArr[j]!) ** 2;
      let d2_yz = (yArr[i]! - yArr[j]!) ** 2;
      let d2_z = 0;
      for (const c of condCols) {
        const diff = data[i]![c]! - data[j]![c]!;
        d2_xz += diff * diff;
        d2_yz += diff * diff;
        d2_z += diff * diff;
      }

      if (d2_xz < rho * rho) n_xz++;
      if (d2_yz < rho * rho) n_yz++;
      if (condCols.length > 0 && d2_z < rho * rho) n_z++;
      else if (condCols.length === 0) n_z++; // unconditional: count all
    }

    // KSG formula
    const psiTerm = digamma(k);
    const nzTerm = condCols.length > 0 ? digamma(n_z + 1) : 0;
    const nxzTerm = digamma(n_xz + 1);
    const nyzTerm = digamma(n_yz + 1);

    sum += psiTerm + nzTerm - nxzTerm - nyzTerm;
    validPoints++;
  }

  if (validPoints === 0) return 0;
  // CMI can be negative due to estimator variance; clamp to 0 for p-value
  return Math.max(0, sum / validPoints);
}
