/**
 * BDeu Score (Bayesian Dirichlet equivalent uniform) for discrete data.
 *
 * Used in score-based causal discovery (GES/BOSS/GRaSP) when variables
 * are discrete/categorical rather than continuous.
 *
 * BDeu formula (Heckerman et al. 1995):
 *   Score = Σ_i [ log Γ(α/q_i) - log Γ(α/q_i + N_j)
 *            + Σ_k ( log Γ(α/(q_i·r_i) + N_ijk) - log Γ(α/(q_i·r_i)) ) ]
 *
 * where α = equivalent sample size (default 1.0).
 *
 * @packageDocumentation
 */

// ── Log-Gamma ───────────────────────────────────────────────────────

/**
 * Stirling's approximation for log-gamma.
 * Accurate to ~10⁻⁶ for x > 1.
 */
export function logGamma(x: number): number {
  if (x <= 0) return 0;
  if (x < 1) return logGamma(x + 1) - Math.log(x);
  // Stirling: ln Γ(x) ≈ (x-0.5)ln(x) - x + 0.5*ln(2π) + 1/(12x)
  return (x - 0.5) * Math.log(x) - x + 0.9189385332046727 + 1 / (12 * x);
}

// ── BDeu Score ──────────────────────────────────────────────────────

/**
 * Compute BDeu score for a target variable given its parents.
 *
 * @param data — discrete data matrix (n × d), values must be integers {0, 1, ..., r-1}
 * @param targetIdx — column index of target variable
 * @param parentIndices — column indices of parent variables
 * @param domainSizes — number of categories for each variable
 * @param alpha — equivalent sample size (default 1.0)
 */
export function bdeuScore(
  data: number[][],
  targetIdx: number,
  parentIndices: number[],
  domainSizes: number[],
  alpha: number = 1.0,
): number {
  const n = data.length;
  const ri = domainSizes[targetIdx] ?? 2; // number of states for target
  const parentDomainSizes = parentIndices.map(p => domainSizes[p] ?? 2);

  // Number of parent configurations
  const qi = parentDomainSizes.reduce((prod, s) => prod * s, 1);

  // Count occurrences: N_ijk = count(target=k | parents=j)
  const counts = new Map<string, number[]>();
  for (let i = 0; i < n; i++) {
    const parentKey = parentIndices.map(p => data[i]![p] ?? 0).join(',');
    if (!counts.has(parentKey)) counts.set(parentKey, new Array<number>(ri));
    const row = counts.get(parentKey)!;
    const val = data[i]![targetIdx] ?? 0;
    const idx = Math.min(val, ri - 1);
    row[idx] = (row[idx] ?? 0) + 1;
  }

  // Total per parent config
  const parentTotals = new Map<string, number>();
  for (const [key, arr] of counts) {
    parentTotals.set(key, arr.reduce((s, v) => s + v, 0));
  }

  // BDeu computation
  const alphaDivQi = alpha / qi;
  const alphaDivQiRi = alpha / (qi * ri);
  let score = 0;

  for (const [key, nijk] of counts) {
    const nij = parentTotals.get(key) ?? 0;
    // log Γ(α/qi) - log Γ(α/qi + nij)
    score += logGamma(alphaDivQi) - logGamma(alphaDivQi + nij);

    for (let k = 0; k < ri; k++) {
      const nijkVal = nijk[k] ?? 0;
      // log Γ(α/(qi·ri) + nijk) - log Γ(α/(qi·ri))
      score += logGamma(alphaDivQiRi + nijkVal) - logGamma(alphaDivQiRi);
    }
  }

  return score;
}

// ── Data Discretization ─────────────────────────────────────────────

/**
 * Discretize continuous data using equal-frequency binning.
 *
 * @param data — continuous data matrix (n × d)
 * @param numBins — number of discrete bins per variable (default 3)
 * @returns discretized data and domain sizes
 */
export function discretizeBDeu(data: number[][], numBins: number = 3): {
  discretized: number[][];
  domainSizes: number[];
} {
  const d = data[0]?.length ?? 0;
  if (d === 0) return { discretized: [], domainSizes: [] };

  const domainSizes: number[] = new Array<number>(d).fill(numBins);
  const discretized: number[][] = data.map((row: number[]): number[] => [...row]);

  for (let col = 0; col < d; col++) {
    const values = discretized.map(r => r[col]!).sort((a, b) => a - b);
    const n = values.length;

    // Equal-width binning
    const min = values[0]!;
    const max = values[n - 1]!;
    const range = max - min || 1e-10;

    for (let r = 0; r < discretized.length; r++) {
      const v = discretized[r]![col]!;
      discretized[r]![col] = Math.min(numBins - 1, Math.floor(((v - min) / range) * numBins));
    }
  }

  return { discretized, domainSizes };
}
