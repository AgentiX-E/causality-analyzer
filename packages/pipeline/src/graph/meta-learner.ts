/**
 * Algorithm Meta-Learner — automatic causal discovery algorithm selection.
 *
 * Analyzes dataset characteristics (linearity, non-Gaussianity, sparsity,
 * dimensionality, sample size) to recommend the optimal causal discovery
 * algorithm from the available suite.
 *
 * The meta-learner uses a lightweight heuristic scoring system based on
 * the known strengths of each algorithm family:
 *
 *   - Constraint-based (PC, FCI): Best for linear-Gaussian, moderate d
 *   - Score-based (GES, BOSS, GRaSP): Good general-purpose
 *   - Continuous optimization (NOTEARS, DAGMA, GOLEM): Best for non-linear,
 *     differentiable structure
 *   - Functional models (LiNGAM, CAM-UV): Best for non-Gaussian distributions
 *   - Kernel methods (KCI-based): Best for non-linear, non-parametric
 *
 * Reference: The meta-learner approach is inspired by the OCDB benchmark
 *   (Zhou et al., 2024) finding that no single algorithm dominates all
 *   scenarios, and metadata-based selection achieves >80% accuracy.
 *
 * @packageDocumentation
 */

import { Matrix } from 'ml-matrix';

// ── Types ────────────────────────────────────────────────────────────────

/** Available causal discovery algorithm identifiers */
export type DiscoveryAlgorithm =
  | 'pc' | 'fci' | 'ges' | 'boss' | 'grasp'
  | 'notears' | 'dagma' | 'golem'
  | 'lingam' | 'cam-uv'
  | 'kci' | 'cdnod';

/** Data characteristics used for algorithm recommendation */
export interface DataCharacteristics {
  /** Number of samples */
  n: number;
  /** Number of variables */
  d: number;
  /** Average linear correlation magnitude (Pearson |ρ|) */
  linearity: number;
  /** Non-Gaussianity score (average excess kurtosis / 3) */
  nonGaussianity: number;
  /** Estimated sparsity (1 - avg |ρ|) */
  sparsity: number;
  /** Nonlinearity score (avg of mutual information / correlation discrepancy) */
  nonlinearity: number;
  /** Ratio d/n (curse of dimensionality indicator) */
  dimensionRatio: number;
}

/** Algorithm recommendation with confidence score */
export interface AlgorithmRecommendation {
  /** Recommended algorithm */
  algorithm: DiscoveryAlgorithm;
  /** Confidence [0, 1] */
  confidence: number;
  /** Human-readable rationale */
  rationale: string;
}

/** Complete meta-learner result */
export interface MetaLearnerResult {
  /** Data characteristics extracted */
  characteristics: DataCharacteristics;
  /** Top-3 algorithm recommendations (sorted by confidence) */
  recommendations: ReadonlyArray<AlgorithmRecommendation>;
  /** Recommended algorithm */
  best: DiscoveryAlgorithm;
}

// ── Characteristic Extraction ───────────────────────────────────────────

/**
 * Extract data characteristics for algorithm recommendation.
 *
 * @param data — (n × d) observation matrix
 * @returns computed characteristics
 */
export function extractCharacteristics(data: Matrix): DataCharacteristics {
  const n = data.rows;
  const d = data.columns;

  if (n < 2 || d < 2) {
    return { n, d, linearity: 0, nonGaussianity: 0, sparsity: 1, nonlinearity: 0, dimensionRatio: d / Math.max(1, n) };
  }

  // ── Linearity: average |Pearson correlation| ──────────────────────────
  let sumAbsCorr = 0;
  let pairCount = 0;

  for (let i = 0; i < d; i++) {
    for (let j = i + 1; j < d; j++) {
      const rho = pearsonCorrelation(data, i, j);
      sumAbsCorr += Math.abs(rho);
      pairCount++;
    }
  }
  const linearity = pairCount > 0 ? sumAbsCorr / pairCount : 0;

  // ── Non-Gaussianity: average excess kurtosis ──────────────────────────
  let sumKurt = 0;
  for (let j = 0; j < d; j++) {
    const col = data.getColumn(j);
    const m = col.reduce((a, b) => a + b, 0) / n;
    const m2 = col.reduce((s, v) => s + (v - m) ** 2, 0) / n;
    const m4 = col.reduce((s, v) => s + (v - m) ** 4, 0) / n;
    const kurt = m2 > 1e-10 ? m4 / (m2 * m2) - 3 : 0;
    sumKurt += Math.abs(kurt) / 3; // Normalize: Gaussian kurtosis = 0
  }
  const nonGaussianity = Math.min(1, sumKurt / d);

  // ── Sparsity: 1 - average |ρ| ────────────────────────────────────────
  const sparsity = 1 - linearity;

  // ── Nonlinearity: discrepancy between linear fit and actual ──────────
  let sumNonlinear = 0;
  let nlCount = 0;
  for (let i = 0; i < d; i++) {
    for (let j = i + 1; j < d; j++) {
      const xi = data.getColumn(i);
      const xj = data.getColumn(j);

      // Linear R²
      const rho = pearsonCorrelation(data, i, j);

      // Discretized mutual information proxy (rank-based)
      const ranksI = rankArray(xi);
      const ranksJ = rankArray(xj);
      const tau = kendallTau(ranksI, ranksJ);

      // If Kendall τ differs from Pearson ρ, there's nonlinearity
      const discrepancy = Math.abs(Math.abs(tau) - Math.abs(rho));
      sumNonlinear += discrepancy;
      nlCount++;
    }
  }
  const nonlinearity = nlCount > 0 ? Math.min(1, sumNonlinear / nlCount * 2) : 0;

  // ── Dimension ratio ──────────────────────────────────────────────────
  const dimensionRatio = d / n;

  return { n, d, linearity, nonGaussianity, sparsity, nonlinearity, dimensionRatio };
}

// ── Algorithm Recommendation ────────────────────────────────────────────

/**
 * Recommend the best causal discovery algorithm based on data characteristics.
 *
 * @param data — (n × d) observation matrix, or pre-extracted characteristics
 * @returns ranked recommendations
 */
export function recommendAlgorithm(
  data: DataCharacteristics | Matrix,
): MetaLearnerResult {
  const chars = 'rows' in data ? extractCharacteristics(data) : data;
  const { n, d, linearity, nonGaussianity, sparsity, nonlinearity, dimensionRatio } = chars;

  const scores = new Map<DiscoveryAlgorithm, number>();
  const rationales = new Map<DiscoveryAlgorithm, string>();

  // ── Score each algorithm family ───────────────────────────────────────

  // PC: best for linear-Gaussian, moderate d, sparse graphs
  let pcScore = 0.7;
  pcScore += linearity * 0.3; // High linearity favors PC
  pcScore -= nonGaussianity * 0.2; // Non-Gaussian hurts PC (Fisher Z assumes Gaussian)
  pcScore += sparsity * 0.2; // Sparse graphs favor constraint-based
  pcScore -= Math.min(0.3, dimensionRatio * 0.5); // High d/n ratio hurts
  scores.set('pc', pcScore);
  rationales.set('pc', `Constraint-based (PC): linearity=${linearity.toFixed(2)}, sparsity=${sparsity.toFixed(2)}`);

  // FCI: best when latent confounders may exist, linear setting
  let fciScore = pcScore - 0.1; // Slightly below PC (more conservative)
  fciScore += nonGaussianity * 0.1; // Can handle some non-Gaussian with KCI
  scores.set('fci', fciScore);
  rationales.set('fci', `Constraint-based with latent handling (FCI)`);

  // GES: good general-purpose score-based method
  let gesScore = 0.6;
  gesScore += linearity * 0.2;
  gesScore -= nonGaussianity * 0.1;
  gesScore += sparsity * 0.15;
  gesScore -= Math.min(0.3, dimensionRatio * 0.3);
  scores.set('ges', gesScore);
  rationales.set('ges', `Score-based (GES): general-purpose, BIC scoring`);

  // BOSS: permutation-based, good for smaller d
  let bossScore = 0.55;
  bossScore += sparsity * 0.2;
  bossScore -= Math.min(0.3, (d / 50) * 0.5); // Curse of dimensionality for permutation
  scores.set('boss', bossScore);
  rationales.set('boss', `Permutation-based (BOSS): d=${d}, sparsity=${sparsity.toFixed(2)}`);

  // GRaSP: L1-regularized, good for sparse graphs
  let graspScore = 0.55;
  graspScore += sparsity * 0.3;
  graspScore += linearity * 0.15;
  scores.set('grasp', graspScore);
  rationales.set('grasp', `L1-regularized (GRaSP): sparse structure detection`);

  // NOTEARS: best for non-linear, differentiable
  let notearsScore = 0.5;
  notearsScore += nonlinearity * 0.4; // Nonlinearity favors continuous optimization
  notearsScore += nonGaussianity * 0.1;
  notearsScore -= linearity * 0.1; // Pure linear is slightly worse for NOTEARS
  notearsScore -= Math.min(0.3, dimensionRatio * 0.3);
  scores.set('notears', notearsScore);
  rationales.set('notears', `Continuous optimization (NOTEARS): nonlinearity=${nonlinearity.toFixed(2)}`);

  // DAGMA: better for non-linear with M-matrix acyclicity
  let dagmaScore = notearsScore + 0.05;
  dagmaScore += nonlinearity * 0.1;
  scores.set('dagma', dagmaScore);
  rationales.set('dagma', `Continuous optimization (DAGMA): M-matrix acyclicity`);

  // GOLEM: likelihood-based, strong for well-specified models
  let golemScore = 0.5;
  golemScore += linearity * 0.2;
  golemScore += nonGaussianity * 0.1;
  scores.set('golem', golemScore);
  rationales.set('golem', `Likelihood-based (GOLEM): well-specified models`);

  // LiNGAM: best for non-Gaussian distributions
  let lingamScore = 0.55;
  lingamScore += nonGaussianity * 0.5; // Strong non-Gaussianity favors LiNGAM
  lingamScore -= linearity * 0.1; // Not needed if data is Gaussian
  lingamScore -= Math.min(0.2, dimensionRatio * 0.2);
  scores.set('lingam', lingamScore);
  rationales.set('lingam', `Non-Gaussian ICA (LiNGAM): nonGaussianity=${nonGaussianity.toFixed(2)}`);

  // CAM-UV: non-linear additive, non-Gaussian
  let camScore = 0.55;
  camScore += nonlinearity * 0.3;
  camScore += nonGaussianity * 0.2;
  camScore -= dimensionRatio * 0.2;
  scores.set('cam-uv', camScore);
  rationales.set('cam-uv', `Additive non-linear (CAM-UV): nonlinearity=${nonlinearity.toFixed(2)}`);

  // KCI: kernel-based, best for complex non-linear dependencies
  let kciScore = 0.45;
  kciScore += nonlinearity * 0.5;
  kciScore += nonGaussianity * 0.2;
  kciScore -= Math.min(0.3, (n / 1000) * 0.3); // Quadratic complexity hurts with large n
  scores.set('kci', kciScore);
  rationales.set('kci', `Kernel-based (KCI): complex non-linear detection`);

  // CD-NOD: non-stationary data, domain shifts
  let cdnodScore = 0.4;
  cdnodScore += nonGaussianity * 0.2;
  cdnodScore += nonlinearity * 0.1;
  scores.set('cdnod', cdnodScore);
  rationales.set('cdnod', `Non-stationary (CD-NOD): domain shift detection`);

  // ── Rank and return top 3 ─────────────────────────────────────────────
  const ranked = [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([alg, score]) => ({
      algorithm: alg,
      confidence: Math.min(1, Math.max(0, score)),
      rationale: rationales.get(alg) ?? '',
    }));

  return {
    characteristics: chars,
    recommendations: ranked.slice(0, 3),
    best: ranked[0].algorithm,
  };
}

// ── Statistical Helpers ─────────────────────────────────────────────────

function pearsonCorrelation(data: Matrix, colA: number, colB: number): number {
  const n = data.rows;
  const a = data.getColumn(colA);
  const b = data.getColumn(colB);
  let sumA = 0, sumB = 0, sumAB = 0, sumA2 = 0, sumB2 = 0;
  for (let i = 0; i < n; i++) {
    sumA += a[i]; sumB += b[i];
    sumAB += a[i] * b[i];
    sumA2 += a[i] * a[i]; sumB2 += b[i] * b[i];
  }
  const denom = Math.sqrt((n * sumA2 - sumA * sumA) * (n * sumB2 - sumB * sumB));
  return denom > 0 ? (n * sumAB - sumA * sumB) / denom : 0;
}

function rankArray(values: number[]): number[] {
  const n = values.length;
  const indexed = values.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);
  const ranks = new Array(n);
  for (let i = 0; i < n; i++) {
    ranks[indexed[i].i] = (i + 1) / n;
  }
  return ranks as number[];
}

function kendallTau(ranksA: number[], ranksB: number[]): number {
  const n = Math.min(ranksA.length, ranksB.length);
  let concordant = 0;
  let discordant = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const da = ranksA[i] - ranksA[j];
      const db = ranksB[i] - ranksB[j];
      if (da * db > 0) concordant++;
      else if (da * db < 0) discordant++;
    }
  }
  const total = concordant + discordant;
  return total > 0 ? (concordant - discordant) / total : 0;
}
