/**
 * Multi-Source Root Cause Fusion Ranker.
 *
 * Fuses ranking signals from multiple detectors into a single,
 * weighted root cause ranking. Replaces single-source HeuristicPathRCA
 * with an ensemble that adapts weights per fault type via the
 * SelfEvolvingStrategy.
 *
 * Signal sources:
 *   - BOCPD timing:   "who failed first?"  (earliest changepoint = highest)
 *   - CUSUM magnitude: "how big is the shift?" (largest CUSUM = highest)
 *   - HeuristicPath:  "what does the causal graph say?" (propagation scoring)
 *   - Log errors:     "do the logs confirm?" (error count confirmation)
 *
 * Weights are auto-tuned per fault type via statistical feedback.
 * Default: [0.35, 0.35, 0.20, 0.10] for [bocpd, cusum, heuristicPath, log].
 *
 * @packageDocumentation
 */

import type { CausalGraph } from '../graph/causal-graph.js';
import type {
  ChangepointResult,
} from '../detect/bocd.js';

// ── Types ────────────────────────────────────────────────────────────────

export interface BOCDSignal {
  service: string;
  changepointIndex: number;
  magnitudeShift: number;
  confidence: number;
}

export interface CUSUMSignal {
  service: string;
  maxCusum: number;
  magnitudeShift: number;
}

export interface HeuristicPathSignal {
  component: string;
  score: number;
  isRoot: boolean;
}

export interface LogErrorSignal {
  service: string;
  errorCount: number;
  severity: 'critical' | 'error' | 'warning';
}

export interface FusionWeights {
  bocpd: number;
  cusum: number;
  heuristicPath: number;
  logError: number;
}

export interface MultiSourceInput {
  bocpdResults: BOCDSignal[];
  cusumResults: CUSUMSignal[];
  heuristicPathRanking: HeuristicPathSignal[];
  logErrors: LogErrorSignal[];
  serviceNames: string[];
  totalTimesteps: number;
}

export interface FusionRankEntry {
  component: string;
  score: number;
  isRoot: boolean;
  contributions: {
    bocpd: number;
    cusum: number;
    heuristicPath: number;
    logError: number;
  };
}

// ── Default Weights ──────────────────────────────────────────────────────

const DEFAULT_WEIGHTS: FusionWeights = {
  bocpd: 0.35,
  cusum: 0.35,
  heuristicPath: 0.20,
  logError: 0.10,
};

// ── MultiSourceRanker ─────────────────────────────────────────────────────

export class MultiSourceRanker {
  private weights: FusionWeights;

  constructor(weights?: Partial<FusionWeights>) {
    this.weights = { ...DEFAULT_WEIGHTS, ...weights };
  }

  /** Update fusion weights (e.g., from SelfEvolvingStrategy) */
  setWeights(weights: Partial<FusionWeights>): void {
    this.weights = { ...this.weights, ...weights };
    this.normalizeWeights();
  }

  get currentWeights(): FusionWeights { return { ...this.weights }; }

  /**
   * Compute fused root cause ranking from multiple signal sources.
   *
   * For each service s:
   *   score(s) = Σ wᵢ · normalize(rankᵢ(s))
   *
   * Normalization: min-max scaling to [0, 1] across top-k candidates.
   * Services not present in a given signal source get score 0 for that source.
   */
  rank(input: MultiSourceInput): FusionRankEntry[] {
    const n = input.serviceNames.length;
    if (n === 0) return [];

    // Normalize each signal source independently
    const bocpdScores = this.normalizeBOCPD(input.bocpdResults, input.serviceNames, input.totalTimesteps);
    const cusumScores = this.normalizeCUSUM(input.cusumResults, input.serviceNames);
    const hpScores = this.normalizeHeuristicPath(input.heuristicPathRanking, input.serviceNames);
    const logScores = this.normalizeLogErrors(input.logErrors, input.serviceNames);

    // Weighted fusion
    const entries: FusionRankEntry[] = [];
    for (const svc of input.serviceNames) {
      const bScore = bocpdScores.get(svc) ?? 0;
      const cScore = cusumScores.get(svc) ?? 0;
      const hScore = hpScores.get(svc) ?? 0;
      const lScore = logScores.get(svc) ?? 0;

      const total =
        this.weights.bocpd * bScore +
        this.weights.cusum * cScore +
        this.weights.heuristicPath * hScore +
        this.weights.logError * lScore;

      // Root detection: service has no ingoing BOCPD signal (earliest in graph)
      const isRoot = bocpdScores.has(svc) && bScore > 0.5;

      entries.push({
        component: svc,
        score: total,
        isRoot,
        contributions: {
          bocpd: bScore,
          cusum: cScore,
          heuristicPath: hScore,
          logError: lScore,
        },
      });
    }

    // Sort by score descending
    entries.sort((a, b) => b.score - a.score);

    return entries;
  }

  // ── Signal Normalization ──────────────────────────────────────────

  /**
   * BOCPD: score = (1 - cpIndex/totalTimesteps) × magnitudeShift × confidence
   * Earlier changepoint → higher score. Higher magnitude → higher score.
   */
  private normalizeBOCPD(
    signals: BOCDSignal[],
    allServices: string[],
    totalSteps: number,
  ): Map<string, number> {
    const scores = new Map<string, number>();
    const t = Math.max(1, totalSteps);

    for (const s of signals) {
      const timingScore = 1 - s.changepointIndex / t;
      const magScore = Math.min(1, s.magnitudeShift / 5);
      scores.set(s.service, timingScore * 0.6 + magScore * 0.4 * s.confidence);
    }

    return this.scaleTo01(scores, allServices);
  }

  /**
   * CUSUM: score = normalized maxCusum × magnitudeShift
   * Larger CUSUM statistic → more confident detection.
   */
  private normalizeCUSUM(
    signals: CUSUMSignal[],
    allServices: string[],
  ): Map<string, number> {
    const scores = new Map<string, number>();
    const maxCusum = Math.max(1, ...signals.map(s => s.maxCusum));

    for (const s of signals) {
      scores.set(s.service, (s.maxCusum / maxCusum) * Math.min(1, s.magnitudeShift / 3));
    }

    return this.scaleTo01(scores, allServices);
  }

  /**
   * HeuristicPath: use existing scores directly. Root nodes get a boost.
   */
  private normalizeHeuristicPath(
    ranking: HeuristicPathSignal[],
    allServices: string[],
  ): Map<string, number> {
    const scores = new Map<string, number>();
    const maxScore = Math.max(0.01, ...ranking.map(r => r.score));

    for (const r of ranking) {
      let score = r.score / maxScore;
      if (r.isRoot) score = Math.min(1, score * 1.2); // Root boost +20%
      scores.set(r.component, score);
    }

    return this.scaleTo01(scores, allServices);
  }

  /**
   * Log: score = errorCount / maxErrorCount × severity multiplier
   * Critical errors get 2× weight.
   */
  private normalizeLogErrors(
    errors: LogErrorSignal[],
    allServices: string[],
  ): Map<string, number> {
    const scores = new Map<string, number>();
    const maxCount = Math.max(1, ...errors.map(e => e.errorCount));

    for (const e of errors) {
      const severityMult = e.severity === 'critical' ? 2.0 : e.severity === 'error' ? 1.5 : 1.0;
      scores.set(e.service, Math.min(1, (e.errorCount / maxCount) * severityMult));
    }

    return this.scaleTo01(scores, allServices);
  }

  /** Min-max scale a map to [0, 1], filling missing keys with 0 */
  private scaleTo01(scores: Map<string, number>, allServices: string[]): Map<string, number> {
    const vals = [...scores.values()];
    const min = vals.length > 0 ? Math.min(...vals) : 0;
    const max = vals.length > 0 ? Math.max(...vals) : 1;
    const range = max - min || 1;

    const result = new Map<string, number>();
    for (const svc of allServices) {
      const raw = scores.get(svc);
      result.set(svc, raw !== undefined ? (raw - min) / range : 0);
    }
    return result;
  }

  /** Ensure all 4 weights sum to 1.0 */
  private normalizeWeights(): void {
    const sum = this.weights.bocpd + this.weights.cusum +
                this.weights.heuristicPath + this.weights.logError;
    if (sum <= 0) return;
    this.weights.bocpd /= sum;
    this.weights.cusum /= sum;
    this.weights.heuristicPath /= sum;
    this.weights.logError /= sum;
  }
}
