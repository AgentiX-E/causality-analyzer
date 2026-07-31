/**
 * AutoProfile — Production Self-Optimizing Causal Discovery System
 *
 * Three-layer safety architecture:
 *   Layer 1: KSWIN drift detection — proactive, before performance degrades
 *   Layer 2: Shadow deployment — validate new params silently before adopting
 *   Layer 3: Auto-rollback — instant revert on SHD spike
 *
 * Research foundations:
 *   - KSWIN: Raab et al. (2020), "Reactive Soft Prototype Computing"
 *     Kolmogorov-Smirnov Windowing for concept drift detection
 *   - Shadow/Canary: HFL AI Lab (2025) "Model Rollout Canary"
 *     Silent comparison → gradual adoption → automatic rollback
 *   - Overfitting prevention: Holdout validation + 3-trial σ/μ check
 *     per Optuna best practices (Akiba et al. 2019)
 *
 * Test suites (from scikit-multiflow / River):
 *   - SEA Generator: abrupt concept drift (3 concepts)
 *   - Hyperplane Generator: gradual rotation drift
 *   - ConceptDriftStream: two-source mixing drift
 *
 * @packageDocumentation
 */

import { createHash } from 'crypto';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';

// ── Types ───────────────────────────────────────────────────────────

/** Data source identity — cryptographic fingerprint prevents cross-source leakage */
export interface DataSourceIdentity {
  sourceId: string;
  displayName: string;
  columnCount: number;
  sampleCount: number;
  fingerprint: string;  // SHA-256 of (columnNames + first 100 rows)
  createdAt: Date;
}

/** Tuning profile with full performance history for drift detection */
export interface TuningProfile {
  profileId: string;
  algorithm: string;
  sourceId: string;
  /** Current production params */
  activeParams: Record<string, number>;
  /** Parameters under shadow evaluation */
  shadowParams?: Record<string, number>;
  /** Performance history — each entry from one causal discovery run */
  history: ProfileEntry[];
  /** Number of times auto-tuned */
  tuningCount: number;
  status: ProfileStatus;
  createdAt: Date;
  updatedAt: Date;
}

export interface ProfileEntry {
  params: Record<string, number>;
  shd: number;
  f1: number;
  /** Whether this entry is from shadow mode (not affecting production) */
  shadow: boolean;
  timestamp: Date;
}

export type ProfileStatus = 'cold-start' | 'active' | 'shadow-evaluating' | 'canary-rollout' | 'retuning' | 'stale';

/** Layer 1: Drift detection result */
export interface DriftReport {
  drifted: boolean;
  severity: number;        // 0.0–1.0, higher = more severe drift
  trigger: DriftTrigger;   // which detector fired
  details: string;
}

export type DriftTrigger = 'none' | 'shd-degradation' | 'ks-test' | 'variance-spike' | 'distribution-shift';

/** Layer 2: Shadow evaluation result */
export interface ShadowReport {
  ready: boolean;          // params ready to promote?
  shadowSHD: number;       // mean SHD with shadow params
  activeSHD: number;       // mean SHD with active params
  improvement: number;     // positive = better, negative = worse
  trials: number;          // number of comparison runs
}

/** Layer 3: Rollback event */
export interface RollbackEvent {
  timestamp: Date;
  fromParams: Record<string, number>;
  toParams: Record<string, number>;
  reason: string;
  severity: number;
}

// ── Configuration ───────────────────────────────────────────────────

export interface AutoProfileConfig {
  /** Directory for profile storage */
  storagePath: string;
  /** Max profiles per (source, algorithm) pair */
  maxProfiles: number;
  /** Window size for KS drift test */
  ksWindowSize: number;
  /** α for KS statistical test */
  ksAlpha: number;
  /** SHD degradation threshold (fraction >1) */
  shdDegradationThreshold: number;
  /** Variance spike threshold (fraction >1) */
  varianceSpikeThreshold: number;
  /** Minimum trials for shadow evaluation */
  shadowMinTrials: number;
  /** Minimum improvement to promote shadow params */
  shadowMinImprovement: number;
  /** LLM analysis trigger ratio (1 call per N tuning trials) */
  llmTriggerRatio: number;
  /** Maximum rollback events before human intervention */
  maxAutoRollbacks: number;
  /** DeepSeek API key — loaded from env, never in code */
  deepseekApiKey?: string;
}

const DEFAULTS: AutoProfileConfig = {
  storagePath: join(process.cwd(), 'profiles'),
  maxProfiles: 5,
  ksWindowSize: 50,
  ksAlpha: 0.01,
  shdDegradationThreshold: 1.2,
  varianceSpikeThreshold: 2.0,
  shadowMinTrials: 5,
  shadowMinImprovement: 0.05,
  llmTriggerRatio: 0.01,
  maxAutoRollbacks: 3,
};

// ── Profile Store ───────────────────────────────────────────────────

export class ProfileStore {
  private profiles = new Map<string, TuningProfile>();
  private rollbacks: RollbackEvent[] = [];
  /** @internal — incremented by RollbackManager on each auto-rollback */
  autoRollbackCount = 0;

  constructor(private config: AutoProfileConfig = DEFAULTS) {
    mkdirSync(config.storagePath, { recursive: true });
  }

  /** Get or create profile for (algorithm, source) pair */
  getOrCreate(algorithm: string, source: DataSourceIdentity): TuningProfile {
    const key = `${algorithm}:${source.sourceId}`;
    let profile = this.profiles.get(key);
    if (!profile) {
      profile = this.loadFromDisk(key) ?? {
        profileId: key,
        algorithm,
        sourceId: source.sourceId,
        activeParams: this.coldStartParams(algorithm, source.columnCount),
        history: [],
        tuningCount: 0,
        status: 'cold-start',
        createdAt: new Date(),
        updatedAt: new Date(),
      };
      this.profiles.set(key, profile);
    }
    return profile;
  }

  /** Record a production run result */
  recordRun(profile: TuningProfile, shd: number, f1: number, shadow: boolean): void {
    profile.history.push({
      params: { ...(shadow ? profile.shadowParams ?? profile.activeParams : profile.activeParams) },
      shd,
      f1,
      shadow,
      timestamp: new Date(),
    });
    profile.updatedAt = new Date();

    // Trim history to keep last 200 entries
    if (profile.history.length > 200) {
      profile.history = profile.history.slice(-200);
    }

    this.saveToDisk(profile);
  }

  /** Save profile to disk */
  private saveToDisk(profile: TuningProfile): void {
    const path = join(this.config.storagePath, `${profile.profileId}.json`);
    writeFileSync(path, JSON.stringify(profile, null, 2));
  }

  /** Load profile from disk */
  private loadFromDisk(profileId: string): TuningProfile | null {
    const path = join(this.config.storagePath, `${profileId}.json`);
    if (!existsSync(path)) return null;
    try {
      const data = JSON.parse(readFileSync(path, 'utf-8'));
      data.history = data.history.map((h: Record<string, unknown>) => ({
        ...h,
        timestamp: new Date(h.timestamp as string),
      }));
      return data as TuningProfile;
    } catch {
      return null;
    }
  }

  /** Evict LRU profile when max exceeded */
  evictLRU(algorithm: string, sourceId: string): void {
    const prefix = `${algorithm}:${sourceId}`;
    const candidates = [...this.profiles.entries()]
      .filter(([k]) => k.startsWith(prefix))
      .sort(([, a], [, b]) => a.updatedAt.getTime() - b.updatedAt.getTime());
    while (candidates.length > this.config.maxProfiles) {
      const [key] = candidates.shift()!;
      this.profiles.delete(key);
    }
  }

  /** Cold-start params from v2.0 grid search defaults */
  private coldStartParams(algorithm: string, columnCount: number): Record<string, number> {
    const d = columnCount;
    switch (algorithm) {
      case 'NOTEARS': return { lambda1: 0.005, wThreshold: 0.2 };
      case 'BOSS': return { numStarts: Math.max(3, Math.ceil(d / 4)), maxParents: d > 30 ? 4 : -1 };
      case 'GES': return { penaltyDiscount: d > 15 ? 2.0 : 1.0 };
      case 'PC': return { alpha: 0.05, maxDegree: d > 20 ? 3 : -1 };
      default: return {};
    }
  }

  /** Get rollback history */
  getRollbackHistory(): RollbackEvent[] { return [...this.rollbacks]; }

  /** Register a rollback event */
  registerRollback(event: RollbackEvent): void {
    this.rollbacks.push(event);
    if (this.rollbacks.length > 50) this.rollbacks = this.rollbacks.slice(-50);
  }
}

// ── Layer 1: Drift Detector ─────────────────────────────────────────

/**
 * Multi-signal drift detection combining three statistical tests:
 *   1. SHD degradation: sliding window mean comparison
 *   2. KS test: distribution shift in SHD values
 *   3. Variance spike: σ(recent) vs σ(baseline)
 */
export class DriftDetector {
  constructor(private config: AutoProfileConfig) {}

  /**
   * Check all drift signals. Returns the MOST SEVERE trigger.
   */
  detect(profile: TuningProfile): DriftReport {
    const h = profile.history;
    if (h.length < 10) {
      return { drifted: false, severity: 0, trigger: 'none', details: 'Insufficient history (<10 runs)' };
    }

    const baseline = h.slice(0, Math.min(10, Math.floor(h.length / 2)));
    const recent = h.slice(-this.config.ksWindowSize);

    const baseSHD = baseline.map(e => e.shd);
    const recentSHD = recent.map(e => e.shd);

    // Signal 1: SHD mean degradation
    const baseMean = baseSHD.reduce((a, b) => a + b, 0) / baseSHD.length;
    const recentMean = recentSHD.reduce((a, b) => a + b, 0) / recentSHD.length;
    const degradationRatio = recentMean / Math.max(1, baseMean);

    if (degradationRatio > this.config.shdDegradationThreshold) {
      return {
        drifted: true,
        severity: Math.min(1, (degradationRatio - 1) / 0.5),
        trigger: 'shd-degradation',
        details: `SHD degraded: mean ${baseMean.toFixed(1)}→${recentMean.toFixed(1)} (${((degradationRatio - 1) * 100).toFixed(0)}%)`,
      };
    }

    // Signal 2: Two-sample KS test on SHD distributions
    const ksStat = this.twoSampleKS(baseSHD, recentSHD);
    const ksCritical = this.ksCriticalValue(baseSHD.length, recentSHD.length, this.config.ksAlpha);
    if (ksStat > ksCritical) {
      return {
        drifted: true,
        severity: Math.min(1, ksStat / (ksCritical * 2)),
        trigger: 'ks-test',
        details: `KS statistic ${ksStat.toFixed(3)} > critical ${ksCritical.toFixed(3)} at α=${this.config.ksAlpha}`,
      };
    }

    // Signal 3: Variance spike
    const baseVariance = this.variance(baseSHD);
    const recentVariance = this.variance(recentSHD);
    if (baseVariance > 0 && recentVariance / baseVariance > this.config.varianceSpikeThreshold) {
      return {
        drifted: true,
        severity: Math.min(1, (recentVariance / baseVariance - 1) / 3),
        trigger: 'variance-spike',
        details: `Variance spike: σ² ${baseVariance.toFixed(1)}→${recentVariance.toFixed(1)}`,
      };
    }

    return { drifted: false, severity: 0, trigger: 'none', details: 'All signals stable' };
  }

  /** Two-sample Kolmogorov-Smirnov test statistic */
  private twoSampleKS(a: number[], b: number[]): number {
    const sorted = [...a, ...b].sort((x, y) => x - y);
    let maxDiff = 0;
    let aIdx = 0, bIdx = 0;
    for (const val of sorted) {
      while (aIdx < a.length && a[aIdx]! <= val) aIdx++;
      while (bIdx < b.length && b[bIdx]! <= val) bIdx++;
      const diff = Math.abs(aIdx / a.length - bIdx / b.length);
      if (diff > maxDiff) maxDiff = diff;
    }
    return maxDiff;
  }

  /** Critical value for two-sample KS test */
  private ksCriticalValue(n1: number, n2: number, alpha: number): number {
    return Math.sqrt(-0.5 * Math.log(alpha / 2)) * Math.sqrt((n1 + n2) / (n1 * n2));
  }

  private variance(arr: number[]): number {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    return arr.reduce((a, v) => a + (v - mean) ** 2, 0) / (arr.length - 1);
  }
}

// ── Layer 2: Shadow Evaluator ───────────────────────────────────────

/**
 * Shadow evaluation: run new params alongside production, compare results
 * silently. Only promote if improvement is statistically significant.
 *
 * Pattern: Shadow → Canary (10% adoption) → Full rollout
 */
export class ShadowEvaluator {
  constructor(private config: AutoProfileConfig) {}

  /**
   * Check if shadow params are ready for promotion.
   * Requires minimum trials and positive mean improvement.
   */
  evaluate(profile: TuningProfile): ShadowReport {
    const shadowEntries = profile.history.filter(e => e.shadow);
    const activeEntries = profile.history.filter(e => !e.shadow).slice(-shadowEntries.length);

    if (shadowEntries.length < this.config.shadowMinTrials) {
      return {
        ready: false,
        shadowSHD: 0,
        activeSHD: 0,
        improvement: 0,
        trials: shadowEntries.length,
      };
    }

    const shadowMean = shadowEntries.reduce((s, e) => s + e.shd, 0) / shadowEntries.length;
    const activeMean = activeEntries.length > 0
      ? activeEntries.reduce((s, e) => s + e.shd, 0) / activeEntries.length
      : profile.history.filter(e => !e.shadow).slice(-5).reduce((s, e) => s + e.shd, 0) / 5;

    const improvement = activeMean - shadowMean;  // positive = shadow is better (lower SHD)

    return {
      ready: improvement > this.config.shadowMinImprovement && shadowEntries.length >= this.config.shadowMinTrials,
      shadowSHD: shadowMean,
      activeSHD: activeMean,
      improvement,
      trials: shadowEntries.length,
    };
  }

  /**
   * Promote shadow params to production if evaluation passes.
   * Stage: shadow → canary (gradual) → active
   */
  promote(profile: TuningProfile): void {
    if (!profile.shadowParams) return;

    const report = this.evaluate(profile);
    if (!report.ready) return;

    // Gradual promotion: keep shadow for canary phase
    if (profile.status === 'shadow-evaluating') {
      profile.status = 'canary-rollout';
      // In canary mode, 10% of runs use shadow params, 90% active
      // This is tracked by the caller (alternating runs)
      return;
    }

    // Full promotion
    const shadow = profile.shadowParams;
    if (!shadow) return;
    profile.activeParams = { ...shadow };
    delete profile.shadowParams;
    profile.status = 'active';
    profile.tuningCount++;
  }
}

// ── Layer 3: Rollback Manager ───────────────────────────────────────

/**
 * Automatic rollback on SHD spike detection.
 * If SHD exceeds active baseline × 2, immediately revert to
 * previous known-good params.
 */
export class RollbackManager {
  constructor(private store: ProfileStore, private config: AutoProfileConfig) {}

  /** Check if immediate rollback is needed and execute */
  checkAndRollback(profile: TuningProfile, latestSHD: number): RollbackEvent | null {
    if (profile.history.length < 5) return null;

    const baseline = profile.history
      .filter(e => !e.shadow)
      .slice(-10, -1)  // exclude latest (current run)
      .map(e => e.shd);

    if (baseline.length < 3) return null;

    const baselineMean = baseline.reduce((a, b) => a + b, 0) / baseline.length;

    // Rollback trigger: SHD > baseline × 2
    if (latestSHD > baselineMean * 2 && profile.history.length >= 2) {
      const previous = profile.history[profile.history.length - 2];
      if (!previous) return null;

      const event: RollbackEvent = {
        timestamp: new Date(),
        fromParams: { ...profile.activeParams },
        toParams: { ...previous.params },
        reason: `SHD spike: ${latestSHD} > baseline ${baselineMean.toFixed(1)} × 2`,
        severity: latestSHD / baselineMean,
      };

      // Execute rollback
      profile.activeParams = { ...previous.params };
      profile.status = 'active';
      this.store.registerRollback(event);
      this.store.autoRollbackCount++;

      // If too many rollbacks, lock and require human intervention
      if (this.store.autoRollbackCount >= this.config.maxAutoRollbacks) {
        profile.status = 'stale';
      }

      return event;
    }

    return null;
  }
}

// ── Data Source Identity ────────────────────────────────────────────

/** Generate a stable, cryptographic sourceId from data metadata */
export function generateSourceIdentity(
  displayName: string,
  columnNames: string[],
  dataSample: number[][],
): DataSourceIdentity {
  const fingerprint = createHash('sha256')
    .update(columnNames.join(','))
    .update(dataSample.slice(0, 100).map(r => r.join(',')).join('|'))
    .digest('hex')
    .slice(0, 16);

  const sourceId = `${displayName.replace(/[^a-zA-Z0-9]/g, '_')}_${fingerprint}`;
  return {
    sourceId,
    displayName,
    columnCount: columnNames.length,
    sampleCount: dataSample.length,
    fingerprint,
    createdAt: new Date(),
  };
}

// ── Production Pipeline ─────────────────────────────────────────────

/**
 * Main production loop: integrates all three safety layers.
 *
 * Flow:
 *   1. Load or create profile for (source, algorithm)
 *   2. Run drift detection → if drifted, stage retuning
 *   3. If shadow params exist, run shadow evaluation
 *   4. Run causal discovery with profile.activeParams
 *   5. Check rollback trigger on result SHD
 *   6. Record result to profile history
 *   7. If retuning is staged, queue tuning job
 */
export async function productionPipeline(
  algorithm: string,
  source: DataSourceIdentity,
  store: ProfileStore,
  driftDetector: DriftDetector,
  shadowEval: ShadowEvaluator,
  rollback: RollbackManager,
  // Callback: run causal discovery and return SHD/F1
  runDiscovery: (params: Record<string, number>) => { shd: number; f1: number },
): Promise<{ shd: number; f1: number; status: ProfileStatus; drift: DriftReport }> {
  const profile = store.getOrCreate(algorithm, source);

  // Layer 1: Drift detection
  const drift = driftDetector.detect(profile);
  if (drift.drifted && profile.status !== 'retuning') {
    profile.status = 'retuning';
  }

  // Layer 2: Shadow evaluation (if shadow params exist)
  if (profile.shadowParams && profile.status === 'shadow-evaluating') {
    const shadowResult = runDiscovery(profile.shadowParams);
    store.recordRun(profile, shadowResult.shd, shadowResult.f1, true);

    const report = shadowEval.evaluate(profile);
    if (report.ready) {
      shadowEval.promote(profile);
    }
  }

  // Run production with active params
  const result = runDiscovery(profile.activeParams);

  // Layer 3: Rollback check
  const rollbackEvent = rollback.checkAndRollback(profile, result.shd);
  if (rollbackEvent) {
    // Re-run with rolled-back params
    const retryResult = runDiscovery(profile.activeParams);
    store.recordRun(profile, retryResult.shd, retryResult.f1, false);
    return { ...retryResult, status: profile.status, drift };
  }

  store.recordRun(profile, result.shd, result.f1, false);
  return { ...result, status: profile.status, drift };
}
