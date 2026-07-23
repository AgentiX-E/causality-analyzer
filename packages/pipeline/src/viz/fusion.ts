/**
 * FusionAnalyzer — multimodal RCA combining Metric + Trace + Log.
 *
 * Implements cascaded (nested) RCA: first coarse-grained Metric RCA,
 * then fine-grained Trace RCA within the identified scope.
 */
import type { RootCause, RootCausePath, RCAResult } from '@agentix-e/causality-analyzer-core';

export type FusionStrategy = 'weighted' | 'nested' | 'voting';

export interface FusionConfig { strategy: FusionStrategy; weights?: { metric: number; trace: number; log: number }; }

export class FusionAnalyzer {
  readonly config: FusionConfig;

  constructor(config: Partial<FusionConfig> = {}) {
    this.config = { strategy: config.strategy ?? 'weighted', weights: config.weights ?? { metric: 0.5, trace: 0.35, log: 0.15 } };
  }

  fuse(
    metricRCA: RCAResult | null,
    traceRCA: RCAResult | null,
    logRCA?: RCAResult | null,
  ): RCAResult {
    if (this.config.strategy === 'nested') {
      return this.nestedFuse(metricRCA, traceRCA, logRCA);
    }
    if (this.config.strategy === 'voting') {
      return this.votingFuse(metricRCA, traceRCA, logRCA);
    }
    return this.weightedFuse(metricRCA, traceRCA, logRCA);
  }

  private weightedFuse(
    metricRCA: RCAResult | null, traceRCA: RCAResult | null, logRCA?: RCAResult | null,
  ): RCAResult {
    const scores = new Map<string, { score: number; confidence: number; evidence: import('@agentix-e/causality-analyzer-core').Evidence[] }>();
    const process = (rca: RCAResult | null, weight: number) => {
      if (!rca) return;
      for (const rc of rca.rootCauses) {
        const cur = scores.get(rc.name) ?? { score: 0, confidence: 0, evidence: [] };
        cur.score += rc.score * weight;
        cur.confidence = Math.max(cur.confidence, rc.confidence);
        cur.evidence.push(...rc.evidence);
        scores.set(rc.name, cur);
      }
    };
    process(metricRCA, this.config.weights!.metric);
    process(traceRCA, this.config.weights!.trace);
    if (logRCA) process(logRCA, this.config.weights!.log);

    const rootCauses: RootCause[] = [];
    for (const [name, s] of scores) {
      rootCauses.push({ name, score: s.score, confidence: s.confidence, rank: 0, evidence: s.evidence });
    }
    rootCauses.sort((a, b) => b.score - a.score);
    rootCauses.forEach((r, i) => (r as { rank: number }).rank = i + 1);

    return {
      rootCauses: rootCauses.slice(0, 5),
      paths: (metricRCA?.paths ?? []).concat(traceRCA?.paths ?? []),
      metadata: { method: 'fusion_weighted', analyzedAt: Date.now(), durationMs: 0, extra: { strategy: 'weighted' } },
      toJSON() { return { rootCauses, paths: this.paths, metadata: this.metadata }; },
    };
  }

  private nestedFuse(
    metricRCA: RCAResult | null, traceRCA: RCAResult | null, _logRCA?: RCAResult | null,
  ): RCAResult {
    // Nested: use Metric RCA to define scope, then Trace RCA for precision
    if (metricRCA && traceRCA) {
      const metricTop = new Set(metricRCA.rootCauses.slice(0, 3).map(r => r.name));
      const filteredTrace = traceRCA.rootCauses.filter(r => metricTop.has(r.name));
      const combined = [...metricRCA.rootCauses, ...filteredTrace];
      combined.sort((a, b) => b.score - a.score);
      combined.forEach((r, i) => (r as { rank: number }).rank = i + 1);
      return {
        rootCauses: combined.slice(0, 5),
        paths: metricRCA.paths,
        metadata: { method: 'fusion_nested', analyzedAt: Date.now(), durationMs: 0, extra: { strategy: 'nested' } },
        toJSON() { return { rootCauses: this.rootCauses, paths: metricRCA.paths, metadata: this.metadata }; },
      };
    }
    return metricRCA ?? traceRCA ?? { rootCauses: [], paths: [], metadata: { method: 'fusion_empty', analyzedAt: Date.now(), durationMs: 0, extra: {} }, toJSON() { return {} as any; } };
  }

  /** Voting: majority vote across RCA methods, tie-breaking by score. */
  private votingFuse(
    metricRCA: RCAResult | null, traceRCA: RCAResult | null, logRCA?: RCAResult | null,
  ): RCAResult {
    const votes = new Map<string, { count: number; maxScore: number; maxConfidence: number; evidence: import('@agentix-e/causality-analyzer-core').Evidence[] }>();
    const tally = (rca: RCAResult | null) => {
      if (!rca) return;
      for (const rc of rca.rootCauses) {
        const cur = votes.get(rc.name) ?? { count: 0, maxScore: 0, maxConfidence: 0, evidence: [] };
        cur.count++;
        cur.maxScore = Math.max(cur.maxScore, rc.score);
        cur.maxConfidence = Math.max(cur.maxConfidence, rc.confidence);
        cur.evidence.push(...rc.evidence);
        votes.set(rc.name, cur);
      }
    };
    tally(metricRCA);
    tally(traceRCA);
    if (logRCA) tally(logRCA);

    const rootCauses: RootCause[] = [];
    for (const [name, v] of votes) {
      rootCauses.push({ name, score: v.count / 3, confidence: v.maxConfidence, rank: 0, evidence: v.evidence });
    }
    // Sort by vote count first, then by maxScore as tie-breaker
    rootCauses.sort((a, b) => {
      const aCnt = Math.round(a.score * 3), bCnt = Math.round(b.score * 3);
      if (aCnt !== bCnt) return bCnt - aCnt;
      // Tie: use maxScore from original votes (confidence as proxy)
      return (votes.get(b.name)?.maxScore ?? 0) - (votes.get(a.name)?.maxScore ?? 0);
    });
    rootCauses.forEach((r, i) => Object.assign(r, { rank: i + 1 }));

    return {
      rootCauses: rootCauses.slice(0, 5),
      paths: [...(metricRCA?.paths ?? []), ...(traceRCA?.paths ?? [])],
      metadata: { method: 'fusion_voting', analyzedAt: Date.now(), durationMs: 0, extra: { strategy: 'voting' } },
      toJSON() { return { rootCauses: this.rootCauses, paths: this.paths, metadata: this.metadata }; },
    };
  }
}
