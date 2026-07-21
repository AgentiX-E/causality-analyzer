/**
 * VotingDetector — ensemble anomaly detector.
 *
 * Combines multiple detectors via configurable voting strategy.
 * Supports: majority vote, weighted average, maximum consensus.
 */
import type { DetectionResult } from '@agentix-e/causality-analyzer-core';

export type VotingStrategy = 'majority' | 'weighted' | 'maximum';

export interface VotingDetectorConfig {
  strategy: VotingStrategy;
  /** Minimum number of detectors that must agree (majority only) */
  minAgreement?: number;
  /** Per-detector weights (weighted only) */
  weights?: number[];
}

export class VotingDetector {
  private detectors: Array<{ detect: (data: number[][]) => DetectionResult[]; update: (point: number[]) => DetectionResult }>;
  readonly config: VotingDetectorConfig;

  constructor(
    detectors: Array<{ detect: (data: number[][]) => DetectionResult[]; update: (point: number[]) => DetectionResult }>,
    config: Partial<VotingDetectorConfig> = {},
  ) {
    this.detectors = detectors;
    this.config = { strategy: config.strategy ?? 'majority', minAgreement: config.minAgreement, weights: config.weights };
  }

  /** Streaming: combine results from all detectors */
  update(point: number[]): DetectionResult {
    const results = this.detectors.map(d => d.update(point));
    return this.combine(results);
  }

  /** Batch detection */
  detect(data: number[][]): DetectionResult[] {
    return data.map(row => this.update(row));
  }

  private combine(results: DetectionResult[]): DetectionResult {
    if (results.length === 0) return { isAnomalous: false, labels: new Float64Array(0), scores: new Float64Array(0), timestamp: Date.now(), metadata: { method: 'voting_empty' } };
    const nLabels = results[0]!.labels.length;
    if (results.some(r => r.labels.length !== nLabels)) {
      throw new Error(`VotingDetector: all detectors must have the same label length (got ${[...new Set(results.map(r => r.labels.length))].join(', ')})`);
    }
    const { strategy } = this.config;
    if (strategy === 'majority') {
      const nAnomalous = results.filter(r => r.isAnomalous).length;
      const threshold = this.config.minAgreement ?? Math.ceil(results.length / 2);
      const isAnomalous = nAnomalous >= threshold;
      const labels = new Float64Array(results[0]?.labels.length ?? 0);
      const n = labels.length;
      for (let i = 0; i < n; i++) {
        let votes = 0;
        for (const r of results) { if (r.labels[i] === 1) votes++; }
        labels[i] = votes >= threshold ? 1 : 0;
      }
      return { isAnomalous, labels, scores: results[0]?.scores ?? new Float64Array(0), timestamp: Date.now(), metadata: { method: 'voting_majority', nDetectors: results.length, nAnomalous } };
    }

    if (strategy === 'maximum') {
      const best = results.reduce((a, b) => (b.scores[0] ?? 0) > (a.scores[0] ?? 0) ? b : a);
      return { ...best, metadata: { ...best.metadata, method: 'voting_maximum', nDetectors: results.length } };
    }

    // weighted
    const weights = this.config.weights ?? results.map(() => 1 / results.length);
    let weightedScore = 0;
    for (let i = 0; i < results.length; i++) {
      weightedScore += (results[i]?.scores[0] ?? 0) * (weights[i] ?? 0);
    }
    const isAnomalous = weightedScore > 0.5;
    return { isAnomalous, labels: results[0]?.labels ?? new Float64Array(0), scores: new Float64Array([weightedScore]), timestamp: Date.now(), metadata: { method: 'voting_weighted', nDetectors: results.length } };
  }
}
