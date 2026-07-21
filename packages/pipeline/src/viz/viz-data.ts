/**
 * Visualization data structures — framework-agnostic, self-describing.
 *
 * Causality Analyzer does NOT render any UI. The viz module outputs
 * typed JSON-compatible data structures that consumers adapt to their
 * own frontend (React, Vue, ECharts, D3, Cytoscape, etc.).
 *
 * Every data point is self-describing — no implicit array-index alignment.
 */
import type { CausalEdge, RootCause, RootCausePath } from '@agentix-e/causality-analyzer-core';
import type {
  GraphVizNode, GraphVisualizationData,
  TimeSeriesDataPoint, AnomalyRegion, TimeSeriesChartData,
  RCARankingData, PropagationPath,
} from '@agentix-e/causality-analyzer-core';

// ── Pipeline-specific visualization types ──────────────────────────

/** Pipeline-specific evidence with richer metadata. */
export interface RankingEvidence {
  type: string;
  description: string;
  value: number;
  metadata?: Record<string, unknown>;
}

/** Pipeline-specific ranking entry (extends core with RankingEvidence). */
export interface RankingEntry {
  rank: number;
  name: string;
  score: number;
  confidence: number;
  evidence: RankingEvidence[];
}

// Re-export shared types for convenience
export type {
  GraphVizNode, GraphVisualizationData,
  TimeSeriesDataPoint, AnomalyRegion, TimeSeriesChartData,
  RCARankingData, PropagationPath,
};

export interface ThresholdLine {
  value: number;
  label: string;
  color: string;
  style: 'solid' | 'dashed' | 'dotted';
}

// ── Builders ──────────────────────────────────────────────────────

/** Convert a CausalGraph + RCA results to visualization data */
export function buildGraphVizData(
  nodes: string[], edges: CausalEdge[],
  rootCauses: RootCause[], anomalousNodes: string[],
): GraphVisualizationData {
  const rcNames = new Set(rootCauses.map(r => r.name));
  const anomSet = new Set(anomalousNodes);
  const vizNodes: GraphVizNode[] = nodes.map(id => ({
    id, label: id,
    type: rcNames.has(id) ? 'root_cause' : anomSet.has(id) ? 'anomaly' : 'intermediate',
    score: rootCauses.find(r => r.name === id)?.score ?? 0,
    isAnomalous: anomSet.has(id),
  }));
  return { nodes: vizNodes, edges: edges.map(e => ({ source: e.source, target: e.target, weight: e.weight, directed: e.directed })) };
}

// ── Time Series Visualization ────────────────────────────────────────

/** Detection threshold line — pipeline-specific with metric context. */
export interface ThresholdLine {
  metric: string; value: number;
  type: 'upper' | 'lower';
}

/** Build timeseries visualization data from metric data and detections */
export function buildTimeseriesVizData(
  metricData: Record<string, number[]>,
  timestamps: number[],
  anomalousIndices: number[],
  rootCause?: string,
): TimeSeriesChartData {
  const anomSet = new Set(anomalousIndices);
  const series = Object.entries(metricData).map(([name, values]) => ({
    name,
    data: values.map((value, i) => ({
      ts: timestamps[i] ?? i,
      value,
    })),
  }));
  const anomalyRegions: AnomalyRegion[] = [];
  if (anomalousIndices.length > 0) {
    const t0 = timestamps[anomalousIndices[0]!] ?? anomalousIndices[0]!;
    const t1 = timestamps[anomalousIndices[anomalousIndices.length - 1]!] ?? anomalousIndices[anomalousIndices.length - 1]!;
    anomalyRegions.push({ start: t0, end: t1, severity: 'critical', rootCause });
  }
  return { series, anomalyRegions };
}

// ── RCA Ranking Visualization — pipeline-specific types ────────────

/** Build ranking visualization data from RCA result */
export function buildRankingVizData(rootCauses: RootCause[], paths: RootCausePath[]): RCARankingData {
  return {
    rootCauses: rootCauses.map(rc => ({ ...rc, evidence: rc.evidence as import('@agentix-e/causality-analyzer-core').Evidence[] })),
    propagationPaths: paths.map(p => ({ root: p.nodes[0] ?? '', path: [...p.nodes], score: p.score })),
  };
}
