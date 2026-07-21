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

// ── Graph Visualization ──────────────────────────────────────────────

/** Node in a causal graph visualization */
export interface GraphVizNode {
  id: string; label: string;
  type: 'root_cause' | 'anomaly' | 'intermediate' | 'healthy';
  score: number;
  isAnomalous: boolean;
}

/** Full causal graph visualization data */
export interface GraphVisualizationData {
  nodes: GraphVizNode[];
  edges: Array<{ source: string; target: string; weight: number; directed: boolean }>;
}

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

/** A single self-describing data point */
export interface TimeSeriesDataPoint {
  ts: number; value: number;
  /** Forecast lower bound (present only for forecasted points) */
  q10?: number;
  /** Forecast upper bound (present only for forecasted points) */
  q90?: number;
}

/** Anomaly region annotation */
export interface AnomalyRegion {
  start: number; end: number;
  severity: 'critical' | 'warning' | 'info';
  rootCause?: string;
}

/** Detection threshold line */
export interface ThresholdLine {
  metric: string; value: number;
  type: 'upper' | 'lower';
}

/** Time series chart data — self-describing, framework-agnostic */
export interface TimeSeriesChartData {
  series: Array<{ name: string; data: TimeSeriesDataPoint[] }>;
  anomalyRegions: AnomalyRegion[];
  thresholds?: ThresholdLine[];
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

// ── RCA Ranking Visualization ────────────────────────────────────────

/** Evidential detail for root cause ranking display */
export interface RankingEvidence {
  type: 'regression_residual' | 'parent_anomaly' | 'descendant_score' | 'frequent_pattern' | 'causal_effect';
  description: string; value: number;
}

/** Root cause ranking entry */
export interface RankingEntry { rank: number; name: string; score: number; confidence: number; evidence: RankingEvidence[]; }

/** Propagation path for Sankey/tree display */
export interface PropagationPath { root: string; path: string[]; score: number; }

/** Root cause ranking data — framework-agnostic */
export interface RCARankingData { rootCauses: RankingEntry[]; propagationPaths: PropagationPath[]; }

/** Build ranking visualization data from RCA result */
export function buildRankingVizData(rootCauses: RootCause[], paths: RootCausePath[]): RCARankingData {
  return {
    rootCauses: rootCauses.map(rc => ({ ...rc, evidence: rc.evidence as import('@agentix-e/causality-analyzer-core').Evidence[] })),
    propagationPaths: paths.map(p => ({ root: p.nodes[0] ?? '', path: [...p.nodes], score: p.score })),
  };
}
