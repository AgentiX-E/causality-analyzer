/**
 * Causal Graph Drift Detection.
 *
 * Monitors a causal graph structure over time to detect when the
 * underlying causal mechanisms have shifted. Essential for AIOps
 * where system behavior changes during incidents, upgrades, or
 * configuration changes.
 *
 * Uses the Structural Hamming Distance (SHD) between successive
 * time windows to quantify drift magnitude.
 *
 * @packageDocumentation
 */
import { CausalGraph } from '../graph/causal-graph.js';

// ── Types ───────────────────────────────────────────────────────────

export interface DriftWindow {
  /** Start index in the time series */
  startIndex: number;
  /** End index (exclusive) */
  endIndex: number;
  /** Causal graph estimated from this window */
  graph: CausalGraph;
  /** SHD from previous window */
  driftScore: number;
  /** Timestamp of this window's center */
  timestamp: number;
}

export interface DriftDetectionResult {
  /** Whether a significant drift was detected */
  drifted: boolean;
  /** Maximum drift observed between any consecutive windows */
  maxDrift: number;
  /** Mean drift across windows */
  meanDrift: number;
  /** Per-window details */
  windows: DriftWindow[];
  /** The window where the largest jump occurred */
  driftPoint?: {
    beforeWindow: number;
    afterWindow: number;
    driftMagnitude: number;
  };
  /** Drift severity classification */
  severity: 'none' | 'mild' | 'moderate' | 'severe';
}

export interface DriftDetectorConfig {
  /** Window size for each sub-period */
  windowSize: number;
  /** Step size between consecutive windows */
  stepSize: number;
  /** Drift threshold for flagging (SHD relative to edge count) */
  threshold: number;
  /** Minimum observations per window */
  minWindowSize: number;
}

const DEFAULT_CONFIG: DriftDetectorConfig = {
  windowSize: 200,
  stepSize: 100,
  threshold: 0.2,
  minWindowSize: 50,
};

// ── Drift Detector ──────────────────────────────────────────────────

/**
 * Compute Structural Hamming Distance (SHD) between two graphs.
 * SHD = (# of edges in g1 but not in g2) + (# of edges in g2 but not in g1)
 *       + (# of incorrect edge directions)
 *
 * For undirected edges, we consider direction matches only for
 * edges present in both graphs.
 */
export function computeSHD(g1: CausalGraph, g2: CausalGraph): {
  shd: number;
  extraEdges: number;
  missingEdges: number;
  reversedEdges: number;
  normalizedSHD: number;
} {
  let extraEdges = 0;
  let missingEdges = 0;
  let reversedEdges = 0;

  const g1EdgeSet = new Set<string>();
  const g2EdgeSet = new Set<string>();

  for (const e of g1.edges) {
    g1EdgeSet.add(`${e.source}→${e.target}`);
  }
  for (const e of g2.edges) {
    g2EdgeSet.add(`${e.source}→${e.target}`);
  }

  // Edges in g1 not in g2 (missing from g2 or direction mismatch)
  for (const edgeStr of g1EdgeSet) {
    const [source, target] = edgeStr.split('→') as [string, string];
    const reverseEdge = `${target}→${source}`;

    if (g2EdgeSet.has(edgeStr)) {
      // Same edge, same direction — perfect match
      continue;
    } else if (g2EdgeSet.has(reverseEdge)) {
      // Same edge, reversed direction
      reversedEdges++;
    } else {
      // Edge missing entirely from g2 (or direction differs and no opposite)
      missingEdges++;
    }
  }

  // Edges in g2 not in g1
  for (const edgeStr of g2EdgeSet) {
    const [source, target] = edgeStr.split('→') as [string, string];
    const reverseEdge = `${target}→${source}`;

    if (!g1EdgeSet.has(edgeStr) && !g1EdgeSet.has(reverseEdge)) {
      extraEdges++;
    }
  }

  const shd = extraEdges + missingEdges + reversedEdges;
  const maxEdges = Math.max(g1.edges.length, g2.edges.length, 1);
  const normalizedSHD = shd / (2 * maxEdges);

  return { shd, extraEdges, missingEdges, reversedEdges, normalizedSHD };
}

/**
 * Detect causal graph drift over time using sliding windows.
 *
 * For each pair of consecutive windows, computes the SHD between
 * their causal graphs. Significant drift is flagged when the
 * normalized SHD exceeds the threshold.
 *
 * @param discoverFn — function that takes (data_window, nodeNames) → CausalGraph
 * @param data — time-series data matrix (rows = observations)
 * @param nodeNames — variable names
 * @param config — drift detection configuration
 */
export function detectCausalDrift(
  discoverFn: (data: number[][], nodeNames: string[]) => CausalGraph,
  data: number[][],
  nodeNames: string[],
  config: Partial<DriftDetectorConfig> = {},
): DriftDetectionResult {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const n = data.length;

  if (n < cfg.windowSize * 2) {
    return {
      drifted: false,
      maxDrift: 0,
      meanDrift: 0,
      windows: [],
      severity: 'none',
    };
  }

  const windows: DriftWindow[] = [];
  const driftScores: number[] = [];

  let prevGraph: CausalGraph | null = null;

  for (let start = 0; start + cfg.windowSize <= n; start += cfg.stepSize) {
    const end = start + cfg.windowSize;
    if (end - start < cfg.minWindowSize) continue;

    const windowData = data.slice(start, end);
    const graph = discoverFn(windowData, nodeNames);

    let driftScore = 0;
    if (prevGraph) {
      const result = computeSHD(prevGraph, graph);
      driftScore = result.normalizedSHD;
      driftScores.push(driftScore);
    }

    windows.push({
      startIndex: start,
      endIndex: end,
      graph,
      driftScore,
      timestamp: Math.floor((start + end) / 2),
    });

    prevGraph = graph;
  }

  if (windows.length < 2) {
    return {
      drifted: false,
      maxDrift: 0,
      meanDrift: 0,
      windows,
      severity: 'none',
    };
  }

  const maxDrift = Math.max(...driftScores);
  const meanDrift = driftScores.reduce((s, v) => s + v, 0) / driftScores.length;

  // Find the drift point
  let driftPoint: DriftDetectionResult['driftPoint'];
  const maxDriftIdx = driftScores.indexOf(maxDrift);
  if (maxDriftIdx >= 0 && maxDrift > cfg.threshold) {
    driftPoint = {
      beforeWindow: maxDriftIdx,
      afterWindow: maxDriftIdx + 1,
      driftMagnitude: maxDrift,
    };
  }

  // Classify severity
  let severity: DriftDetectionResult['severity'];
  if (maxDrift < cfg.threshold) severity = 'none';
  else if (maxDrift < cfg.threshold * 2) severity = 'mild';
  else if (maxDrift < cfg.threshold * 4) severity = 'moderate';
  else severity = 'severe';

  return {
    drifted: maxDrift >= cfg.threshold,
    maxDrift,
    meanDrift,
    windows,
    driftPoint,
    severity,
  };
}

/**
 * Convenience: detect drift using a pre-computed list of graphs
 * (one per time window).  Avoids calling the discovery function.
 */
export function detectDriftFromGraphs(
  graphs: CausalGraph[],
  config: Partial<DriftDetectorConfig> = {},
): DriftDetectionResult {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  if (graphs.length < 2) {
    return {
      drifted: false,
      maxDrift: 0,
      meanDrift: 0,
      windows: [],
      severity: 'none',
    };
  }

  const windows: DriftWindow[] = [];
  const driftScores: number[] = [];

  for (let i = 0; i < graphs.length; i++) {
    const graph = graphs[i]!;
    let driftScore = 0;
    if (i > 0) {
      const result = computeSHD(graphs[i - 1]!, graph);
      driftScore = result.normalizedSHD;
      driftScores.push(driftScore);
    }

    windows.push({
      startIndex: i * cfg.stepSize,
      endIndex: i * cfg.stepSize + cfg.windowSize,
      graph,
      driftScore,
      timestamp: i,
    });
  }

  const maxDrift = Math.max(...driftScores, 0);
  const meanDrift = driftScores.length > 0
    ? driftScores.reduce((s, v) => s + v, 0) / driftScores.length
    : 0;

  const maxDriftIdx = driftScores.indexOf(maxDrift);
  const driftPoint = maxDriftIdx >= 0 && maxDrift > cfg.threshold
    ? { beforeWindow: maxDriftIdx, afterWindow: maxDriftIdx + 1, driftMagnitude: maxDrift }
    : undefined;

  let severity: DriftDetectionResult['severity'];
  if (maxDrift < cfg.threshold) severity = 'none';
  else if (maxDrift < cfg.threshold * 2) severity = 'mild';
  else if (maxDrift < cfg.threshold * 4) severity = 'moderate';
  else severity = 'severe';

  return {
    drifted: maxDrift >= cfg.threshold,
    maxDrift,
    meanDrift,
    windows,
    driftPoint,
    severity,
  };
}
