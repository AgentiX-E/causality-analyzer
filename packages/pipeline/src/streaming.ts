/**
 * Streaming Data Processing for Causality Analyzer.
 *
 * Supports online (streaming) causal analysis with configurable
 * sliding-window batch accumulation. Compatible with AsyncIterable
 * data sources (WebSocket, Kafka, EventEmitter, etc.).
 *
 * @packageDocumentation
 */
import { CausalGraph } from './graph/causal-graph.js';
import { StatsDetector } from './detect/stats-detector.js';
import { HeuristicPathRCA } from './analyze/rca.js';
import type { RCAResult } from '@agentix-e/causality-analyzer-core';

// ── Types ────────────────────────────────────────────────────────────

export interface StreamingConfig {
  /** Window size in data points (batch accumulation) */
  windowSize: number;
  /** Slide interval — how often to emit results (in points) */
  slideInterval: number;
  /** Max history to keep for anomaly detection (points) */
  maxHistory: number;
}

const DEFAULT_CONFIG: StreamingConfig = {
  windowSize: 100,
  slideInterval: 10,
  maxHistory: 1000,
};

export interface StreamingResult {
  /** Detected anomalies in the current window */
  anomalies: Array<{ node: string; timestamp?: number; score: number }>;
  /** Root cause analysis for detected anomalies */
  rootCauses: RCAResult | null;
}

// ── Streaming Pipeline ───────────────────────────────────────────────

/**
 * Streaming causal analysis pipeline.
 *
 * Accumulates data points in a sliding window, runs anomaly detection
 * per-window, and triggers RCA when anomalies are detected.
 */
export class StreamingPipeline {
  private readonly graph: CausalGraph;
  private readonly config: StreamingConfig;
  private readonly detector: StatsDetector;
  private buffer: Map<string, number[]>;
  private pointCount = 0;

  constructor(graph: CausalGraph, config: Partial<StreamingConfig> = {}) {
    this.graph = graph;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.detector = new StatsDetector({ method: 'zscore' });
    this.buffer = new Map();
    for (const node of graph.nodes) this.buffer.set(node, []);
  }

  /**
   * Feed a single data point into the pipeline.
   * Returns analysis result if a full window has been processed.
   */
  ingest(point: Record<string, number>): StreamingResult | null {
    this.pointCount++;

    // Append to buffer
    for (const node of this.graph.nodes) {
      const values = this.buffer.get(node)!;
      values.push(point[node] ?? NaN);
      if (values.length > this.config.maxHistory) values.shift();
    }

    // Process when window is full
    if (this.pointCount % this.config.slideInterval !== 0) return null;
    if (this.pointCount < this.config.windowSize) return null;

    return this.processWindow();
  }

  /**
   * Process accumulated data as a single batch (AsyncIterable source).
   */
  async processStream(
    source: AsyncIterable<Record<string, number>>,
  ): Promise<StreamingResult[]> {
    const results: StreamingResult[] = [];
    for await (const point of source) {
      const result = this.ingest(point);
      if (result) results.push(result);
    }
    return results;
  }

  // ── Internal ───────────────────────────────────────────────────────

  private processWindow(): StreamingResult {
    // Extract window data
    const nodeNames = [...this.graph.nodes];
    const data: number[][] = [];
    const n = this.buffer.get(nodeNames[0]!)!.length;
    const windowSize = Math.min(this.config.windowSize, n);
    const offset = n - windowSize;

    for (let r = offset; r < n; r++) {
      const row: number[] = [];
      for (const node of nodeNames) {
        const values = this.buffer.get(node)!;
        row.push(values[r] ?? NaN);
      }
      data.push(row);
    }

    // Clean NaN rows
    const cleanData = data.filter(row => row.every(v => !Number.isNaN(v)));
    if (cleanData.length < 10) return { anomalies: [], rootCauses: null };

    // Train detector and detect anomalies
    this.detector.train(cleanData);
    const detection = this.detector.update(
      cleanData[cleanData.length - 1]!,
    );

    const anomalies: StreamingResult['anomalies'] = [];
    let hasAnomalies = false;

    if (detection.isAnomalous && detection.scores) {
      for (let i = 0; i < nodeNames.length; i++) {
        if (Math.abs(detection.scores[i] ?? 0) > 2) {
          hasAnomalies = true;
          anomalies.push({ node: nodeNames[i]!, score: detection.scores[i]! });
        }
      }
    }

    if (!hasAnomalies) return { anomalies: [], rootCauses: null };

    // Run RCA on anomalies
    const anomalousNodes = anomalies.map(a => a.node);
    const matrix = this.toMatrix(cleanData);
    const rca = new HeuristicPathRCA();
    rca.train(this.graph, new Set(anomalousNodes), matrix);
    const rootCauses = rca.findRootCauses(anomalousNodes);

    return { anomalies, rootCauses };
  }

  private toMatrix(data: number[][]): import('ml-matrix').Matrix {
    const { Matrix } = require('ml-matrix');
    return new Matrix(data);
  }
}
