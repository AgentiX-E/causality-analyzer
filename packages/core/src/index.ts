/**
 * @agentix-e/causality-analyzer-core
 *
 * Core package for Causality Analyzer — type definitions, interfaces,
 * plugin registry, configuration system, and universal data primitives.
 *
 * This package defines the contracts that every other package in the
 * causality-analyzer ecosystem depends on. It contains exactly ONE
 * implementation class (ColumnarTable) — by explicit design exception,
 * as it serves the same foundational role as `Array` in the runtime.
 */

// ── Types ──────────────────────────────────────────────────────────
export type {
  CausalEdge,
  CausalGraph,
  GraphMetadata,
  GraphVersion,
  Evidence,
  RootCause,
  RootCausePath,
  AnalysisMetadata,
  RCAResult,
  DetectionResult,
  IdentifiedEstimand,
  CausalEstimate,
  DomainKnowledge,
  MetricQuery,
  ResultQuery,
  ConditionalProbabilityTable,
  RegressionParams,
} from './types/index.js';

export { PipelineStage } from './types/index.js';

// ── Table ─────────────────────────────────────────────────────────
export type {
  TableSchema,
  ColumnNames,
  DataRow,
  StandardizeMethod,
  DiscretizeStrategy,
} from './table/index.js';

export { ColumnarTable } from './table/index.js';

// ── Interfaces ────────────────────────────────────────────────────
export type { IRelationalStore, IGraphStore } from './interfaces/index.js';

// ── Registry ──────────────────────────────────────────────────────
export {
  PluginRegistry,
  PluginCategory,
  RegisterDetector,
  RegisterGraph,
  RegisterAnalyzer,
} from './registry/index.js';

// ── Config ────────────────────────────────────────────────────────
export type { ValidationResult, BaseConfigOptions } from './config/index.js';
export { BaseConfig } from './config/index.js';

// ── DI ────────────────────────────────────────────────────────────
export type { CausalityAnalyzerConfig } from './di/index.js';

// ── Math ─────────────────────────────────────────────────────────
export { solveLinear, normalTail, normalCDFTail, normalCDF, erf, colMean, createRNG, combinations } from './math.js';

// ── Visualization Types ───────────────────────────────────────────
export type {
  GraphVizNode,
  GraphVisualizationData,
  TimeSeriesDataPoint,
  AnomalyRegion,
  TimeSeriesChartData,
  RankingEntry,
  PropagationPath,
  RCARankingData,
} from './types/index.js';
