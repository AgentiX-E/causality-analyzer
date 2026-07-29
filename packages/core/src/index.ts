/**
 * @agentix-e/causality-analyzer-core
 *
 * Core package for Causality Analyzer â€? type definitions, interfaces,
 * plugin registry, configuration system, and universal data primitives.
 *
 * This package defines the contracts that every other package in the
 * causality-analyzer ecosystem depends on. It contains exactly ONE
 * implementation class (ColumnarTable) â€? by explicit design exception,
 * as it serves the same foundational role as `Array` in the runtime.
 */

// â”€â”€ Types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

// â”€â”€ Table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export type {
  TableSchema,
  ColumnNames,
  DataRow,
  StandardizeMethod,
  DiscretizeStrategy,
} from './table/index.js';

export { ColumnarTable } from './table/index.js';

// â”€â”€ Interfaces â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export type {
  IRelationalStore,
  IGraphStore,
  IMetricStore,
  IModelStore,
  IResultStore,
  ITransactionStore,
} from './interfaces/index.js';

// â”€â”€ Registry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export {
  PluginRegistry,
  PluginCategory,
  RegisterDetector,
  RegisterGraph,
  RegisterAnalyzer,
} from './registry/index.js';

// â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export type { ValidationResult, BaseConfigOptions } from './config/index.js';
export { BaseConfig } from './config/index.js';

// â”€â”€ Logger â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export { ConsoleLogger, NoopLogger, LogLevel } from './logger.js';
export type { Logger } from './logger.js';

// â”€â”€ DI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export type { CausalityAnalyzerConfig } from './di/index.js';

// â”€â”€ Errors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export {
  CausalityError,
  StoreError,
  ValidationError,
  ConfigError,
  NotFoundError,
  ConvergenceError,
  ErrorCode,
} from './errors.js';
export type { ErrorCodeType } from './errors.js';

// â”€â”€ Math â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export { solveLinear, solveLinearSafe, solveLinearCholesky, normalTail, normalCDFTail, normalCDF, erf, colMean, createRNG, combinations, fisherZTest, precomputeCorrelation, isMatrixSingular, partialCorrelationFromCov, invertMatrix, solveOLS, bicScore, gicScore, isBicScore, chiSquareTest, gSquareTest, _resetFisherZCache, _setFisherZCacheMax, digamma, partialCorrelationRaw, chiSquareCDF } from './math.js';
export { logGamma, bdeuScore, discretizeBDeu } from './bdeu.js';

// â”€â”€ Visualization Types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

// â”€â”€ Optimization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export { lbfgs, adam } from './optimize.js';
export type { LBFGSConfig, LBFGSResult, AdamConfig } from './optimize.js';

// ©¤©¤ Telemetry ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
export { Telemetry } from './telemetry.js';
export type { TelemetrySpan, TelemetryTracer, TelemetrySpanOptions, TelemetryCounter, TelemetryHistogram, TelemetryMeter } from './telemetry.js';

// ©¤©¤ Graph Similarity ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
export { computeFingerprint, cosineSimilarity, graphSimilarity } from './graph-similarity.js';
export type { CausalFingerprint } from './graph-similarity.js';

// ©¤©¤ Time-Series Types ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
export type {
  EdgeMark,
  CIBackend,
  CITestResult,
  CITestObserver,
  CPDAGInput,
  PCMCIPlusConfig,
  PCMCIPlusEdgeSummary,
  PCMCIPlusResult,
  TimeSeriesEdge,
  TimeSeriesGraph,
} from './types/timeseries.js';

// ©¤©¤ Distributed Types ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
export type {
  VectorClock,
  ClockOrder,
  DistributedCITask,
  DistributedCIResult,
  DistributedCITaskBatch,
  DistributedCIBatchResult,
  DistributedGraphVersion,
  DistributedDiscoveryConfig,
  SQLClusterConfig,
  GraphClusterConfig,
  FederatedConfig,
  FederatedStatistic,
  ClusterMode,
} from './types/distributed.js';

export {
  compareClocks,
  mergeClocks,
  incrementClock,
} from './types/distributed.js';
