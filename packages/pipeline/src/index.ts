export { standardize, discretize, extractWindows, imputeMean } from './data/standardizer.js';
export type { StandardizeMethod } from './data/standardizer.js';
export { StatsDetector } from './detect/stats-detector.js';
export type { StatsMethod, StatsDetectorConfig } from './detect/stats-detector.js';
export { SpectralResidualDetector } from './detect/spectral-residual.js';
export type { SRConfig } from './detect/spectral-residual.js';
export { SPOTDetector, DSPOTDetector } from './detect/spot.js';
export type { SPOTConfig, DSPOTConfig } from './detect/spot.js';
export { VotingDetector } from './detect/voting-detector.js';
export type { VotingStrategy, VotingDetectorConfig } from './detect/voting-detector.js';

// Graph — all 32 causal discovery algorithms
export {
  CausalGraph, pcAlgorithm, fisherZTest, pcMaxAlgorithm,
  gesAlgorithm, fciAlgorithm, gfciAlgorithm, rfciAlgorithm,
  directLiNGAM, icaLiNGAM, notearsAlgorithm, golemAlgorithm,
  dagmaAlgorithm, graspAlgorithm, bossAlgorithm,
  rcdAlgorithm, cdnodAlgorithm, mvpcAlgorithm,
  tsIcdAlgorithm, varLingam, pcmciAlgorithm, tsFciAlgorithm,
  timinoAlgorithm, ginDetect, faskAlgorithm, ccdAlgorithm,
  imagesAlgorithm, discoverClusters,
  stabilitySelection, starsSelection, exactSearch, kciTest,
  growShrink, targetedDiscovery, OnlinePC, detectCausalDrift, detectDriftFromGraphs,
} from './graph/index.js';
export type {
  PCConfig, GESConfig, TSConfig, TimeSeriesEdge, TSResult,
  KCIConfig, NOTEARSConfig, GRaSPConfig, RCDConfig,
  CDNODConfig, MVPCConfig, BOSSConfig, GFCIConfig,
  RFCIConfig, StabilityResult, StARSResult, FASKConfig,
  DAGMAConfig, CCDConfig, ClusterResult, GOLEMConfig,
  ICALiNGAMConfig, PCMCIEdge, PCMCIResult, PCMCIconfig,
  VARLiNGAMConfig, VARLiNGAMResult, TsFCIResult, TsFCIConfig,
  TiMINoResult, GINResult, OnlinePCConfig, StreamingGraphState, GraphChangeEvent,
  DriftDetectionResult,
} from './graph/index.js';

// Analyze (RCA)
export { HeuristicPathRCA, RandomWalkRCA, HTRCA, FPGrowthRCA } from './analyze/index.js';
export { RHTScorer, DAScorer, CIRCAPipeline } from './analyze/index.js';
export type { RHTConfig, DAConfig } from './analyze/index.js';

// Causal Inference
export {
  CausalAnalysis, identifyBackdoor, identifyFrontdoor,
  estimateLinearRegression, refutePlaceboTreatment,
  refuteDataSubset, refuteBootstrap,
} from './infer/index.js';
export type { RefutationResult, LinearRegressionEstimate } from './infer/index.js';

// Effect estimation (I7)
export {
  findBackdoorSet, adjustBackdoor, estimateFrontdoor,
  estimateIV, estimatePropensityScore, estimatePSMatching,
  estimateDoublyRobust,
} from './infer/index.js';

// GCM
export {
  StructuralCausalModel, cateToRCA,
  evaluateMechanismR2, evaluateMSE,
  shapleyAttribute, bootstrapRCA,
  detectMechanismChanges, distributionChangeRobust, changeAttributionCI,
  resitTest,
} from './gcm/index.js';
export type { RESITResult, RESITConfig, MechanismChangeResult } from './gcm/index.js';

// Visualization
export {
  buildGraphVizData, buildTimeseriesVizData, buildRankingVizData,
} from './viz/index.js';
export type {
  GraphVisualizationData, GraphVizNode,
  TimeSeriesChartData, TimeSeriesDataPoint, AnomalyRegion, ThresholdLine,
  RCARankingData, RankingEntry, RankingEvidence, PropagationPath,
} from './viz/index.js';
export { FusionAnalyzer } from './viz/index.js';
export type { FusionConfig, FusionStrategy } from './viz/index.js';

// ── Bayesian Network Inference (all engines + factor algebra) ──
export {
  cptToFactor, factorMultiply, factorMarginalize,
  factorReduce, factorNormalize,
  variableElimination, junctionTreeInference,
  loopyBeliefPropagation, likelihoodWeighting, gibbsSampling,
  estimateCPTs, bruteForceOracle,
  DirichletLearner,
} from './infer/index.js';
export type { Factor, CPT, Evidence, JunctionTreeResult, CredibleInterval } from './infer/index.js';

// ── Audit Trail ────────────────────────────────────────────────
export { AuditTrail } from './audit-trail.js';
export type { AuditEntryType, AuditVerifyResult } from './audit-trail.js';

// ── NL Explainer ───────────────────────────────────────────────
export {
  explainRCA, explainSensitivity, explainEstimate, explainDetection,
} from './explainer.js';
export type {
  RCAExplanation, SensitivityExplanation, EstimateExplanation,
} from './explainer.js';

// ── LLM-Powered Explainer ────────────────────────────────────
export {
  explainRCAWithLLM,
  explainSensitivityWithLLM,
  explainEstimateWithLLM,
} from './explain/llm-explainer.js';

// ── Observability ──────────────────────────────────────────────
export { AuditLogger, MetricsRegistry } from './observability.js';
export type { AuditEntry, MetricCounter, MetricHistogram } from './observability.js';

// ── Infrastructure ───────────────────────────────────────────
export { RateLimiter } from './rate-limiter.js';
export type { RateLimiterConfig, RateLimitResult, OverflowStrategy } from './rate-limiter.js';
export { EncryptedStore } from './encrypted-store.js';
export type { EncryptedStoreConfig } from './encrypted-store.js';

// ── HTTP Server ──────────────────────────────────────────────
export { CausalityServer } from './server.js';

// ── Streaming ────────────────────────────────────────────────
export { StreamingPipeline } from './streaming.js';
export type { StreamingConfig, StreamingResult } from './streaming.js';

// ── Health Check ─────────────────────────────────────────────
export { HealthChecker } from './health.js';
export type { HealthStatus, HealthCheckResult } from './health.js';

// ── Benchmark ────────────────────────────────────────────────
export { runBenchmark, computeSHD, formatBenchmarkTable, asiaGraph, sachsGraph, mBiasGraph, butterflyGraph, childGraph, alarmGraph, randomDAG, generateLinearData } from './benchmark.js';
export type { BenchmarkResult, AlgorithmResult } from './benchmark.js';
