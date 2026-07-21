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

// Graph
export { CausalGraph, pcAlgorithm, fisherZTest } from './graph/index.js';
export type { PCConfig } from './graph/index.js';

// Analyze (RCA)
export { BayesianRCA, RandomWalkRCA, HTRCA, FPGrowthRCA } from './analyze/index.js';
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
export { StructuralCausalModel, cateToRCA } from './gcm/index.js';

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
