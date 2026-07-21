export {
  CausalAnalysis,
  identifyBackdoor,
  identifyFrontdoor,
  estimateLinearRegression,
  refutePlaceboTreatment,
  refuteDataSubset,
  refuteBootstrap,
} from './causal-inference.js';
export type { RefutationResult, LinearRegressionEstimate } from './causal-inference.js';

// Effect estimation (I7)
export {
  findBackdoorSet,
  adjustBackdoor,
  estimateFrontdoor,
  estimateIV,
  estimatePropensityScore,
  estimatePSMatching,
  estimateDoublyRobust,
} from './effect-estimation.js';
