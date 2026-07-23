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

// Sensitivity + do-calculus (I13)
export {
  eValueSensitivity, partialRSensitivity, robustnessValue,
} from './sensitivity.js';
export { identifyByDoCalculus } from './do-calculus.js';
export type { DoCalculusResult } from './do-calculus.js';

// Collider bias detection (I8)
export {
  detectColliderBias, findColliders, isColliderBias, removeColliderBiasedAdjustments,
} from './collider-bias.js';
export type { ColliderBiasWarning } from './collider-bias.js';
