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

// ── Causal Forest + Double ML ───────────────────────────────────
export { CausalForest } from './causal-forest.js';
export type { CausalForestConfig, CausalForestPrediction, FeatureImportance, CausalForestResult } from './causal-forest.js';
export { DoubleMLPLR, DoubleMLPLIV } from './double-ml.js';
export type { DoubleMLConfig } from './double-ml.js';

// ── Refutation Methods ─────────────────────────────────────────
export { runRefutationBatch, summarizeRefutation } from './refutation-base.js';
export type { Refuter, RefutationBatch } from './refutation-base.js';
export { refuteAddUnobservedCommonCause } from './refutation-advanced.js';
export { refuteRandomCommonCause, refuteDummyOutcome } from './refutation-extensions.js';

// ── Mediation ───────────────────────────────────────────────────
export { naturalDirectEffect, arrowStrength } from './mediation.js';
export type { MediationResult } from './mediation.js';

// ── Uplift Modeling ─────────────────────────────────────────────
export { evaluateUplift, upliftAtK, compareUpliftModels } from './uplift.js';
export type { UpliftObservation, UpliftCurvePoint, UpliftEvaluation } from './uplift.js';

// ── CATE + Fairness ─────────────────────────────────────────────
export { estimateCATE, estimateIPW, checkFairness } from './cate-fairness.js';

// ── Unified Refutation Portal ───────────────────────────────────
export { runAllRefutations, generateRefutationReport } from './refutation-portal.js';
export type { RefutationPortalResult, RefutationReport } from './refutation-portal.js';

// ── CATE Unification ────────────────────────────────────────────
export { unifiedCATE, compareCATEModels } from './cate-unified.js';
export type { CATEstimator, CATEstimate, CATEModelComparison } from './cate-unified.js';

// ── DML Estimator Family (I8-P1) ─────────────────────────────────
export { LinearDML, CausalForestDML, NonParamDML } from './dml-estimators.js';
export type { EffectInterval, DMLEstimatorConfig } from './dml-estimators.js';

// ── DR Estimator Family (I8-P1) ──────────────────────────────────
export { LinearDRLearner, ForestDRLearner } from './dr-estimators.js';
export type { DRConfig } from './dr-estimators.js';

// ── Meta-Learner Family (I8-P2) ───────────────────────────────────
export { SLearner, TLearner, XLearner } from './meta-learners.js';
export type { BaseLearner, MetaLearnerConfig } from './meta-learners.js';

// ── Policy Learning (I8-P3) ───────────────────────────────────────
export { PolicyTree, PolicyForest } from './policy-learning.js';
export type { PolicyConfig, PolicyEvaluation } from './policy-learning.js';

// ── Federated Learning + DP (I8-P3) ───────────────────────────────
export {
  laplaceMechanism,
  gaussianMechanism,
  computeATESensitivity,
  federatedDMLWithDP,
  secureAggregate,
  totalPrivacyCost,
} from './federated-dp.js';
export type {
  FederatedLearningConfig,
  FederatedNodeResult,
  FederatedAggregation,
} from './federated-dp.js';

// ── Bayesian Network Inference ─────────────────────────────────
export {
  cptToFactor,
  factorMultiply,
  factorMarginalize,
  factorReduce,
  factorNormalize,
  variableElimination,
  junctionTreeInference,
  loopyBeliefPropagation,
  likelihoodWeighting,
  gibbsSampling,
  estimateCPTs,
  bruteForceOracle,
  DirichletLearner,
} from './bayesian-network.js';
export type { Factor, CPT, Evidence, JunctionTreeResult, CredibleInterval } from './bayesian-network.js';
