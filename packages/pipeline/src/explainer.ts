/**
 * Natural Language Explanation Generator.
 *
 * Produces deterministic, templated human-readable explanations for
 * causal analysis results. No LLM dependency — pure template logic.
 *
 * Covers: RCA, sensitivity analysis, causal effect estimation,
 * counterfactual queries, and anomaly detection.
 *
 * Design decisions:
 * - Deterministic output (same input → same explanation)
 * - No external dependencies
 * - Plain-text output suitable for logs, dashboards, and audit trails
 * - Honest about model limitations and assumptions
 *
 * @packageDocumentation
 */

import type { RCAResult, DetectionResult } from '@agentix-e/causality-analyzer-core';

// ── Types ───────────────────────────────────────────────────────────────

export interface RCAExplanation {
  /** One-sentence summary */
  summary: string;
  /** Detailed reasoning (multi-paragraph) */
  reasoning: string;
  /** Ranked root causes with interpretation */
  ranking: Array<{ name: string; rank: number; interpretation: string }>;
  /** Model limitations and caveats */
  caveats: string[];
  /** Confidence level: high/medium/low */
  confidence: 'high' | 'medium' | 'low';
}

export interface SensitivityExplanation {
  summary: string;
  interpretation: string;
  actionableAdvice: string;
}

export interface EstimateExplanation {
  summary: string;
  interpretation: string;
  confidenceStatement: string;
}

// ── RCA Explanation ─────────────────────────────────────────────────────

/**
 * Generate a natural language explanation for RCA results.
 *
 * @param result — RCA result from any RCA analyzer
 * @param method — name of the RCA method used
 * @returns structured explanation with summary, reasoning, and caveats
 */
export function explainRCA(
  result: RCAResult,
  method: string,
  nodeCount?: number,
): RCAExplanation {
  const causes = result.rootCauses;
  if (causes.length === 0) {
    return {
      summary: 'No root causes were identified.',
      reasoning: `The analysis using ${method} did not find any statistically significant root causes. This could mean the anomalies are noise, the causal graph is incorrect, or the evidence is insufficient.`,
      ranking: [],
      caveats: [
        'Empty result may indicate insufficient data or model misspecification',
        'Consider checking the causal graph topology and anomaly thresholds',
      ],
      confidence: 'low',
    };
  }

  const topCause = causes[0]!;
  const confidenceLevel = topCause.score > 0.7 ? 'high' : topCause.score > 0.4 ? 'medium' : 'low';

  // Summary
  const summary = topCause.score > 0.6
    ? `The primary root cause is **${topCause.name}** with a confidence score of ${(topCause.score * 100).toFixed(0)}%.`
    : `The most likely root cause is **${topCause.name}** (score: ${(topCause.score * 100).toFixed(0)}%), though confidence is limited.`;

  // Reasoning
  let reasoning = '';
  reasoning += `Analysis performed using ${method}${nodeCount ? ` across ${nodeCount} nodes` : ''}.\n\n`;

  if (causes.length > 1 && causes[1]) {
    const gap = topCause.score - causes[1].score;
    if (gap > 0.3) {
      reasoning += `${topCause.name} is the clear primary root cause, with a score significantly higher (by ${(gap * 100).toFixed(0)} percentage points) than the next candidate. `;
    } else if (gap > 0.1) {
      reasoning += `${topCause.name} is the top-ranked root cause, with a moderate margin over ${causes[1].name} (difference: ${(gap * 100).toFixed(0)} percentage points). `;
    } else {
      reasoning += `${topCause.name} is only marginally ahead of ${causes[1].name} — the top candidates are close and the ranking may be sensitive to small changes in evidence. `;
    }
  }

  reasoning += `The root cause was identified by analyzing causal propagation paths from root nodes to the observed anomalous indicators. `;
  reasoning += `A higher score indicates stronger evidence that this node's anomalous behavior propagated to and caused the observed symptoms.`;

  // Ranking
  const ranking = causes.slice(0, 5).map(c => ({
    name: c.name,
    rank: c.rank,
    interpretation: interpretScore(c.score, c.name, c === topCause),
  }));

  // Caveats
  const caveats: string[] = [
    `This analysis is only as good as the declared causal graph. If the true causal structure differs, root cause attribution may be incorrect.`,
    `Confidence scores are model estimates, not frequentist probabilities. They should be interpreted in context.`,
  ];

  if (confidenceLevel === 'low') {
    caveats.push('Low confidence — investigate the top-ranked candidates manually before acting on results.');
  }

  return { summary, reasoning, ranking, caveats, confidence: confidenceLevel };
}

function interpretScore(score: number, name: string, isTop: boolean): string {
  if (score > 0.8) return `Very strong evidence — ${name} is likely the root cause`;
  if (score > 0.6) return `Strong evidence — ${name} is a probable root cause`;
  if (score > 0.4) return `Moderate evidence — ${name} should be investigated`;
  if (score > 0.2) return `Weak evidence — ${name} is a possible but uncertain cause`;
  return `Very weak evidence — ${name} is unlikely to be the cause`;
}

// ── Sensitivity Explanation ─────────────────────────────────────────────

/**
 * Explain sensitivity analysis results in plain English.
 */
export function explainSensitivity(
  eValue: number,
  partialR2Treatment: number,
  partialR2Outcome: number,
  robustnessValue: number,
): SensitivityExplanation {
  let summary: string;
  let interpretation: string;
  let actionableAdvice: string;

  if (robustnessValue > 2) {
    summary = `The causal conclusion is **robust** (RV = ${robustnessValue.toFixed(2)}).`;
    interpretation = `An unmeasured confounder would need to explain ${(partialR2Treatment * 100).toFixed(1)}% of the treatment variance and ${(partialR2Outcome * 100).toFixed(1)}% of the outcome variance to alter the conclusion. `;
    interpretation += `This corresponds to an E-value of ${eValue.toFixed(2)}, meaning the confounder would need a risk ratio of at least ${eValue.toFixed(2)} with both treatment and outcome to explain away the observed effect.`;
    actionableAdvice = 'The result is trustworthy. No further sensitivity checks are needed unless the domain suggests very strong unmeasured confounders.';
  } else if (robustnessValue > 1.5) {
    summary = `The causal conclusion is **moderately robust** (RV = ${robustnessValue.toFixed(2)}).`;
    interpretation = `A confounder explaining ${(partialR2Treatment * 100).toFixed(1)}% of treatment variance could alter the result. The E-value is ${eValue.toFixed(2)}.`;
    actionableAdvice = 'Proceed with caution. If there are known unmeasured confounders in your domain, consider collecting additional data or using instrumental variable methods.';
  } else {
    summary = `The causal conclusion is **sensitive** to unmeasured confounding (RV = ${robustnessValue.toFixed(2)}).`;
    interpretation = `Even a weak confounder explaining just ${(partialR2Treatment * 100).toFixed(1)}% of treatment variance could change the conclusion. The E-value of ${eValue.toFixed(2)} indicates low robustness.`;
    actionableAdvice = 'The result should NOT be treated as definitive. Collect data on potential confounders, use sensitivity bounds to qualify conclusions, or consider randomized experiments.';
  }

  return { summary, interpretation, actionableAdvice };
}

// ── Estimate Explanation ────────────────────────────────────────────────

/**
 * Explain causal effect estimates in plain English.
 */
export function explainEstimate(
  ate: number,
  se: number,
  method: string,
  outcomeName: string,
  treatmentName: string,
): EstimateExplanation {
  const ciLow = ate - 1.96 * se;
  const ciHigh = ate + 1.96 * se;
  const significant = ciLow > 0 || ciHigh < 0;

  const direction = ate > 0 ? 'increases' : 'decreases';
  const absATE = Math.abs(ate);
  const ciStr = `95% CI: [${ciLow.toFixed(3)}, ${ciHigh.toFixed(3)}]`;

  let summary: string;
  if (significant) {
    summary = `${treatmentName} ${direction} ${outcomeName} by ${absATE.toFixed(3)} on average (${method}). ${ciStr}.`;
  } else {
    summary = `No statistically significant effect detected: ${treatmentName} → ${outcomeName} = ${ate.toFixed(3)} (${method}). ${ciStr} — interval includes zero.`;
  }

  let interpretation = '';
  if (significant) {
    if (Math.abs(ate) > 2 * se) {
      interpretation = `The effect is both statistically significant and practically meaningful (effect size > 2× standard error). `;
    } else {
      interpretation = `The effect is statistically significant but may be small in practical terms. `;
    }
    interpretation += `There is a 95% probability that the true effect lies between ${ciLow.toFixed(3)} and ${ciHigh.toFixed(3)}.`;
  } else {
    interpretation = `The confidence interval includes zero, indicating that random chance could explain the observed difference. `;
    interpretation += `More data or a different identification strategy may be needed.`;
  }

  let confidenceStatement: string;
  if (significant && Math.abs(ate) > 2 * se) {
    confidenceStatement = 'High confidence in the direction and approximate magnitude of the effect.';
  } else if (significant) {
    confidenceStatement = 'Confident in the direction of the effect; magnitude may be imprecise.';
  } else {
    confidenceStatement = 'Insufficient evidence to draw a conclusion about the effect.';
  }

  return { summary, interpretation, confidenceStatement };
}

// ── Anomaly Explanation ─────────────────────────────────────────────────

/**
 * Explain anomaly detection results.
 */
export function explainDetection(result: DetectionResult): string {
  if (!result.isAnomalous) {
    return 'No anomaly detected. The metric value is within the expected range based on the trained model.';
  }

  const method = result.metadata?.method ?? 'statistical';
  const numAnomalous = result.labels.filter(l => l === 1).length;

  if (numAnomalous === 0) {
    return 'No anomaly detected across any metric dimension.';
  }

  return `Anomaly detected using ${method} method — ${numAnomalous} of ${result.labels.length} metrics exceeded the threshold. ` +
    `Max score: ${Math.max(...result.scores).toFixed(2)}. ` +
    'This suggests an unusual pattern that deviates from historical norms and may warrant causal root cause analysis.';
}
