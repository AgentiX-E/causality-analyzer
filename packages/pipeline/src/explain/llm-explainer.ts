/**
 * LLM-powered Causal Analysis Natural Language Explainer.
 *
 * Uses DeepSeek API to generate human-readable explanations for
 * causal analysis results (RCA, effect estimation, sensitivity).
 * Falls back gracefully to templated explanations when the LLM
 * is unavailable or an API key is not configured.
 *
 * API Key: Configured via DEEPSEEK_API_KEY environment variable.
 * Never hardcoded — .env is gitignored.
 *
 * @packageDocumentation
 */
import type { RCAResult } from '@agentix-e/causality-analyzer-core';
import { explainRCA as templateExplainRCA } from '../explainer.js';
import type {
  RCAExplanation,
  SensitivityExplanation,
  EstimateExplanation,
} from '../explainer.js';

const DEEPSEEK_API_BASE = 'https://api.deepseek.com/v1';
const DEFAULT_TIMEOUT_MS = 15000;
const MAX_RETRIES = 2;
const MAX_TOKENS = 1024;

interface LLMResponse {
  choices: Array<{ message: { content: string } }>;
}

/**
 * Call DeepSeek API with timeout and retry logic.
 */
async function callDeepSeek(
  prompt: string,
  apiKey: string,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<string | null> {
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(`${DEEPSEEK_API_BASE}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: 'deepseek-chat',
          messages: [
            {
              role: 'system',
              content:
                'You are an expert in causal inference and root cause analysis. Explain results concisely, accurately, and in plain language. Include confidence levels, potential caveats, and actionable recommendations.',
            },
            { role: 'user', content: prompt },
          ],
          max_tokens: MAX_TOKENS,
          temperature: 0.3,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!response.ok) {
        if (attempt < MAX_RETRIES) continue;
        return null;
      }

      const data = (await response.json()) as LLMResponse;
      return data.choices[0]?.message?.content ?? null;
    } catch {
      clearTimeout(timeout);
      if (attempt < MAX_RETRIES) continue;
      return null;
    }
  }
  return null;
}

/**
 * Build a structured prompt for RCA explanation.
 */
function buildRCAPrompt(result: RCAResult, method: string, nodeCount?: number): string {
  const rootCausesDesc = result.rootCauses
    .slice(0, 5)
    .map((rc, i) => `${i + 1}. ${rc.name} (score: ${rc.score?.toFixed(4) ?? 'N/A'})`)
    .join('\n');

  return `Explain the following root cause analysis result in plain language:

Analysis method: ${method}
Total nodes analyzed: ${nodeCount ?? 'unknown'}
Top root causes:\n${rootCausesDesc || '(none found)'}

Provide:
1. A one-sentence summary of the finding
2. Reasoning about why these were identified as root causes
3. Confidence level (high/medium/low)
4. Any caveats or limitations of the analysis
5. Actionable recommendations

Respond in plain text. Keep it concise.`;
}

/**
 * Build a structured prompt for sensitivity explanation.
 */
function buildSensitivityPrompt(
  eValue: number,
  partialR2: number,
  robustnessValue: number,
  estimate: number,
): string {
  return `Explain the following sensitivity analysis result:

E-value: ${eValue.toFixed(3)}
Partial R²: ${partialR2.toFixed(4)}
Robustness value: ${robustnessValue.toFixed(3)}
Estimated causal effect: ${estimate.toFixed(4)}

Provide:
1. How robust is this causal estimate to unmeasured confounding?
2. What does the E-value mean in plain language?
3. Is the robustness value acceptable?
4. Any recommendations

Respond concisely in plain text.`;
}

/**
 * Build a structured prompt for effect estimate explanation.
 */
function buildEstimatePrompt(
  method: string,
  ate: number,
  se: number,
  adjustmentSet: string[],
): string {
  return `Explain the following causal effect estimate:

Method: ${method}
ATE (Average Treatment Effect): ${ate.toFixed(4)}
Standard Error: ${se.toFixed(4)}
95% CI: [${(ate - 1.96 * se).toFixed(4)}, ${(ate + 1.96 * se).toFixed(4)}]
Adjustment variables: ${adjustmentSet.join(', ') || 'none'}

Provide:
1. What does this causal effect mean in plain language?
2. Is the effect statistically significant?
3. How confident should we be in this estimate?
4. Any caveats about the estimation method

Respond concisely in plain text.`;
}

// ── Public API ────────────────────────────────────────────────────────

/**
 * Generate an NL explanation for RCA results using LLM when available,
 * falling back to the deterministic templated explainer.
 */
export async function explainRCAWithLLM(
  result: RCAResult,
  method: string,
  nodeCount?: number,
): Promise<RCAExplanation> {
  const apiKey = process.env.DEEPSEEK_API_KEY;

  if (!apiKey || result.rootCauses.length === 0) {
    return templateExplainRCA(result, method, nodeCount);
  }

  const prompt = buildRCAPrompt(result, method, nodeCount);
  const llmResponse = await callDeepSeek(prompt, apiKey);

  if (!llmResponse) {
    return templateExplainRCA(result, method, nodeCount);
  }

  // Parse LLM response into structured format
  const template = templateExplainRCA(result, method, nodeCount);

  return {
    summary: extractSection(llmResponse, 'summary') ?? template.summary,
    reasoning: extractSection(llmResponse, 'reasoning') ?? llmResponse,
    ranking: result.rootCauses.slice(0, 5).map((rc, i) => ({
      name: rc.name,
      rank: i + 1,
      interpretation: `${rc.name} ranked #${i + 1} with score ${rc.score?.toFixed(4) ?? 'N/A'}`,
    })),
    caveats: template.caveats,
    confidence: estimateConfidence(llmResponse),
  };
}

/**
 * Generate an NL explanation for sensitivity analysis using LLM when available.
 */
export async function explainSensitivityWithLLM(
  eValue: number,
  partialR2: number,
  robustnessValue: number,
  estimate: number,
): Promise<SensitivityExplanation> {
  const apiKey = process.env.DEEPSEEK_API_KEY;

  if (!apiKey) {
    return {
      summary: `E-value = ${eValue.toFixed(3)}; Robustness = ${robustnessValue.toFixed(3)}`,
      interpretation: getSensitivityInterpretation(eValue, robustnessValue),
      actionableAdvice:
        eValue < 1.5
          ? 'Consider collecting additional data on potential confounders.'
          : 'The causal estimate is reasonably robust to unmeasured confounding.',
    };
  }

  const prompt = buildSensitivityPrompt(eValue, partialR2, robustnessValue, estimate);
  const llmResponse = await callDeepSeek(prompt, apiKey);

  if (!llmResponse) {
    return {
      summary: `E-value = ${eValue.toFixed(3)}; Robustness = ${robustnessValue.toFixed(3)}`,
      interpretation: getSensitivityInterpretation(eValue, robustnessValue),
      actionableAdvice:
        eValue < 1.5
          ? 'Consider collecting additional data on potential confounders.'
          : 'The causal estimate is reasonably robust to unmeasured confounding.',
    };
  }

  return {
    summary: llmResponse.split('\n')[0] ?? `E-value = ${eValue.toFixed(3)}`,
    interpretation: llmResponse,
    actionableAdvice:
      eValue < 1.5
        ? 'Collect more confounder data; E-value below threshold.'
        : 'Estimate is robust; no additional confounder data needed.',
  };
}

/**
 * Generate an NL explanation for effect estimates using LLM when available.
 */
export async function explainEstimateWithLLM(
  method: string,
  ate: number,
  se: number,
  adjustmentSet: string[],
): Promise<EstimateExplanation> {
  const apiKey = process.env.DEEPSEEK_API_KEY;

  if (!apiKey) {
    const sigStr = Math.abs(ate / Math.max(se, 1e-10)) > 1.96 ? 'significant' : 'not significant';
    return {
      summary: `ATE = ${ate.toFixed(4)} (SE=${se.toFixed(4)}, ${sigStr})`,
      interpretation: `The estimated causal effect is ${ate.toFixed(4)} with standard error ${se.toFixed(4)}. This is ${sigStr} at the 95% level.`,
      confidenceStatement: `Estimated using ${method} with adjustment for ${adjustmentSet.length} variable(s).`,
    };
  }

  const prompt = buildEstimatePrompt(method, ate, se, adjustmentSet);
  const llmResponse = await callDeepSeek(prompt, apiKey);

  if (!llmResponse) {
    const sigStr = Math.abs(ate / Math.max(se, 1e-10)) > 1.96 ? 'significant' : 'not significant';
    return {
      summary: `ATE = ${ate.toFixed(4)} (SE=${se.toFixed(4)}, ${sigStr})`,
      interpretation: `The estimated causal effect is ${ate.toFixed(4)} with standard error ${se.toFixed(4)}. This is ${sigStr} at the 95% level.`,
      confidenceStatement: `Estimated using ${method} with adjustment for ${adjustmentSet.length} variable(s).`,
    };
  }

  return {
    summary: llmResponse.split('\n')[0] ?? `ATE = ${ate.toFixed(4)}`,
    interpretation: llmResponse,
    confidenceStatement: `Estimated using ${method}; results interpreted by LLM.`,
  };
}

// ── Internal Helpers ──────────────────────────────────────────────────

function extractSection(text: string, sectionName: string): string | null {
  const lines = text.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]?.toLowerCase() ?? '';
    if (
      line.includes(sectionName) ||
      line.startsWith(`${i + 1}.`) ||
      line.startsWith('-')
    ) {
      return lines.slice(i, Math.min(i + 3, lines.length)).join('\n');
    }
  }
  return null;
}

function estimateConfidence(text: string): 'high' | 'medium' | 'low' {
  const lower = text.toLowerCase();
  if (lower.includes('high confidence') || lower.includes('strong')) return 'high';
  if (lower.includes('low confidence') || lower.includes('uncertain')) return 'low';
  return 'medium';
}

function getSensitivityInterpretation(eValue: number, robustness: number): string {
  const parts: string[] = [];
  parts.push(`E-value of ${eValue.toFixed(3)}: `);
  if (eValue >= 2) {
    parts.push('Strong robustness — even strong unmeasured confounders cannot explain the effect.');
  } else if (eValue >= 1.5) {
    parts.push('Moderate robustness — moderately strong confounders could potentially explain the effect.');
  } else {
    parts.push('Low robustness — relatively weak confounders could explain away the effect.');
  }
  parts.push(` Robustness value: ${robustness.toFixed(3)}.`);
  return parts.join('');
}
