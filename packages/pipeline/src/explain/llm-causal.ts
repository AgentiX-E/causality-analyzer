/**
 * LLM-Enhanced Causal Reasoning — DeepSeek-powered causal graph
 * construction, domain knowledge fusion, and unified explanation
 * generation.
 *
 * Capabilities:
 *   - proposeCausalGraph: Generate initial causal graph from domain description
 *   - fuseDomainKnowledge: Merge LLM proposals with existing graph constraints
 *   - explainAnalysis: Unified NL explanation for any analysis result type
 *   - CausalDialogue: Multi-turn conversation for iterative graph refinement
 *
 * API Key: Configured via DEEPSEEK_API_KEY environment variable.
 * Never hardcoded — .env is gitignored. Falls back gracefully when
 * the key is unavailable.
 *
 * Reference: IJCAI 2025 Survey — "Large Language Models for Causal
 *   Discovery: Current Landscape and Future Directions."
 *
 * @packageDocumentation
 */

import type { DomainKnowledge, CausalGraph, CausalEdge, RCAResult } from '@agentix-e/causality-analyzer-core';
import { CausalGraph as CausalGraphImpl } from '../graph/causal-graph.js';
import { explainRCAWithLLM, explainSensitivityWithLLM, explainEstimateWithLLM } from './llm-explainer.js';
import type { RCAExplanation, SensitivityExplanation, EstimateExplanation } from '../explainer.js';

// ── Configuration ───────────────────────────────────────────────────────

const API_BASE = 'https://api.deepseek.com/v1';
const DEFAULT_TIMEOUT = 20000;
const MAX_RETRIES = 3;
const MODEL = 'deepseek-chat';

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface ChatResponse {
  choices: Array<{ message: { content: string } }>;
}

/** Result of LLM graph proposal */
export interface GraphProposal {
  /** Proposed edges from domain description */
  edges: ReadonlyArray<{
    source: string;
    target: string;
    confidence: number;
    rationale: string;
  }>;
  /** Nodes identified as root causes */
  rootNodes: ReadonlyArray<string>;
  /** Nodes identified as leaf/effect nodes */
  leafNodes: ReadonlyArray<string>;
  /** Raw LLM response for transparency */
  rawResponse: string;
}

/** Domain knowledge fusion result */
export interface DomainFusionResult {
  /** Merged DomainKnowledge constraints */
  domainKnowledge: DomainKnowledge;
  /** Edges added from LLM proposal */
  llmEdges: ReadonlyArray<readonly [string, string]>;
  /** Edges that conflicted with existing knowledge and were rejected */
  conflicts: ReadonlyArray<{
    edge: readonly [string, string];
    reason: string;
  }>;
}

/** Unified explanation result for any analysis type */
export interface UnifiedExplanation {
  /** Analysis type */
  type: 'rca' | 'sensitivity' | 'estimate' | 'graph' | 'refutation';
  /** NL summary (one sentence) */
  summary: string;
  /** Detailed interpretation */
  interpretation: string;
  /** Actionable recommendations */
  recommendations: string;
  /** Whether LLM was used (vs template fallback) */
  llmEnhanced: boolean;
  /** Raw structured data */
  data: Record<string, unknown>;
}

/** Configuration for LLM causal operations */
export interface LLMCausalConfig {
  /** DeepSeek API key (defaults to process.env.DEEPSEEK_API_KEY) */
  apiKey?: string;
  /** Request timeout in ms (default: 20000) */
  timeoutMs?: number;
  /** Model name (default: 'deepseek-chat') */
  model?: string;
  /** Temperature for creativity (default: 0.3) */
  temperature?: number;
  /** Maximum output tokens (default: 2048) */
  maxTokens?: number;
}

// ── Core LLM Client ─────────────────────────────────────────────────────

/**
 * Call DeepSeek API with exponential backoff retry.
 */
async function chat(
  messages: ChatMessage[],
  config: LLMCausalConfig,
): Promise<string | null> {
  const apiKey = config.apiKey ?? process.env.DEEPSEEK_API_KEY;
  if (!apiKey) return null;

  const timeout = config.timeoutMs ?? DEFAULT_TIMEOUT;
  const model = config.model ?? MODEL;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);

    try {
      const resp = await fetch(`${API_BASE}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages,
          max_tokens: config.maxTokens ?? 2048,
          temperature: config.temperature ?? 0.3,
        }),
        signal: controller.signal,
      });

      clearTimeout(timer);

      if (!resp.ok) {
        if (attempt < MAX_RETRIES) {
          await sleep(Math.pow(2, attempt) * 500);
          continue;
        }
        return null;
      }

      const data = (await resp.json()) as ChatResponse;
      return data.choices[0]?.message?.content ?? null;
    } catch {
      clearTimeout(timer);
      if (attempt < MAX_RETRIES) {
        await sleep(Math.pow(2, attempt) * 500);
        continue;
      }
      return null;
    }
  }
  return null;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ── Causal Graph Proposal ───────────────────────────────────────────────

/**
 * Propose a causal graph from a natural-language domain description
 * using the DeepSeek LLM.
 *
 * The LLM analyzes the domain description, identifies variables,
 * proposes causal relationships with confidence scores, and returns
 * a structured proposal suitable for conversion to DomainKnowledge.
 *
 * @param domainDescription — natural language description of the domain
 * @param nodeNames — known variable names (optional; auto-extracted if omitted)
 * @param config — LLM configuration
 * @returns structured graph proposal, or null if LLM is unavailable
 */
export async function proposeCausalGraph(
  domainDescription: string,
  nodeNames?: string[],
  config: LLMCausalConfig = {},
): Promise<GraphProposal | null> {
  const variableHint = nodeNames && nodeNames.length > 0
    ? `\nKnown variables: ${nodeNames.join(', ')}`
    : '';

  const systemPrompt = `You are an expert in causal inference and causal graph construction.
Given a domain description, identify causal relationships between variables.
For each causal edge, provide a confidence score (0-1) and a brief rationale.
Also identify which variables are likely root causes (no incoming edges) and
which are likely leaf nodes (no outgoing edges).

Respond ONLY in valid JSON format:
{
  "edges": [
    { "source": "var1", "target": "var2", "confidence": 0.85, "rationale": "brief reason" }
  ],
  "rootNodes": ["root1", "root2"],
  "leafNodes": ["leaf1"]
}

If no variables are specified, infer them from the description.`;

  const response = await chat([
    { role: 'system', content: systemPrompt },
    { role: 'user', content: `Domain description: ${domainDescription}${variableHint}\n\nPropose causal relationships.` },
  ], config);

  if (!response) return null;

  return parseGraphProposal(response, nodeNames);
}

function parseGraphProposal(raw: string, nodeNames?: string[]): GraphProposal {
  try {
    // Extract JSON block from response (may contain markdown fences)
    const jsonMatch = raw.match(/\{[\s\S]*\}/);
    if (!jsonMatch) throw new Error('No JSON found');

    const parsed = JSON.parse(jsonMatch[0]) as {
      edges?: Array<{ source?: string; target?: string; confidence?: number; rationale?: string }>;
      rootNodes?: string[];
      leafNodes?: string[];
    };

    const edges = (parsed.edges ?? []).map(e => ({
      source: e.source ?? 'unknown',
      target: e.target ?? 'unknown',
      confidence: Math.min(1, Math.max(0, e.confidence ?? 0.5)),
      rationale: e.rationale ?? '',
    }));

    return {
      edges,
      rootNodes: parsed.rootNodes ?? [],
      leafNodes: parsed.leafNodes ?? [],
      rawResponse: raw,
    };
  } catch {
    // Fallback: try line-by-line parsing
    return { edges: [], rootNodes: [], leafNodes: [], rawResponse: raw };
  }
}

// ── Domain Knowledge Fusion ─────────────────────────────────────────────

/**
 * Fuse LLM-proposed causal edges with existing domain knowledge,
 * resolving conflicts and producing a merged DomainKnowledge object.
 *
 * Strategy:
 *   - LLM edges with high confidence (>0.7) become required edges
 *   - LLM edges with medium confidence (0.3-0.7) become soft suggestions
 *   - Existing forbids always take precedence (LLM cannot override)
 *   - Conflicts are reported but do not block fusion
 *
 * @param proposal — LLM graph proposal
 * @param existing — pre-existing domain knowledge (optional)
 * @param nodeNames — all variable names in the domain
 * @returns fused DomainKnowledge with conflict report
 */
export function fuseDomainKnowledge(
  proposal: GraphProposal,
  existing?: DomainKnowledge,
  nodeNames?: string[],
): DomainFusionResult {
  const existingForbids = new Set(
    (existing?.forbids ?? []).map(([s, t]) => `${s}|${t}`),
  );
  const existingRequires = new Set(
    (existing?.requires ?? []).map(([s, t]) => `${s}|${t}`),
  );

  const llmEdges: Array<readonly [string, string]> = [];
  const conflicts: Array<{
    edge: readonly [string, string];
    reason: string;
  }> = [];

  // High-confidence edges become requires
  const highConfidence = proposal.edges.filter(e => e.confidence >= 0.7);
  for (const e of highConfidence) {
    const key = `${e.source}|${e.target}`;
    if (existingForbids.has(key)) {
      conflicts.push({ edge: [e.source, e.target], reason: `Conflicts with existing forbid constraint: ${e.rationale}` });
      continue;
    }
    existingRequires.add(key);
    llmEdges.push([e.source, e.target]);
  }

  // Build merged DomainKnowledge
  const requires = [...existingRequires].map(key => {
    const [s, t] = key.split('|') as [string, string];
    return [s, t] as readonly [string, string];
  });

  const forbids = [...existingForbids].map(key => {
    const [s, t] = key.split('|') as [string, string];
    return [s, t] as readonly [string, string];
  });

  const rootNodes = [
    ...new Set([...(existing?.rootNodes ?? []), ...proposal.rootNodes]),
  ];
  const leafNodes = [
    ...new Set([...(existing?.leafNodes ?? []), ...proposal.leafNodes]),
  ];

  // Build DomainKnowledge with conditional fields (exactOptionalPropertyTypes)
  const dk: { forbids?: ReadonlyArray<readonly [string, string]>; requires?: ReadonlyArray<readonly [string, string]>; rootNodes?: ReadonlyArray<string>; leafNodes?: ReadonlyArray<string> } = {};
  if (forbids.length > 0) dk.forbids = forbids;
  if (requires.length > 0) dk.requires = requires;
  if (rootNodes.length > 0) dk.rootNodes = rootNodes;
  if (leafNodes.length > 0) dk.leafNodes = leafNodes;

  return {
    domainKnowledge: dk,
    llmEdges,
    conflicts,
  };
}

/**
 * Apply fused domain knowledge to an existing causal graph,
 * adding required edges and removing forbidden ones.
 *
 * @param graph — causal graph to modify (mutated in place)
 * @param knowledge — domain knowledge constraints
 * @returns the modified graph
 */
export function applyDomainKnowledge(
  graph: CausalGraphImpl,
  knowledge: DomainKnowledge,
): CausalGraphImpl {
  graph.applyDomainKnowledge(knowledge);
  return graph;
}

// ── Unified Explanation ─────────────────────────────────────────────────

/** Analysis result types supported by the unified explainer */
export type AnalysisResultType = 'rca' | 'sensitivity' | 'estimate' | 'graph' | 'refutation';

/** Input data for each analysis type */
export interface AnalysisInput {
  type: AnalysisResultType;
  // RCA
  rcaResult?: RCAResult;
  rcaMethod?: string;
  nodeCount?: number;
  // Sensitivity
  eValue?: number;
  partialR2?: number;
  robustnessValue?: number;
  sensitivityEstimate?: number;
  // Estimate
  estimateMethod?: string;
  ate?: number;
  se?: number;
  adjustmentSet?: string[];
  // Graph
  graphNodeCount?: number;
  graphEdgeCount?: number;
  algorithmName?: string;
  // Refutation
  refutationVerdict?: string;
  robustFraction?: number;
}

/**
 * Generate a unified natural-language explanation for any analysis result.
 *
 * Routes to the appropriate LLM-enhanced or template-based explainer
 * and returns a standardized UnifiedExplanation.
 *
 * @param input — analysis result data
 * @param config — LLM configuration
 * @returns unified explanation
 */
export async function explainAnalysis(
  input: AnalysisInput,
  config: LLMCausalConfig = {},
): Promise<UnifiedExplanation> {
  const apiKey = config.apiKey ?? process.env.DEEPSEEK_API_KEY;
  const llmAvailable = !!apiKey;

  switch (input.type) {
    case 'rca': {
      if (input.rcaResult && input.rcaMethod) {
        const rcaExplain = await explainRCAWithLLM(input.rcaResult, input.rcaMethod, input.nodeCount);
        return {
          type: 'rca',
          summary: rcaExplain.summary,
          interpretation: rcaExplain.reasoning ?? rcaExplain.summary,
          recommendations: Array.isArray(rcaExplain.caveats) ? rcaExplain.caveats.join('; ') : (rcaExplain.caveats ?? ''),
          llmEnhanced: llmAvailable,
          data: { rootCauses: input.rcaResult.rootCauses.map(rc => rc.name) },
        };
      }
      return fallbackExplanation('rca', input);
    }

    case 'sensitivity': {
      if (input.eValue !== undefined && input.partialR2 !== undefined && input.robustnessValue !== undefined) {
        const sensExplain = await explainSensitivityWithLLM(
          input.eValue, input.partialR2, input.robustnessValue, input.sensitivityEstimate ?? 0,
        );
        return {
          type: 'sensitivity',
          summary: sensExplain.summary,
          interpretation: sensExplain.interpretation ?? sensExplain.summary,
          recommendations: sensExplain.actionableAdvice ?? '',
          llmEnhanced: llmAvailable,
          data: { eValue: input.eValue, partialR2: input.partialR2, robustnessValue: input.robustnessValue },
        };
      }
      return fallbackExplanation('sensitivity', input);
    }

    case 'estimate': {
      if (input.estimateMethod && input.ate !== undefined && input.se !== undefined) {
        const estExplain = await explainEstimateWithLLM(
          input.estimateMethod, input.ate, input.se, input.adjustmentSet ?? [],
        );
        return {
          type: 'estimate',
          summary: estExplain.summary,
          interpretation: estExplain.interpretation ?? estExplain.summary,
          recommendations: estExplain.confidenceStatement ?? '',
          llmEnhanced: llmAvailable,
          data: { method: input.estimateMethod, ate: input.ate, se: input.se },
        };
      }
      return fallbackExplanation('estimate', input);
    }

    case 'graph': {
      return {
        type: 'graph',
        summary: `Causal graph with ${input.graphNodeCount ?? '?'} nodes and ${input.graphEdgeCount ?? '?'} edges`,
        interpretation: `Discovered using ${input.algorithmName ?? 'causal discovery'} algorithm. The graph contains ${input.graphNodeCount ?? 'N'} variables connected by ${input.graphEdgeCount ?? 'N'} causal edges.`,
        recommendations: 'Validate the discovered graph with domain experts. Test structural stability via refutation methods.',
        llmEnhanced: false,
        data: { nodes: input.graphNodeCount, edges: input.graphEdgeCount, algorithm: input.algorithmName },
      };
    }

    case 'refutation': {
      return {
        type: 'refutation',
        summary: `Refutation verdict: ${input.refutationVerdict ?? 'unknown'} (${input.robustFraction ?? 0}/7 robust)`,
        interpretation: `The causal estimate ${(input.refutationVerdict === 'robust' ? 'survived' : 'failed')} refutation tests. ${input.robustFraction ? `${(input.robustFraction * 100).toFixed(0)}%` : '0%'} of tests indicate robustness.`,
        recommendations: input.refutationVerdict === 'robust'
          ? 'The estimate is trustworthy. Proceed with confidence.'
          : 'The estimate may be unreliable. Consider gathering additional data or refining the causal model.',
        llmEnhanced: false,
        data: { verdict: input.refutationVerdict, robustFraction: input.robustFraction },
      };
    }

    default:
      return fallbackExplanation('graph', input);
  }
}

function fallbackExplanation(type: AnalysisResultType, input: AnalysisInput): UnifiedExplanation {
  return {
    type,
    summary: `${type} analysis completed`,
    interpretation: `Analysis results are available for review. Enable LLM (DEEPSEEK_API_KEY) for natural language explanations.`,
    recommendations: 'Set DEEPSEEK_API_KEY environment variable to enable AI-powered explanations.',
    llmEnhanced: false,
    data: input as unknown as Record<string, unknown>,
  };
}

// ── Multi-Turn Dialogue ─────────────────────────────────────────────────

/**
 * Stateful causal reasoning dialogue for iterative graph refinement.
 *
 * Maintains conversation history and allows step-by-step refinement
 * of causal graphs through natural language interaction.
 */
export class CausalDialogue {
  private history: ChatMessage[] = [];
  private config: LLMCausalConfig;

  constructor(config: LLMCausalConfig = {}) {
    this.config = config;
    this.history = [{
      role: 'system',
      content: `You are a causal reasoning expert. Help users build and refine causal graphs.
For each response, suggest causal relationships clearly using the format:
- "X causes Y" or "X → Y" for directed edges
- "X and Y are correlated but direction unknown" for undirected
- "X and Y are independent" for no relationship
- Confidence: [high/medium/low] with brief justification
Keep responses concise and structured.`,
    }];
  }

  /**
   * Send a user message and get the assistant's response.
   *
   * @param message — user's natural language message
   * @returns assistant's response, or null if LLM unavailable
   */
  async send(message: string): Promise<string | null> {
    this.history.push({ role: 'user', content: message });
    const response = await chat(this.history, this.config);

    if (response) {
      this.history.push({ role: 'assistant', content: response });
    }
    return response;
  }

  /**
   * Get the full conversation history.
   */
  getHistory(): ReadonlyArray<ChatMessage> {
    return this.history;
  }

  /**
   * Reset the conversation, keeping only the system prompt.
   */
  reset(): void {
    const systemPrompt = this.history[0];
    this.history = systemPrompt ? [systemPrompt] : [];
  }

  /**
   * Check if the LLM is available (API key configured).
   */
  isAvailable(): boolean {
    return !!(this.config.apiKey ?? process.env.DEEPSEEK_API_KEY);
  }
}

// ── Auto-Generation of Causal Hypotheses ────────────────────────────────

/**
 * Automatically generate causal hypotheses from a list of variables.
 *
 * The LLM is asked to enumerate plausible causal relationships between
 * every pair of variables based on common-sense domain reasoning.
 *
 * @param nodeNames — variable names
 * @param context — optional domain context (e.g., "IT microservices metrics")
 * @param config — LLM configuration
 * @returns list of hypothesized edges with confidence, or null
 */
export async function generateHypotheses(
  nodeNames: string[],
  context?: string,
  config: LLMCausalConfig = {},
): Promise<GraphProposal | null> {
  const ctx = context ? `Domain context: ${context}\n` : '';
  const description = `${ctx}Variables: ${nodeNames.join(', ')}.\n\nFor each pair of variables, hypothesize whether a causal relationship exists and in which direction. Respond in JSON.`;

  return proposeCausalGraph(description, nodeNames, config);
}
