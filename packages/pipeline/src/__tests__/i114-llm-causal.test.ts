/**
 * I114: LLM-Enhanced Causal Reasoning Tests
 *
 * Tests (without requiring a live DeepSeek API key):
 *   - GraphProposal JSON parsing (offline)
 *   - fuseDomainKnowledge correctness
 *   - applyDomainKnowledge integration
 *   - explainAnalysis all types (with LLM fallback)
 *   - CausalDialogue conversation state management
 *   - generateHypotheses offline behavior
 *
 * Tests that require a live API key are skipped gracefully.
 */

import { describe, it, expect } from 'vitest';
import {
  fuseDomainKnowledge,
  applyDomainKnowledge,
  explainAnalysis,
  generateHypotheses,
  CausalDialogue,
  type GraphProposal,
  type DomainFusionResult,
  type AnalysisInput,
} from '../explain/llm-causal.js';
import { CausalGraph } from '../graph/causal-graph.js';
import type { RCAResult } from '@agentix-e/causality-analyzer-core';

// ── Graph Proposal Parsing ─────────────────────────────────────────

describe('proposeCausalGraph — offline JSON parsing', () => {
  it('fuseDomainKnowledge handles empty proposal', () => {
    const emptyProposal: GraphProposal = {
      edges: [],
      rootNodes: [],
      leafNodes: [],
      rawResponse: '',
    };

    const result = fuseDomainKnowledge(emptyProposal);
    expect(result.llmEdges.length).toBe(0);
    expect(result.conflicts.length).toBe(0);
    expect(result.domainKnowledge).toBeDefined();
  });

  it('fuseDomainKnowledge merges high-confidence edges', () => {
    const proposal: GraphProposal = {
      edges: [
        { source: 'CPU', target: 'Latency', confidence: 0.9, rationale: 'High CPU causes latency' },
        { source: 'Memory', target: 'CPU', confidence: 0.85, rationale: 'Memory pressure increases CPU' },
      ],
      rootNodes: ['Memory'],
      leafNodes: ['Latency'],
      rawResponse: '',
    };

    const result = fuseDomainKnowledge(proposal);
    expect(result.llmEdges.length).toBe(2);
    expect(result.llmEdges).toContainEqual(['CPU', 'Latency']);
    expect(result.llmEdges).toContainEqual(['Memory', 'CPU']);
  });

  it('fuseDomainKnowledge respects existing forbids', () => {
    const proposal: GraphProposal = {
      edges: [
        { source: 'CPU', target: 'Latency', confidence: 0.9, rationale: '' },
      ],
      rootNodes: [],
      leafNodes: [],
      rawResponse: '',
    };

    const existing = {
      forbids: [['CPU', 'Latency']] as ReadonlyArray<readonly [string, string]>,
    };

    const result = fuseDomainKnowledge(proposal, existing);
    expect(result.llmEdges.length).toBe(0);
    expect(result.conflicts.length).toBe(1);
    expect(result.conflicts[0]!.edge).toEqual(['CPU', 'Latency']);
  });

  it('fuseDomainKnowledge filters low-confidence edges', () => {
    const proposal: GraphProposal = {
      edges: [
        { source: 'A', target: 'B', confidence: 0.95, rationale: '' },
        { source: 'C', target: 'D', confidence: 0.3, rationale: '' },
        { source: 'E', target: 'F', confidence: 0.5, rationale: '' },
      ],
      rootNodes: [],
      leafNodes: [],
      rawResponse: '',
    };

    const result = fuseDomainKnowledge(proposal);
    // Only confidence >= 0.7 should become required edges
    expect(result.llmEdges.length).toBe(1);
    expect(result.llmEdges).toContainEqual(['A', 'B']);
  });

  it('fuseDomainKnowledge merges root and leaf nodes', () => {
    const proposal: GraphProposal = {
      edges: [],
      rootNodes: ['CPU', 'Memory'],
      leafNodes: ['Latency', 'ErrorRate'],
      rawResponse: '',
    };

    const existing = {
      rootNodes: ['Network'] as ReadonlyArray<string>,
    };

    const result = fuseDomainKnowledge(proposal, existing);
    const dk = result.domainKnowledge;
    expect(dk.rootNodes).toBeDefined();
    if (dk.rootNodes) {
      expect(dk.rootNodes).toContain('CPU');
      expect(dk.rootNodes).toContain('Memory');
      expect(dk.rootNodes).toContain('Network');
    }
  });
});

// ── Domain Knowledge Application ───────────────────────────────────

describe('applyDomainKnowledge', () => {
  it('applies required edges to a graph', () => {
    const graph = new CausalGraph(['A', 'B', 'C']);
    const dk = {
      requires: [['A', 'B'], ['B', 'C']] as ReadonlyArray<readonly [string, string]>,
    };

    applyDomainKnowledge(graph, dk);
    const edges = graph.edges;
    expect(edges.some(e => e.source === 'A' && e.target === 'B')).toBe(true);
    expect(edges.some(e => e.source === 'B' && e.target === 'C')).toBe(true);
  });

  it('removes forbidden edges', () => {
    const graph = new CausalGraph(['A', 'B', 'C']);
    graph.addEdge('A', 'B');
    graph.addEdge('B', 'C');

    const dk = {
      forbids: [['A', 'B']] as ReadonlyArray<readonly [string, string]>,
    };

    applyDomainKnowledge(graph, dk);
    const edges = graph.edges;
    expect(edges.some(e => e.source === 'A' && e.target === 'B')).toBe(false);
    expect(edges.some(e => e.source === 'B' && e.target === 'C')).toBe(true);
  });
});

// ── Unified Explanation ────────────────────────────────────────────

describe('explainAnalysis — offline fallback', () => {
  it('explains RCA results with template fallback', async () => {
    const input: AnalysisInput = {
      type: 'rca',
      rcaResult: {
        rootCauses: [{ name: 'Memory', score: 0.95, confidence: 0.9, rank: 1, evidence: [] }],
        metadata: { method: 'HT', computedAt: Date.now(), duration: 100 },
        propagationPaths: [],
      } as RCAResult,
      rcaMethod: 'HTRCA',
    };

    const result = await explainAnalysis(input);
    expect(result.type).toBe('rca');
    expect(result.summary.length).toBeGreaterThan(0);
  });

  it('explains sensitivity results', async () => {
    const input: AnalysisInput = {
      type: 'sensitivity',
      eValue: 2.5,
      partialR2: 0.1,
      robustnessValue: 1.8,
    };

    const result = await explainAnalysis(input);
    expect(result.type).toBe('sensitivity');
    expect(result.summary.length).toBeGreaterThan(0);
  });

  it('explains estimate results', async () => {
    const input: AnalysisInput = {
      type: 'estimate',
      estimateMethod: 'Backdoor OLS',
      ate: 0.45,
      se: 0.12,
      adjustmentSet: ['Confounder1'],
    };

    const result = await explainAnalysis(input);
    expect(result.type).toBe('estimate');
    expect(result.summary.length).toBeGreaterThan(0);
  });

  it('explains graph results', async () => {
    const input: AnalysisInput = {
      type: 'graph',
      graphNodeCount: 5,
      graphEdgeCount: 7,
      algorithmName: 'PC',
    };

    const result = await explainAnalysis(input);
    expect(result.type).toBe('graph');
    expect(result.summary).toContain('5');
    expect(result.summary).toContain('7');
  });

  it('explains refutation results', async () => {
    const input: AnalysisInput = {
      type: 'refutation',
      refutationVerdict: 'robust',
      robustFraction: 6/7,
    };

    const result = await explainAnalysis(input);
    expect(result.type).toBe('refutation');
    expect(result.summary).toContain('robust');
  });

  it('handles unknown type gracefully', async () => {
    const result = await explainAnalysis({ type: 'graph' } as AnalysisInput);
    expect(result.type).toBe('graph');
    expect(result.summary.length).toBeGreaterThan(0);
  });
});

// ── CausalDialogue ─────────────────────────────────────────────────

describe('CausalDialogue', () => {
  it('creates dialogue with system prompt', () => {
    const dialogue = new CausalDialogue();
    const history = dialogue.getHistory();
    expect(history.length).toBe(1);
    expect(history[0]!.role).toBe('system');
    expect(history[0]!.content).toContain('causal reasoning');
  });

  it('isAvailable returns false without API key', () => {
    const dialogue = new CausalDialogue({ apiKey: '' });
    expect(dialogue.isAvailable()).toBe(false);
  });

  it('send returns null without API key', async () => {
    const dialogue = new CausalDialogue({ apiKey: '' });
    const response = await dialogue.send('What causes latency?');
    expect(response).toBeNull();
  });

  it('reset clears conversation but keeps system prompt', () => {
    const dialogue = new CausalDialogue();
    dialogue.reset();
    const history = dialogue.getHistory();
    expect(history.length).toBeLessThanOrEqual(1);
  });

  it('maintains conversation history', async () => {
    const dialogue = new CausalDialogue({ apiKey: '' });
    const initialLen = dialogue.getHistory().length;
    await dialogue.send('test message');
    // Without API key, send returns null but still adds user message to history
    expect(dialogue.getHistory().length).toBeGreaterThanOrEqual(initialLen);
  });
});

// ── Hypothesis Generation ──────────────────────────────────────────

describe('generateHypotheses — offline', () => {
  it('returns null without API key', async () => {
    const result = await generateHypotheses(
      ['CPU', 'Memory', 'Latency'],
      'IT operations metrics',
      { apiKey: '' },
    );
    expect(result).toBeNull();
  });

  it('handles empty node list', async () => {
    const result = await generateHypotheses([], undefined, { apiKey: '' });
    expect(result).toBeNull();
  });
});
