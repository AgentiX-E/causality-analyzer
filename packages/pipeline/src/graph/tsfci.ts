/**
 * tsFCI — Time-Series Fast Causal Inference with latent confounders.
 *
 * Extends FCI to temporal data.  Uses a lagged data representation
 * and FCI's skeleton + orientation rules to produce a Partial
 * Ancestral Graph (PAG) that accounts for both temporally-lagged
 * and contemporaneous latent confounding.
 *
 * Reference: Entner & Hoyer (2010). "On causal discovery from
 *   time series data using FCI." PGM 2010.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import { fciAlgorithm } from './advanced-discovery.js';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';

export interface TsFCIResult {
  /** Instantaneous PAG (same-time edges) */
  instantaneousGraph: CausalGraph;
  /** Lagged edges: Map<lag, CausalGraph> */
  laggedGraphs: Map<number, CausalGraph>;
  /** All PAG edge types */
  pagEdges: Map<string, string>;
}

export interface TsFCIConfig {
  alpha?: number;
  maxLag?: number;
}

/**
 * Run tsFCI on time-series data.
 *
 * @param data — (T × d) matrix
 * @param nodeNames — variable names
 */
export function tsFciAlgorithm(
  data: number[][],
  nodeNames: string[],
  config: TsFCIConfig = {},
  domainKnowledge?: DomainKnowledge,
): TsFCIResult {
  const alpha = config.alpha ?? 0.05;
  const maxLag = config.maxLag ?? Math.min(3, Math.floor(data.length / 20));
  const T = data.length;
  const d = nodeNames.length;
  const effT = T - maxLag;

  const pagEdges = new Map<string, string>();

  if (effT < 10 || d < 2) {
    return {
      instantaneousGraph: new CausalGraph([...nodeNames]),
      laggedGraphs: new Map(),
      pagEdges,
    };
  }

  // ── Contemporaneous structure ──
  const instData = new Matrix(effT, d);
  for (let t = 0; t < effT; t++)
    for (let j = 0; j < d; j++)
      instData.set(t, j, data[t + maxLag][j]);

  const instResult = fciAlgorithm(instData, nodeNames, { alpha });
  const instantaneousGraph = instResult.graph;
  for (const [k, v] of instResult.pagEdges) pagEdges.set(k, v);

  // ── Lagged structure ──
  const laggedGraphs = new Map<number, CausalGraph>();
  for (let tau = 0; tau < maxLag; tau++) {
    // Build data: each lag τ creates a "new" variable in the design
    // Use FCI on augmented data (current + lagged as distinct variables)
    const lagNames = nodeNames.map((n, j) => `${n}_lag${tau + 1}`);
    const allNames = [...nodeNames, ...lagNames];
    const lagData = new Matrix(effT, d * 2);

    for (let t = 0; t < effT; t++) {
      for (let j = 0; j < d; j++) {
        lagData.set(t, j, data[t + maxLag][j]); // current
        lagData.set(t, d + j, data[t + maxLag - tau - 1][j]); // lagged
      }
    }

    const lagResult = fciAlgorithm(lagData, allNames, { alpha, maxDegree: 5 });

    // Extract only cross-time edges (current → lagged or lagged → current)
    const lagGraph = new CausalGraph(nodeNames);
    for (const e of lagResult.graph.edges) {
      const isCurrent1 = nodeNames.includes(e.source);
      const isCurrent2 = nodeNames.includes(e.target);
      if (isCurrent1 !== isCurrent2) {
        if (isCurrent1) {
          lagGraph.addEdge(e.source, lagToOriginal(e.target, tau + 1));
        } else {
          lagGraph.addEdge(lagToOriginal(e.source, tau + 1), e.target);
        }
      }
    }
    laggedGraphs.set(tau + 1, lagGraph);
  }

  return { instantaneousGraph, laggedGraphs, pagEdges };
}

function lagToOriginal(name: string, lag: number): string {
  // Remove "_lagN" suffix to get original variable name
  return name.replace(new RegExp(`_lag${lag}$`), '');
}
