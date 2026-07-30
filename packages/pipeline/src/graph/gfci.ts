/**
 * GFCI — Greedy Fast Causal Inference.
 *
 * Hybrid causal discovery algorithm combining score-based FGES with
 * constraint-based FCI refinement. Designed for realistic scenarios
 * where latent confounders may exist.
 *
 * Algorithm:
 *  1. Run FGES to obtain an initial sparse CPDAG
 *  2. Run FCI's Fast Adjacency Search (FAS) on the FGES skeleton
 *     + Possible-D-SEP for edges remaining after FAS
 *  3. Apply full FCI orientation rules (R0-R10) to produce a PAG
 *
 * GFCI is significantly faster than pure FCI because FGES provides
 * a high-quality skeleton, dramatically reducing the CI test budget.
 *
 * Reference: Ogarrio, Spirtes & Ramsey (UAI 2016).
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import { gesAlgorithm } from './ges.js';
import { combinations, fisherZTest } from '@agentix-e/causality-analyzer-core';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';

export interface GFCIConfig {
  alpha: number;
  maxDegree: number;
  /** Use Possible-D-SEP refinement (slower but more accurate) */
  usePDS: boolean;
}

// ── Public API ──────────────────────────────────────────────────────

export function gfciAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: Partial<GFCIConfig> = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph; pagEdges: Map<string, string> } {
  const n = nodeNames.length;
  const alpha = config.alpha ?? (n > 15 ? 0.01 : 0.05);
  // Cap maxDegree on large graphs to avoid PDS combinatorial explosion
  const effectiveMaxDegree = config.maxDegree ?? (n > 30 ? 5 : n > 15 ? 8 : -1);
  const cfg: GFCIConfig = { alpha, maxDegree: effectiveMaxDegree, usePDS: config.usePDS ?? true };
  const _N = data.rows;
  const pagEdges = new Map<string, string>();

  if (data.rows === 0) return { graph: new CausalGraph(nodeNames), pagEdges };

  // Phase 1: FGES → initial CPDAG
  const gesGraph = gesAlgorithm(data, nodeNames);

  // Phase 2: FCI-style refinement on FGES skeleton
  // Start with FGES edges as the initial graph (not complete graph)
  const g = new CausalGraph(nodeNames);
  for (const e of gesGraph.edges) {
    if (e.directed) {
      g.addEdge(e.source, e.target);
    } else {
      g.undirectedEdge(e.source, e.target);
    }
  }

  const sepSet = new Map<string, Set<string>>();
  const maxDepth = cfg.maxDegree === -1 ? n : cfg.maxDegree;

  // ── FAS (Fast Adjacency Search) on FGES skeleton ──
  // Convert Matrix → number[][] for core fisherZTest
  const dataArr = matrixTo2D(data);

  let depth = 0;
  let changed = true;

  while (changed && depth <= maxDepth) {
    changed = false;
    for (let i = 0; i < n; i++) {
      const neighbors = g.neighbors(nodeNames[i]);
      if (neighbors.length - 1 < depth) continue;
      for (const jName of neighbors) {
        if (jName <= nodeNames[i]) continue;
        const j = nodeNames.indexOf(jName);
        if (j < 0 || !g.hasEdge(nodeNames[i], jName)) continue;
        const otherNeighbors = neighbors.filter(n => n !== jName);
        if (otherNeighbors.length < depth) continue;

        const subsets = combinations(otherNeighbors, depth);
        for (const S of subsets) {
          const sIndices = S.map(s => nodeNames.indexOf(s));
          const p = fisherZTest(dataArr, i, j, sIndices);
          if (p > cfg.alpha) {
            g.removeEdge(nodeNames[i], jName);
            g.removeEdge(jName, nodeNames[i]);
            sepSet.set(`${Math.min(i, j)}-${Math.max(i, j)}`, new Set(S));
            changed = true;
            break;
          }
        }
      }
    }
    depth++;
  }

  // ── Possible-D-SEP (optional) ──
  if (cfg.usePDS) {
    const pdsDepthMax = 3;
    for (let i = 0; i < n; i++) {
      const iName = nodeNames[i];
      const pds = new Set<number>();
      const visited = new Set<number>();
      const queue: number[] = [i];
      visited.add(i);
      let pdsDepth = 0;

      while (queue.length > 0 && pdsDepth < pdsDepthMax) {
        const sz = queue.length;
        for (let _s = 0; _s < sz; _s++) {
          const v = queue.shift()!;
          if (v !== i) pds.add(v);
          for (let w = 0; w < n; w++) {
            if (visited.has(w)) continue;
            if (g.hasEdge(nodeNames[v], nodeNames[w]) || g.hasEdge(nodeNames[w], nodeNames[v])) {
              visited.add(w);
              queue.push(w);
            }
          }
        }
        pdsDepth++;
      }

      for (let j = i + 1; j < n; j++) {
        if (!g.hasEdge(iName, nodeNames[j])) continue;
        const candidates = [...pds].filter(k => k !== j);
        for (let cSize = 1; cSize <= Math.min(3, candidates.length); cSize++) {
          const subsets = combinations(candidates.map(String).map(Number), cSize);
          let removed = false;
          for (const S of subsets) {
            const p = fisherZTest(dataArr, i, j, S);
            if (p > cfg.alpha) {
              g.removeEdge(iName, nodeNames[j]);
              g.removeEdge(nodeNames[j], iName);
              const key = `${Math.min(i, j)}-${Math.max(i, j)}`;
              sepSet.set(key, new Set(S.map(s => nodeNames[s])));
              removed = true;
              break;
            }
          }
          if (removed) break;
        }
      }
    }
  }

  // Phase 3: FCI orientation rules (R0-R10)
  orientRules(g, nodeNames, n, sepSet);

  // Build PAG edge notation
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const hasIJ = g.hasEdge(nodeNames[i], nodeNames[j]);
      const hasJI = g.hasEdge(nodeNames[j], nodeNames[i]);
      if (!hasIJ && !hasJI) pagEdges.set(`${nodeNames[i]}-${nodeNames[j]}`, 'none');
      else if (hasIJ && hasJI) pagEdges.set(`${nodeNames[i]}-${nodeNames[j]}`, 'undirected');
      else if (hasIJ && !hasJI) pagEdges.set(`${nodeNames[i]}-${nodeNames[j]}`, `${nodeNames[i]}→${nodeNames[j]}`);
      else pagEdges.set(`${nodeNames[i]}-${nodeNames[j]}`, `${nodeNames[j]}→${nodeNames[i]}`);
    }
  }

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);

  return { graph: g, pagEdges };
}

// ── FCI Orientation Rules (R0-R10) ──────────────────────────────────

function orientRules(
  g: CausalGraph, nodeNames: string[], n: number,
  sepSet: Map<string, Set<string>>,
): void {
  // Phase A: V-structure (R0)
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (g.hasEdge(nodeNames[i], nodeNames[j])) continue;
      for (let k = 0; k < n; k++) {
        if (k === i || k === j) continue;
        if (!g.hasEdge(nodeNames[i], nodeNames[k]) || !g.hasEdge(nodeNames[j], nodeNames[k])) continue;
        const key = `${Math.min(i, j)}-${Math.max(i, j)}`;
        if (!sepSet.get(key)?.has(nodeNames[k])) {
          g.orientEdge(nodeNames[i], nodeNames[k]);
          g.orientEdge(nodeNames[j], nodeNames[k]);
        }
      }
    }
  }

  // Phase B: R1-R3
  let changed = true;
  let iter = 0;
  const maxIter = 30;
  while (changed && iter++ < maxIter) {
    changed = false;

    // R1: i→j∘—k, i∗∗k non-adjacent → j→k
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!g.hasEdge(nodeNames[i], nodeNames[j]) || g.hasEdge(nodeNames[j], nodeNames[i])) continue;
        for (let k = 0; k < n; k++) {
          if (!g.hasEdge(nodeNames[j], nodeNames[k]) || !g.hasEdge(nodeNames[k], nodeNames[j])) continue;
          if (g.hasEdge(nodeNames[i], nodeNames[k]) || g.hasEdge(nodeNames[k], nodeNames[i])) continue;
          g.orientEdge(nodeNames[j], nodeNames[k]);
          changed = true;
        }
      }
    }

    // R2: i→j→k and i∘—∘k → i→k
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < n; k++) {
        if (!g.hasEdge(nodeNames[i], nodeNames[k]) || !g.hasEdge(nodeNames[k], nodeNames[i])) continue;
        for (let j = 0; j < n; j++) {
          if (!g.hasEdge(nodeNames[i], nodeNames[j]) || g.hasEdge(nodeNames[j], nodeNames[i])) continue;
          if (!g.hasEdge(nodeNames[j], nodeNames[k]) || g.hasEdge(nodeNames[k], nodeNames[j])) continue;
          g.orientEdge(nodeNames[i], nodeNames[k]);
          changed = true;
        }
      }
    }

    // R3: i∘—k→j, i∘—l→j, k,l non-adjacent → i→j
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!g.hasEdge(nodeNames[i], nodeNames[j]) || !g.hasEdge(nodeNames[j], nodeNames[i])) continue;
        for (let k = 0; k < n; k++) {
          if (!g.hasEdge(nodeNames[i], nodeNames[k]) || !g.hasEdge(nodeNames[k], nodeNames[i])) continue;
          if (!g.hasEdge(nodeNames[k], nodeNames[j]) || g.hasEdge(nodeNames[j], nodeNames[k])) continue;
          for (let l = 0; l < n; l++) {
            if (l === k) continue;
            if (!g.hasEdge(nodeNames[i], nodeNames[l]) || !g.hasEdge(nodeNames[l], nodeNames[i])) continue;
            if (!g.hasEdge(nodeNames[l], nodeNames[j]) || g.hasEdge(nodeNames[j], nodeNames[l])) continue;
            if (g.hasEdge(nodeNames[k], nodeNames[l]) || g.hasEdge(nodeNames[l], nodeNames[k])) continue;
            g.orientEdge(nodeNames[i], nodeNames[j]);
            changed = true;
            break;
          }
        }
      }
    }
  }
}

// ── Helpers ─────────────────────────────────────────────────────────

function matrixTo2D(data: Matrix): number[][] {
  const rows: number[][] = [];
  for (let r = 0; r < data.rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < data.columns; c++) row.push(data.get(r, c));
    rows.push(row);
  }
  return rows;
}
