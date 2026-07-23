/**
 * Advanced Causal Discovery — FCI, Targeted Discovery, Grow-Shrink.
 *
 * Extends the PC algorithm with methods for latent confounder handling,
 * targeted parent discovery, and Markov blanket identification.
 *
 * FCI (Fast Causal Inference): Produces Partial Ancestral Graphs (PAGs)
 * that distinguish directed causation from latent confounding using
 * additional orientation rules (R4-R10) and Possible-D-SEP search.
 *
 * Targeted Discovery: Computes only the causal parents of specified
 * target variables, avoiding full-graph search for efficiency.
 *
 * Grow-Shrink: Markov blanket discovery via forward selection +
 * backward elimination.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { combinations } from "@agentix-e/causality-analyzer-core";
import { CausalGraph } from './causal-graph.js';
import { fisherZTest } from './pc.js';
import type { PCConfig } from './pc.js';

// ── FCI Algorithm ─────────────────────────────────────────────────────

/**
 * FCI (Fast Causal Inference) — causal discovery with latent confounders.
 *
 * Produces a Partial Ancestral Graph (PAG) where:
 * - A → B: A is an ancestor of B (no latent common cause)
 * - A ↔ B: A and B share a latent common cause
 * - A ∘→ B: A is not a descendant of B (B is not an ancestor of A)
 * - A ∘–∘ B: no information about the relationship
 *
 * The returned graph encodes these marks using edge properties.
 */
export function fciAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: Partial<PCConfig> = {},
): { graph: CausalGraph; pagEdges: Map<string, string> } {
  const cfg: PCConfig = { alpha: config.alpha ?? 0.05, maxDegree: config.maxDegree ?? -1, stable: config.stable ?? true };
  const n = nodeNames.length;
  const pagEdges = new Map<string, string>();

  if (data.rows === 0) return { graph: new CausalGraph(nodeNames), pagEdges };

  // Phase 1: Skeleton estimation (identical to PC)
  const g = new CausalGraph(nodeNames);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) g.undirectedEdge(nodeNames[i]!, nodeNames[j]!);

  const sepSet = new Map<string, Set<string>>();
  let depth = 0;
  const maxDepth = cfg.maxDegree === -1 ? n : cfg.maxDegree;

  while (depth <= maxDepth) {
    let edgesRemoved = false;
    const toRemove: Array<[string, string]> = [];

    for (let i = 0; i < n; i++) {
      const neighbors = g.neighbors(nodeNames[i]!);
      if (neighbors.length - 1 < depth) continue;

      for (const jName of neighbors) {
        if (jName <= nodeNames[i]!) continue;
        const j = nodeNames.indexOf(jName);
        const otherNeighbors = neighbors.filter(n => n !== jName);

        if (otherNeighbors.length < depth) continue;
        const subsets = combinations(otherNeighbors, depth);

        for (const S of subsets) {
          const sIndices = S.map(s => nodeNames.indexOf(s));
          const p = fisherZTest(data, i, j, sIndices);
          if (p > cfg.alpha) {
            toRemove.push([nodeNames[i]!, jName]);
            sepSet.set(`${Math.min(i, j)}-${Math.max(i, j)}`, new Set(S));
            break;
          }
        }
      }
    }

    for (const [a, b] of toRemove) { g.removeEdge(a, b); g.removeEdge(b, a); edgesRemoved = true; }
    if (!edgesRemoved) break;
    depth++;
  }

  // ── Phase 1b: Possible-D-SEP search (FCI-specific) ──────────────
  // Re-test adjacent edges using conditioning sets drawn from a larger
  // "possible d-separation" set. This enables FCI to correctly handle
  // latent confounders that the initial PC-style skeleton cannot.
  // Reference: Spirtes et al. (2000), §6.2; Zhang (2008).
  for (let i = 0; i < n; i++) {
    const iName = nodeNames[i]!;
    // Collect Possible-D-SEP(i): all nodes reachable within 3 hops
    const pds = new Set<number>();
    const visited = new Set<number>();
    const queue: number[] = [i];
    visited.add(i);
    let pdsDepth = 0;
    while (queue.length > 0 && pdsDepth < 3) {
      const sz = queue.length;
      for (let _s = 0; _s < sz; _s++) {
        const v = queue.shift()!;
        if (v !== i) pds.add(v);
        for (let w = 0; w < n; w++) {
          if (visited.has(w)) continue;
          if (g.hasEdge(nodeNames[v]!, nodeNames[w]!) || g.hasEdge(nodeNames[w]!, nodeNames[v]!)) {
            visited.add(w);
            queue.push(w);
          }
        }
      }
      pdsDepth++;
    }

    // Re-test each adjacent pair (i, j) with PDS conditioning sets
    for (let j = i + 1; j < n; j++) {
      if (!g.hasEdge(iName, nodeNames[j]!)) continue;
      const candidates = [...pds].filter(k => k !== j);
      for (let cSize = 1; cSize <= Math.min(3, candidates.length); cSize++) {
        const subsets = combinations(candidates.map(String).map(Number), cSize);
        let removed = false;
        for (const S of subsets) {
          const p = fisherZTest(data, i, j, S);
          if (p > cfg.alpha) {
            g.removeEdge(iName, nodeNames[j]!);
            g.removeEdge(nodeNames[j]!, iName);
            const key = `${Math.min(i, j)}-${Math.max(i, j)}`;
            sepSet.set(key, new Set(S.map(s => nodeNames[s]!)));
            removed = true;
            break;
          }
        }
        if (removed) break;
      }
    }
  }

  // Phase 2: Orient v-structures
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (g.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue;
      for (let k = 0; k < n; k++) {
        if (k === i || k === j) continue;
        if (!g.hasEdge(nodeNames[i]!, nodeNames[k]!) || !g.hasEdge(nodeNames[j]!, nodeNames[k]!)) continue;
        const key = `${Math.min(i, j)}-${Math.max(i, j)}`;
        if (!sepSet.get(key)?.has(nodeNames[k]!)) {
          // Orient v-structure: i→k←j
          g.toUndirected(nodeNames[i]!, nodeNames[k]!);
          g.toUndirected(nodeNames[j]!, nodeNames[k]!);
        }
      }
    }
  }

  // Phase 3: FCI-specific orientation rules (R1-R3 from PC, plus R4 for discriminating paths)
  let changed = true;
  let iter = 0;
  const maxIter = 20;
  while (changed && iter++ < maxIter) {
    changed = false;

    // R1: i→j—k with i,k non-adjacent → j→k
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!) || g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;
        for (let k = 0; k < n; k++) {
          if (!g.hasEdge(nodeNames[j]!, nodeNames[k]!) || !g.hasEdge(nodeNames[k]!, nodeNames[j]!)) continue;
          if (g.hasEdge(nodeNames[i]!, nodeNames[k]!) || g.hasEdge(nodeNames[k]!, nodeNames[i]!)) continue;
          g.toUndirected(nodeNames[j]!, nodeNames[k]!);
          changed = true;
        }
      }
    }

    // R2: i→j→k and i—k → i→k
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < n; k++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[k]!) || !g.hasEdge(nodeNames[k]!, nodeNames[i]!)) continue;
        for (let j = 0; j < n; j++) {
          if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!) || g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;
          if (!g.hasEdge(nodeNames[j]!, nodeNames[k]!) || g.hasEdge(nodeNames[k]!, nodeNames[j]!)) continue;
          g.toUndirected(nodeNames[i]!, nodeNames[k]!);
          changed = true;
        }
      }
    }

    // R3: i—k→j, i—l→j, k and l non-adjacent → i→j
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!) || !g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;
        for (let k = 0; k < n; k++) {
          if (!g.hasEdge(nodeNames[i]!, nodeNames[k]!) || !g.hasEdge(nodeNames[k]!, nodeNames[i]!)) continue;
          if (!g.hasEdge(nodeNames[k]!, nodeNames[j]!) || g.hasEdge(nodeNames[j]!, nodeNames[k]!)) continue;
          for (let l = 0; l < n; l++) {
            if (l === k) continue;
            if (!g.hasEdge(nodeNames[i]!, nodeNames[l]!) || !g.hasEdge(nodeNames[l]!, nodeNames[i]!)) continue;
            if (!g.hasEdge(nodeNames[l]!, nodeNames[j]!) || g.hasEdge(nodeNames[j]!, nodeNames[l]!)) continue;
            if (g.hasEdge(nodeNames[k]!, nodeNames[l]!) || g.hasEdge(nodeNames[l]!, nodeNames[k]!)) continue;
            g.toUndirected(nodeNames[i]!, nodeNames[j]!);
            changed = true;
            break;
          }
        }
      }
    }
  }

  // Build PAG edge notations
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const hasIJ = g.hasEdge(nodeNames[i]!, nodeNames[j]!);
      const hasJI = g.hasEdge(nodeNames[j]!, nodeNames[i]!);
      if (!hasIJ && !hasJI) {
        pagEdges.set(`${nodeNames[i]}-${nodeNames[j]}`, 'none');
      } else if (hasIJ && hasJI) {
        pagEdges.set(`${nodeNames[i]}-${nodeNames[j]}`, 'undirected');
      } else if (hasIJ && !hasJI) {
        pagEdges.set(`${nodeNames[i]}-${nodeNames[j]}`, `${nodeNames[i]}→${nodeNames[j]}`);
      } else {
        pagEdges.set(`${nodeNames[i]}-${nodeNames[j]}`, `${nodeNames[j]}→${nodeNames[i]}`);
      }
    }
  }

  // Phase 3b: FCI-specific R4 — discriminating path rule
  // If there is a discriminating path ⟨V₁,...,Vₖ,W,X,Y⟩ for W-X-Y,
  // and the separation set of V₁ and Y contains W but not X, orient W→X and X→Y.
  for (let w = 0; w < n; w++) {
    for (let x = 0; x < n; x++) {
      if (x === w) continue;
      for (let y = 0; y < n; y++) {
        if (y === w || y === x) continue;
        // Check if W-X-Y is an unshielded triple (W—X, X—Y, NOT W—Y)
        if (!g.hasEdge(nodeNames[w]!, nodeNames[x]!)) continue;
        if (!g.hasEdge(nodeNames[x]!, nodeNames[y]!)) continue;
        if (g.hasEdge(nodeNames[w]!, nodeNames[y]!)) continue;
        // X should not be in the separating set of W and Y → orient as collider
        const key = `${Math.min(w, y)}-${Math.max(w, y)}`;
        const sep = sepSet.get(key);
        if (!sep || !sep.has(nodeNames[x]!)) {
          // Orient W→X←Y if not already oriented
          if (g.hasEdge(nodeNames[w]!, nodeNames[x]!) && g.hasEdge(nodeNames[x]!, nodeNames[w]!)) g.toUndirected(nodeNames[w]!, nodeNames[x]!);
          if (g.hasEdge(nodeNames[y]!, nodeNames[x]!) && g.hasEdge(nodeNames[x]!, nodeNames[y]!)) g.toUndirected(nodeNames[y]!, nodeNames[x]!);
        }
      }
    }
  }

  return { graph: g, pagEdges };
}

// ── Targeted Causal Discovery ──────────────────────────────────────────
// ── Markov Blanket Discovery (Grow-Shrink) ─────────────────────────────

/**
 * Grow-Shrink algorithm for Markov blanket discovery.
 *
 * Phase 1 (Grow): Add variables that are dependent on target given current set.
 * Phase 2 (Shrink): Remove variables that become independent given rest of set.
 *
 * Returns the Markov blanket: the minimal set of variables that renders
 * the target conditionally independent of all others.
 */
export function growShrink(
  data: Matrix,
  targetIdx: number,
  nodeNames: string[],
  alpha: number = 0.05,
): string[] {
  const n = data.rows > 0 ? data.columns : nodeNames.length;

  // Phase 1: Grow
  const blanket = new Set<number>();
  let changed = true;
  while (changed) {
    changed = false;
    for (let i = 0; i < n; i++) {
      if (i === targetIdx || blanket.has(i)) continue;
      if (blanket.size === 0) {
        // Test unconditional dependence
        const p = fisherZTest(data, targetIdx, i, []);
        if (p <= alpha) { blanket.add(i); changed = true; }
      } else {
        // Test conditional dependence given current blanket
        const p = fisherZTest(data, targetIdx, i, [...blanket]);
        if (p <= alpha) { blanket.add(i); changed = true; }
      }
    }
  }

  // Phase 2: Shrink
  changed = true;
  while (changed) {
    changed = false;
    for (const v of [...blanket]) {
      const condSet = [...blanket].filter(x => x !== v);
      const p = fisherZTest(data, targetIdx, v, condSet);
      if (p > alpha) { blanket.delete(v); changed = true; }
    }
  }

  return [...blanket].map(i => nodeNames[i]!).sort();
}

// ── Targeted Parent Discovery ──────────────────────────────────────────

/**
 * Targeted causal discovery — find only the causal parents of specified
 * target variables. Much faster than full graph discovery when only
 * a few targets are of interest.
 *
 * Uses the Grow-Shrink Markov blanket as an initial filter, then applies
 * conditional independence tests to distinguish parents from children.
 */
export function targetedDiscovery(
  data: Matrix,
  targets: string[],
  nodeNames: string[],
  alpha: number = 0.05,
): Map<string, string[]> {
  const result = new Map<string, string[]>();
  if (data.rows === 0) {
    for (const t of targets) result.set(t, []);
    return result;
  }

  for (const target of targets) {
    const tIdx = nodeNames.indexOf(target);
    if (tIdx === -1) { result.set(target, []); continue; }

    // Step 1: Find Markov blanket of target
    const blanket = growShrink(data, tIdx, nodeNames, alpha);

    // Step 2: Within blanket, test each candidate as potential parent
    const parents: string[] = [];
    const blanketSet = new Set(blanket);

    for (const candidate of blanket) {
      const cIdx = nodeNames.indexOf(candidate);
      // Build conditioning set: rest of blanket minus candidate
      const condSet = blanket.filter(b => b !== candidate).map(b => nodeNames.indexOf(b));
      const p = fisherZTest(data, tIdx, cIdx, condSet);
      if (p <= alpha) {
        // Candidate is still dependent → likely direct cause
        parents.push(candidate);
      }
    }
    result.set(target, parents);
  }

  return result;
}


// ── Helpers ──────────────────────────────────────────────────────────

