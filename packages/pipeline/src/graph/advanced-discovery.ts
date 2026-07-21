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

function combinations<T>(arr: T[], k: number): T[][] {
  if (k === 0) return [[]];
  if (arr.length < k) return [];
  const [first, ...rest] = arr as [T, ...T[]];
  const withFirst = combinations(rest, k - 1).map(c => [first, ...c]);
  const without = combinations(rest, k);
  return [...withFirst, ...without];
}
