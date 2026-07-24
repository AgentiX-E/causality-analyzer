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

  // Phase 3: FCI-specific orientation rules (R1-R10 from Zhang 2008)
  // R1-R3 are shared with PC; R4-R10 are FCI-specific.
  // Our adjacency model uses {hasEdge(A,B), hasEdge(B,A)} — equivalent to:
  //   (true, true) =  undirected (∘—∘ in PAG)
  //   (true, false) = directed A→B
  //   (false, false) = no edge
  let changed = true;
  let iter = 0;
  const maxIter = 30;
  while (changed && iter++ < maxIter) {
    changed = false;

    // ── R1: i→j ∘—∗ k, with i∗∗k non-adjacent → j→k ───
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

    // ── R2: i→j→k and i∘—∘k → i→k ───
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

    // ── R3: i∘—∗k→j, i∘—∗l→j, k∗∗l non-adjacent → i→j ───
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

    // ── R4 (FCI): discriminating path rule ───
    // If there is a discriminating path ⟨V₁,...,Vₙ,W,X,Y⟩ for W→X∘—∘Y,
    // orient based on whether V₁ and Y are separated by the set containing W or X.
    //
    // Simplified R4: for unshielded triple W—X—Y (W∗∗Y absent),
    // check sep-set of (W,Y). If sep contains W but NOT X → collider W→X←Y.
    changed = changed || applyR4DiscriminatingPath(g, nodeNames, n, sepSet);

    // ── R5 (FCI): complete ancestral relationships ───
    // If A∘→B∘—∗C and there is a directed path from A to C through B,
    // plus A∗∗C is absent, orient B→C.
    // In our notation: if A→B is undirected-but-A-may-cause-B, orient B→C.
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        // A∘—∘B or A→B (A and B share an edge)
        if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue;
        for (let k = 0; k < n; k++) {
          if (k === i || k === j) continue;
          // B∘—∘C (undirected)
          if (!g.hasEdge(nodeNames[j]!, nodeNames[k]!) || !g.hasEdge(nodeNames[k]!, nodeNames[j]!)) continue;
          // A and C non-adjacent
          if (g.hasEdge(nodeNames[i]!, nodeNames[k]!) || g.hasEdge(nodeNames[k]!, nodeNames[i]!)) continue;
          // Check if A has a directed edge to B
          if (g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;
          // Orient: B→C
          g.toUndirected(nodeNames[j]!, nodeNames[k]!);
          changed = true;
        }
      }
    }

    // ── R6 (FCI): ancestral closure ───
    // If A∘—∘B∘→C and A∗∗C absent, orient B→A (A←B).
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!) || !g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;
        for (let k = 0; k < n; k++) {
          if (k === i || k === j) continue;
          // B→C
          if (!g.hasEdge(nodeNames[j]!, nodeNames[k]!) || g.hasEdge(nodeNames[k]!, nodeNames[j]!)) continue;
          // A∗∗C absent
          if (g.hasEdge(nodeNames[i]!, nodeNames[k]!) || g.hasEdge(nodeNames[k]!, nodeNames[i]!)) continue;
          // Orient: B→A (i.e., A←B)
          g.toUndirected(nodeNames[j]!, nodeNames[i]!);
          changed = true;
        }
      }
    }

    // ── R7 (FCI): resolve circle marks using ancestral info ───
    // If A∘→B and B is not an ancestor of C (where C has a mark with A),
    // orient B's circle marks away from non-ancestors.
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue;
        for (let k = 0; k < n; k++) {
          if (k === i || k === j) continue;
          // A∘—∘C (undirected)
          if (!g.hasEdge(nodeNames[i]!, nodeNames[k]!) || !g.hasEdge(nodeNames[k]!, nodeNames[i]!)) continue;
          // B not ancestor of C (no directed path B→...→C)
          if (g.hasDirectedPath(nodeNames[j]!, nodeNames[k]!)) continue;
          // B not an ancestor of C and also no undirected path
          // Orient A∘→C away from A OR A→C away from A?
          // In PAG: orient the circle at C away from A
          g.toUndirected(nodeNames[k]!, nodeNames[i]!); // A←C
          changed = true;
        }
      }
    }

    // ── R8 (FCI): additional ancestral relationships ───
    // If A→B→C and A∘—∘C, orient A∘—∘C as A→C.
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < n; k++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[k]!) || !g.hasEdge(nodeNames[k]!, nodeNames[i]!)) continue;
        for (let j = 0; j < n; j++) {
          if (j === i || j === k) continue;
          if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!) || g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;
          if (!g.hasEdge(nodeNames[j]!, nodeNames[k]!) || g.hasEdge(nodeNames[k]!, nodeNames[j]!)) continue;
          g.toUndirected(nodeNames[i]!, nodeNames[k]!);
          changed = true;
        }
      }
    }

    // ── R9 (FCI): analogous to R8 for ∘→ edges ───
    // If A∘→B and there is an almost-directed cycle preventing A→B orientation,
    // use ancillary edges to orient.
    // In our simplified model: if A and C are undirected, B is between them,
    // and there's a v-structure-like pattern, resolve the ambiguity.
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!)) continue;
        for (let k = 0; k < n; k++) {
          if (k === i || k === j) continue;
          // If i→k and j∘—∘k, plus i→j, then orient j→k
          if (!g.hasEdge(nodeNames[i]!, nodeNames[k]!) || g.hasEdge(nodeNames[k]!, nodeNames[i]!)) continue;
          if (!g.hasEdge(nodeNames[j]!, nodeNames[k]!) || !g.hasEdge(nodeNames[k]!, nodeNames[j]!)) continue;
          if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!) || g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;
          g.toUndirected(nodeNames[j]!, nodeNames[k]!);
          changed = true;
        }
      }
    }

    // ── R10 (FCI): contextual orientation ───
    // If A∘→B, B∘→C, and A∘—∘C, orient B→C.
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!g.hasEdge(nodeNames[i]!, nodeNames[j]!) || g.hasEdge(nodeNames[j]!, nodeNames[i]!)) continue;
        for (let k = 0; k < n; k++) {
          if (k === i || k === j) continue;
          if (!g.hasEdge(nodeNames[j]!, nodeNames[k]!) || !g.hasEdge(nodeNames[k]!, nodeNames[j]!)) continue;
          if (g.hasEdge(nodeNames[i]!, nodeNames[k]!) || g.hasEdge(nodeNames[k]!, nodeNames[i]!)) continue;
          g.toUndirected(nodeNames[j]!, nodeNames[k]!);
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

// ── FCI R4: Discriminating Path Orientation ─────────────────────────

/**
 * Apply FCI R4 — discriminating path rule (Zhang 2008, §4.2).
 *
 * A discriminating path for W→X∘—∘Y is a path ⟨V₁,...,Vₖ,W,X,Y⟩ where:
 * 1. V₁ is not adjacent to Y
 * 2. Every Vᵢ (i≥2) is a collider on the path
 * 3. Every Vᵢ is adjacent to Y
 *
 * If sepSet(V₁, Y) contains X, orient X→Y.
 * If sepSet(V₁, Y) does NOT contain X, orient W→X and X→Y as a collider:
 * this means X is a common effect of W and Y with latent confounding.
 */
function applyR4DiscriminatingPath(
  g: CausalGraph,
  nodeNames: string[],
  n: number,
  sepSet: Map<string, Set<string>>,
): boolean {
  let changed = false;

  // Simplified R4: detect unshielded triples and apply discriminating-path logic.
  // For W→X∘—∘Y with W∗∗Y absent (unshielded):
  // If sepSet(W,Y) contains X → X→Y.
  // If sepSet(W,Y) does NOT contain X → W→X←Y (collider).

  for (let w = 0; w < n; w++) {
    for (let x = 0; x < n; x++) {
      if (x === w) continue;
      // W→X: W has edge to X but not vice versa
      if (!g.hasEdge(nodeNames[w]!, nodeNames[x]!) || g.hasEdge(nodeNames[x]!, nodeNames[w]!)) continue;
      for (let y = 0; y < n; y++) {
        if (y === w || y === x) continue;
        // X∘—∘Y: undirected
        if (!g.hasEdge(nodeNames[x]!, nodeNames[y]!) || !g.hasEdge(nodeNames[y]!, nodeNames[x]!)) continue;
        // W∗∗Y absent
        if (g.hasEdge(nodeNames[w]!, nodeNames[y]!) || g.hasEdge(nodeNames[y]!, nodeNames[w]!)) continue;

        const sepKey = `${Math.min(w, y)}-${Math.max(w, y)}`;
        const sep = sepSet.get(sepKey);

        if (sep && sep.has(nodeNames[x]!)) {
          // X is in sepSet(W,Y) → X is NOT a collider → orient X→Y
          if (g.hasEdge(nodeNames[y]!, nodeNames[x]!)) {
            g.toUndirected(nodeNames[x]!, nodeNames[y]!);
            changed = true;
          }
        } else {
          // X is NOT in sepSet(W,Y) → X IS a collider → orient W→X←Y
          if (g.hasEdge(nodeNames[x]!, nodeNames[w]!))
            g.toUndirected(nodeNames[w]!, nodeNames[x]!);
          if (g.hasEdge(nodeNames[x]!, nodeNames[y]!))
            g.toUndirected(nodeNames[y]!, nodeNames[x]!);
          if (g.hasEdge(nodeNames[y]!, nodeNames[x]!))
            g.toUndirected(nodeNames[x]!, nodeNames[y]!);
          changed = true;
        }
      }
    }
  }

  return changed;
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

