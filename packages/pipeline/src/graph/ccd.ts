/**
 * CCD — Cyclic Causal Discovery (Richardson 1996).
 *
 * Unlike PC/FCI/GES which assume acyclicity (DAGs), CCD discovers
 * cyclic causal structures (graphs with feedback loops). This is
 * essential for domains like economics, biology, and control systems
 * where feedback loops are common.
 *
 * Algorithm:
 *  1. Build moral graph (connect all pairs with common child)
 *  2. Apply conditional independence tests to prune edges
 *  3. Build PAG-like output with directed and bidirected edges
 *     (allows X→Y and Y→X simultaneously for cycles)
 *  4. Orientation: use collider detection for partial orientation
 *
 * Reference: Richardson (1996). "A Discovery Algorithm for Directed
 *            Cyclic Graphs." UAI 1996.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';
import { CausalGraph } from './causal-graph.js';
import { combinations, fisherZTest } from '@agentix-e/causality-analyzer-core';
import type { DomainKnowledge } from '@agentix-e/causality-analyzer-core';

export interface CCDConfig {
  alpha?: number;
  maxDegree?: number;
  /** Maximum number of feedback loop iterations */
  maxLoopIter?: number;
}

export function ccdAlgorithm(
  data: Matrix,
  nodeNames: string[],
  config: CCDConfig = {},
  domainKnowledge?: DomainKnowledge,
): { graph: CausalGraph; cycleEdges: Map<string, boolean> } {
  const alpha = config.alpha ?? 0.05;
  const maxDegree = config.maxDegree ?? -1;
  const n = nodeNames.length;
  const N = data.rows;
  const cycleEdges = new Map<string, boolean>();
  const dataArr = matrixTo2D(data);

  if (N < 5) return { graph: new CausalGraph(nodeNames), cycleEdges };

  const g = new CausalGraph(nodeNames);

  // Phase 1: Complete graph (CCD starts with all possible edges)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (i !== j) g.addEdge(nodeNames[i], nodeNames[j]);

  // Phase 2: Unconditional CI pruning
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (i !== j && fisherZTest(dataArr, i, j, []) > alpha) {
        g.removeEdge(nodeNames[i], nodeNames[j]);
        g.removeEdge(nodeNames[j], nodeNames[i]);
      }

  // Phase 3: Conditional CI pruning
  let depth = 1;
  const maxDepth = maxDegree === -1 ? Math.min(n - 1, 5) : maxDegree;
  let changed = true;

  while (changed && depth <= maxDepth) {
    changed = false;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j || !g.hasEdge(nodeNames[i], nodeNames[j])) continue;
        const neighbors = g.neighbors(nodeNames[i]).filter(c => c !== nodeNames[j]);
        if (neighbors.length < depth) continue;

        const subsets = combinations(neighbors, depth);
        for (const S of subsets) {
          const sIdx = S.map(s => nodeNames.indexOf(s));
          const p = fisherZTest(dataArr, i, j, sIdx);
          if (p > alpha) {
            g.removeEdge(nodeNames[i], nodeNames[j]);
            changed = true; break;
          }
        }
      }
    }
    depth++;
  }

  // Phase 4: Cycle detection — identify edges that form feedback loops
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      if (g.hasEdge(nodeNames[i], nodeNames[j]) && g.hasEdge(nodeNames[j], nodeNames[i])) {
        // Bidirectional edges suggest a cycle
        cycleEdges.set(`${nodeNames[i]}↔${nodeNames[j]}`, true);
      }
    }
  }

  // Phase 5: Collider detection for partial orientation
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      if (!g.hasEdge(nodeNames[i], nodeNames[j])) continue;
      for (let k = 0; k < n; k++) {
        if (k === i || k === j) continue;
        if (!g.hasEdge(nodeNames[k], nodeNames[j])) continue;
        // i→j and k→j but i and k not adjacent → possible collider
        if (!g.hasEdge(nodeNames[i], nodeNames[k]) && !g.hasEdge(nodeNames[k], nodeNames[i])) {
          // Orient: keep i→j and k→j as-is (they're already present)
          // Mark as directed (remove j→i if present)
          if (g.hasEdge(nodeNames[j], nodeNames[i])) {
            g.orientEdge(nodeNames[i], nodeNames[j]);
          }
          if (g.hasEdge(nodeNames[j], nodeNames[k])) {
            g.orientEdge(nodeNames[k], nodeNames[j]);
          }
        }
      }
    }
  }

  if (domainKnowledge) g.applyDomainKnowledge(domainKnowledge);

  return { graph: g, cycleEdges };
}

function matrixTo2D(data: Matrix): number[][] {
  const rows: number[][] = [];
  for (let r = 0; r < data.rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < data.columns; c++) row.push(data.get(r, c));
    rows.push(row);
  }
  return rows;
}
