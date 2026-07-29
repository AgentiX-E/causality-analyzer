import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { gesAlgorithm } from '../graph/ges.js';
import { asiaGraph, generateLinearData, computeSHD } from '../benchmark.js';
import { solveLinear } from '@agentix-e/causality-analyzer-core';

function f(n: number, d: number): string { return n.toFixed(d); }

describe('GES Root Cause', () => {
  it('ALL BIC delta candidates — first iteration', () => {
    const truth = asiaGraph();
    const { data, nodeNames } = generateLinearData(truth, 5000, 42);
    const N = 5000;
    const n = nodeNames.length;

    function bicLoc(nodeIdx: number, parentIdxs: number[]): number {
      const k = parentIdxs.length;
      let rss: number;
      if (k === 0) {
        let ss = 0, sum = 0;
        for (let r = 0; r < N; r++) { const v = data[r]![nodeIdx]!; sum += v; }
        const mean = sum / N;
        for (let r = 0; r < N; r++) { const v = data[r]![nodeIdx]!; ss += (v - mean) ** 2; }
        rss = ss;
      } else {
        const XtX: number[][] = []; for (let i=0;i<k;i++) XtX.push(new Array(k).fill(0));
        const Xty: number[] = new Array(k).fill(0);
        for (let r = 0; r < N; r++) {
          const y = data[r]![nodeIdx]!;
          for (let i = 0; i < k; i++) {
            const xi = data[r]![parentIdxs[i]!]!;
            Xty[i] += xi * y;
            for (let j = 0; j < k; j++) XtX[i]![j] += xi * data[r]![parentIdxs[j]!]!;
          }
        }
        const beta = solveLinear(XtX, Xty);
        rss = 0;
        for (let r = 0; r < N; r++) {
          const y = data[r]![nodeIdx]!;
          let pred = 0;
          for (let i = 0; i < k; i++) pred += (beta[i] ?? 0) * data[r]![parentIdxs[i]!]!;
          rss += (y - pred) ** 2;
        }
      }
      return -(N * Math.log(Math.max(1e-10, rss / N)) + k * Math.log(Math.max(2, N)));
    }

    // Rank ALL candidate edges by BIC improvement
    const candidates: Array<{ from: string; to: string; delta: number; isTrue: boolean }> = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const u = nodeNames[i]!, v = nodeNames[j]!;
        const delta = bicLoc(j, [i]) - bicLoc(j, []);
        candidates.push({ from: u, to: v, delta, isTrue: truth.hasEdge(u, v) });
      }
    }
    candidates.sort((a, b) => b.delta - a.delta);

    console.log('Top 15 candidates by BIC delta:');
    for (let i = 0; i < Math.min(15, candidates.length); i++) {
      const c = candidates[i]!;
      console.log(`  ${c.from}->${c.to} delta=${f(c.delta,1)} [${c.isTrue ? 'TRUE' : 'spurious'}]`);
    }
    expect(candidates.length).toBe(n * (n - 1));
  });

  it('GES result', () => {
    const truth = asiaGraph();
    const nodeNames = [...truth.nodes];
    const { data } = generateLinearData(truth, 5000, 42);
    const result = gesAlgorithm(new Matrix(data), nodeNames);
    console.log('GES result:', result.edges.length, '/', truth.edges.length, 'edges');
    for (const e of result.edges) console.log('  '+e.source+'->'+e.target);
    expect(result.nodes.length).toBe(nodeNames.length);
  });
});
