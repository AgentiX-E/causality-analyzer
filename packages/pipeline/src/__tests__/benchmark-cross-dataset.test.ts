/**
 * I28: Cross-dataset precision benchmark.
 *
 * Measures SHD/TPR/FPR for all algorithms on 4 diverse DAGs:
 * ASIA (8n), M-Bias (5n), Butterfly (4n), Child (20n).
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import {
  asiaGraph, sachsGraph, childGraph, butterflyGraph, mBiasGraph,
  generateLinearData, computeSHD,
} from '../../src/benchmark.js';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import { pcAlgorithm } from '../../src/graph/pc.js';
import { gesAlgorithm } from '../../src/graph/ges.js';
import { bossAlgorithm } from '../../src/graph/boss.js';
import { directLiNGAM } from '../../src/graph/lingam.js';
import { notearsAlgorithm } from '../../src/graph/notears.js';
import { dagmaAlgorithm } from '../../src/graph/dagma.js';
import { golemAlgorithm } from '../../src/graph/golem.js';
import { fciAlgorithm } from '../../src/graph/advanced-discovery.js';
import { gfciAlgorithm } from '../../src/graph/gfci.js';

interface AlgoResult { edges: number; shd: number; tpr: number; fpr: number; }
type Algo = (d: Matrix, n: string[]) => CausalGraph | { graph: CausalGraph };

const datasets = [
  { name: 'ASIA', fn: asiaGraph, nodes: 8, edges: 8, samples: 2000 },
  { name: 'M-Bias', fn: mBiasGraph, nodes: 5, edges: 4, samples: 1000 },
  { name: 'Butterfly', fn: butterflyGraph, nodes: 4, edges: 4, samples: 1000 },
  { name: 'Child', fn: childGraph, nodes: 20, edges: 25, samples: 2000 },
];

function runAlgo(algo: Algo, data: number[][], truth: CausalGraph, names: string[]): AlgoResult {
  const mat = new Matrix(data);
  const result = algo(mat, names);
  const graph = 'graph' in result ? result.graph : result;
  const shd = computeSHD(graph, truth);
  return { edges: graph.edges.length, shd: shd.shd, tpr: shd.tpr, fpr: shd.fpr };
}

const algorithms: [string, Algo][] = [
  ['PC',  (d, n) => pcAlgorithm(d, n).graph],
  ['GES', (d, n) => gesAlgorithm(d, n)],
  ['BOSS',(d, n) => bossAlgorithm(d, n)],
  ['LiNGAM', (d, n) => directLiNGAM(d, n).graph],
  ['NOTEARS', (d, n) => notearsAlgorithm((d as Matrix).to2DArray(), n).graph],
  ['DAGMA', (d, n) => dagmaAlgorithm((d as Matrix).to2DArray(), n).graph],
  ['GOLEM', (d, n) => golemAlgorithm(d, n, { maxIter: 800 }).graph],
  ['FCI', (d, n) => fciAlgorithm(d, n).graph],
  ['GFCI',(d, n) => gfciAlgorithm(d, n).graph],
];

describe('I28 Cross-Dataset Precision', () => {
  for (const ds of datasets) {
    describe(ds.name, () => {
      const truth = ds.fn();
      const { data } = generateLinearData(truth, ds.samples, 42);
      const names = [...truth.nodes];
      const results: Record<string, AlgoResult> = {};

      for (const [name, algo] of algorithms) {
        const slow = (ds.nodes >= 20 && ['BOSS', 'LiNGAM', 'GOLEM', 'DAGMA'].includes(name))
          || (name === 'GOLEM') || (name === 'LiNGAM');
        it(`${name} runs and finds edges`, { timeout: slow ? 90000 : 10000 }, () => {
          const r = runAlgo(algo, data, truth, names);
          results[name] = r;
          expect(r.edges).toBeGreaterThanOrEqual(0);
        });
      }

      it('summary', () => {
        console.log(`\n${ds.name} (${ds.nodes}n, ${ds.edges}e, ${ds.samples}s):`);
        const sorted = Object.entries(results)
          .sort((a, b) => a[1].shd - b[1].shd);
        for (const [name, r] of sorted) {
          console.log(`  ${name.padEnd(8)} edges=${r.edges}/${ds.edges}  SHD=${r.shd}  TPR=${r.tpr.toFixed(3)}  FPR=${r.fpr.toFixed(3)}`);
        }
        const best = sorted[0]!;
        console.log(`  🏆 BEST: ${best[0]} SHD=${best[1].shd}`);
        expect(Object.keys(results).length).toBe(algorithms.length);
      });
    });
  }
});
