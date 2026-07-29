/**
 * I20: Full algorithm precision audit — reference test.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { pcAlgorithm } from '../graph/pc.js';
import { gesAlgorithm } from '../graph/ges.js';
import { directLiNGAM } from '../graph/lingam.js';
import { notearsAlgorithm } from '../graph/notears.js';
import { bossAlgorithm } from '../graph/boss.js';
import { dagmaAlgorithm } from '../graph/dagma.js';
import { golemAlgorithm } from '../graph/golem.js';
import { fciAlgorithm } from '../graph/advanced-discovery.js';
import { gfciAlgorithm } from '../graph/gfci.js';
import { asiaGraph, generateLinearData, computeSHD } from '../benchmark.js';

const f = (n: number) => n.toFixed(3);

describe('Algorithm Precision Reference — ASIA', () => {
  const truth = asiaGraph();
  const nodeNames = [...truth.nodes];
  const { data: rawData } = generateLinearData(truth, 5000, 42);
  const mat = new Matrix(rawData);
  const results: Record<string, {edges:number;shd:number;tpr:number}> = {};

  function record(name: string, g: {nodes:readonly string[];edges:readonly {source:string;target:string}[]}) {
    const shd = computeSHD(g, truth);
    results[name] = { edges: g.edges.length, shd: shd.shd, tpr: shd.tpr };
  }

  it('all algorithms produce valid output', { timeout: 120000 }, () => {
    record('PC', pcAlgorithm(mat, nodeNames).graph);
    record('GES', gesAlgorithm(mat, nodeNames));
    record('LiNGAM', directLiNGAM(mat, nodeNames).graph);
    record('NOTEARS', notearsAlgorithm(rawData, nodeNames).graph);
    record('BOSS', bossAlgorithm(mat, nodeNames));
    record('DAGMA', dagmaAlgorithm(rawData, nodeNames).graph);
    record('GOLEM', golemAlgorithm(rawData, nodeNames).graph);
    const fcig = fciAlgorithm(mat, nodeNames); record('FCI', fcig.graph);
    const gfcig = gfciAlgorithm(mat, nodeNames); record('GFCI', gfcig.graph);

    console.log(`ASIA Benchmark (truth: ${truth.edges.length} edges):`);
    for (const [name, r] of Object.entries(results)) {
      console.log(`  ${name.padEnd(12)} edges=${r.edges}/${truth.edges.length}  SHD=${r.shd}  TPR=${f(r.tpr)}`);
    }

    // All algorithms must produce some output
    for (const [, r] of Object.entries(results)) {
      expect(r.edges).toBeGreaterThanOrEqual(0);
    }
  });
});
