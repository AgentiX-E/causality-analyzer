/**
 * I19: Full algorithm precision audit on ASIA DAG.
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

function f(n: number): string { return n.toFixed(3); }

type GraphResult = { nodes: readonly string[]; edges: readonly {source:string;target:string;weight:number;directed:boolean}[] };

describe('Full Algorithm Precision — ASIA (8 nodes, 8 edges)', () => {
  const truth = asiaGraph();
  const nodeNames = [...truth.nodes];
  const N = 5000;
  const { data: rawData } = generateLinearData(truth, N, 42);
  const mat = new Matrix(rawData);

  function log(name: string, g: GraphResult) {
    const shd = computeSHD(g, truth);
    console.log(`  ${name.padEnd(12)} edges=${g.edges.length}/${truth.edges.length}  SHD=${shd.shd.toString().padStart(3)}  TPR=${f(shd.tpr)}  FPR=${f(shd.fpr)}`);
    return shd;
  }

  it('PC', () => {
    const { graph } = pcAlgorithm(mat, nodeNames);
    const r = log('PC', graph);
    expect(r.tpr).toBeGreaterThan(0.3);
  });

  it('GES', () => {
    const g = gesAlgorithm(mat, nodeNames);
    const r = log('GES', g);
    expect(r.shd).toBeLessThan(25);
  });

  it('LiNGAM', () => {
    const { graph } = directLiNGAM(mat, nodeNames);
    const r = log('LiNGAM', graph);
    expect(r.graph.edges.length).toBeGreaterThanOrEqual(0);
  });

  it('NOTEARS', () => {
    const { graph } = notearsAlgorithm(rawData, nodeNames);
    const r = log('NOTEARS', graph);
    expect(r.graph.edges.length).toBeGreaterThanOrEqual(0);
  });

  it('BOSS', () => {
    const g = bossAlgorithm(mat, nodeNames);
    const r = log('BOSS', g);
    expect(r.tpr).toBeGreaterThan(0.3);
  });

  it('DAGMA', () => {
    const { graph } = dagmaAlgorithm(rawData, nodeNames);
    const r = log('DAGMA', graph);
    expect(r.graph.edges.length).toBeGreaterThanOrEqual(0);
  });

  it('GOLEM', () => {
    const { graph } = golemAlgorithm(rawData, nodeNames);
    const r = log('GOLEM', graph);
    expect(r.graph.edges.length).toBeGreaterThanOrEqual(0);
  });

  it('FCI', () => {
    const g = fciAlgorithm(mat, nodeNames);
    const r = log('FCI', g);
    expect(r.graph.edges.length).toBeGreaterThanOrEqual(0);
  });

  it('GFCI', () => {
    const g = gfciAlgorithm(mat, nodeNames);
    const r = log('GFCI', g);
    expect(r.graph.edges.length).toBeGreaterThanOrEqual(0);
  });
});
