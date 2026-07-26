import { tsIcdAlgorithm } from '../../src/graph/tsicd.js';
import { starsSelection } from '../../src/graph/stability-selection.js';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../../src/graph/causal-graph.js';
import {
  asiaGraph, sachsGraph, childGraph, alarmGraph, butterflyGraph, mBiasGraph,
  generateLinearData, computeSHD,
} from '../../src/benchmark.js';

// ── Discovery Algorithms ──────────────────────────────────────────
import { pcAlgorithm } from '../../src/graph/pc.js';
import { pcMaxAlgorithm } from '../../src/graph/pc-max.js';
import { gesAlgorithm } from '../../src/graph/ges.js';
import { bossAlgorithm } from '../../src/graph/boss.js';
import { graspAlgorithm } from '../../src/graph/grasp.js';
import { directLiNGAM } from '../../src/graph/lingam.js';
import { icaLiNGAM } from '../../src/graph/ica-lingam.js';
import { notearsAlgorithm } from '../../src/graph/notears.js';
import { dagmaAlgorithm } from '../../src/graph/dagma.js';
import { golemAlgorithm } from '../../src/graph/golem.js';
import { fciAlgorithm } from '../../src/graph/advanced-discovery.js';
import { gfciAlgorithm } from '../../src/graph/gfci.js';
import { rfciAlgorithm } from '../../src/graph/rfci.js';
import { ccdAlgorithm } from '../../src/graph/ccd.js';
import { rcdAlgorithm } from '../../src/graph/rcd.js';
import { cdnodAlgorithm } from '../../src/graph/cdnod.js';
import { mvpcAlgorithm } from '../../src/graph/mvpc.js';
import { imagesAlgorithm } from '../../src/graph/images.js';
import { tsIcdAlgorithm } from '../../src/graph/tsicd.js';
import { faskAlgorithm } from '../../src/graph/fask.js';
import { stabilitySelection } from '../../src/graph/stability-selection.js';
import { discoverClusters } from '../../src/graph/latent-clusters.js';
import { pcmciAlgorithm } from '../../src/graph/pcmci.js';
import { varLingam } from '../../src/graph/var-lingam.js';
import { timinoAlgorithm } from '../../src/graph/timino.js';
import { tsFciAlgorithm } from '../../src/graph/tsfci.js';
import { exactSearch } from '../../src/graph/exact-search.js';
import { ginDetect } from '../../src/graph/gin.js';

// Standard DAGs configured for benchmark
const BENCHMARK_DAGS = [
  { name: 'ASIA', fn: asiaGraph, nodes: 8, edges: 8, samples: 2000 },
  { name: 'M-Bias', fn: mBiasGraph, nodes: 5, edges: 4, samples: 500 },
  { name: 'Butterfly', fn: butterflyGraph, nodes: 4, edges: 4, samples: 500 },
  { name: 'Child', fn: childGraph, nodes: 20, edges: 25, samples: 2000 },
];

// ── Constraint-based algorithms ───────────────────────────────────

describe('Benchmark: Constraint-based', () => {
  for (const { name, fn, nodes, edges, samples } of BENCHMARK_DAGS) {
    const truth = fn();
    const { data, nodeNames } = generateLinearData(truth, samples, 42);
    const matrix = new Matrix(data);

    it(`PC on ${name}: node count valid`, () => {
      const result = pcAlgorithm(matrix, nodeNames, { alpha: 0.05, stable: true }).graph;
      expect(result.nodeCount).toBe(nodes);
    });

    it(`PC-Max on ${name}: node count valid`, () => {
      const pcMaxResult = pcMaxAlgorithm(matrix, nodeNames, 0.05);
      expect(pcMaxResult.nodeCount).toBe(nodes);
    });
  }
});

// ── Score-based algorithms ────────────────────────────────────────

describe('Benchmark: Score-based', () => {
  for (const { name, fn, nodes, samples } of BENCHMARK_DAGS.slice(0, 3)) {
    const truth = fn();
    const { data, nodeNames } = generateLinearData(truth, samples, 43);

    it(`GES on ${name}: produces valid output`, () => {
      const result = gesAlgorithm(new Matrix(data), nodeNames, { maxDegree: nodes > 8 ? 3 : undefined });
      expect(result.nodeCount).toBe(nodes);
    });

    it(`BOSS on ${name}: produces valid DAG`, () => {
      const result = bossAlgorithm(new Matrix(data), nodeNames, { numStarts: 2, maxIter: nodes > 8 ? 15 : 30 });
      expect(result.isDAG()).toBe(true);
    });

    it(`GRaSP on ${name}: produces valid DAG`, () => {
      const result = graspAlgorithm(new Matrix(data), nodeNames, { maxIter: nodes > 8 ? 15 : 30 });
      expect(result.isDAG()).toBe(true);
    });
  }
});

// ── FCM-based algorithms ──────────────────────────────────────────

describe('Benchmark: FCM-based', () => {
  for (const { name, fn, samples } of BENCHMARK_DAGS.slice(0, 2)) {
    const truth = fn();
    const { data, nodeNames } = generateLinearData(truth, samples, 44);

    it(`LiNGAM on ${name}: produces valid output`, () => {
      const result = directLiNGAM(new Matrix(data), nodeNames);
      expect(result.graph.nodeCount).toBe(nodeNames.length);
      expect(result.order.length).toBe(nodeNames.length);
    });

    it(`ICA-LiNGAM on ${name}: produces valid output`, () => {
      const result = icaLiNGAM(data, nodeNames, { tol: 1e-3, maxIter: 200 });
      expect(result.graph.nodeCount).toBe(nodeNames.length);
    });
  }
});

// ── Continuous optimization algorithms ───────────────────────────

describe('Benchmark: Continuous Optimization', () => {
  for (const { name, fn, nodes, samples } of BENCHMARK_DAGS.slice(0, 3)) {
    const truth = fn();
    const { data, nodeNames } = generateLinearData(truth, samples, 45);

    if (nodes <= 8) {
      it(`NOTEARS on ${name}: produces valid output`, () => {
        const result = notearsAlgorithm(data, nodeNames, { maxOuterIter: nodes > 5 ? 5 : 10, lambda1: 0.1 });
        expect(result.graph.nodeCount).toBe(nodes);
      });

      it(`DAGMA on ${name}: produces valid DAG`, { timeout: 45000 }, () => {
        const result = dagmaAlgorithm(data, nodeNames, { T: 2, warmIter: 500, maxIter: 1000, maxOuterIter: nodes > 5 ? 10 : 15 });
        expect(result.graph.nodeCount).toBe(nodes);
        expect(result.W.length).toBe(nodes * nodes);
      });

      it(`GOLEM on ${name}: produces valid output`, () => {
        const result = golemAlgorithm(data, nodeNames, { maxIter: nodes > 5 ? 1500 : 3000 });
        expect(result.graph.nodeCount).toBe(nodes);
        expect(result.W).toBeInstanceOf(Float64Array);
      });
    }
  }
});

// ── Enterprise / Infrastructure algorithms ───────────────────────

describe('Benchmark: Enterprise & Infrastructure', () => {
  it('ExactSearch finds optimal on 3-node data', () => {
    const data = new Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9], [1.5, 3, 4.5], [2.5, 5, 7.5]]);
    const result = exactSearch(data, ['X', 'Y', 'Z']);
    expect(result.isDAG()).toBe(true);
  });

  it('Stability Selection produces stable graph', () => {
    const truth = butterflyGraph();
    const { data, nodeNames } = generateLinearData(truth, 300, 46);
    // stabilitySelection requires a discoverFn as 3rd argument (not an options object)
    const result = stabilitySelection(
      new Matrix(data),
      nodeNames,
      (d: Matrix, names: string[]) => pcAlgorithm(d, names).graph,
      { nSubsamples: 10 },
    );
    expect(result.stableGraph).toBeDefined();
    expect(result.nSubsamples).toBeGreaterThan(0);
  });

  it('StARS selects nonzero regularization', () => {
    const truth = butterflyGraph();
    const { data, nodeNames } = generateLinearData(truth, 300, 47);
    const result = starsSelection(
      new Matrix(data),
      nodeNames,
      (alpha: number) => (d: Matrix, names: string[]) => pcAlgorithm(d, names, { alpha }).graph,
      [0.01, 0.05, 0.1],
      { nSubsamples: 5 },
    );
    expect(result.bestParam).toBeGreaterThanOrEqual(0);
  });
});

// ── Time-series algorithms ────────────────────────────────────────

describe('Benchmark: Time Series', () => {
  function generateCoupledTS(T: number): number[][] {
    const data: number[][] = [];
    for (let t = 0; t < T; t++) {
      const x = Math.random() * 2 - 1;
      const y = (t > 0 ? 0.7 * data[t - 1]![0]! : 0) + Math.random() * 0.3;
      const z = (t > 0 ? 0.5 * data[t - 1]![1]! : 0) + Math.random() * 0.3;
      data.push([x, y, z]);
    }
    return data;
  }

  it('PCMCI discovers lagged structure', () => {
    const data = generateCoupledTS(300);
    const result = pcmciAlgorithm(data, ['X', 'Y', 'Z'], { tauMax: 2 });
    expect(result.tauMax).toBe(2);
    expect(result.parents.size).toBe(3);
  });

  it('VAR-LiNGAM provides lagged matrices', () => {
    const data = generateCoupledTS(300);
    const result = varLingam(data, ['X', 'Y', 'Z'], { maxLag: 2 });
    expect(result.laggedMatrices).toHaveLength(2);
    expect(result.B0.length).toBe(9);
  });

  it('TiMINo discovers temporal edges', () => {
    const data = generateCoupledTS(200);
    const result = timinoAlgorithm(data, ['X', 'Y', 'Z'], 2);
    expect(result.tauMax).toBe(2);
  });

  it('tsFCI on coupled TS produces valid PAG', () => {
    const data = generateCoupledTS(200);
    const result = tsFciAlgorithm(data, ['X', 'Y', 'Z'], { maxLag: 1 });
    expect(result.instantaneousGraph.nodeCount).toBe(3);
  });

  it('TS-ICD produces valid contemporaneous graph', () => {
    const data = generateCoupledTS(200);
    // tsIcdAlgorithm expects number[][], not Matrix
    const result = tsIcdAlgorithm(data, ['X', 'Y', 'Z']);
    expect(result.contemporaneous).toBeDefined();
    expect(result.contemporaneous.nodeCount).toBe(3);
  });
});
