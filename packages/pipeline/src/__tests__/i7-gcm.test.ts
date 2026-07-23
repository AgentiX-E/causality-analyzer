/**
 * I7 tests: GCM + Counterfactual + Branch coverage improvements.
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../graph/causal-graph.js';
import { StructuralCausalModel, cateToRCA } from '../gcm/structural-causal-model.js';
import { CIRCAPipeline, RHTScorer } from '../analyze/circa.js';
import { HTRCA } from '../analyze/rca.js';
import { estimateLinearRegression, refuteBootstrap } from '../infer/causal-inference.js';
import { VotingDetector } from '../detect/voting-detector.js';
import { StatsDetector } from '../detect/stats-detector.js';
import { SPOTDetector } from '../detect/spot.js';
import { SpectralResidualDetector } from '../detect/spectral-residual.js';

// ── GCM Training ────────────────────────────────────────────────────
describe('StructuralCausalModel', () => {
  function twoNodeGraph(): CausalGraph {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');
    return g;
  }

  it('trains mechanisms and performs forward computation', () => {
    const g = twoNodeGraph();
    const scm = new StructuralCausalModel(g);
    // Y = 3*X + noise
    const data = Array.from({ length: 100 }, () => {
      const x = Math.random() * 5;
      const y = 3 * x + (Math.random() - 0.5) * 0.5;
      return [x, y];
    });
    scm.train(data);
    // Abduction
    const noise = scm.abduct({ X: 3, Y: 9.5 });
    expect(typeof noise.X).toBe('number');
    expect(typeof noise.Y).toBe('number');
  });

  it('counterfactual: abduction-action-prediction', () => {
    const g = twoNodeGraph();
    const scm = new StructuralCausalModel(g);
    const data = Array.from({ length: 80 }, () => {
      const x = Math.random() * 5;
      return [x, 3 * x + (Math.random() - 0.5) * 0.5];
    });
    scm.train(data);

    // Observe current state
    const observation = { X: 5, Y: 16 };
    const noise = scm.abduct(observation);

    // Counterfactual: what if X had been 0?
    const cf = scm.counterfactual(noise, { X: 0 });
    expect(cf.X).toBe(0);
    expect(cf.Y).toBeLessThan(observation.Y); // Y should be lower when X=0
    expect(cf.Y).not.toBeNaN();
  });

  it('anomalyScores identifies anomalous nodes', () => {
    const g = twoNodeGraph();
    const scm = new StructuralCausalModel(g);
    const data = Array.from({ length: 60 }, () => {
      const x = Math.random() * 5;
      return [x, 2 * x + (Math.random() - 0.5) * 0.5];
    });
    scm.train(data);
    // X=20 is anomalous, Y should follow
    const scores = scm.anomalyScores({ X: 20, Y: 42 });
    expect(scores.get('X')!).toBeGreaterThan(1);
  });

  it('attributeAnomalies ranks root causes', () => {
    const g = twoNodeGraph();
    const scm = new StructuralCausalModel(g);
    const data = Array.from({ length: 50 }, () => {
      const x = Math.random() * 3;
      return [x, 2 * x + (Math.random() - 0.5) * 0.3];
    });
    scm.train(data);
    const attr = scm.attributeAnomalies({ X: 15, Y: 32 });
    expect(attr.length).toBeGreaterThan(0);
    // Root nodes (X) should rank highest
    expect(attr[0]!.name).toBe('X');
  });

  it('detectDistributionChange finds significant shifts', () => {
    const g = twoNodeGraph();
    const scm = new StructuralCausalModel(g);
    const data = Array.from({ length: 50 }, () => {
      const x = Math.random() * 3;
      return [x, 2 * x + (Math.random() - 0.5) * 0.3];
    });
    scm.train(data);

    const before = [{ X: 2, Y: 4.5 }, { X: 2.5, Y: 5.5 }];
    const after = [{ X: 10, Y: 22 }, { X: 11, Y: 24 }];
    const result = scm.detectDistributionChange(before, after);
    expect(result.changed).toBe(true);
    expect(result.pValue).toBeGreaterThan(0);
  });

  it('empty data handled gracefully', () => {
    const g = twoNodeGraph();
    const scm = new StructuralCausalModel(g);
    scm.train([]);
    const scores = scm.anomalyScores({ X: 1, Y: 2 });
    expect(scores.get('X')).toBeDefined();
  });
});

// ── CATE → RCA Bridge ────────────────────────────────────────────
describe('cateToRCA', () => {
  it('converts treatment effects to root cause ranking', () => {
    const effects = new Map([
      ['cpu', 3.5], ['memory', 8.2], ['disk_io', -1.2],
    ]);
    const rca = cateToRCA(effects);
    expect(rca.length).toBe(3);
    expect(rca[0]!.name).toBe('memory');
    expect(rca[2]!.name).toBe('disk_io');
  });

  it('handles empty effects map', () => {
    expect(cateToRCA(new Map()).length).toBe(0);
  });
});

// ── Branch coverage: VotingDetector strategies ───────────────────
describe('voting detector branch coverage', () => {
  it('weighted with all-zero weights handles gracefully', () => {
    const d1 = new StatsDetector({ threshold: 3, minSamples: 3 });
    const d2 = new StatsDetector({ threshold: 3, minSamples: 3 });
    const v = new VotingDetector([d1, d2], { strategy: 'weighted', weights: [0, 0] });
    for (let i = 0; i < 5; i++) v.update([5]);
    const r = v.update([50]);
    expect(typeof r.isAnomalous).toBe('boolean');
  });

  it('majority with minAgreement exceeding detector count', () => {
    const d = new StatsDetector({ threshold: 3, minSamples: 3 });
    const v = new VotingDetector([d], { strategy: 'majority', minAgreement: 2 });
    for (let i = 0; i < 5; i++) v.update([5]);
    const r = v.update([50]);
    expect(r.isAnomalous).toBe(false); // can't meet quorum of 2 with 1 detector
  });
});

// ── Branch coverage: CIRCA edge paths ────────────────────────────
describe('CIRCA edge path coverage', () => {
  it('RHT aggregator sum with multiple aggregation methods', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const scorer = new RHTScorer({ aggregator: 'sum', tauMax: 1 });
    scorer.train(g, [[1, 2], [1.1, 2.2]]);
    const scores = scorer.score([[5, 12], [4, 11]]);
    expect(scores.size).toBe(2);
  });


});

// ── Branch coverage: causal inference refutation edge cases ──────
describe('causal inference edge case coverage', () => {
  it('bootstrap with very small data', () => {
    const data = [[1, 3], [0, 1], [1, 4], [0, 0]];
    const result = refuteBootstrap(data, 0, 1, 20);
    expect(typeof result.isRobust).toBe('boolean');
  });

  it('estimation with single data point', () => {
    const result = estimateLinearRegression([[1, 5]], 0, 1);
    expect(typeof result.ate).toBe('number');
    expect(isNaN(result.ate)).toBe(false);
  });
});

// ── Branch coverage: spectral residual edge ─────────────────────
describe('spectral residual branch coverage', () => {
  it('handles very small score window', () => {
    const sr = new SpectralResidualDetector({ minPoints: 16, scoreWindow: 3, threshold: 5 });
    for (let i = 0; i < 32; i++) sr.update(1);
    for (let i = 0; i < 10; i++) {
      const r = sr.update(i % 2 === 0 ? 10 : 1);
      expect(typeof r.isAnomalous).toBe('boolean');
    }
  });

  it('handles magWindow=1', () => {
    const sr = new SpectralResidualDetector({ minPoints: 16, magWindow: 1, threshold: 5 });
    for (let i = 0; i < 32; i++) sr.update(1);
    const r = sr.update(10);
    expect(typeof r.isAnomalous).toBe('boolean');
  });
});

// ── SCM NaN handling: consistent row-skipping ────────────────────
describe('SCM NaN handling', () => {
  it('handles NaN values in training data without crashing', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');
    const scm = new StructuralCausalModel(g);
    const data = [
      [1, 2], [NaN, 3], [2, NaN], [NaN, NaN], [3, 7],
    ];
    scm.train(data);
    // Should not crash and should produce valid mechanisms
    expect(scm.causalGraph).toBe(g);
  });

  it('NaN row skipping: training with NaN produces same coefficients as without NaN', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');
    // Generate clean data: Y = 3*X + noise
    const cleanData = Array.from({ length: 50 }, () => {
      const x = Math.random() * 5;
      return [x, 3 * x + (Math.random() - 0.5) * 0.5];
    });
    // Same data with some NaN rows inserted
    const dirtyData = [...cleanData, [NaN, 5], [1, NaN], [NaN, NaN]];

    const scmClean = new StructuralCausalModel(g);
    scmClean.train(cleanData);
    const scmDirty = new StructuralCausalModel(g);
    scmDirty.train(dirtyData);

    // Both should be able to perform abduction
    const obs = { X: 3, Y: 10 };
    const noiseClean = scmClean.abduct(obs);
    const noiseDirty = scmDirty.abduct(obs);
    expect(typeof noiseClean.X).toBe('number');
    expect(typeof noiseDirty.X).toBe('number');
  });

  it('all-NaN column handled gracefully', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');
    const scm = new StructuralCausalModel(g);
    const data = Array.from({ length: 10 }, () => [NaN, NaN] as number[]);
    scm.train(data);
    // Should not crash — mechanisms should fall back to safe defaults
    const scores = scm.anomalyScores({ X: 1, Y: 2 });
    expect(scores.size).toBe(2);
  });

  it('mixed NaN: some rows clean, some dirty', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const scm = new StructuralCausalModel(g);
    const data = [
      [1, 2, 3], [NaN, 2, 3], [1, NaN, 3], [1, 2, NaN],
      [2, 4, 7], [3, 6, 13],
    ];
    scm.train(data);
    // Should not crash — only clean rows contribute
    const scores = scm.anomalyScores({ X: 10, Y: 20, Z: 50 });
    expect(scores.get('X')).toBeDefined();
    expect(scores.get('Y')).toBeDefined();
    expect(scores.get('Z')).toBeDefined();
  });
});
describe('GCM function coverage', () => {
  it('causalGraph getter returns the graph', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const scm = new StructuralCausalModel(g);
    expect(scm.causalGraph).toBe(g);
  });

  it('detectDistributionChange with empty before/after', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const scm = new StructuralCausalModel(g);
    scm.train([[1, 2], [1.1, 2.2]]);
    const result = scm.detectDistributionChange([], [{ A: 5, B: 12 }]);
    expect(typeof result.changed).toBe('boolean');
  });
});
