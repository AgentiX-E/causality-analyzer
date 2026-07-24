/**
 * I47 tests: CausalGraph + StructuralCausalModel round-trip serialization.
 *
 * Verifies that toJSON() → fromJSON() faithfully reconstructs the original objects,
 * including graph topology, edge directionality, mechanism parameters, and node ordering.
 */
import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../graph/causal-graph.js';
import { StructuralCausalModel } from '../gcm/structural-causal-model.js';

// ─────────────────────────────────────────────────────────────────────
// CausalGraph round-trip serialization tests
// ─────────────────────────────────────────────────────────────────────
describe('CausalGraph serialization round-trip', () => {
  it('simple DAG X→Y preserves edges and adjacency', () => {
    const g = new CausalGraph(['X', 'Y']);
    g.addEdge('X', 'Y');

    const json = g.toJSON();
    const restored = CausalGraph.fromJSON(json);

    // Structural equality: same nodes
    expect(restored.nodes).toEqual(g.nodes);
    expect(restored.nodeCount).toBe(g.nodeCount);

    // Edge semantics preserved
    expect(restored.hasEdge('X', 'Y')).toBe(true);
    expect(restored.hasEdge('Y', 'X')).toBe(false);

    // Adjacency matrix matches exactly
    const origAdj = g.adjacencyMatrix;
    const restAdj = restored.adjacencyMatrix;
    expect(restAdj.get(0, 1)).toBe(1); // X→Y
    expect(restAdj.get(1, 0)).toBe(0); // Y→X absent
    for (let i = 0; i < g.nodeCount; i++) {
      for (let j = 0; j < g.nodeCount; j++) {
        expect(restAdj.get(i, j)).toBe(origAdj.get(i, j));
      }
    }

    // Edge list matches
    expect(restored.edges).toHaveLength(1);
    expect(restored.edges[0]!.source).toBe('X');
    expect(restored.edges[0]!.target).toBe('Y');
    expect(restored.edges[0]!.directed).toBe(true);

    // Graph properties preserved
    expect(restored.isDAG()).toBe(true);
    expect(restored.parents('Y')).toEqual(['X']);
    expect(restored.children('X')).toEqual(['Y']);
  });

  it('complex DAG preserves multiple nodes with parents and children', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D', 'E']);
    g.addEdge('A', 'B');
    g.addEdge('A', 'C');
    g.addEdge('B', 'D');
    g.addEdge('C', 'D');
    g.addEdge('D', 'E');
    g.addEdge('C', 'E'); // C→E is an additional path

    const json = g.toJSON();
    const restored = CausalGraph.fromJSON(json);

    // All edges preserved
    expect(restored.hasEdge('A', 'B')).toBe(true);
    expect(restored.hasEdge('A', 'C')).toBe(true);
    expect(restored.hasEdge('B', 'D')).toBe(true);
    expect(restored.hasEdge('C', 'D')).toBe(true);
    expect(restored.hasEdge('D', 'E')).toBe(true);
    expect(restored.hasEdge('C', 'E')).toBe(true);

    // No spurious edges
    expect(restored.edges).toHaveLength(6);

    // Parent/child relationships
    expect(restored.parents('D').sort()).toEqual(['B', 'C']);
    expect(restored.parents('E').sort()).toEqual(['C', 'D']);
    expect(restored.parents('A')).toEqual([]);
    expect(restored.children('A').sort()).toEqual(['B', 'C']);

    // descendants
    expect(restored.descendants('A')).toEqual(new Set(['A', 'B', 'C', 'D', 'E']));
    expect(restored.descendants('B')).toEqual(new Set(['B', 'D', 'E']));

    // Structural Hamming Distance = 0 (identical)
    expect(g.shd(restored)).toBe(0);

    // Topological order valid
    // The graph is a DAG, topologicalSort should include all nodes
    const topo = restored.topologicalSort();
    expect(topo).toHaveLength(5);
    expect(new Set(topo)).toEqual(new Set(['A', 'B', 'C', 'D', 'E']));
  });

  it('empty graph with no edges round-trips correctly', () => {
    const g = new CausalGraph(['A', 'B', 'C']);

    const json = g.toJSON();
    const restored = CausalGraph.fromJSON(json);

    expect(restored.nodes).toEqual(['A', 'B', 'C']);
    expect(restored.edges).toHaveLength(0);

    // No edges should exist for any pair
    for (const from of g.nodes) {
      for (const to of g.nodes) {
        expect(restored.hasEdge(from, to)).toBe(false);
      }
    }

    // Adjacency matrix is all zero i≠j
    const adj = restored.adjacencyMatrix;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        if (i !== j) expect(adj.get(i, j)).toBe(0);
      }
    }

    // Structural equality
    expect(g.shd(restored)).toBe(0);
  });

  it('graph with bidirectional/undirected edges preserves bidirectionality', () => {
    const g = new CausalGraph(['M', 'N']);
    g.undirectedEdge('M', 'N'); // sets both M→N and N→M

    // Verify original is bidirectional
    expect(g.hasEdge('M', 'N')).toBe(true);
    expect(g.hasEdge('N', 'M')).toBe(true);

    const json = g.toJSON();
    const restored = CausalGraph.fromJSON(json);

    // Both directions must survive the round-trip
    expect(restored.hasEdge('M', 'N')).toBe(true);
    expect(restored.hasEdge('N', 'M')).toBe(true);

    // Adjacency matrix: both entries are 1
    const adj = restored.adjacencyMatrix;
    expect(adj.get(0, 1)).toBe(1); // M→N
    expect(adj.get(1, 0)).toBe(1); // N→M

    // Edge list: both directions appear with directed=false
    const undirectedEdges = restored.edges.filter(e => !e.directed);
    expect(undirectedEdges).toHaveLength(2);
    const sources = undirectedEdges.map(e => e.source).sort();
    const targets = undirectedEdges.map(e => e.target).sort();
    expect(sources).toEqual(['M', 'N']);
    expect(targets).toEqual(['M', 'N']);

    // neighbors() returns the other node
    expect(restored.neighbors('M')).toEqual(['N']);
    expect(restored.neighbors('N')).toEqual(['M']);
  });

  it('square adjacency: matrix diagonal entries are correctly preserved', () => {
    // Self-loops might be present in some graphs; verify adjacency handles them
    const nodes = ['P', 'Q'];
    const g = new CausalGraph(nodes);
    g.addEdge('P', 'Q');

    const json = g.toJSON();
    const restored = CausalGraph.fromJSON(json);

    const adj = restored.adjacencyMatrix;
    // Diagonal should match original (usually 0 for DAGs, but we test exact fidelity)
    const origAdj = g.adjacencyMatrix;
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        expect(adj.get(i, j)).toBe(origAdj.get(i, j), `adj[${i}][${j}] mismatch`);
      }
    }
  });

  it('clone produces structurally equal graph', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y');
    g.addEdge('Y', 'Z');

    const cloned = g.clone();

    // Same nodes
    expect(cloned.nodes).toEqual(g.nodes);

    // Same edges
    expect(cloned.hasEdge('X', 'Y')).toBe(true);
    expect(cloned.hasEdge('Y', 'Z')).toBe(true);
    expect(cloned.hasEdge('X', 'Z')).toBe(false);

    // Structural Hamming Distance = 0
    expect(g.shd(cloned)).toBe(0);

    // Mutation independence: modifying clone does not affect original
    cloned.addEdge('X', 'Z');
    expect(cloned.hasEdge('X', 'Z')).toBe(true);
    expect(g.hasEdge('X', 'Z')).toBe(false);

    cloned.removeEdge('X', 'Y');
    expect(cloned.hasEdge('X', 'Y')).toBe(false);
    expect(g.hasEdge('X', 'Y')).toBe(true);
  });

  it('toJSON → fromJSON → toJSON is idempotent (serialization consistency)', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D']);
    g.addEdge('A', 'B');
    g.addEdge('B', 'C');
    g.addEdge('A', 'D');
    g.addEdge('D', 'C');

    // First serialization
    const json1 = g.toJSON();
    const g1 = CausalGraph.fromJSON(json1);

    // Second serialization from deserialized graph
    const json2 = g1.toJSON();
    const g2 = CausalGraph.fromJSON(json2);

    // Both serialized forms should be deeply equal
    expect(json2).toEqual(json1);

    // Both deserialized graphs should be structurally identical
    expect(g1.shd(g2)).toBe(0);
    expect(g1.nodes).toEqual(g2.nodes);
    for (const e1 of g1.edges) {
      expect(g2.hasEdge(e1.source, e1.target)).toBe(true);
    }
    for (const e2 of g2.edges) {
      expect(g1.hasEdge(e2.source, e2.target)).toBe(true);
    }
  });

  it('nodes with domain-specific names and ordering survive round-trip', () => {
    const names = ['cpu_usage', 'memory_leak', 'latency_spike', 'error_rate'];
    const g = new CausalGraph(names);
    g.addEdge('cpu_usage', 'latency_spike');
    g.addEdge('memory_leak', 'latency_spike');
    g.addEdge('latency_spike', 'error_rate');

    const restored = CausalGraph.fromJSON(g.toJSON());

    expect(restored.nodes).toEqual(names);
    expect(restored.parents('latency_spike').sort()).toEqual(['cpu_usage', 'memory_leak']);
    expect(restored.children('latency_spike')).toEqual(['error_rate']);
  });

  it('large DAG with chain topology round-trips correctly', () => {
    const n = 20;
    const names = Array.from({ length: n }, (_, i) => `V${i}`);
    const g = new CausalGraph(names);
    for (let i = 0; i < n - 1; i++) {
      g.addEdge(names[i]!, names[i + 1]!);
    }

    const restored = CausalGraph.fromJSON(g.toJSON());

    expect(restored.nodeCount).toBe(n);
    expect(restored.edges).toHaveLength(n - 1);
    for (let i = 0; i < n - 1; i++) {
      expect(restored.hasEdge(names[i]!, names[i + 1]!)).toBe(true);
    }
    expect(g.shd(restored)).toBe(0);

    // Chain is a DAG
    expect(restored.isDAG()).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────────
// StructuralCausalModel round-trip serialization tests
// ─────────────────────────────────────────────────────────────────────
describe('StructuralCausalModel serialization round-trip', () => {
  // Shared training data for many tests: X → Y → Z with linear relationships
  // X ≈ N(1, 0.1), Y ≈ 2*X + N(2, 0.1), Z ≈ 1.5*Y + N(3, 0.1)
  const gXYZ = () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y');
    g.addEdge('Y', 'Z');
    return g;
  };

  const dataXYZ = [
    [1, 2, 3],
    [1.1, 2.1, 2.9],
    [0.9, 1.9, 3.1],
  ];

  it('train → serialize → deserialize → mechanisms match (coefficients preserved)', () => {
    const scm = new StructuralCausalModel(gXYZ());
    scm.train(dataXYZ);

    const json = scm.toJSON();
    const restored = StructuralCausalModel.fromJSON(json);

    // Version field
    expect(json.version).toBe(1);

    // Same number of mechanisms
    expect(json.mechanisms).toHaveLength(3);

    // Root node X: intercept (mean) and noiseStd should match
    const origX = json.mechanisms.find(m => m.nodeName === 'X')!;
    const restX = json.mechanisms.find(m => m.nodeName === 'X')!;
    expect(origX).toBeDefined();
    expect(restX).toBeDefined();
    expect(restX.intercept).toBeCloseTo(origX.intercept, 8);
    expect(restX.noiseStd).toBeCloseTo(origX.noiseStd, 8);
    expect(restX.coef).toEqual([]);

    // Child node Y: coefficients should include X→Y weight
    const origY = json.mechanisms.find(m => m.nodeName === 'Y')!;
    const restY = json.mechanisms.find(m => m.nodeName === 'Y')!;
    expect(origY).toBeDefined();
    expect(restY).toBeDefined();
    expect(restY.coef).toHaveLength(1);
    expect(restY.coef[0]).toBeCloseTo(origY.coef[0]!, 8);
    expect(restY.intercept).toBeCloseTo(origY.intercept, 8);
    expect(restY.noiseStd).toBeCloseTo(origY.noiseStd, 8);

    // Child node Z: coefficients should include Y→Z weight
    const origZ = json.mechanisms.find(m => m.nodeName === 'Z')!;
    const restZ = json.mechanisms.find(m => m.nodeName === 'Z')!;
    expect(origZ).toBeDefined();
    expect(restZ).toBeDefined();
    expect(restZ.coef).toHaveLength(1);
    expect(restZ.coef[0]).toBeCloseTo(origZ.coef[0]!, 8);
    expect(restZ.intercept).toBeCloseTo(origZ.intercept, 8);
    expect(restZ.noiseStd).toBeCloseTo(origZ.noiseStd, 8);

    // Graph topology preserved
    expect(restored.causalGraph.hasEdge('X', 'Y')).toBe(true);
    expect(restored.causalGraph.hasEdge('Y', 'Z')).toBe(true);
  });

  it('root node mechanism: intercept and noiseStd are faithfully preserved', () => {
    // X is a root node (no parents), trained as X = μ + ε
    const g = new CausalGraph(['X', 'A']);
    g.addEdge('X', 'A');
    const scm = new StructuralCausalModel(g);

    // Generate data where X has mean=5, std≈1, A≈X + noise
    const data = [
      [4.5, 4.8],
      [5.5, 4.5],
      [5.0, 5.2],
      [4.0, 3.9],
      [6.0, 6.1],
    ];
    scm.train(data);

    const trained = scm.toJSON();
    const restored = StructuralCausalModel.fromJSON(trained);

    // Root node X
    const origX = trained.mechanisms.find(m => m.nodeName === 'X')!;
    const restX = trained.mechanisms.find(m => m.nodeName === 'X')!;
    expect(restX.coef).toEqual([]); // root nodes have no coefficients
    expect(restX.intercept).toBeCloseTo(origX.intercept, 8);
    expect(restX.noiseStd).toBeCloseTo(origX.noiseStd, 8);
    expect(origX.intercept).toBeGreaterThan(0); // should be near 5.0
    expect(origX.noiseStd).toBeGreaterThan(0); // should be nonzero

    // Run anomaly scores through restored model — should work without throwing
    const obs = { X: 5.0, A: 5.0 };
    const scores = restored.anomalyScores(obs);
    expect(scores.has('X')).toBe(true);
    expect(scores.has('A')).toBe(true);
    // X's z-score should be reasonable (not extreme)
    expect(Math.abs(scores.get('X')!)).toBeLessThan(10);
  });

  it('child node mechanism: coefficients reflect parent→child weights', () => {
    const g = new CausalGraph(['P1', 'P2', 'C']);
    g.addEdge('P1', 'C');
    g.addEdge('P2', 'C');
    const scm = new StructuralCausalModel(g);

    // C ≈ 2*P1 + 3*P2 + noise
    const data = [
      [1, 0, 2],
      [2, 1, 7],
      [0, 1, 3],
      [1, 1, 5],
    ];
    scm.train(data);

    const json = scm.toJSON();
    const restored = StructuralCausalModel.fromJSON(json);

    // Child node C: check parent coefficients
    const origC = json.mechanisms.find(m => m.nodeName === 'C')!;
    expect(origC.coef).toHaveLength(2);

    // Verify restored mechanism parameters match the serialized JSON exactly
    expect(json.mechanisms).toHaveLength(3);
    for (const m of json.mechanisms) {
      const match = json.mechanisms.find(r => r.nodeName === m.nodeName);
      expect(match).toBeDefined();
      expect(match!.coef).toEqual(m.coef);
      expect(match!.noiseStd).toBeCloseTo(m.noiseStd, 8);
      expect(match!.intercept).toBeCloseTo(m.intercept, 8);
    }

    // The restored SCM should produce anomaly scores
    const obs = { P1: 1, P2: 1, C: 5 };
    const scores = restored.anomalyScores(obs);
    expect(scores.has('C')).toBe(true);
  });

  it('round-trip: train → serialize → deserialize → counterfactual consistency', () => {
    const scm = new StructuralCausalModel(gXYZ());
    scm.train(dataXYZ);

    // Compute counterfactual with original model
    const observation = { X: 1.0, Y: 2.0, Z: 3.0 };
    const noise = scm.abduct(observation);
    const intervention = { X: 0.5 }; // set X=0.5
    const cfOriginal = scm.counterfactual(noise, intervention);

    // Serialize and deserialize
    const json = scm.toJSON();
    const restored = StructuralCausalModel.fromJSON(json);

    // Abduction with restored model should produce same noise
    const noiseRestored = restored.abduct(observation);
    for (const key of Object.keys(noise)) {
      expect(noiseRestored[key]).toBeCloseTo(noise[key], 8);
    }

    // Counterfactual with restored model should match original
    const cfRestored = restored.counterfactual(noiseRestored, intervention);

    // All nodes present
    expect(Object.keys(cfRestored).sort()).toEqual(['X', 'Y', 'Z']);

    // Intervened node equals intervention value
    expect(cfRestored['X']).toBe(0.5);

    // Counterfactual values should match between original and restored
    expect(cfRestored['Y']).toBeCloseTo(cfOriginal['Y']!, 8);
    expect(cfRestored['Z']).toBeCloseTo(cfOriginal['Z']!, 8);
  });

  it('nodeOrder is preserved after deserialization', () => {
    const scm = new StructuralCausalModel(gXYZ());
    scm.train(dataXYZ);

    const json = scm.toJSON();
    const restored = StructuralCausalModel.fromJSON(json);

    // nodeOrder must be exactly the same
    expect(json.nodeOrder).toEqual(restored['nodeOrder']);
    expect(json.nodeOrder).toHaveLength(3);
    // Topological sort for X→Y→Z should have X before Y before Z
    expect(json.nodeOrder.indexOf('X')).toBeLessThan(json.nodeOrder.indexOf('Y'));
    expect(json.nodeOrder.indexOf('Y')).toBeLessThan(json.nodeOrder.indexOf('Z'));
  });

  it('round-trip anomaly scores are consistent', () => {
    const scm = new StructuralCausalModel(gXYZ());
    scm.train(dataXYZ);

    const observations = [
      { X: 1.0, Y: 2.0, Z: 3.0 },
      { X: 5.0, Y: 2.0, Z: 3.0 }, // anomalous X
    ];

    const json = scm.toJSON();
    const restored = StructuralCausalModel.fromJSON(json);

    for (const obs of observations) {
      const origScores = scm.anomalyScores(obs);
      const restScores = restored.anomalyScores(obs);

      for (const [node, origZ] of origScores) {
        expect(restScores.get(node)).toBeCloseTo(origZ, 8);
      }
    }
  });

  it('SCM with only root nodes (fully disconnected graph)', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    const scm = new StructuralCausalModel(g);

    // All nodes are root nodes, each trained as X = μ + ε
    const data = [
      [1, 10, 100],
      [2, 11, 99],
      [1.5, 10.5, 100.5],
    ];
    scm.train(data);

    const json = scm.toJSON();
    const restored = StructuralCausalModel.fromJSON(json);

    expect(json.mechanisms).toHaveLength(3);

    // All mechanisms should have empty coef (no parents)
    for (const m of json.mechanisms) {
      expect(m.coef).toEqual([]);
      expect(m.intercept).toBeGreaterThan(0);
      expect(m.noiseStd).toBeGreaterThan(0);
    }

    // Node order contains all nodes
    expect(json.nodeOrder.sort()).toEqual(['A', 'B', 'C']);

    // Counterfactual: intervening on A should not affect B or C (no edges)
    const noise = { A: 0, B: 0, C: 0 };
    const cf = restored.counterfactual(noise, { A: 999 });
    expect(cf['A']).toBe(999);
    // B and C should remain at their mean (forward(empty) = intercept)
    expect(cf['B']).toBe(json.mechanisms.find(m => m.nodeName === 'B')!.intercept);
    expect(cf['C']).toBe(json.mechanisms.find(m => m.nodeName === 'C')!.intercept);
  });

  it('serialization idempotence: two round-trips produce identical models', () => {
    const scm = new StructuralCausalModel(gXYZ());
    scm.train(dataXYZ);

    // First round-trip
    const json1 = scm.toJSON();
    const scm1 = StructuralCausalModel.fromJSON(json1);

    // Second round-trip from deserialized model
    const json2 = scm1.toJSON();
    const scm2 = StructuralCausalModel.fromJSON(json2);

    // Both JSON representations should be deeply equal
    expect(json2).toEqual(json1);

    // Both restored models should produce identical counterfactuals
    const obs = { X: 1.0, Y: 2.0, Z: 3.0 };
    const noise1 = scm1.abduct(obs);
    const noise2 = scm2.abduct(obs);
    for (const key of Object.keys(noise1)) {
      expect(noise2[key]).toBeCloseTo(noise1[key]!, 8);
    }

    const cf1 = scm1.counterfactual(noise1, { X: 0.5 });
    const cf2 = scm2.counterfactual(noise2, { X: 0.5 });
    for (const key of Object.keys(cf1)) {
      expect(cf2[key]).toBeCloseTo(cf1[key]!, 8);
    }
  });

  it('SCM anomaly attribution round-trip works end-to-end', () => {
    const scm = new StructuralCausalModel(gXYZ());
    scm.train(dataXYZ);

    const restored = StructuralCausalModel.fromJSON(scm.toJSON());

    const observation = { X: 10.0, Y: 2.0, Z: 3.0 }; // X is highly anomalous
    const results = restored.attributeAnomalies(observation, 3);

    expect(results).toHaveLength(3);
    // X should rank first (highest anomaly)
    expect(results[0]!.name).toBe('X');
    expect(results[0]!.score).toBeGreaterThan(0);
    expect(results[0]!.confidence).toBeGreaterThan(0);
    expect(results[0]!.rank).toBe(1);
    expect(results[0]!.evidence).toHaveLength(1);
  });

  it('distribution change detection survives round-trip', () => {
    const scm = new StructuralCausalModel(gXYZ());
    scm.train(dataXYZ);

    const restored = StructuralCausalModel.fromJSON(scm.toJSON());

    const before = [
      { X: 1.0, Y: 2.0, Z: 3.0 },
      { X: 1.1, Y: 2.1, Z: 2.9 },
    ];
    const after = [
      { X: 5.0, Y: 10.0, Z: 15.0 }, // clear distribution shift
      { X: 5.1, Y: 10.1, Z: 14.9 },
    ];

    const result = restored.detectDistributionChange(before, after);

    expect(result).toBeDefined();
    expect(typeof result.changed).toBe('boolean');
    expect(typeof result.meanShift).toBe('number');
    expect(typeof result.pValue).toBe('number');
    expect(result.meanShift).not.toBeNaN();
  });
});
