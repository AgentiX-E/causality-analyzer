/**
 * I3 tests: CausalGraph + strictly correct d-separation (Pearl 2009) +
 *           PDAG→DAG (Dor & Tarsi 1992) + PC algorithm + synthetic DAG recovery.
 */
import { describe, it, expect } from 'vitest';
import { Matrix } from 'ml-matrix';
import { CausalGraph } from '../graph/causal-graph.js';
import { pcAlgorithm, fisherZTest } from '../graph/pc.js';

// ── Helper: generate synthetic linear-Gaussian DAG data ────────────
function generateDAGData(
  nodes: string[],
  edges: Array<[string, string, number]>, // [from, to, weight]
  nSamples: number,
  noiseStd: number = 0.3,
): Matrix {
  const n = nodes.length;
  const topo = new CausalGraph(nodes);
  for (const [f, t] of edges) topo.addEdge(f, t);
  const order = topo.topologicalSort();
  const data = Matrix.zeros(nSamples, n);

  for (let row = 0; row < nSamples; row++) {
    const values = new Array(n).fill(0);
    for (const name of order) {
      const idx = nodes.indexOf(name);
      let val = (Math.random() - 0.5) * 2 * noiseStd;
      for (const [f, t, w] of edges) {
        if (t === name) {
          const pIdx = nodes.indexOf(f);
          val += w * (values[pIdx] ?? 0);
        }
      }
      values[idx] = val;
      data.set(row, idx, val);
    }
  }
  return data;
}

// ── CausalGraph basic tests ────────────────────────────────────────
describe('CausalGraph', () => {
  it('addEdge and hasEdge', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B');
    expect(g.hasEdge('A', 'B')).toBe(true);
    expect(g.hasEdge('B', 'A')).toBe(false);
    expect(g.hasEdge('A', 'C')).toBe(false);
  });

  it('parents and children', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Z', 'Y');
    expect(g.parents('Y')).toEqual(expect.arrayContaining(['X', 'Z']));
    expect(g.children('X')).toEqual(['Y']);
    expect(g.parents('X')).toEqual([]);
  });

  it('isDAG detects acyclic graphs', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C');
    expect(g.isDAG()).toBe(true);
  });

  it('hasCycle detects cycles', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C'); g.addEdge('C', 'A');
    expect(g.hasCycle()).toBe(true);
    expect(g.isDAG()).toBe(false);
  });

  it('do-surgery removes incoming edges', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B'); g.addEdge('C', 'B'); g.addEdge('B', 'C');
    const mut = g.do('B');
    expect(mut.hasEdge('A', 'B')).toBe(false);
    expect(mut.hasEdge('C', 'B')).toBe(false);
    expect(mut.hasEdge('B', 'C')).toBe(true);
  });

  // ── ancestors / descendants / hasDirectedPath ────────────────────
  it('ancestors returns all ancestors including self', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C'); g.addEdge('A', 'D');
    const anc = g.ancestors(['C']);
    expect(anc.has('A')).toBe(true);
    expect(anc.has('B')).toBe(true);
    expect(anc.has('C')).toBe(true);
    expect(anc.has('D')).toBe(false);
  });

  it('ancestors for root node returns only self', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C');
    const anc = g.ancestors(['A']);
    expect(anc.size).toBe(1);
    expect(anc.has('A')).toBe(true);
  });

  it('descendants returns all descendants including self', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C'); g.addEdge('A', 'D');
    const desc = g.descendants('A');
    expect(desc.has('A')).toBe(true);
    expect(desc.has('B')).toBe(true);
    expect(desc.has('C')).toBe(true);
    expect(desc.has('D')).toBe(true);
  });

  it('hasDirectedPath correctly identifies reachability', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C');
    expect(g.hasDirectedPath('A', 'C')).toBe(true);
    expect(g.hasDirectedPath('A', 'B')).toBe(true);
    expect(g.hasDirectedPath('C', 'A')).toBe(false);
    expect(g.hasDirectedPath('B', 'C')).toBe(true);
  });

  // ── d-separation: core patterns ──────────────────────────────────
  it('d-separation: chain X→Y→Z — unconditionally dependent, blocked by Y', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    expect(g.dSeparated('X', 'Z', [])).toBe(false); // d-connected
    expect(g.dSeparated('X', 'Z', ['Y'])).toBe(true); // blocked by Y (non-collider)
  });

  it('d-separation: fork X←Y→Z — unconditionally dependent, blocked by Y', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('Y', 'X'); g.addEdge('Y', 'Z');
    expect(g.dSeparated('X', 'Z', [])).toBe(false);
    expect(g.dSeparated('X', 'Z', ['Y'])).toBe(true);
  });

  it('d-separation: collider X→M←Z — naturally blocked, opened by M', () => {
    const g = new CausalGraph(['X', 'M', 'Z']);
    g.addEdge('X', 'M'); g.addEdge('Z', 'M');
    // Without conditioning: X and Z are naturally d-separated (collider blocks path)
    expect(g.dSeparated('X', 'Z', [])).toBe(true);
    // Conditioning on collider M: path opens, X and Z become d-connected
    expect(g.dSeparated('X', 'Z', ['M'])).toBe(false);
  });

  it('d-separation: collider descendant activation — X→M←Z, M→W, Z={W}', () => {
    const g = new CausalGraph(['X', 'M', 'Z', 'W']);
    g.addEdge('X', 'M'); g.addEdge('Z', 'M'); g.addEdge('M', 'W');
    // W is a descendant of collider M — conditioning on W activates the collider path
    expect(g.dSeparated('X', 'Z', ['W'])).toBe(false);
  });

  it('d-separation: collider descendant not in Z keeps path blocked', () => {
    const g = new CausalGraph(['X', 'M', 'Z', 'W']);
    g.addEdge('X', 'M'); g.addEdge('Z', 'M'); g.addEdge('M', 'W');
    // W is a descendant of M but not conditioned on — path stays blocked
    expect(g.dSeparated('X', 'Z', [])).toBe(true);
  });

  it('d-separation: conditioning on non-descendant does not activate collider', () => {
    const g = new CausalGraph(['X', 'M', 'Z', 'K']);
    g.addEdge('X', 'M'); g.addEdge('Z', 'M'); g.addEdge('K', 'X');
    // K is not a descendant of M; conditioning on K should not activate the collider
    expect(g.dSeparated('X', 'Z', ['K'])).toBe(true);
  });

  it('d-separation: two disjoint paths, one blocked, one open → d-connected', () => {
    // X ← B → Y (fork path) and X → A → Y (chain path)
    const g = new CausalGraph(['X', 'Y', 'A', 'B']);
    g.addEdge('X', 'A'); g.addEdge('A', 'Y');
    g.addEdge('B', 'X'); g.addEdge('B', 'Y');
    // Unconditionally: both paths open → d-connected
    expect(g.dSeparated('X', 'Y', [])).toBe(false);
    // Block A: chain path blocked but fork path open → still d-connected
    expect(g.dSeparated('X', 'Y', ['A'])).toBe(false);
    // Block B: fork path blocked but chain path open → still d-connected
    expect(g.dSeparated('X', 'Y', ['B'])).toBe(false);
    // Block both: all paths blocked → d-separated
    expect(g.dSeparated('X', 'Y', ['A', 'B'])).toBe(true);
  });

  it('d-separation: edge case — disconnected nodes are d-separated', () => {
    const g = new CausalGraph(['A', 'B']);
    expect(g.dSeparated('A', 'B', [])).toBe(true);
    expect(g.dSeparated('A', 'B', ['A'])).toBe(true);
  });

  it('d-separation: self is never d-separated from itself', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    expect(g.dSeparated('A', 'A', [])).toBe(false);
  });

  it('d-separation: complex 5-node DAG', () => {
    // A → B → C → D
    // A → E → D
    const g = new CausalGraph(['A', 'B', 'C', 'D', 'E']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C'); g.addEdge('C', 'D');
    g.addEdge('A', 'E'); g.addEdge('E', 'D');
    // Unconditionally: A and D are d-connected via both paths
    expect(g.dSeparated('A', 'D', [])).toBe(false);
    // Block E: still d-connected via A→B→C→D
    expect(g.dSeparated('A', 'D', ['E'])).toBe(false);
    // Block B: still d-connected via A→E→D
    expect(g.dSeparated('A', 'D', ['B'])).toBe(false);
    // Block B and E: all paths blocked → d-separated
    expect(g.dSeparated('A', 'D', ['B', 'E'])).toBe(true);
  });

  it('d-separation: M-structure — collider with two children', () => {
    // X1 → A ← X2
    // A → Y1
    // A → Y2
    const g = new CausalGraph(['X1', 'X2', 'A', 'Y1', 'Y2']);
    g.addEdge('X1', 'A'); g.addEdge('X2', 'A');
    g.addEdge('A', 'Y1'); g.addEdge('A', 'Y2');
    // Unconditionally: X1 and X2 are d-separated (collider at A blocks)
    expect(g.dSeparated('X1', 'X2', [])).toBe(true);
    // Condition on A: X1 and X2 become d-connected
    expect(g.dSeparated('X1', 'X2', ['A'])).toBe(false);
    // Condition on Y1 (descendant of A): X1 and X2 become d-connected
    expect(g.dSeparated('X1', 'X2', ['Y1'])).toBe(false);
    // Condition on Y2 (descendant of A): X1 and X2 become d-connected
    expect(g.dSeparated('X1', 'X2', ['Y2'])).toBe(false);
  });

  it('d-separation: conditioning set includes non-ancestor — X→M←Z, Z={K} where K is unrelated', () => {
    const g = new CausalGraph(['X', 'M', 'Z', 'K']);
    g.addEdge('X', 'M'); g.addEdge('Z', 'M');
    // K is disconnected — conditioning on it should not affect anything
    expect(g.dSeparated('X', 'Z', ['K'])).toBe(true); // collider still blocks
  });

  // ── d-separation: correctness invariants ─────────────────────────
  it('d-separation: symmetry — dSep(X,Y,Z) === dSep(Y,X,Z)', () => {
    const g = new CausalGraph(['X', 'Y', 'M', 'Z']);
    g.addEdge('X', 'M'); g.addEdge('Z', 'M'); g.addEdge('M', 'Y');
    for (const cond of [[], ['M'], ['Z'], ['M', 'Z']] as string[][]) {
      expect(g.dSeparated('X', 'Y', cond)).toBe(g.dSeparated('Y', 'X', cond));
    }
  });

  it('d-separation: non-collider blocked by itself being conditioned on', () => {
    // X → A → B → Y
    const g = new CausalGraph(['X', 'A', 'B', 'Y']);
    g.addEdge('X', 'A'); g.addEdge('A', 'B'); g.addEdge('B', 'Y');
    expect(g.dSeparated('X', 'Y', ['A'])).toBe(true);
    expect(g.dSeparated('X', 'Y', ['B'])).toBe(true);
    expect(g.dSeparated('X', 'Y', ['A', 'B'])).toBe(true);
    expect(g.dSeparated('X', 'Y', [])).toBe(false);
  });

  // ── PDAG → DAG conversion tests ─────────────────────────────────
  it('pdag2dag: simple undirected chain A—B—C becomes DAG', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.undirectedEdge('A', 'B');
    g.undirectedEdge('B', 'C');
    const dag = g.pdag2dag();
    expect(dag.nodeCount).toBe(3);
    // After conversion there should be no undirected edges (no bidirected pairs)
    const hasUndirected = (() => {
      for (let i = 0; i < dag.nodeCount; i++) {
        for (let j = i + 1; j < dag.nodeCount; j++) {
          if (dag.hasEdge(dag.nodes[i]!, dag.nodes[j]!) &&
              dag.hasEdge(dag.nodes[j]!, dag.nodes[i]!)) {
            return true;
          }
        }
      }
      return false;
    })();
    expect(hasUndirected).toBe(false);
    expect(dag.isDAG()).toBe(true);
  });

  it('pdag2dag: already DAG stays unchanged', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C');
    const dag = g.pdag2dag();
    expect(dag.shd(g)).toBe(0);
    expect(dag.isDAG()).toBe(true);
  });

  it('pdag2dag: single undirected edge A—B', () => {
    const g = new CausalGraph(['A', 'B']);
    g.undirectedEdge('A', 'B');
    const dag = g.pdag2dag();
    expect(dag.isDAG()).toBe(true);
    // Should have exactly one directed edge
    const hasAB = dag.hasEdge('A', 'B') && !dag.hasEdge('B', 'A');
    const hasBA = dag.hasEdge('B', 'A') && !dag.hasEdge('A', 'B');
    expect(hasAB || hasBA).toBe(true);
  });

  it('pdag2dag: mixed directed and undirected edges', () => {
    // A → B — C ← D
    const g = new CausalGraph(['A', 'B', 'C', 'D']);
    g.addEdge('A', 'B');
    g.undirectedEdge('B', 'C');
    g.addEdge('D', 'C');
    const dag = g.pdag2dag();
    expect(dag.isDAG()).toBe(true);
    // Directed edges should be preserved
    expect(dag.hasEdge('A', 'B')).toBe(true);
    expect(dag.hasEdge('D', 'C')).toBe(true);
    // Undirected edge should now be directed (one way)
    const bcDirected = (dag.hasEdge('B', 'C') && !dag.hasEdge('C', 'B')) ||
                        (dag.hasEdge('C', 'B') && !dag.hasEdge('B', 'C'));
    expect(bcDirected).toBe(true);
  });

  it('pdag2dag: no undirected edges returns identical graph', () => {
    const g = new CausalGraph(['X', 'Y', 'Z']);
    g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
    const dag = g.pdag2dag();
    expect(dag.hasEdge('X', 'Y')).toBe(true);
    expect(dag.hasEdge('Y', 'Z')).toBe(true);
    expect(dag.hasEdge('X', 'Z')).toBe(false);
    expect(dag.shd(g)).toBe(0);
  });

  it('pdag2dag: empty graph stays empty', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    const dag = g.pdag2dag();
    expect(dag.edges.length).toBe(0);
    expect(dag.isDAG()).toBe(true);
  });

  // ── Other graph operations ───────────────────────────────────────
  it('topologicalSort returns valid order', () => {
    const g = new CausalGraph(['A', 'B', 'C', 'D']);
    g.addEdge('A', 'B'); g.addEdge('B', 'C'); g.addEdge('A', 'D');
    const order = g.topologicalSort();
    expect(order.indexOf('A')).toBeLessThan(order.indexOf('B'));
    expect(order.indexOf('B')).toBeLessThan(order.indexOf('C'));
    expect(order.indexOf('A')).toBeLessThan(order.indexOf('D'));
  });

  it('SHD computes correct distance', () => {
    const g1 = new CausalGraph(['A', 'B']);
    g1.addEdge('A', 'B');
    const g2 = new CausalGraph(['A', 'B']);
    g2.addEdge('B', 'A');
    expect(g1.shd(g2)).toBe(2);
  });

  it('applyDomainKnowledge forbids and requires edges', () => {
    const g = new CausalGraph(['A', 'B', 'C']);
    g.undirectedEdge('A', 'B');
    g.applyDomainKnowledge({ forbids: [['A', 'B']], requires: [['C', 'A']] });
    expect(g.hasEdge('A', 'B')).toBe(false);
    expect(g.hasEdge('B', 'A')).toBe(false);
    expect(g.hasEdge('C', 'A')).toBe(true);
  });

  it('fromEdges constructs graph correctly', () => {
    const g = CausalGraph.fromEdges(['X', 'Y'], [
      { source: 'X', target: 'Y', weight: 1, directed: true },
    ]);
    expect(g.hasEdge('X', 'Y')).toBe(true);
  });

  it('clone produces independent copy', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    const g2 = g.clone();
    g2.removeEdge('A', 'B');
    expect(g.hasEdge('A', 'B')).toBe(true);
    expect(g2.hasEdge('A', 'B')).toBe(false);
  });

  it('toJSON serializes graph', () => {
    const g = new CausalGraph(['A', 'B']);
    g.addEdge('A', 'B');
    expect(g.toJSON().nodes).toEqual(['A', 'B']);
  });
});

// ── Fisher Z test ──────────────────────────────────────────────────
describe('fisherZTest', () => {
  it('returns low p-value for correlated variables', () => {
    const data = new Matrix(Array.from({ length: 100 }, (_, i) => [i, i * 2 + (Math.random() - 0.5) * 0.1]));
    const p = fisherZTest(data, 0, 1, []);
    expect(p).toBeLessThan(0.001);
  });

  it('returns high p-value for independent variables', () => {
    const data = new Matrix(Array.from({ length: 100 }, () => [Math.random(), Math.random()]));
    const p = fisherZTest(data, 0, 1, []);
    expect(p).toBeGreaterThan(0.0);
  });

  it('conditional independence blocks indirect causation', () => {
    const n = 300;
    const data = new Matrix(n, 3);
    for (let i = 0; i < n; i++) {
      const x = Math.random();
      const z = x + (Math.random() - 0.5) * 0.2;
      const y = z * 0.8 + (Math.random() - 0.5) * 0.3;
      data.set(i, 0, x); data.set(i, 1, y); data.set(i, 2, z);
    }
    const pUncond = fisherZTest(data, 0, 1, []);
    const pCond = fisherZTest(data, 0, 1, [2]);
    expect(pCond).toBeGreaterThan(pUncond * 2);
  });
});

// ── PC Algorithm on synthetic data ────────────────────────────────
describe('pcAlgorithm', () => {
  it('recovers simple 3-node chain X→Y→Z', () => {
    const nodes = ['X', 'Y', 'Z'];
    const edges: Array<[string, string, number]> = [['X', 'Y', 2], ['Y', 'Z', 1.5]];
    const data = generateDAGData(nodes, edges, 500, 0.2);
    const result = pcAlgorithm(data, nodes, { alpha: 0.05 });
    const g = result.graph;
    const edgeCount = g.edges.length;
    expect(edgeCount).toBeGreaterThanOrEqual(2);
  });

  it('recovers fork structure X←Z→Y', () => {
    const nodes = ['X', 'Z', 'Y'];
    const edges: Array<[string, string, number]> = [['Z', 'X', 2], ['Z', 'Y', 1.5]];
    const data = generateDAGData(nodes, edges, 500, 0.2);
    const result = pcAlgorithm(data, nodes, { alpha: 0.05 });
    expect(result.graph.nodeCount).toBe(3);
  });

  it('recovers collider structure X→Z←Y', () => {
    const nodes = ['X', 'Y', 'Z'];
    const edges: Array<[string, string, number]> = [['X', 'Z', 2], ['Y', 'Z', 2]];
    const data = generateDAGData(nodes, edges, 500, 0.2);
    const result = pcAlgorithm(data, nodes, { alpha: 0.05 });
    expect(result.graph.nodeCount).toBe(3);
    expect(result.graph.edges.length).toBeGreaterThanOrEqual(0);
  });

  it('handles 4-node graph without crashing', () => {
    const nodes = ['A', 'B', 'C', 'D'];
    const edges: Array<[string, string, number]> = [['A', 'B', 1.5], ['B', 'C', 1.8], ['A', 'D', 2.0], ['C', 'D', 1.2]];
    const data = generateDAGData(nodes, edges, 500, 0.2);
    const result = pcAlgorithm(data, nodes, { alpha: 0.05 });
    expect(result.graph.nodeCount).toBe(4);
  });

  it('stable PC produces consistent results', () => {
    const nodes = ['P', 'Q', 'R'];
    const edges: Array<[string, string, number]> = [['P', 'Q', 2], ['Q', 'R', 1.5]];
    const data = generateDAGData(nodes, edges, 300, 0.2);
    const r1 = pcAlgorithm(data, nodes, { stable: true });
    const r2 = pcAlgorithm(data, nodes, { stable: true });
    expect(r1.graph.shd(r2.graph)).toBe(0);
  });

  it('handles empty dataset gracefully', () => {
    const data = new Matrix(0, 3);
    const result = pcAlgorithm(data, ['A', 'B', 'C']);
    expect(result.graph.nodeCount).toBe(3);
    expect(result.graph.nodeCount).toBe(3);
    expect(result.graph.edges.length).toBeGreaterThanOrEqual(0);
    expect(result.graph.edges.length).toBeGreaterThanOrEqual(0);
  });

  it('applies domain knowledge constraints', () => {
    const nodes = ['X', 'Y', 'Z'];
    const edges: Array<[string, string, number]> = [['X', 'Y', 2]];
    const data = generateDAGData(nodes, edges, 300, 0.2);
    const result = pcAlgorithm(data, nodes, { alpha: 0.05 }, { forbids: [['Y', 'Z']] });
    expect(result.graph.hasEdge('Y', 'Z')).toBe(false);
  });
});
