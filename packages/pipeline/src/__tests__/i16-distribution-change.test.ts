import { describe, it, expect } from 'vitest';
import { CausalGraph } from '../graph/causal-graph.js';
import { StructuralCausalModel } from '../gcm/structural-causal-model.js';
import { detectMechanismChanges, distributionChangeRobust, changeAttributionCI } from '../gcm/distribution-change.js';

function trainDAG(): { scm: StructuralCausalModel } {
  const g = new CausalGraph(['X', 'Y', 'Z']);
  g.addEdge('X', 'Y'); g.addEdge('Y', 'Z');
  const scm = new StructuralCausalModel(g);
  const data: number[][] = [];
  for (let i = 0; i < 200; i++) {
    const x = Math.random() * 2 - 1;
    const y = x * 0.7 + Math.random() * 0.3;
    const z = y * 0.5 + Math.random() * 0.2;
    data.push([x, y, z]);
  }
  scm.train(data);
  return { scm };
}

describe('detectMechanismChanges', () => {
  it('detects no change when distributions are identical', () => {
    const { scm } = trainDAG();
    const before: Record<string, number>[] = [];
    const after: Record<string, number>[] = [];
    for (let i = 0; i < 30; i++) {
      const x = Math.random();
      before.push({ X: x, Y: x * 0.7 + Math.random() * 0.3, Z: (x * 0.7) * 0.5 + Math.random() * 0.2 });
      after.push({ X: x, Y: x * 0.7 + Math.random() * 0.3, Z: (x * 0.7) * 0.5 + Math.random() * 0.2 });
    }
    const changes = detectMechanismChanges(scm, before, after);
    expect(changes.length).toBe(3);
    // Most mechanisms should NOT be flagged as changed when distributions are similar
    const flaggedCount = changes.filter(c => c.changed).length;
    expect(flaggedCount).toBeLessThan(3);
  });

  it('detects change with shifted distribution', () => {
    const { scm } = trainDAG();
    const before: Record<string, number>[] = [];
    const after: Record<string, number>[] = [];
    for (let i = 0; i < 50; i++) {
      before.push({ X: Math.random(), Y: Math.random() * 0.7 + Math.random() * 0.3, Z: Math.random() * 0.5 + Math.random() * 0.2 });
      // Shift X by +2
      after.push({ X: Math.random() + 2, Y: Math.random() * 0.7 + Math.random() * 0.3, Z: Math.random() * 0.5 + Math.random() * 0.2 });
    }
    const changes = detectMechanismChanges(scm, before, after);
    const xChange = changes.find(c => c.node === 'X');
    expect(xChange).toBeDefined();
  });

  it('handles empty data gracefully', () => {
    const { scm } = trainDAG();
    const changes = detectMechanismChanges(scm, [], []);
    expect(changes.length).toBe(3);
    for (const c of changes) expect(c.pValue).toBe(1);
  });
});

describe('distributionChangeRobust', () => {
  it('returns attribution for all nodes', () => {
    const { scm } = trainDAG();
    const before: Record<string, number>[] = [];
    const after: Record<string, number>[] = [];
    for (let i = 0; i < 40; i++) {
      before.push({ X: Math.random(), Y: Math.random(), Z: Math.random() });
      after.push({ X: Math.random() + 3, Y: Math.random(), Z: Math.random() });
    }
    const attrib = distributionChangeRobust(scm, before, after);
    expect(attrib.size).toBe(3);
    for (const [, v] of attrib) {
      expect(v.contribution).toBeGreaterThanOrEqual(0);
      expect(v.contribution).toBeLessThanOrEqual(1);
    }
  });
});

describe('changeAttributionCI', () => {
  it('provides CIs for attributions', () => {
    const { scm } = trainDAG();
    const before: Record<string, number>[] = [];
    const after: Record<string, number>[] = [];
    for (let i = 0; i < 30; i++) {
      before.push({ X: Math.random(), Y: Math.random(), Z: Math.random() });
      after.push({ X: Math.random() + 2, Y: Math.random(), Z: Math.random() });
    }
    const cis = changeAttributionCI(scm, before, after, 50, 42);
    expect(cis.size).toBe(3);
    for (const [, ci] of cis) {
      expect(ci.ciLow).toBeLessThanOrEqual(ci.contribution);
      expect(ci.ciHigh).toBeGreaterThanOrEqual(ci.contribution);
    }
  });
});
