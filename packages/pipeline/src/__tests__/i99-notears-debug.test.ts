import { describe, it, expect } from 'vitest';
import { notearsAlgorithm } from '../graph/notears.js';
import { asiaGraph, generateLinearData } from '../benchmark.js';

describe('NOTEARS Debug', () => {
  it('DEFAULTS param test', () => {
    const truth = asiaGraph();
    const { data, nodeNames } = generateLinearData(truth, 5000, 42);
    // Use explicit default-equivalent params
    const result = notearsAlgorithm(data, nodeNames, { lambda1: 0.001, wThreshold: 0.1 });
    console.log('DEFAULT params — edges:', result.graph.edges.length);
    const maxAbs = Math.max(...Array.from(result.W).map(Math.abs));
    console.log('Max |weight|:', maxAbs);
    expect(maxAbs).toBeGreaterThan(0.01);
  });

  it('explicit params test', () => {
    const truth = asiaGraph();
    const { data, nodeNames } = generateLinearData(truth, 5000, 42);
    const result = notearsAlgorithm(data, nodeNames, { lambda1: 0.001, wThreshold: 0.01 });
    console.log('EXPLICIT params — edges:', result.graph.edges.length);
    expect(result.graph.edges.length).toBeGreaterThan(0);
  });
});
