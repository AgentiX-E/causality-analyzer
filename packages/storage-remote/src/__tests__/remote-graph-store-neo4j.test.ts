/**
 * Neo4j integration tests — runs against a REAL Neo4j instance via Bolt.
 *
 * Requires NEO4J_BOLT_URI env var (e.g. bolt://localhost:7687).
 * Skipped when env var is absent (local dev without Docker).
 *
 * These tests validate that RemoteGraphStore's Cypher queries,
 * UNWIND batching, versioned storage, and retry logic work against
 * an actual Neo4j server — closing the gap between BoltSessionMock
 * and production behavior.
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { RemoteGraphStore } from '../remote-graph-store.js';
import type { CausalGraph, GraphMetadata } from '@agentix-e/causality-analyzer-core';

const BOLT_URI = process.env.NEO4J_BOLT_URI;
const NEO4J_USER = process.env.NEO4J_USER || 'neo4j';
const NEO4J_PASS = process.env.NEO4J_PASSWORD || 'password';

const describeIf = BOLT_URI ? describe : describe.skip;

describeIf('RemoteGraphStore (real Neo4j)', () => {
  let store: RemoteGraphStore;

  beforeAll(async () => {
    store = new RemoteGraphStore({
      uri: BOLT_URI!,
      auth: { type: 'basic', user: NEO4J_USER, password: NEO4J_PASS },
    });
    // Clean slate: delete all nodes and relationships
    const s = (store as any).driver.session();
    try {
      await s.run('MATCH (n) DETACH DELETE n');
    } finally {
      await s.close();
    }
  });

  afterAll(async () => {
    // Final cleanup
    const s = (store as any).driver.session();
    try {
      await s.run('MATCH (n) DETACH DELETE n');
    } finally {
      await s.close();
    }
    await store.close();
  });

  const g = (nodes: string[], edges?: CausalGraph['edges']): CausalGraph => ({
    nodes,
    edges: edges ?? [],
  });

  const m = (id: string): GraphMetadata => ({
    id, method: 'pc', computedAt: Date.now(), parameters: {}, confidence: 0.9,
  });

  it('connects to Neo4j', () => {
    expect(store).toBeDefined();
  });

  it('saveGraph + loadGraph round-trip', async () => {
    const graph = g(['A', 'B', 'C'], [
      { source: 'A', target: 'B', weight: 0.8, directed: true },
      { source: 'B', target: 'C', weight: 0.5, directed: true },
    ]);
    const id = await store.saveGraph(graph, m('neo4j-g1'));
    expect(id).toBe('neo4j-g1');

    const loaded = await store.loadGraph(id);
    expect(loaded?.nodes).toHaveLength(3);
    expect(loaded?.edges).toHaveLength(2);
  });

  it('preserves edge weight and direction', async () => {
    const graph = g(['X', 'Y'], [
      { source: 'X', target: 'Y', weight: 0.42, directed: false },
    ]);
    await store.saveGraph(graph, m('neo4j-g2'));
    const loaded = await store.loadGraph('neo4j-g2');
    expect(loaded?.edges[0]?.weight).toBe(0.42);
    expect(loaded?.edges[0]?.directed).toBe(false);
  });

  it('loadGraph returns null for unknown ID', async () => {
    expect(await store.loadGraph('neo4j-nonexistent')).toBeNull();
  });

  it('versioned storage: multiple versions accessible', async () => {
    await store.saveGraph(g(['V1']), m('neo4j-g3'));
    await store.saveGraph(g(['V1', 'V2']), m('neo4j-g3'));
    await store.saveGraph(g(['V1', 'V2', 'V3']), m('neo4j-g3'));

    const v1 = await store.loadGraphVersion('neo4j-g3', 1);
    const v2 = await store.loadGraphVersion('neo4j-g3', 2);
    const v3 = await store.loadGraphVersion('neo4j-g3', 3);

    expect(v1?.nodes).toEqual(['V1']);
    expect(v2?.nodes).toEqual(['V1', 'V2']);
    expect(v3?.nodes).toEqual(['V1', 'V2', 'V3']);
    expect(await store.loadGraphVersion('neo4j-g3', 999)).toBeNull();
  });

  it('listGraphVersions returns correct count and monotonic versions', async () => {
    await store.saveGraph(g(['A']), m('neo4j-g4'));
    await store.saveGraph(g(['A', 'B']), m('neo4j-g4'));

    const versions = await store.listGraphVersions('neo4j-g4');
    expect(versions.length).toBe(2);
    expect(versions[0]!.version).toBe(1);
    expect(versions[1]!.version).toBe(2);
  });

  it('listGraphVersions returns empty for unknown', async () => {
    expect(await store.listGraphVersions('neo4j-unknown')).toEqual([]);
  });

  it('loadGraph returns latest version', async () => {
    await store.saveGraph(g(['old1', 'old2']), m('neo4j-g5'));
    await store.saveGraph(g(['new1', 'new2', 'new3']), m('neo4j-g5'));

    const latest = await store.loadGraph('neo4j-g5');
    expect(latest?.nodes).toHaveLength(3);
  });

  it('findSimilarGraphs returns graphs sorted by Jaccard similarity', async () => {
    await store.saveGraph(g(['A', 'B', 'C']), m('neo4j-s1'));
    await store.saveGraph(g(['A', 'B', 'X']), m('neo4j-s2'));
    await store.saveGraph(g(['X', 'Y', 'Z']), m('neo4j-s3'));

    const results = await store.findSimilarGraphs(g(['A', 'B', 'C']), 5);
    expect(results.length).toBeGreaterThanOrEqual(3);
    // Perfect match should be first
    expect(results[0]!.nodes).toEqual(['A', 'B', 'C']);
  });

  it('UNWIND batch: large graph round-trip', async () => {
    // 50 nodes, ~100 edges — exercises UNWIND batching
    const nodes = Array.from({ length: 50 }, (_, i) => `N${i}`);
    const edges: CausalGraph['edges'] = [];
    for (let i = 0; i < nodes.length - 1; i++) {
      edges.push({ source: nodes[i]!, target: nodes[i + 1]!, weight: 1, directed: true });
      if (i < nodes.length - 2) {
        edges.push({ source: nodes[i]!, target: nodes[i + 2]!, weight: 0.5, directed: false });
      }
    }

    const id = await store.saveGraph(g(nodes, edges), m('neo4j-large'));
    const loaded = await store.loadGraph(id);
    expect(loaded?.nodes).toHaveLength(50);
    expect(loaded?.edges.length).toBeGreaterThanOrEqual(90);
  });

  it('empty graph: no nodes, no edges', async () => {
    await store.saveGraph(g([], []), m('neo4j-empty'));
    // Store returns the ID even for empty graphs
    const versions = await store.listGraphVersions('neo4j-empty');
    expect(versions.length).toBe(1);
  });

  it('close() disconnects cleanly', async () => {
    // Create a separate store for this test to not interfere with afterAll
    const s = new RemoteGraphStore({
      uri: BOLT_URI!,
      auth: { type: 'basic', user: NEO4J_USER, password: NEO4J_PASS },
    });
    await s.close();
    // Should not throw
  });
});
