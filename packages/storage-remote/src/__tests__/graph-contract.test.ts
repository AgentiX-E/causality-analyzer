/**
 * IGraphStore Contract Test Suite.
 *
 * Validates that EmbedGraphStore and RemoteGraphStore (stub)
 * implement identical IGraphStore semantics. The contract test
 * is the authoritative verification — any IGraphStore backend
 * must pass all tests in this suite.
 *
 * For production Cypher/Bolt testing, spin up Neo4j via Docker:
 *   docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
 * Then test RemoteGraphStore with neo4j-driver-lite.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import type { IGraphStore, CausalGraph, GraphMetadata, GraphVersion } from '@agentix-e/causality-analyzer-core';

/** Contract-compliant implementations to test */
import { EmbedGraphStore } from '../../../storage-embed/src/embed-graph-store.js';
import { RemoteGraphStore } from '../../../storage-remote/src/remote-graph-store.js';

function makeGraph(): CausalGraph {
  return { nodes: ['A', 'B', 'C'], edges: [
    { source: 'A', target: 'B', weight: 1, directed: true },
    { source: 'B', target: 'C', weight: 1, directed: true },
  ]};
}
function makeMeta(id: string): GraphMetadata {
  return { id, method: 'pc', computedAt: Date.now(), parameters: {}, confidence: 0.9 };
}

/** Test runner: runs the same test suite against any IGraphStore implementation */
function testIGraphStore(name: string, factory: () => IGraphStore) {
  describe(`${name} IGraphStore contract`, () => {
    let store: IGraphStore;
    beforeEach(() => { store = factory(); });

    it('saveGraph returns graph ID', async () => {
      const id = await store.saveGraph(makeGraph(), makeMeta('g1'));
      expect(typeof id).toBe('string');
    });

    it('loadGraph retrieves saved graph', async () => {
      const id = await store.saveGraph(makeGraph(), makeMeta('g2'));
      const loaded = await store.loadGraph(id);
      expect(loaded).not.toBeNull();
      expect(loaded!.nodes).toEqual(['A', 'B', 'C']);
      expect(loaded!.edges).toHaveLength(2);
    });

    it('loadGraph returns null for unknown graph', async () => {
      expect(await store.loadGraph('nonexistent')).toBeNull();
    });

    it('loadGraphVersion retrieves specific version', async () => {
      const id = await store.saveGraph(makeGraph(), makeMeta('g3'));
      // Save second version
      const g2: CausalGraph = { nodes: ['A', 'B', 'C', 'D'], edges: [] };
      await store.saveGraph(g2, { ...makeMeta('g3'), id });
      const v1 = await store.loadGraphVersion(id, 1);
      const v2 = await store.loadGraphVersion(id, 2);
      expect(v1?.nodes.length).toBe(3);
      expect(v2?.nodes.length).toBe(4);
    });

    it('loadGraphVersion returns null for missing version', async () => {
      const id = await store.saveGraph(makeGraph(), makeMeta('g4'));
      expect(await store.loadGraphVersion(id, 999)).toBeNull();
    });

    it('listGraphVersions returns ordered versions', async () => {
      const id = await store.saveGraph(makeGraph(), makeMeta('g5'));
      await store.saveGraph(makeGraph(), { ...makeMeta('g5'), id });
      await store.saveGraph(makeGraph(), { ...makeMeta('g5'), id });
      const versions = await store.listGraphVersions(id);
      expect(versions.length).toBeGreaterThanOrEqual(2);
      // Versions should be monotonically increasing
      for (let i = 1; i < versions.length; i++) {
        expect(versions[i]!.version).toBeGreaterThan(versions[i-1]!.version);
      }
    });

    it('listGraphVersions returns empty for unknown', async () => {
      expect(await store.listGraphVersions('unknown')).toEqual([]);
    });

    it('findSimilarGraphs returns list of graphs', async () => {
      await store.saveGraph(makeGraph(), makeMeta('g6'));
      await store.saveGraph(makeGraph(), { ...makeMeta('g6'), id: 'g7' });
      const similar = await store.findSimilarGraphs(makeGraph(), 5);
      expect(Array.isArray(similar)).toBe(true);
    });
  });
}

// ════════════════════════════════════════════════════════════════════
// Run contract tests against BOTH implementations
// ════════════════════════════════════════════════════════════════════
testIGraphStore('EmbedGraphStore', () => new EmbedGraphStore());
testIGraphStore('RemoteGraphStore', () => new RemoteGraphStore() as unknown as IGraphStore);
