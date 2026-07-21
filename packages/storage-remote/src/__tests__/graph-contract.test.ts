/**
 * IGraphStore Contract Test Suite.
 *
 * Validates that IGraphStore implementations obey the same interface
 * contract. Currently tests ContractGraphStore (test-only reference).
 *
 * Production implementations (EmbedGraphStore, RemoteGraphStore) are
 * tested in their respective package test files with real backends
 * or backend-specific mocks.
 *
 * The ContractGraphStore lives in __tests__/ and is NEVER imported
 * by production code.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import type { IGraphStore, CausalGraph, GraphMetadata, GraphVersion } from '@agentix-e/causality-analyzer-core';
import { ContractGraphStore } from './contract-graph-store.js';

function makeGraph(): CausalGraph {
  return { nodes: ['A', 'B', 'C'], edges: [
    { source: 'A', target: 'B', weight: 1, directed: true },
    { source: 'B', target: 'C', weight: 1, directed: true },
  ]};
}
function makeMeta(id: string): GraphMetadata {
  return { id, method: 'pc', computedAt: Date.now(), parameters: {}, confidence: 0.9 };
}

function testContract(name: string, factory: () => IGraphStore) {
  describe(`${name} IGraphStore contract`, () => {
    let store: IGraphStore;
    beforeEach(() => { store = factory(); });

    it('saveGraph returns string ID', async () => {
      expect(typeof await store.saveGraph(makeGraph(), makeMeta('g1'))).toBe('string');
    });

    it('loadGraph retrieves saved graph', async () => {
      const id = await store.saveGraph(makeGraph(), makeMeta('g2'));
      const loaded = await store.loadGraph(id);
      expect(loaded?.nodes).toEqual(['A', 'B', 'C']);
      expect(loaded?.edges).toHaveLength(2);
    });

    it('loadGraph returns null for unknown', async () => {
      expect(await store.loadGraph('nonexistent')).toBeNull();
    });

    it('versioned storage preserves all versions', async () => {
      const id = await store.saveGraph(makeGraph(), makeMeta('g3'));
      await store.saveGraph({ nodes: ['A','B','C','D'], edges:[] }, { ...makeMeta('g3'), id });
      expect((await store.loadGraphVersion(id, 1))?.nodes.length).toBe(3);
      expect((await store.loadGraphVersion(id, 2))?.nodes.length).toBe(4);
      expect(await store.loadGraphVersion(id, 999)).toBeNull();
    });

    it('listGraphVersions returns monotonic versions', async () => {
      const id = await store.saveGraph(makeGraph(), makeMeta('g4'));
      await store.saveGraph(makeGraph(), { ...makeMeta('g4'), id });
      const versions = await store.listGraphVersions(id);
      expect(versions.length).toBeGreaterThanOrEqual(2);
      for (let i = 1; i < versions.length; i++) {
        expect(versions[i]!.version).toBeGreaterThan(versions[i-1]!.version);
      }
    });

    it('listGraphVersions returns empty for unknown', async () => {
      expect(await store.listGraphVersions('unknown')).toEqual([]);
    });

    it('findSimilarGraphs returns array', async () => {
      await store.saveGraph(makeGraph(), makeMeta('g5'));
      expect(Array.isArray(await store.findSimilarGraphs(makeGraph(), 5))).toBe(true);
    });
  });
}

// Run contract suite against the test-only reference implementation
testContract('ContractGraphStore', () => new ContractGraphStore());
