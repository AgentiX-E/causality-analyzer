/**
 * TEST-ONLY: In-process IGraphStore for contract validation.
 *
 * NEVER imported by production code. This file lives in __tests__/
 * and provides an IGraphStore reference implementation that works
 * without Docker/Neo4j. Used by the contract test suite to validate
 * RemoteGraphStore and EmbedGraphStore against the same IGraphStore API.
 *
 * Production code uses:
 *   - RemoteGraphStore + neo4j-driver-lite (Bolt → Neo4j)
 *   - EmbedGraphStore + overgraph (embedded persistent)
 *
 * Test code uses this ONLY for contract validation — it's not a
 * "fallback" or "mock" mixed into production.
 */
import type { IGraphStore, CausalGraph, GraphMetadata, GraphVersion } from '@agentix-e/causality-analyzer-core';

export class ContractGraphStore implements IGraphStore {
  private graphs = new Map<string, Array<{ graph: CausalGraph; metadata: GraphMetadata; version: number; timestamp: number }>>();

  async saveGraph(graph: CausalGraph, metadata: GraphMetadata): Promise<string> {
    const id = metadata.id;
    const versions = this.graphs.get(id) ?? [];
    versions.push({ graph, metadata, version: versions.length + 1, timestamp: Date.now() });
    this.graphs.set(id, versions);
    return id;
  }
  async loadGraph(graphId: string): Promise<CausalGraph | null> {
    return this.graphs.get(graphId)?.[this.graphs.get(graphId)!.length - 1]?.graph ?? null;
  }
  async loadGraphVersion(graphId: string, version: number): Promise<CausalGraph | null> {
    return this.graphs.get(graphId)?.find(v => v.version === version)?.graph ?? null;
  }
  async listGraphVersions(graphId: string): Promise<GraphVersion[]> {
    return (this.graphs.get(graphId) ?? []).map(v => ({ graphId, version: v.version, timestamp: v.timestamp }));
  }
  async findSimilarGraphs(_graph: CausalGraph, limit: number): Promise<CausalGraph[]> {
    const all: CausalGraph[] = [];
    for (const versions of this.graphs.values()) all.push(...versions.map(v => v.graph));
    return all.slice(0, limit);
  }
}
