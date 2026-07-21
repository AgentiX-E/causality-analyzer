import type { IGraphStore, CausalGraph, GraphMetadata, GraphVersion } from '@agentix-e/causality-analyzer-core';

/**
 * Remote Graph Store — Bolt protocol (neo4j-driver-lite).
 *
 * Present implementation delegates to an in-memory EmbedGraphStore
 * for IGraphStore contract compliance. When a real Neo4j instance
 * is available, replace with actual Bolt queries.
 *
 * Production use:
 *   const store = new RemoteGraphStore({ uri: 'bolt://localhost:7687', user: 'neo4j', password: 'xxx' });
 */
export class RemoteGraphStore implements IGraphStore {
  private graphs = new Map<string, Array<{ graph: CausalGraph; metadata: GraphMetadata; version: number; timestamp: number }>>();

  async saveGraph(graph: CausalGraph, metadata: GraphMetadata): Promise<string> {
    const id = metadata.id;
    const versions = this.graphs.get(id) ?? [];
    versions.push({ graph, metadata, version: versions.length + 1, timestamp: Date.now() });
    this.graphs.set(id, versions);
    return id;
  }
  async loadGraph(graphId: string): Promise<CausalGraph | null> {
    const versions = this.graphs.get(graphId);
    return versions?.[versions.length - 1]?.graph ?? null;
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
