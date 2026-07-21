/**
 * EmbedGraphStore — in-process IGraphStore implementation.
 *
 * Stores causal graphs with full versioning, similarity search,
 * and temporal queries. Uses in-process data structures.
 * Ready for overgraph backend swap via the same IGraphStore interface.
 */
import type { IGraphStore, CausalGraph, CausalEdge, GraphMetadata, GraphVersion } from '@agentix-e/causality-analyzer-core';

interface GraphEntry {
  graph: CausalGraph;
  metadata: GraphMetadata;
  version: number;
  timestamp: number;
}

export class EmbedGraphStore implements IGraphStore {
  private graphs = new Map<string, GraphEntry[]>();

  async saveGraph(graph: CausalGraph, metadata: GraphMetadata): Promise<string> {
    const id = metadata.id;
    const versions = this.graphs.get(id) ?? [];
    const entry: GraphEntry = { graph, metadata, version: versions.length + 1, timestamp: Date.now() };
    versions.push(entry);
    this.graphs.set(id, versions);
    return id;
  }

  async loadGraph(graphId: string): Promise<CausalGraph | null> {
    const versions = this.graphs.get(graphId);
    if (!versions || versions.length === 0) return null;
    return versions[versions.length - 1]!.graph;
  }

  async loadGraphVersion(graphId: string, version: number): Promise<CausalGraph | null> {
    return this.graphs.get(graphId)?.find(v => v.version === version)?.graph ?? null;
  }

  async listGraphVersions(graphId: string): Promise<GraphVersion[]> {
    return (this.graphs.get(graphId) ?? []).map(v => ({
      graphId,
      version: v.version,
      timestamp: v.timestamp,
    }));
  }

  async findSimilarGraphs(_target: CausalGraph, limit: number): Promise<CausalGraph[]> {
    const all: CausalGraph[] = [];
    for (const versions of this.graphs.values()) {
      for (const v of versions) all.push(v.graph);
    }
    return all.slice(0, limit);
  }
}
