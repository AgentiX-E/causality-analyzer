/**
 * Embedded Graph Store — in-memory IGraphStore for CI/testing.
 *
 * Production use should switch to overgraph-backed implementation.
 */
import type { IGraphStore, CausalGraph, CausalEdge, GraphMetadata, GraphVersion } from '@agentix-e/causality-analyzer-core';

interface GraphEntry { graph: CausalGraph; metadata: GraphMetadata; version: number; timestamp: number; }

export class EmbedGraphStore implements IGraphStore {
  private graphs = new Map<string, GraphEntry[]>();

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
    return (this.graphs.get(graphId) ?? []).map(v => ({
      graphId, version: v.version, timestamp: v.timestamp, changeDescription: undefined,
    }));
  }

  async findSimilarGraphs(graph: CausalGraph, limit: number): Promise<CausalGraph[]> {
    const allGraphs: CausalGraph[] = [];
    for (const versions of this.graphs.values()) {
      allGraphs.push(...versions.map(v => v.graph));
    }
    return allGraphs.slice(0, limit);
  }
}
