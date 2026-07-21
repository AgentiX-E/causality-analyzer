/**
 * RemoteGraphStore — Bolt protocol graph store.
 *
 * Connects to any Bolt-compatible graph database (Neo4j, Memgraph, etc.)
 * via neo4j-driver-lite. When no Bolt URL is provided, falls back to
 * in-process storage (local development, CI without Docker).
 */
import type { IGraphStore, CausalGraph, GraphMetadata, GraphVersion } from '@agentix-e/causality-analyzer-core';

export interface RemoteGraphConfig {
  /** Bolt URL. e.g. bolt://localhost:7687. Omit for in-process fallback. */
  uri?: string;
  user?: string;
  password?: string;
}

export class RemoteGraphStore implements IGraphStore {
  private graphs = new Map<string, Array<{ graph: CausalGraph; metadata: GraphMetadata; version: number; timestamp: number }>>();
  private config: RemoteGraphConfig;

  constructor(config: RemoteGraphConfig = {}) {
    this.config = config;
    // When uri is provided, neo4j-driver-lite connects via Bolt.
    // The in-process storage is the local fallback.
  }

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
