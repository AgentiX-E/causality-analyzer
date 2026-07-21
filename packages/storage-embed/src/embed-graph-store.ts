/**
 * EmbedGraphStore — file-persisted IGraphStore.
 *
 * Default: ./causality-analyzer-graph.json (JSON file persistence).
 * :memory: mode available via { path: ':memory:' }.
 * Overgraph backend ready: swap via the same IGraphStore interface.
 */
import { readFileSync, writeFileSync, existsSync, unlinkSync } from 'fs';
import { join } from 'path';
import type { IGraphStore, CausalGraph, GraphMetadata, GraphVersion } from '@agentix-e/causality-analyzer-core';

interface GraphEntry {
  graph: CausalGraph;
  metadata: GraphMetadata;
  version: number;
  timestamp: number;
}

export interface EmbedGraphOptions {
  /** File path. Default: ./causality-analyzer-graph.json. ':memory:' for CI. */
  path?: string;
}

export class EmbedGraphStore implements IGraphStore {
  private path: string;
  private graphs: Map<string, GraphEntry[]>;

  constructor(opts: EmbedGraphOptions = {}) {
    this.path = opts.path === ':memory:' ? '' : (opts.path || './causality-analyzer-graph.json');
    if (this.path && existsSync(this.path)) {
      try {
        const data = JSON.parse(readFileSync(this.path, 'utf-8'));
        this.graphs = new Map(Object.entries(data).map(([k, v]) => [k, v as GraphEntry[]]));
      } catch { this.graphs = new Map(); }
    } else {
      this.graphs = new Map();
    }
    if (!this.path) this.path = '';
  }

  private persist() {
    if (this.path) {
      const obj: Record<string, GraphEntry[]> = {};
      for (const [k, v] of this.graphs) obj[k] = v;
      writeFileSync(this.path, JSON.stringify(obj, null, 2));
    }
  }

  async saveGraph(graph: CausalGraph, metadata: GraphMetadata): Promise<string> {
    const id = metadata.id;
    const versions = this.graphs.get(id) ?? [];
    versions.push({ graph, metadata, version: versions.length + 1, timestamp: Date.now() });
    this.graphs.set(id, versions);
    this.persist();
    return id;
  }

  async loadGraph(graphId: string): Promise<CausalGraph | null> {
    const v = this.graphs.get(graphId);
    return v?.[v.length - 1]?.graph ?? null;
  }

  async loadGraphVersion(graphId: string, version: number): Promise<CausalGraph | null> {
    return this.graphs.get(graphId)?.find(e => e.version === version)?.graph ?? null;
  }

  async listGraphVersions(graphId: string): Promise<GraphVersion[]> {
    return (this.graphs.get(graphId) ?? []).map(e => ({ graphId, version: e.version, timestamp: e.timestamp }));
  }

  async findSimilarGraphs(_target: CausalGraph, limit: number): Promise<CausalGraph[]> {
    const all: CausalGraph[] = [];
    for (const versions of this.graphs.values()) for (const e of versions) all.push(e.graph);
    return all.slice(0, limit);
  }

  /** Delete the persisted file (for test cleanup) */
  deleteFile(): void { if (this.path) try { unlinkSync(this.path); } catch {} }
}
