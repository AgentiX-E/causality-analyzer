/**
 * Browser Graph Store — IGraphStore backed by SQLite tables via SqlitePort.
 *
 * Stores causal graphs in SQLite tables (graph_nodes, graph_edges)
 * within the same WASM SQLite instance as the relational store.
 * This enables browser-side graph persistence (OPFS) without OverGraph.
 *
 * Features:
 * - Full IGraphStore contract: save/load/version/findSimilar
 * - SQL-based graph queries for loadGraph (single query per graph)
 * - Jaccard similarity for findSimilarGraphs
 * - Compatible with WasmRelationalStore (same SqlitePort instance)
 *
 * @packageDocumentation
 */
import type { IGraphStore, CausalGraph, GraphMetadata, GraphVersion } from '@agentix-e/causality-analyzer-core';
import type { SqlitePort, SqliteRow } from './sqlite-port.js';

const DDL = [
  `CREATE TABLE IF NOT EXISTS graph_nodes (
    graph_id TEXT NOT NULL, name TEXT NOT NULL, version INTEGER NOT NULL,
    PRIMARY KEY (graph_id, name, version)
  )`,
  `CREATE TABLE IF NOT EXISTS graph_edges (
    graph_id TEXT NOT NULL,
    source TEXT NOT NULL, target TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    directed INTEGER NOT NULL DEFAULT 1,
    version INTEGER NOT NULL,
    PRIMARY KEY (graph_id, source, target, version)
  )`,
];

interface NodeRow extends SqliteRow { name: string; }
interface EdgeRow extends SqliteRow { source: string; target: string; weight: number; directed: number; }
interface VersionRow extends SqliteRow { version: number; graphId: string; }

function jaccardSimilarity(a: ReadonlyArray<string>, b: ReadonlyArray<string>): number {
  const sa = new Set(a), sb = new Set(b);
  const intersection = [...sa].filter(x => sb.has(x)).length;
  const union = new Set([...sa, ...sb]).size;
  return union === 0 ? 0 : intersection / union;
}

export class WasmGraphStore implements IGraphStore {
  private port: SqlitePort;
  private _versionCounter = new Map<string, number>();

  constructor(port: SqlitePort) {
    this.port = port;
    for (const d of DDL) this.port.exec(d);
  }

  async saveGraph(graph: CausalGraph, metadata: GraphMetadata, signal?: AbortSignal): Promise<string> {
    signal?.throwIfAborted();
    const id = metadata.id;
    const ver = (this._versionCounter.get(id) ?? 0) + 1;
    this._versionCounter.set(id, ver);

    // Batch insert nodes
    for (const name of graph.nodes) {
      await this.port.run(
        "INSERT OR REPLACE INTO graph_nodes VALUES (?, ?, ?)",
        [id, name, ver],
      );
    }

    // Batch insert edges
    for (const e of graph.edges) {
      await this.port.run(
        "INSERT OR REPLACE INTO graph_edges VALUES (?, ?, ?, ?, ?, ?)",
        [id, e.source, e.target, e.weight, e.directed ? 1 : 0, ver],
      );
    }

    return id;
  }

  async loadGraph(graphId: string, signal?: AbortSignal): Promise<CausalGraph | null> {
    signal?.throwIfAborted();
    const ver = this._versionCounter.get(graphId);
    if (ver === undefined || ver === 0) return null;
    return this._loadVersion(graphId, ver);
  }

  async loadGraphVersion(graphId: string, ver: number, signal?: AbortSignal): Promise<CausalGraph | null> {
    signal?.throwIfAborted();
    return this._loadVersion(graphId, ver);
  }

  private async _loadVersion(graphId: string, ver: number): Promise<CausalGraph | null> {
    const nodes = await this.port.all(
      "SELECT name FROM graph_nodes WHERE graph_id = ? AND version = ?",
      [graphId, ver],
    ) as unknown as NodeRow[];

    if (!nodes.length) return null;

    const edges = await this.port.all(
      "SELECT source, target, weight, directed FROM graph_edges WHERE graph_id = ? AND version = ?",
      [graphId, ver],
    ) as unknown as EdgeRow[];

    return {
      nodes: nodes.map(n => n.name),
      edges: edges.map(e => ({
        source: e.source,
        target: e.target,
        weight: e.weight,
        directed: e.directed === 1,
      })),
    };
  }

  async listGraphVersions(graphId: string, signal?: AbortSignal): Promise<GraphVersion[]> {
    signal?.throwIfAborted();
    const rows = await this.port.all(
      "SELECT DISTINCT version, graph_id as graphId FROM graph_nodes WHERE graph_id = ? ORDER BY version",
      [graphId],
    ) as unknown as VersionRow[];

    return rows.map(r => ({
      graphId: r.graphId,
      version: r.version,
      timestamp: Date.now(),
    }));
  }

  async findSimilarGraphs(target: CausalGraph, limit: number, signal?: AbortSignal): Promise<CausalGraph[]> {
    signal?.throwIfAborted();
    // Get all distinct graph IDs
    const ids = await this.port.all(
      "SELECT DISTINCT graph_id as graphId FROM graph_nodes",
    ) as unknown as VersionRow[];

    const scored: Array<{ graph: CausalGraph; score: number }> = [];
    for (const row of ids) {
      const g = await this.loadGraph(row.graphId);
      if (g) {
        scored.push({ graph: g, score: jaccardSimilarity(target.nodes, g.nodes) });
      }
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit).map(s => s.graph);
  }

  close(): void { this.port.close(); }
}
