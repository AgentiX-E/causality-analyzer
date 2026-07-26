import { createRequire } from 'module';
const _require = createRequire(import.meta.url);
import type { IGraphStore, CausalGraph, GraphMetadata, GraphVersion } from '@agentix-e/causality-analyzer-core';
import { graphSimilarity } from '@agentix-e/causality-analyzer-core';
import { existsSync, mkdirSync } from 'fs';

// ── OverGraph type interface ────────────────────────────────────────

interface OverGraphNode {
  id: number;
  key: string;
}

interface OverGraphEdge {
  src: number;
  tgt: number;
  label: string;
}

interface OverGraphLabel {
  label: string;
}

interface OverGraphInstance {
  upsertNode(label: string, key: string, props: Record<string, unknown>): number;
  upsertEdge(src: number, tgt: number, label: string): void;
  getNodesByLabels(label: string): OverGraphNode[];
  listEdgeLabels(): OverGraphLabel[];
  getEdgesByLabel(label: string): OverGraphEdge[];
  close(): void;
}

// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
const OverGraph: { open(path: string): OverGraphInstance } = _require('overgraph').OverGraph;

export interface EmbedGraphOptions {
  dbPath?: string;
}

export class EmbedGraphStore implements IGraphStore {
  private g: OverGraphInstance;
  private vers: Map<string, number>;

  constructor(opts: EmbedGraphOptions = {}) {
    const dir = opts.dbPath || './causality-analyzer-graph';
    if (!opts.dbPath && !existsSync(dir)) mkdirSync(dir, { recursive: true });
    this.g = OverGraph.open(dir);
    this.vers = new Map();
  }

  async saveGraph(graph: CausalGraph, m: GraphMetadata, signal?: AbortSignal): Promise<string> {
    signal?.throwIfAborted();
    const id = m.id;
    const v = (this.vers.get(id) ?? 0) + 1;
    this.vers.set(id, v);
    const lid = `g_${id}_v${v}`;
    const nodeIds: Record<string, number> = {};
    for (const n of graph.nodes) {
      signal?.throwIfAborted();
      nodeIds[n] = this.g.upsertNode(lid, `${id}_${n}`, {});
    }
    for (const e of graph.edges) {
      signal?.throwIfAborted();
      const f = nodeIds[e.source];
      const t = nodeIds[e.target];
      if (f !== undefined && t !== undefined) {
        this.g.upsertEdge(f, t, e.directed ? 'DEPENDS_ON_dir' : 'DEPENDS_ON_undir');
      }
    }
    return id;
  }

  async loadGraph(id: string, signal?: AbortSignal): Promise<CausalGraph | null> {
    signal?.throwIfAborted();
    const v = this.vers.get(id) ?? 0;
    if (v === 0) return null;
    return this._loadGraphByLabel(`g_${id}_v${v}`);
  }

  private _loadGraphByLabel(label: string): CausalGraph | null {
    const result = this.g.getNodesByLabels(label);
    const graphNodes: string[] = [];
    const nodeMap = new Map<string, number>();
    for (const n of result) {
      const k: string = n.key;
      if (!k) continue;
      const sep = k.indexOf('_');
      if (sep === -1) continue;
      const name = k.slice(sep + 1);
      graphNodes.push(name);
      nodeMap.set(name, n.id);
    }
    if (graphNodes.length === 0) return null;

    const edges: Array<{ source: string; target: string; weight: number; directed: boolean }> = [];
    const edgeLabels = this.g.listEdgeLabels();
    for (const el of edgeLabels) {
      if (!el.label.includes('DEPENDS_ON')) continue;
      const es = this.g.getEdgesByLabel(el.label);
      for (const e of es) {
        if (!nodeMap.has(String(e.src)) && !nodeMap.has(String(e.tgt))) continue;
        const sn = graphNodes.find(n => nodeMap.get(n) === e.src);
        const tn = graphNodes.find(n => nodeMap.get(n) === e.tgt);
        if (sn && tn) edges.push({
          source: sn,
          target: tn,
          weight: 1,
          directed: el.label.includes('dir'),
        });
      }
    }
    return { nodes: graphNodes, edges };
  }

  async loadGraphVersion(id: string, ver: number, signal?: AbortSignal): Promise<CausalGraph | null> {
    signal?.throwIfAborted();
    return this._loadGraphByLabel(`g_${id}_v${ver}`);
  }

  async listGraphVersions(id: string, signal?: AbortSignal): Promise<GraphVersion[]> {
    signal?.throwIfAborted();
    const count = this.vers.get(id) ?? 0;
    if (count === 0) return [];
    return Array.from({ length: count }, (_, i) => ({
      graphId: id,
      version: i + 1,
      timestamp: Date.now(),
    }));
  }

  async findSimilarGraphs(target: CausalGraph, lim: number, signal?: AbortSignal): Promise<CausalGraph[]> {
    signal?.throwIfAborted();
    const scored: Array<{ graph: CausalGraph; score: number }> = [];
    for (const [id] of this.vers) {
      const g = await this.loadGraph(id);
      if (g) scored.push({ graph: g, score: graphSimilarity(target, g as unknown as Parameters<typeof graphSimilarity>[1]) });
      if (scored.length > lim * 3) break; // cap iterations
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, lim).map(s => s.graph);
  }

  close(): void { this.g.close(); }
  healthCheck(): boolean { return this.g != null; }
}
