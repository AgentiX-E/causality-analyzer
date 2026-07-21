import { createRequire } from 'module';
const _require = createRequire(import.meta.url);
const { OverGraph } = _require('overgraph');
import type { IGraphStore, CausalGraph, GraphMetadata, GraphVersion } from '@agentix-e/causality-analyzer-core';
import { readFileSync, writeFileSync, existsSync, unlinkSync, mkdirSync } from 'fs';

export interface EmbedGraphOptions {
  dbPath?: string;
}

export class EmbedGraphStore implements IGraphStore {
  private g: any;
  private vers: Map<string, number>;

  constructor(opts: EmbedGraphOptions = {}) {
    const dir = opts.dbPath || './causality-analyzer-graph';
    if (!opts.dbPath && !existsSync(dir)) mkdirSync(dir, { recursive: true });
    this.g = OverGraph.open(dir);
    this.vers = new Map(); // graphId → version count
  }

  async saveGraph(graph: CausalGraph, m: GraphMetadata): Promise<string> {
    const id = m.id;
    const v = (this.vers.get(id) ?? 0) + 1;
    this.vers.set(id, v);
    const lid = `g${v}`; // graph version label
    const nodeIds: Record<string, number> = {};
    for (const n of graph.nodes) {
      nodeIds[n] = this.g.upsertNode(lid, `${id}_${n}`, {});
    }
    for (const e of graph.edges) {
      const f = nodeIds[e.source], t = nodeIds[e.target];
      if (f != null && t != null) this.g.upsertEdge(f, t, e.directed ? 'DEPENDS_ON_dir' : 'DEPENDS_ON_undir');
    }
    return id;
  }


  async loadGraph(id: string): Promise<CausalGraph | null> {
    const nodes: string[] = [];
    const nodeMap = new Map<string, number>();
    const labels = this.g.listNodeLabels();
    for (const l of labels) {
      if (!l.label.startsWith('g')) continue;
      const result = this.g.getNodesByLabels(l.label);
      for (const n of result) {
        const k = n.key;
        if (k && k.startsWith(id + '_')) {
          const name = k.replace(id + '_', '');
          nodes.push(name);
          nodeMap.set(name, n.id);
        }
      }
    }
    if (nodes.length === 0) return null;
    const edges: CausalGraph['edges'] = [];
    const edgeLabels = this.g.listEdgeLabels();
    for (const el of edgeLabels) {
      if (!el.label.includes('DEPENDS_ON')) continue;
      const es = this.g.getEdgesByLabel(el.label);
      for (const e of es) {
        if (!nodeMap.has(e.src) && !nodeMap.has(e.tgt)) continue;
        const sn = nodes.find(n => nodeMap.get(n) === e.src);
        const tn = nodes.find(n => nodeMap.get(n) === e.tgt);
        if (sn && tn) edges.push({ source: sn, target: tn, weight: 1, directed: el.label.includes('dir') });
      }
    }
    return { nodes, edges };
  }


  async loadGraphVersion(id: string, _ver: number): Promise<CausalGraph | null> {
    return this.loadGraph(id);
  }

  async listGraphVersions(id: string): Promise<GraphVersion[]> {
    const count = this.vers.get(id) ?? 0;
    if (count === 0) return [];
    return Array.from({ length: count }, (_, i) => ({ graphId: id, version: i + 1, timestamp: Date.now() }));
  }

  async findSimilarGraphs(_t: CausalGraph, lim: number): Promise<CausalGraph[]> {
    const r: CausalGraph[] = [];
    for (const [id] of this.vers) { const g = await this.loadGraph(id); if (g) r.push(g); if (r.length >= lim) break; }
    return r;
  }

  close(): void { this.g?.close?.(); }
}
