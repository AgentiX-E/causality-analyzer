/**
 * GraphRenderer — pluggable graph rendering interface.
 *
 * Canvas2DRenderer is the default (zero deps, < 200 nodes).
 * Swap to WebGLRenderer for 500+ node graphs.
 */
import type { GraphVisualizationData, GraphVizNode } from '@agentix-e/causality-analyzer-core';

export interface ThemeVars {
  primary: string; anomaly: string; rootCause: string; healthy: string; edgeWeight: string;
}

export interface GraphRenderer {
  render(canvas: HTMLCanvasElement, data: GraphVisualizationData, theme: ThemeVars): void;
  hitTest(x: number, y: number, data: GraphVisualizationData): string | null;
  dispose(): void;
}

const DEFAULT_THEME: ThemeVars = {
  primary: '#2563eb', anomaly: '#dc2626', rootCause: '#f59e0b',
  healthy: '#22c55e', edgeWeight: '#94a3b8',
};

export class Canvas2DRenderer implements GraphRenderer {
  private layoutCache: Map<string, { x: number; y: number }> | null = null;

  render(canvas: HTMLCanvasElement, data: GraphVisualizationData, theme: ThemeVars = DEFAULT_THEME): void {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);

    const layout = this.computeLayout(data, canvas.clientWidth, canvas.clientHeight);
    this.layoutCache = layout;

    // Draw edges
    ctx.strokeStyle = theme.edgeWeight;
    ctx.lineWidth = 1;
    for (const edge of data.edges) {
      const s = layout.get(edge.source);
      const t = layout.get(edge.target);
      if (s && t) {
        ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(t.x, t.y); ctx.stroke();
        // Arrowhead
        if (edge.directed) this.drawArrow(ctx, s.x, s.y, t.x, t.y);
      }
    }

    // Draw nodes
    for (const node of data.nodes) {
      const pos = layout.get(node.id);
      if (!pos) continue;
      const color = node.type === 'root_cause' ? theme.rootCause :
        node.type === 'anomaly' ? theme.anomaly : theme.healthy;
      const radius = node.isAnomalous ? 12 + node.score * 4 : 8;

      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();

      // Label
      ctx.fillStyle = '#1e293b'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(node.label, pos.x, pos.y + radius + 12);
    }
  }

  hitTest(x: number, y: number, _data: GraphVisualizationData): string | null {
    if (!this.layoutCache) return null;
    for (const [id, pos] of this.layoutCache) {
      if (Math.hypot(x - pos.x, y - pos.y) < 14) return id;
    }
    return null;
  }

  dispose(): void { this.layoutCache = null; }

  private computeLayout(data: GraphVisualizationData, w: number, h: number): Map<string, { x: number; y: number }> {
    const layout = new Map<string, { x: number; y: number }>();
    const { nodes, edges } = data;

    // Build adjacency maps
    const children = new Map<string, string[]>();
    const parents = new Map<string, string[]>();
    for (const n of nodes) { children.set(n.id, []); parents.set(n.id, []); }
    for (const e of edges) {
      children.get(e.source)?.push(e.target);
      parents.get(e.target)?.push(e.source);
    }

    // BFS layered layout: source nodes at top, sinks at bottom
    const layers: GraphVizNode[][] = [];
    const assigned = new Set<string>();
    let frontier = nodes.filter(n => (parents.get(n.id)?.length ?? 0) === 0);
    if (frontier.length === 0) frontier = [nodes[0]!];

    while (frontier.length > 0) {
      layers.push(frontier);
      for (const n of frontier) assigned.add(n.id);
      const nextIds = new Set<string>();
      for (const n of frontier) {
        for (const c of children.get(n.id) ?? []) {
          if (!assigned.has(c)) nextIds.add(c);
        }
      }
      frontier = [...nextIds].map(id => nodes.find(n => n.id === id)!).filter(Boolean);
    }

    // Position nodes by layer
    const pad = 40;
    const layerH = layers.length > 1 ? (h - pad * 2) / (layers.length - 1) : h - pad * 2;
    for (let li = 0; li < layers.length; li++) {
      const ln = layers[li]!;
      const layerW = ln.length > 1 ? (w - pad * 2) / (ln.length - 1) : w - pad * 2;
      for (let ni = 0; ni < ln.length; ni++) {
        const n = ln[ni]!;
        layout.set(n.id, {
          x: pad + layerW * (ln.length > 1 ? ni : 0),
          y: pad + layerH * li,
        });
      }
    }
    return layout;
  }

  private drawArrow(ctx: CanvasRenderingContext2D, sx: number, sy: number, tx: number, ty: number) {
    const angle = Math.atan2(ty - sy, tx - sx);
    const len = 8;
    ctx.beginPath();
    ctx.moveTo(tx - Math.cos(angle) * 12, ty - Math.sin(angle) * 12);
    ctx.lineTo(tx - Math.cos(angle - 0.4) * len, ty - Math.sin(angle - 0.4) * len);
    ctx.lineTo(tx - Math.cos(angle + 0.4) * len, ty - Math.sin(angle + 0.4) * len);
    ctx.closePath(); ctx.fill();
  }
}
