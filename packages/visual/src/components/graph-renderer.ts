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
    // Simple radial layout for DAG
    const rootNodes = data.nodes.filter(n => n.type === 'root_cause');
    const others = data.nodes.filter(n => n.type !== 'root_cause');
    const allNodes = [...rootNodes, ...others];

    const cx = w / 2, cy = h / 2;
    const maxR = Math.min(w, h) / 2 - 30;
    allNodes.forEach((n: GraphVizNode, i: number) => {
      const angle = (i / allNodes.length) * Math.PI * 2 - Math.PI / 2;
      const r = i < rootNodes.length ? maxR * 0.3 : maxR * 0.7;
      layout.set(n.id, { x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r });
    });
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
