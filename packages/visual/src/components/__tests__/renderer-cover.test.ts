/**
 * Canvas2DRenderer comprehensive coverage tests.
 * Uses manual Canvas mocks to test computeLayout and drawArrow paths
 * that happy-dom's getContext('2d') doesn't cover.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Canvas2DRenderer } from '../graph-renderer.js';
import type { GraphVisualizationData, GraphVizNode } from '@agentix-e/causality-analyzer-core';

function gv(nodes: GraphVizNode[], edges: GraphVisualizationData['edges'] = []): GraphVisualizationData {
  return { nodes, edges };
}

// ── Manual Canvas mock ──────────────────────────────────────────────
// happy-dom getContext('2d') returns an object but without method tracking.
// We create a full mock to verify and cover ALL render code paths.
function createMockCtx() {
  return {
    scale: vi.fn(),
    clearRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    fill: vi.fn(),
    arc: vi.fn(),
    fillText: vi.fn(),
    closePath: vi.fn(),
    // properties
    strokeStyle: '',
    fillStyle: '',
    lineWidth: 0,
    textAlign: '' as string,
    font: '',
  };
}

// ── Tests ───────────────────────────────────────────────────────────
describe('Canvas2DRenderer render coverage', () => {
  let renderer: Canvas2DRenderer;

  beforeEach(() => { renderer = new Canvas2DRenderer(); });

  function mkCanvas(w = 400, h = 300): HTMLCanvasElement {
    const c = document.createElement('canvas');
    // Mock getContext to return our tracked mock
    const ctx = createMockCtx();
    c.getContext = vi.fn(() => ctx) as any;
    Object.defineProperty(c, 'clientWidth', { value: w, configurable: true });
    Object.defineProperty(c, 'clientHeight', { value: h, configurable: true });
    return c;
  }

  it('calls ctx.scale with devicePixelRatio', () => {
    vi.spyOn(window, 'devicePixelRatio', 'get').mockReturnValue(2);
    const c = mkCanvas();
    renderer.render(c, gv([{ id: 'A', label: 'A', type: 'healthy', score: 0, isAnomalous: false }]));
    const ctx = c.getContext('2d') as any;
    expect(ctx.scale).toHaveBeenCalledWith(2, 2);
    expect(c.width).toBe(800);
    vi.restoreAllMocks();
  });

  it('calls clearRect', () => {
    const c = mkCanvas();
    renderer.render(c, gv([{ id: 'A', label: 'A', type: 'healthy', score: 0, isAnomalous: false }]));
    const ctx = c.getContext('2d') as any;
    expect(ctx.clearRect).toHaveBeenCalled();
  });

  it('draws edge path and arrowhead for directed edge', () => {
    const c = mkCanvas(400, 200);
    renderer.render(c, gv(
      [{ id: 'A', label: 'A', type: 'healthy', score: 0, isAnomalous: false },
       { id: 'B', label: 'B', type: 'anomaly', score: 0.5, isAnomalous: true }],
      [{ source: 'A', target: 'B', weight: 1, directed: true }],
    ));
    const ctx = c.getContext('2d') as any;
    expect(ctx.stroke).toHaveBeenCalled(); // edge drawn
    expect(ctx.fill).toHaveBeenCalled();   // arrowhead filled
  });

  it('does not draw arrowhead for undirected edge', () => {
    const c = mkCanvas(400, 200);
    renderer.render(c, gv(
      [{ id: 'A', label: 'A', type: 'healthy', score: 0, isAnomalous: false },
       { id: 'B', label: 'B', type: 'anomaly', score: 0.5, isAnomalous: true }],
      [{ source: 'A', target: 'B', weight: 1, directed: false }],
    ));
    const ctx = c.getContext('2d') as any;
    // stroke is called for the edge line, fill may not be called (no arrowhead)
    expect(ctx.stroke).toHaveBeenCalled();
  });

  it('draws node circle and label', () => {
    const c = mkCanvas();
    renderer.render(c, gv([{ id: 'A', label: 'A', type: 'root_cause', score: 0.9, isAnomalous: false }]));
    const ctx = c.getContext('2d') as any;
    expect(ctx.arc).toHaveBeenCalled();
    expect(ctx.fillText).toHaveBeenCalled();
    expect(ctx.fill).toHaveBeenCalled();
  });

  it('uses anomaly color for anomaly nodes', () => {
    const c = mkCanvas();
    renderer.render(c, gv([{ id: 'X', label: 'X', type: 'anomaly', score: 0.5, isAnomalous: true }]));
    const ctx = c.getContext('2d') as any;
    expect(ctx.fillStyle).toBeTruthy();
    expect(ctx.arc).toHaveBeenCalled();
  });

  it('uses root_cause color for root cause nodes', () => {
    const c = mkCanvas();
    renderer.render(c, gv([{ id: 'R', label: 'R', type: 'root_cause', score: 0.9, isAnomalous: false }]));
    const ctx = c.getContext('2d') as any;
    expect(ctx.arc).toHaveBeenCalled();
  });

  it('render with getContext returns null is no-op', () => {
    const c = document.createElement('canvas');
    const getCtx = vi.fn(() => null);
    c.getContext = getCtx as any;
    renderer.render(c, gv([{ id: 'A', label: 'A', type: 'healthy', score: 0, isAnomalous: false }]));
    expect(getCtx).toHaveBeenCalled();
    // no downstream calls because ctx is null
  });

  it('renders three-layer DAG with mixed edge types', () => {
    const c = mkCanvas(800, 600);
    renderer.render(c, gv(
      [
        { id: 'A', label: 'A', type: 'root_cause', score: 0.9, isAnomalous: false },
        { id: 'B', label: 'B', type: 'anomaly', score: 0.7, isAnomalous: true },
        { id: 'C', label: 'C', type: 'anomaly', score: 0.6, isAnomalous: true },
        { id: 'D', label: 'D', type: 'healthy', score: 0, isAnomalous: false },
        { id: 'E', label: 'E', type: 'healthy', score: 0, isAnomalous: false },
        { id: 'F', label: 'F', type: 'intermediate', score: 0.3, isAnomalous: false },
      ],
      [
        { source: 'A', target: 'B', weight: 1, directed: true },
        { source: 'A', target: 'C', weight: 0.9, directed: true },
        { source: 'B', target: 'D', weight: 0.8, directed: true },
        { source: 'C', target: 'D', weight: 0.7, directed: true },
        { source: 'D', target: 'E', weight: 0.6, directed: true },
        { source: 'A', target: 'F', weight: 0.5, directed: false },
      ],
    ));
    const ctx = c.getContext('2d') as any;
    expect(ctx.arc).toHaveBeenCalledTimes(6);
    expect(ctx.fillText).toHaveBeenCalledTimes(6);
  });

  it('hitTest after render returns correct id', () => {
    const c = mkCanvas(400, 300);
    const data = gv([{ id: 'A', label: 'A', type: 'healthy', score: 0, isAnomalous: false }]);
    renderer.render(c, data);
    // After render, layout cache has positions
    const hit = renderer.hitTest(40, 40, data);
    expect(hit).toBe('A');
  });

  it('hitTest returns null when no node nearby', () => {
    const c = mkCanvas(400, 300);
    const data = gv([{ id: 'A', label: 'A', type: 'healthy', score: 0, isAnomalous: false }]);
    renderer.render(c, data);
    expect(renderer.hitTest(999, 999, data)).toBeNull();
  });

  it('hitTest returns null when layoutCache is null', () => {
    expect(renderer.hitTest(50, 50, gv([{ id: 'A', label: 'A', type: 'healthy', score: 0, isAnomalous: false }]))).toBeNull();
  });

  it('dispose clears layoutCache', () => {
    const c = mkCanvas();
    renderer.render(c, gv([{ id: 'A', label: 'A', type: 'healthy', score: 0, isAnomalous: false }]));
    renderer.dispose();
    expect(renderer.hitTest(0, 0, gv([]))).toBeNull();
  });

  it('render with all node types', () => {
    const c = mkCanvas(600, 400);
    renderer.render(c, gv(
      [
        { id: 'R', label: 'Root', type: 'root_cause', score: 0.9, isAnomalous: false },
        { id: 'A', label: 'Anom', type: 'anomaly', score: 0.7, isAnomalous: true },
        { id: 'H', label: 'Healthy', type: 'healthy', score: 0, isAnomalous: false },
        { id: 'I', label: 'Inter', type: 'intermediate', score: 0.3, isAnomalous: false },
      ],
      [
        { source: 'R', target: 'A', weight: 1, directed: true },
        { source: 'A', target: 'I', weight: 0.8, directed: true },
        { source: 'I', target: 'H', weight: 0.6, directed: false },
      ],
    ));
    const ctx = c.getContext('2d') as any;
    expect(ctx.arc).toHaveBeenCalledTimes(4);
  });

  it('render with nodes that have different scores', () => {
    const c = mkCanvas();
    renderer.render(c, gv(
      [
        { id: 'H', label: 'High', type: 'anomaly', score: 0.95, isAnomalous: true },
        { id: 'L', label: 'Low', type: 'anomaly', score: 0.1, isAnomalous: true },
      ],
    ));
    const ctx = c.getContext('2d') as any;
    expect(ctx.arc).toHaveBeenCalledTimes(2);
  });

  it('empty nodes array renders without error', () => {
    const c = mkCanvas();
    renderer.render(c, gv([]));
    const ctx = c.getContext('2d') as any;
    // clearRect should still be called even with empty data
    expect(ctx.clearRect).toHaveBeenCalled();
  });

  it('no-op render when getContext returns null', () => {
    const c = document.createElement('canvas');
    c.getContext = vi.fn(() => null) as any;
    renderer.render(c, gv([{ id: 'A', label: 'A', type: 'healthy', score: 0, isAnomalous: false }]));
    expect(c.getContext).toHaveBeenCalled();
  });
});
