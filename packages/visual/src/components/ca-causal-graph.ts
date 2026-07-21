/**
 * <ca-causal-graph> — Web Component for DAG visualization.
 *
 * Framework-agnostic. Accepts GraphVisualizationData as attribute or property.
 * Emits 'node-click' event with node ID on click.
 */
import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import type { GraphVisualizationData } from '@agentix-e/causality-analyzer-core';
import { Canvas2DRenderer, type GraphRenderer } from './graph-renderer.js';

@customElement('ca-causal-graph')
export class CaCausalGraph extends LitElement {
  static override styles = css`
    :host { display: block; width: 100%; height: 320px; position: relative; }
    canvas { width: 100%; height: 100%; }
  `;

  @property({ type: Object }) data: GraphVisualizationData | null = null;
  @property({ type: Object }) renderer: GraphRenderer = new Canvas2DRenderer();
  @state() private _canvas: HTMLCanvasElement | null = null;
  private _resizeObserver: ResizeObserver | null = null;

  override updated() {
    if (this._canvas && this.data) {
      this.renderer.render(this._canvas, this.data, this._readTheme());
    }
  }

  override firstUpdated() {
    this._canvas = this.renderRoot.querySelector('canvas');
    this._resizeObserver = new ResizeObserver(() => {
      if (this._canvas && this.data) this.renderer.render(this._canvas, this.data, this._readTheme());
    });
    if (this._canvas) this._resizeObserver.observe(this._canvas);
  }

  override render() {
    return html`<canvas @click=${this._onClick}></canvas>`;
  }

  private _onClick(e: MouseEvent) {
    if (!this._canvas || !this.data) return;
    const rect = this._canvas.getBoundingClientRect();
    const id = this.renderer.hitTest(e.clientX - rect.left, e.clientY - rect.top, this.data);
    if (id) this.dispatchEvent(new CustomEvent('node-click', { detail: { id }, bubbles: true, composed: true }));
  }

  private _readTheme() {
    const style = getComputedStyle(this);
    return {
      primary: style.getPropertyValue('--ca-primary').trim() || '#2563eb',
      anomaly: style.getPropertyValue('--ca-anomaly').trim() || '#dc2626',
      rootCause: style.getPropertyValue('--ca-root-cause').trim() || '#f59e0b',
      healthy: style.getPropertyValue('--ca-healthy').trim() || '#22c55e',
      edgeWeight: style.getPropertyValue('--ca-edge-weight').trim() || '#94a3b8',
    };
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    this._resizeObserver?.disconnect();
    this._resizeObserver = null;
    this.renderer.dispose();
  }
}
