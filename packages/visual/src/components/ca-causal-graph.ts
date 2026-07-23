/**
 * <ca-causal-graph> — Web Component for DAG visualization.
 *
 * Framework-agnostic. Accepts GraphVisualizationData as attribute or property.
 * Emits 'node-click' event with node ID on click.
 *
 * Accessibility: Canvas has role="img" and aria-label for screen readers.
 * Keyboard navigation: Tab to focus, arrow keys to navigate nodes, Enter/Space to select.
 */
import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import type { GraphVisualizationData } from '@agentix-e/causality-analyzer-core';
import { Canvas2DRenderer, type GraphRenderer } from './graph-renderer.js';

const srOnly = css`.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }`;

@customElement('ca-causal-graph')
export class CaCausalGraph extends LitElement {
  static override styles = [
    css`
      :host { display: block; width: 100%; height: 320px; position: relative; outline: none; }
      :host(:focus-visible) { outline: 2px solid var(--ca-primary, #2563eb); outline-offset: 2px; }
      canvas { width: 100%; height: 100%; }
    `,
    srOnly,
  ];

  @property({ type: Object }) data: GraphVisualizationData | null = null;
  @property({ type: Object }) renderer: GraphRenderer = new Canvas2DRenderer();
  @property({ type: String }) accessibleLabel = 'Causal graph visualization';

  @state() private _canvas: HTMLCanvasElement | null = null;
  private _resizeObserver: ResizeObserver | null = null;
  private _selectedIndex = -1;

  override updated() {
    if (this._canvas && this.data) {
      this.renderer.render(this._canvas, this.data, this._readTheme());
    }
  }

  override firstUpdated() {
    this._canvas = this.renderRoot.querySelector('canvas');
    if (this._canvas) {
      this._canvas.setAttribute('role', 'img');
      this._canvas.setAttribute('aria-label', this.accessibleLabel);
      this._canvas.setAttribute('tabindex', '-1');
    }
    this._resizeObserver = new ResizeObserver(() => {
      if (this._canvas && this.data) this.renderer.render(this._canvas, this.data, this._readTheme());
    });
    if (this._canvas) this._resizeObserver.observe(this._canvas);
    this.setAttribute('tabindex', '0');
    this.setAttribute('role', 'figure');
    this.addEventListener('keydown', this._onKeyDown.bind(this));
  }

  override render() {
    const nodeList = this.data?.nodes.map((n: { id: string; label?: string }) => n.label ?? n.id).join(', ') ?? '';
    return html`
      <canvas @click=${this._onClick}></canvas>
      <span class="sr-only" aria-live="polite">${nodeList}</span>
    `;
  }

  private _onClick(e: MouseEvent) {
    if (!this._canvas || !this.data) return;
    const rect = this._canvas.getBoundingClientRect();
    const id = this.renderer.hitTest(e.clientX - rect.left, e.clientY - rect.top, this.data);
    if (id) this.dispatchEvent(new CustomEvent('node-click', { detail: { id }, bubbles: true, composed: true }));
  }

  private _onKeyDown(e: KeyboardEvent) {
    if (!this.data || !this.data.nodes.length) return;
    const nodes = this.data.nodes;
    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        e.preventDefault();
        this._selectedIndex = Math.min(this._selectedIndex + 1, nodes.length - 1);
        this._announceNode(nodes[this._selectedIndex]!);
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
        e.preventDefault();
        this._selectedIndex = Math.max(this._selectedIndex - 1, 0);
        this._announceNode(nodes[this._selectedIndex]!);
        break;
      case 'Enter':
      case ' ':
        e.preventDefault();
        if (this._selectedIndex >= 0 && nodes[this._selectedIndex]) {
          this.dispatchEvent(new CustomEvent('node-click', {
            detail: { id: nodes[this._selectedIndex]!.id },
            bubbles: true,
            composed: true,
          }));
        }
        break;
    }
  }

  private _announceNode(node: { id: string; label?: string }) {
    const label = node.label ?? node.id;
    const sr = this.renderRoot.querySelector('.sr-only');
    if (sr) sr.textContent = `Selected: ${label}`;
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
    this.removeEventListener('keydown', this._onKeyDown.bind(this));
    this._resizeObserver?.disconnect();
    this._resizeObserver = null;
    this.renderer.dispose();
  }
}
