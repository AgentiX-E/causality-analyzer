/**
 * <ca-time-series> — Web Component for time series anomaly visualization.
 *
 * Built on uPlot 1.6 for high-performance Canvas rendering.
 * Accepts TimeSeriesChartData. Emits 'region-select' for anomaly range.
 */
import { LitElement, html, css, type PropertyValues } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import uPlot from 'uplot';
import type { AlignedData } from 'uplot';
import type { TimeSeriesChartData } from '@agentix-e/causality-analyzer-core';

@customElement('ca-time-series')
export class CaTimeSeries extends LitElement {
  static override styles = css`
    :host { display: block; width: 100%; height: 240px; }
    .uplot-wrap { width: 100%; height: 100%; }
  `;

  @property({ type: Object }) data: TimeSeriesChartData | null = null;
  private plot: uPlot | null = null;
  private container: HTMLDivElement | null = null;

  override firstUpdated() {
    this.container = this.renderRoot.querySelector('.uplot-wrap');
    this._render();
    new ResizeObserver(() => this._render()).observe(this.container!);
  }

  override updated(changed: PropertyValues) {
    if (changed.has('data')) this._render();
  }

  override render() { return html`<div class="uplot-wrap"></div>`; }

  private _render() {
    if (!this.container || !this.data || this.data.series.length === 0) return;
    if (this.plot) { this.plot.destroy(); this.plot = null; }

    const series = this.data.series;
    const ts = series[0]?.data.map(d => d.ts) ?? [];
    const aligned: AlignedData = [ts] as AlignedData;
    for (const s of series) aligned.push(Float64Array.from(s.data.map(d => d.value)));

    const opts: uPlot.Options = {
      width: this.container.clientWidth,
      height: this.container.clientHeight,
      series: [{}, ...series.map((s: {name: string}) => ({ label: s.name, stroke: this._color(s.name) }))],
      axes: [{}, { values: (_u: any, vals: number[]) => vals.map((v: number) => v.toFixed(1)) }],
      cursor: { show: true },
    };

    this.plot = new uPlot(opts, aligned, this.container);
  }

  private _color(name: string): string {
    const colors = ['#2563eb', '#f59e0b', '#22c55e', '#dc2626', '#8b5cf6'];
    const idx = name.charCodeAt(0) % colors.length;
    return colors[idx] ?? '#2563eb';
  }

  override disconnectedCallback() { super.disconnectedCallback(); this.plot?.destroy(); }
}
