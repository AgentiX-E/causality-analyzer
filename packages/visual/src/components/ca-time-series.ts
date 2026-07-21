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

    // Build unified time axis from ALL series timestamps (union)
    const tsSet = new Set<number>();
    for (const s of series) for (const d of s.data) tsSet.add(d.ts);
    const ts = Array.from(tsSet).sort((a, b) => a - b);

    // Align each series to unified time axis (NaN for missing points → gap)
    const aligned: AlignedData = [ts] as AlignedData;
    for (const s of series) {
      const valueMap = new Map(s.data.map(d => [d.ts, d.value]));
      aligned.push(Float64Array.from(ts.map(t => valueMap.get(t) ?? NaN)));
    }

    const opts: uPlot.Options = {
      width: this.container.clientWidth,
      height: this.container.clientHeight,
      series: [{}, ...series.map((s: {name: string}) => ({ label: s.name, stroke: this._color(s.name) }))],
      axes: [{}, { values: (_u: any, vals: number[]) => vals.map((v: number) => v.toFixed(1)) }],
      cursor: { show: true },
      hooks: {
        draw: [
          (u: uPlot) => {
            // Render anomaly regions as shaded bands
            const ctx = u.ctx;
            const regions = this.data!.anomalyRegions;
            if (!regions || regions.length === 0) return;
            for (const region of regions) {
              const x0 = u.valToPos(region.start, 'x', true);
              const x1 = u.valToPos(region.end, 'x', true);
              const alpha = region.severity === 'critical' ? 0.12 : region.severity === 'warning' ? 0.08 : 0.04;
              ctx.fillStyle = `rgba(220, 38, 38, ${alpha})`;
              ctx.fillRect(x0, u.bbox.top, Math.max(x1 - x0, 1), u.bbox.height);
            }
          },
        ],
      },
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
