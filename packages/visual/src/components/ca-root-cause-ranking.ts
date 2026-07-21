/**
 * <ca-root-cause-ranking> — Web Component for root cause ranking display.
 *
 * Framework-agnostic. Accepts RCARankingData.
 * Emits 'cause-hover' event on mouseover.
 */
import { LitElement, html, css } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import type { RCARankingData } from '@agentix-e/causality-analyzer-core';

@customElement('ca-root-cause-ranking')
export class CaRootCauseRanking extends LitElement {
  static override styles = css`
    :host { display: block; font-family: sans-serif; }
    .rc-item { display: flex; align-items: center; padding: 8px 12px; border-bottom: 1px solid #e2e8f0; cursor: pointer; }
    .rc-item:hover { background: #f8fafc; }
    .rc-rank { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px; margin-right: 12px; }
    .rc-rank.top { background: var(--ca-root-cause, #f59e0b); color: #fff; }
    .rc-rank.normal { background: #e2e8f0; color: #475569; }
    .rc-name { flex: 1; font-weight: 600; color: #1e293b; }
    .rc-score { font-size: 13px; color: #64748b; margin-right: 8px; }
    .rc-score-bar { width: 80px; height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden; }
    .rc-score-fill { height: 100%; border-radius: 3px; background: var(--ca-primary, #2563eb); }
  `;

  @property({ type: Object }) data: RCARankingData | null = null;

  override render() {
    if (!this.data) return html`<div></div>`;
    return html`
      ${this.data.rootCauses.map(rc => html`
        <div class="rc-item" @mouseenter=${() => this._onHover(rc.name)}>
          <div class="rc-rank ${rc.rank <= 3 ? 'top' : 'normal'}">${rc.rank}</div>
          <div class="rc-name">${rc.name}</div>
          <div class="rc-score">${(rc.score * 100).toFixed(0)}%</div>
          <div class="rc-score-bar"><div class="rc-score-fill" style="width:${rc.score * 100}%"></div></div>
        </div>
      `)}
    `;
  }

  private _onHover(name: string) {
    this.dispatchEvent(new CustomEvent('cause-hover', { detail: { name }, bubbles: true, composed: true }));
  }
}
