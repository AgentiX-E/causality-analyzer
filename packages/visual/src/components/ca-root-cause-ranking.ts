/**
 * <ca-root-cause-ranking> — Web Component for root cause ranking display.
 *
 * Framework-agnostic. Accepts RCARankingData.
 * Emits 'cause-hover' event on mouseover.
 *
 * Accessibility: Uses role="listbox" and role="option" for screen readers.
 * Full keyboard navigation: Tab to focus, Up/Down arrows, Enter to select.
 */
import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import type { RCARankingData } from '@agentix-e/causality-analyzer-core';

@customElement('ca-root-cause-ranking')
export class CaRootCauseRanking extends LitElement {
  static override styles = css`
    :host { display: block; font-family: sans-serif; outline: none; }
    :host(:focus-visible) { outline: 2px solid var(--ca-primary, #2563eb); outline-offset: 2px; }

    .rc-list { list-style: none; margin: 0; padding: 0; }
    .rc-item { display: flex; align-items: center; padding: 8px 12px; border-bottom: 1px solid #e2e8f0; cursor: pointer; outline: none; }
    .rc-item:hover, .rc-item:focus-visible { background: #f8fafc; }
    .rc-item:focus-visible { outline: 2px solid var(--ca-primary, #2563eb); outline-offset: -2px; }
    .rc-item[aria-selected="true"] { background: #eff6ff; }

    .rc-rank { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px; margin-right: 12px; flex-shrink: 0; }
    .rc-rank.top { background: var(--ca-root-cause, #f59e0b); color: #fff; }
    .rc-rank.normal { background: #e2e8f0; color: #475569; }
    .rc-name { flex: 1; font-weight: 600; color: #1e293b; }
    .rc-score { font-size: 13px; color: #64748b; margin-right: 8px; }
    .rc-score-bar { width: 80px; height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden; }
    .rc-score-fill { height: 100%; border-radius: 3px; background: var(--ca-primary, #2563eb); }
  `;

  @property({ type: Object }) data: RCARankingData | null = null;
  @property({ type: String }) accessibleLabel = 'Root cause ranking';

  @state() private _activeIndex = -1;

  override firstUpdated() {
    this.setAttribute('tabindex', '0');
    this.setAttribute('role', 'listbox');
    this.setAttribute('aria-label', this.accessibleLabel);
    this.addEventListener('keydown', this._onKeyDown.bind(this));
  }

  override render() {
    if (!this.data) return html``;
    const causes = this.data.rootCauses;
    return html`
      <div class="rc-list" role="listbox" aria-label=${this.accessibleLabel}>
        ${causes.map((rc, i) => html`
          <div
            class="rc-item"
            role="option"
            aria-selected=${i === this._activeIndex ? 'true' : 'false'}
            tabindex=${i === this._activeIndex ? '0' : '-1'}
            @mouseenter=${() => this._onHover(rc.name)}
            @click=${() => this._onSelect(rc.name, i)}
          >
            <div class="rc-rank ${rc.rank <= 3 ? 'top' : 'normal'}">${rc.rank}</div>
            <div class="rc-name">${rc.name}</div>
            <div class="rc-score">${(rc.score * 100).toFixed(0)}%</div>
            <div class="rc-score-bar"><div class="rc-score-fill" style="width:${rc.score * 100}%"></div></div>
          </div>
        `)}
      </div>
    `;
  }

  private _onKeyDown(e: KeyboardEvent) {
    if (!this.data) return;
    const causes = this.data.rootCauses;
    if (!causes.length) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        this._activeIndex = Math.min(this._activeIndex + 1, causes.length - 1);
        this.requestUpdate();
        break;
      case 'ArrowUp':
        e.preventDefault();
        this._activeIndex = Math.max(this._activeIndex - 1, 0);
        this.requestUpdate();
        break;
      case 'Home':
        e.preventDefault();
        this._activeIndex = 0;
        this.requestUpdate();
        break;
      case 'End':
        e.preventDefault();
        this._activeIndex = causes.length - 1;
        this.requestUpdate();
        break;
      case 'Enter':
      case ' ':
        e.preventDefault();
        if (this._activeIndex >= 0 && causes[this._activeIndex]) {
          this._onSelect(causes[this._activeIndex]!.name, this._activeIndex);
        }
        break;
    }
  }

  private _onSelect(name: string, _index: number) {
    this.dispatchEvent(new CustomEvent('cause-select', { detail: { name }, bubbles: true, composed: true }));
  }

  private _onHover(name: string) {
    this.dispatchEvent(new CustomEvent('cause-hover', { detail: { name }, bubbles: true, composed: true }));
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    this.removeEventListener('keydown', this._onKeyDown.bind(this));
  }
}
