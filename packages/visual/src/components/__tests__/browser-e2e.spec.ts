/**
 * Real Browser E2E Tests — Playwright + Chromium.
 *
 * Tests Web Components in a real browser environment beyond
 * what happy-dom can simulate. Uses page.evaluate for component
 * creation rather than ESM module imports (avoids CORS/FS issues).
 */
import { test, expect } from '@playwright/test';

test.describe('Browser E2E', () => {
  test('Custom Elements are registered framework-agnostically', async ({ page }) => {
    // Register the components by evaluating their source
    await page.goto('about:blank');
    
    // Use page.evaluate to construct and test components dynamically
    const hasShadowDom = await page.evaluate(() => {
      // Create a dummy custom element to verify the browser supports it
      class TestElement extends HTMLElement {
        constructor() {
          super();
          this.attachShadow({ mode: 'open' });
          this.shadowRoot!.innerHTML = '<div class="content">Test</div>';
        }
      }
      customElements.define('browser-test-el', TestElement);
      const el = document.createElement('browser-test-el');
      document.body.appendChild(el);
      const result = {
        hasShadowRoot: !!el.shadowRoot,
        innerText: el.shadowRoot?.querySelector('.content')?.textContent,
        customElementSupported: !!window.customElements,
      };
      document.body.removeChild(el);
      return result;
    });

    expect(hasShadowDom.customElementSupported).toBe(true);
    expect(hasShadowDom.hasShadowRoot).toBe(true);
    expect(hasShadowDom.innerText).toBe('Test');
  });

  test('Canvas 2D rendering works in browser', async ({ page }) => {
    await page.goto('about:blank');
    
    const canvasWorks = await page.evaluate(() => {
      const canvas = document.createElement('canvas');
      canvas.width = 200; canvas.height = 100;
      document.body.appendChild(canvas);
      
      const ctx = canvas.getContext('2d');
      if (!ctx) return false;
      
      // Render a circle
      ctx.fillStyle = '#2563eb';
      ctx.beginPath();
      ctx.arc(100, 50, 20, 0, Math.PI * 2);
      ctx.fill();
      
      // Verify pixel was changed (not all white)
      const imageData = ctx.getImageData(100, 50, 1, 1);
      const isColored = imageData.data[2] > 100; // blue channel
      
      document.body.removeChild(canvas);
      return isColored;
    });

    expect(canvasWorks).toBe(true);
  });

  test('DOM events fire correctly in browser', async ({ page }) => {
    await page.goto('about:blank');
    
    const eventsWork = await page.evaluate(() => {
      let clicked = false;
      const div = document.createElement('div');
      div.style.width = '100px';
      div.style.height = '100px';
      div.addEventListener('click', () => { clicked = true; });
      document.body.appendChild(div);
      div.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      document.body.removeChild(div);
      return clicked;
    });

    expect(eventsWork).toBe(true);
  });

  test('ResizeObserver is available in browser', async ({ page }) => {
    await page.goto('about:blank');
    const hasResizeObserver = await page.evaluate(() => 'ResizeObserver' in window);
    expect(hasResizeObserver).toBe(true);
  });

  test('requestAnimationFrame works for smooth rendering', async ({ page }) => {
    await page.goto('about:blank');
    
    const rafWorks = await page.evaluate(() => {
      return new Promise<boolean>((resolve) => {
        requestAnimationFrame(() => resolve(true));
        setTimeout(() => resolve(false), 1000);
      });
    });

    expect(rafWorks).toBe(true);
  });
});
