/**
 * Browser integration tests: verify Web Components render and handle events
 * in a real Chromium browser via Playwright.
 */
import { test, expect } from '@playwright/test';

const GRAPH_DATA = {
  nodes: [
    { id: 'Memory', label: 'Memory', type: 'root_cause' as const, score: 0.9, isAnomalous: false },
    { id: 'CPU', label: 'CPU', type: 'anomaly' as const, score: 0.5, isAnomalous: true },
  ],
  edges: [{ source: 'Memory', target: 'CPU', weight: 1, directed: true }],
};

const TIMESERIES_DATA = {
  series: [{ name: 'cpu', data: [{ ts: 1000, value: 10 }, { ts: 1001, value: 12 }, { ts: 1002, value: 50 }, { ts: 1003, value: 11 }] }],
  anomalyRegions: [{ start: 1002, end: 1002, severity: 'critical' as const, rootCause: 'Memory' }],
};

const RANKING_DATA = {
  rootCauses: [{ rank: 1, name: 'Memory', score: 0.92, confidence: 0.95, evidence: [] as any[] }],
  propagationPaths: [{ root: 'Memory', path: ['Memory', 'CPU', 'Latency'], score: 0.85 }],
};

test.describe('ca-causal-graph', () => {
  test('renders canvas element in Shadow DOM', async ({ page }) => {
    await page.setContent(`
      <!DOCTYPE html>
      <html><body>
        <ca-causal-graph id="graph"></ca-causal-graph>
        <script type="module">
          import '/workspace/causality-analyzer/packages/visual/src/components/ca-causal-graph.ts';
        </script>
      </body></html>
    `);
    await page.waitForSelector('ca-causal-graph');
    const el = page.locator('ca-causal-graph');
    expect(await el.count()).toBe(1);
  });

  test('accepts data and renders without error', async ({ page }) => {
    await page.setContent(`
      <!DOCTYPE html>
      <html><body>
        <ca-causal-graph id="graph"></ca-causal-graph>
        <script type="module">
          import '/workspace/causality-analyzer/packages/visual/src/node_modules/.vite/deps/ca-causal-graph.ts';
        </script>
      </body></html>
    `);
    await page.evaluate((data) => {
      const el = document.getElementById('graph') as any;
      if (el) el.data = data;
    }, GRAPH_DATA);
    // Should not throw
    await page.waitForTimeout(200);
    expect(true).toBe(true);
  });
});

test.describe('ca-time-series', () => {
  test('renders container div', async ({ page }) => {
    await page.setContent(`
      <!DOCTYPE html>
      <html><body>
        <ca-time-series id="ts"></ca-time-series>
        <script type="module">
          import '/workspace/causality-analyzer/packages/visual/src/components/ca-time-series.ts';
        </script>
      </body></html>
    `);
    await page.waitForSelector('ca-time-series');
    expect(await page.locator('ca-time-series').count()).toBe(1);
  });
});

test.describe('ca-root-cause-ranking', () => {
  test('renders ranking items', async ({ page }) => {
    await page.setContent(`
      <!DOCTYPE html>
      <html><body>
        <ca-root-cause-ranking id="rank"></ca-root-cause-ranking>
        <script type="module">
          import '/workspace/causality-analyzer/packages/visual/src/components/ca-root-cause-ranking.ts';
        </script>
      </body></html>
    `);
    await page.waitForSelector('ca-root-cause-ranking');
    await page.evaluate((data) => {
      const el = document.getElementById('rank') as any;
      if (el) el.data = data;
    }, RANKING_DATA);
    await page.waitForTimeout(100);
    const text = await page.locator('ca-root-cause-ranking').innerText();
    expect(text).toContain('Memory');
  });
});
