/**
 * Real Browser E2E — wa-sqlite + OPFS + WorkerSqlitePort + WasmGraphStore.
 *
 * Starts a Vite dev server with COOP/COEP headers (required for
 * SharedArrayBuffer-based OPFS synchronous I/O), loads the test
 * page that exercises the full storage stack in a real browser,
 * and validates SQL execution, graph persistence, and data survival
 * across port close/reopen cycles.
 *
 * @packageDocumentation
 */
import { test, expect } from '@playwright/test';
import { createServer } from 'vite';

let server: Awaited<ReturnType<typeof createServer>>['httpServer'] | null = null;

test.beforeAll(async () => {
  const viteServer = await createServer({
    configFile: 'browser-test/vite.config.ts',
    root: '.',
    server: { port: 0 }, // random port
  });
  await viteServer.listen();
  server = viteServer.httpServer;
});

test.afterAll(async () => {
  if (server) server.close();
});

test.describe('Real Browser E2E — wa-sqlite + OPFS + GraphStore', () => {
  test('full storage stack: SQL insert, graph save, close, reopen, verify', async ({ page }) => {
    const baseUrl = (server as any).address()
      ? `http://localhost:${(server as any).address().port}`
      : 'http://localhost:4173';

    await page.goto(`${baseUrl}/browser-test/e2e-real.html`);

    // Wait for test completion (OPFS init + SQL + graph store ~10s)
    await page.waitForFunction(
      () => (window as unknown as Record<string, unknown>).__testResults !== undefined,
      { timeout: 30000 },
    );

    const results = await page.evaluate(
      () => (window as unknown as Record<string, unknown>).__testResults,
    ) as Record<string, unknown> | null;

    if (!results) {
      const error = await page.evaluate(
        () => (window as unknown as Record<string, string>).__testError,
      );
      throw new Error(`Browser test failed: ${error}`);
    }

    expect(results.sqliteOk).toBe(true);
    expect(results.storeOk).toBe(true);
    expect(results.persistenceOk).toBe(true);
    expect(results.edgeCount).toBe(3);
    expect(typeof results.graphId).toBe('string');
  });
});
