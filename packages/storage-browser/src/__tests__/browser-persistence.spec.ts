/**
 * I13: Playwright E2E — Worker protocol + OPFS readiness.
 *
 * Tests the WorkerSqlitePort message protocol in a real browser,
 * and verifies that the page environment supports the COOP/COEP
 * headers required for OPFS synchronous access handles.
 */
import { test, expect } from '@playwright/test';

test.describe('Storage-Browser Worker Protocol', () => {
  test('Worker message protocol: exec, run, all, concurrent, terminate', async ({ page }) => {
    await page.goto('/browser-test.html');

    // Wait for test completion
    await page.waitForFunction(
      () => (window as unknown as Record<string, unknown>).__testResults !== undefined,
      { timeout: 15000 },
    );

    const results = await page.evaluate(
      () => (window as unknown as Record<string, unknown>).__testResults,
    ) as Record<string, unknown>;

    expect(results).not.toBeNull();
    expect(results.execOk).toBe(true);
    expect(results.insertOk).toBe(true);
    expect(results.selectOk).toBe(true);
    expect(results.selectCount).toBeGreaterThanOrEqual(0);
    expect(results.concurrentOk).toBe(true);
    expect(results.closeOk).toBe(true);
  });

  test('COOP/COEP headers present for OPFS', async ({ page }) => {
    const response = await page.goto('/browser-test.html');
    const headers = response?.headers() ?? {};

    // These headers are required for SharedArrayBuffer and OPFS
    expect(headers['cross-origin-opener-policy']).toBe('same-origin');
    expect(headers['cross-origin-embedder-policy']).toBe('require-corp');
  });

  test('page reload does not lose test harness', async ({ page }) => {
    // First load
    await page.goto('/browser-test.html');
    await page.waitForFunction(
      () => (window as unknown as Record<string, unknown>).__testResults !== undefined,
      { timeout: 15000 },
    );

    // Reload
    await page.reload();
    await page.waitForFunction(
      () => (window as unknown as Record<string, unknown>).__testResults !== undefined,
      { timeout: 15000 },
    );

    const results = await page.evaluate(
      () => (window as unknown as Record<string, unknown>).__testResults,
    ) as Record<string, unknown>;
    expect(results.execOk).toBe(true);
  });

  test('handles malformed SQL gracefully', async ({ page }) => {
    await page.goto('/browser-test.html');
    await page.waitForFunction(
      () => (window as unknown as Record<string, unknown>).__testResults !== undefined,
      { timeout: 15000 },
    );

    const results = await page.evaluate(
      () => (window as unknown as Record<string, unknown>).__testResults,
    ) as Record<string, unknown>;
    expect(results.errorHandlingOk).toBe(true);
  });
});
