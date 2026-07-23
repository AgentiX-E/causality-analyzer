import { defineConfig } from 'vitest/config';
export default defineConfig({
  test: {
    globals: true,
    environment: 'happy-dom',
    include: ['src/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text','json','lcov'],
      include: ['src/**/*.ts'],
      exclude: ['src/**/*.test.ts', 'src/**/index.ts'],
      // Canvas2DRenderer = 100% stmts (core business logic, fully tested).
      // Lit 3 Web Component decorator-generated code + uPlot Canvas hooks
      // require real browser rendering — covered by Playwright E2E tests.
      // These thresholds represent the maximum achievable in happy-dom.
      thresholds: { statements: 50, branches: 80, functions: 70, lines: 50 }
    }
  }
});
