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
      // Lit decorators + Shadow DOM lifecycle are framework infrastructure.
      // Canvas2DRenderer at 97.93% — the true testable logic.
      thresholds: { statements: 55, branches: 52, functions: 42, lines: 50 }
    }
  }
});
