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
      // Canvas2DRenderer is the testable logic; browser E2E covers rendering.
      thresholds: { statements: 30, branches: 28, functions: 35, lines: 30 }
    }
  }
});
