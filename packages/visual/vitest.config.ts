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
      // Lit decorators (@customElement, @property, @state) and Shadow DOM
      // lifecycle generate framework infrastructure code, not application logic.
      // CI happy-dom coverage is lower than local due to v8 variance.
      // Canvas2DRenderer and GraphRenderer interface are the testable logic.
      thresholds: { statements: 45, branches: 70, functions: 70, lines: 45 }
    }
  }
});
