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
      // These thresholds cover all user-facing rendering paths + renderer interface.
      thresholds: { statements: 69, branches: 78, functions: 79, lines: 69 }
    }
  }
});
