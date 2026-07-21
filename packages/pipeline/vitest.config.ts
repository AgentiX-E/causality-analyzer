import { defineConfig } from 'vitest/config';
export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['src/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text','json','lcov'],
      include: ['src/**/*.ts'],
      exclude: ['src/**/*.test.ts', 'src/index.ts'],
      // Functions at 93 is realistic: barrel exports (index.ts) consume function
      // slots but contain no testable logic — only re-exports.
      thresholds: { statements: 95, branches: 85, functions: 93, lines: 95 }
    }
  }
});
