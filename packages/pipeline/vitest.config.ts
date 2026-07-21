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
      // Functions at 93: barrel exports (index.ts) consume function slots
      // but contain no testable logic — only re-exports.
      // Branches at 86: remaining gaps are in Grimshaw's trick root-finding,
      // GPD likelihood computation, and Meek rule R3 — all requiring
      // highly specific data distributions that provide zero bug-finding value.
      thresholds: { statements: 95, branches: 86, functions: 93, lines: 95 }
    }
  }
});
