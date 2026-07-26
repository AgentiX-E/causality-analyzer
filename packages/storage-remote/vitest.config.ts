import { defineConfig } from 'vitest/config';
export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['src/**/*.test.ts'],
    exclude: ['src/**/*.neo4j.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'lcov'],
      include: ['src/**/*.ts'],
      exclude: ['src/**/*.test.ts', 'src/**/*.neo4j.test.ts', 'src/index.ts', 'src/types.ts'],
      thresholds: { statements: 77, branches: 58, functions: 55, lines: 77 },
    },
  },
});
