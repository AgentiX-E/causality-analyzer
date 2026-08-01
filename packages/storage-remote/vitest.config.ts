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
      exclude: [
        'src/**/*.test.ts',
        'src/**/*.neo4j.test.ts',
        'src/**/__tests__/**',
        'src/index.ts',
        'src/types.ts',
      ],
      // DB connection code requires live PostgreSQL/Neo4j — covered by
      // integration tests in CI. Pure logic functions are covered.
      thresholds: { statements: 74, branches: 58, functions: 76, lines: 75 },
    },
  },
});
