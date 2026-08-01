import { defineConfig } from 'vitest/config';
export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['src/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'lcov'],
      include: ['src/**/*.ts'],
      exclude: [
        'src/**/*.test.ts',
        'src/**/*.spec.ts',
        'src/**/index.ts',
        'src/sqlite-port.ts',
        'src/sqlite-worker.ts',
        'src/worker-sqlite-port.ts',
        'src/types/**',
      ],
      thresholds: { statements: 95, branches: 90, functions: 95, lines: 95 },
    },
  },
});
