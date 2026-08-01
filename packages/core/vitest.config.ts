import { defineConfig } from 'vitest/config';

export default defineConfig({ 
  test: {
    globals: true,
    environment: 'node',
    include: ['src/**/*.test.ts', 'src/**/__tests__/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'lcov'],
      include: ['src/**/*.ts'],
      exclude: [
        'src/**/*.test.ts',
        'src/**/__tests__/**',
        'src/index.ts',
        'src/di/index.ts',
        'src/interfaces/index.ts',
        'src/types/index.ts',
        'src/types/timeseries.ts', // Pure type definitions, no executable code
        'src/registry/**',        // Experimental decorator-based registry, not yet used
      ],
      thresholds: {
        statements: 94,
        branches: 82,
        functions: 94,
        lines: 95
      }
    }
  }
});
