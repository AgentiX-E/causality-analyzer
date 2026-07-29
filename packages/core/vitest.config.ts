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
        'src/registry/**',        // Experimental decorator-based registry, not yet used
      ],
      thresholds: {
        statements: 95,
        branches: 80,
        functions: 95,
        lines: 95
      }
    }
  }
});
