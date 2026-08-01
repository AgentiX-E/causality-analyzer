import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    testTimeout: 30000,
    hookTimeout: 60000,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'lcov'],
      include: ['src/**/*.ts'],
      exclude: [
        'src/**/*.test.ts',
        'src/**/__tests__/**',
        'src/**/index.ts',
        'src/cli.ts',
        'src/server.ts',
        'src/smart-runner.ts',
        'src/ci-worker.ts',
        'src/parallel/**',
        // Infrastructure modules requiring integration environment
        'src/audit-trail.ts',       // Security audit — needs full stack
        'src/encrypted-store.ts',   // AES-256-GCM — key management integration
        'src/llm-causal.ts',        // LLM integration — needs API key
        'src/llm-explainer.ts',     // LLM explainer — needs API key
        'src/auto-profile.ts',      // Auto-profiling — needs telemetry backend
        'benchmark/**',
      ],
      thresholds: {
        statements: 90,
        branches: 75,
        functions: 90,
        lines: 90,
      },
    },
  },
});
