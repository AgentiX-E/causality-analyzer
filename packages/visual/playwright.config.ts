import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './src/components/__tests__',
  testMatch: '**/*.spec.ts',
  timeout: 15000,
  retries: 1,
  use: { baseURL: 'file:///workspace/causality-analyzer/packages/visual/', headless: true },
});
