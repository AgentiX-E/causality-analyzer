import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './src/__tests__',
  testMatch: '*.spec.ts',
  timeout: 60000,
  use: {
    baseURL: 'http://localhost:4173',
    headless: true,
  },
  webServer: {
    command: 'node test-server.mjs',
    port: 4173,
    reuseExistingServer: false,
  },
});
