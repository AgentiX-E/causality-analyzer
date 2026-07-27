import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  server: {
    port: 4173,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  resolve: {
    conditions: ['browser', 'import'],
  },
  worker: {
    format: 'es',
  },
});
