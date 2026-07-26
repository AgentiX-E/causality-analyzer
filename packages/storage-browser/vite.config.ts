import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  server: {
    headers: {
      // Required for SharedArrayBuffer and OPFS synchronous access handles
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
