/**
 * Minimal test HTTP server for Playwright E2E.
 *
 * Serves files from the current directory with COOP/COEP headers
 * required for SharedArrayBuffer and OPFS synchronous access handles.
 */
import { createServer } from 'node:http';
import { readFileSync, existsSync } from 'node:fs';
import { join, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PORT = 4173;

const MIME = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.ts': 'application/javascript',
  '.wasm': 'application/wasm',
  '.css': 'text/css',
};

createServer((req, res) => {
  const url = new URL(req.url ?? '/', `http://localhost:${PORT}`);
  let filePath = join(__dirname, url.pathname === '/' ? 'browser-test.html' : url.pathname);

  if (!existsSync(filePath)) {
    // Try fallback to workspace root for node_modules resolution
    const workspaceRoot = join(__dirname, '..', '..');
    filePath = join(workspaceRoot, url.pathname);
  }

  // Serve TypeScript files as JavaScript MIME type
  const ext = extname(filePath);
  res.setHeader('Content-Type', MIME[ext] ?? 'application/octet-stream');
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');

  try {
    const content = readFileSync(filePath);
    res.writeHead(200);
    res.end(content);
  } catch {
    res.writeHead(404);
    res.end('Not found');
  }
}).listen(PORT, () => {
  console.log(`Test server running at http://localhost:${PORT}`);
});
