#!/usr/bin/env node
/**
 * Docker entrypoint for Causality Analyzer Pipeline.
 *
 * Starts the REST API server on the configured PORT (default 3000)
 * with proper error handling and graceful shutdown.
 */

const PORT = parseInt(process.env.PORT || '3000', 10);

async function main() {
  try {
    const { CausalityServer } = await import('@agentix-e/causality-analyzer-pipeline');
    const server = new CausalityServer();
    await server.start(PORT);
    console.log(`Causality Analyzer API v1.0.0 started on port ${PORT}`);
    console.log('Endpoints: /health /ready /live /metrics /discover /analyze /estimate');
  } catch (err) {
    console.error('Failed to start Causality Analyzer server:', err);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('Received SIGTERM — shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('Received SIGINT — shutting down');
  process.exit(0);
});

main();
