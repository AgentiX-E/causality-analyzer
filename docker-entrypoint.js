#!/usr/bin/env node
/**
 * Docker entrypoint for Causality Analyzer Pipeline.
 *
 * Starts the REST API server on the configured PORT (default 3000)
 * with proper error handling and graceful shutdown.
 */

const PORT = parseInt(process.env.PORT || '3000', 10);

async function main() {
  let server = null;
  try {
    const { CausalityServer } = await import('@agentix-e/causality-analyzer-pipeline');
    server = new CausalityServer();
    await server.start(PORT);
    console.log(`Causality Analyzer API v1.0.0 started on port ${PORT}`);
    console.log('Endpoints: /health /ready /live /metrics /discover /analyze /estimate');

    // Graceful shutdown — drain connections before exit
    const shutdown = async (signal: string) => {
      console.log(`Received ${signal} — shutting down gracefully`);
      try {
        if (server) await server.stop();
        console.log('Server stopped cleanly');
      } catch (err) {
        console.error('Error during shutdown:', err);
      }
      process.exit(0);
    };

    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));
  } catch (err) {
    console.error('Failed to start Causality Analyzer server:', err);
    process.exit(1);
  }
}

main();
