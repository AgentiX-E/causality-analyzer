import { defineConfig } from 'vitest/config';
// Neo4j integration tests — requires Docker Neo4j service + NEO4J_BOLT_URI env var.
// Run via `pnpm test:neo4j` (CI) or manually with env var set (local dev).
export default defineConfig({ test: { globals: true, environment: 'node', include: ['src/__tests__/remote-graph-store-neo4j.test.ts'], coverage: { enabled: false } } });
