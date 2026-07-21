import { defineConfig } from 'vitest/config';
// Default: unit + mock integration tests. Excludes *.neo4j.test.ts (needs Docker).
export default defineConfig({ test: { globals: true, environment: 'node', include: ['src/**/*.test.ts'], exclude: ['src/**/*.neo4j.test.ts'], coverage: { provider:'v8', reporter:['text','json','lcov'], include:['src/**/*.ts'], exclude:['src/**/*.test.ts','src/**/*.neo4j.test.ts','src/index.ts','src/types.ts'], thresholds:{ statements:85, branches:80, functions:55, lines:85 } } } });
