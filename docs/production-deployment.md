# Production Deployment Guide

> How to run Causality Analyzer in production: security hardening, capacity planning, monitoring, and disaster recovery.

## Quick Start (Docker Compose)

```bash
docker compose up -d
# Starts: pipeline (HTTP API) + PostgreSQL + Neo4j
# Health check: curl http://localhost:3000/health
```

## Security Hardening

### API Authentication

Set the `CAUSALITY_API_TOKEN` environment variable to enable Bearer token authentication:

```bash
export CAUSALITY_API_TOKEN=$(openssl rand -hex 32)
npx causal-analyzer serve --port 3000
```

All `/v1/*` endpoints will require `Authorization: Bearer <token>`. Health/live/ready/metrics remain public.

### mTLS for Storage

**PostgreSQL:**
```typescript
import { RemoteRelationalStore } from '@agentix-e/causality-analyzer-storage-remote';
const store = new RemoteRelationalStore({
  connectionString: process.env.DATABASE_URL,
  mtls: {
    cert: process.env.PG_CLIENT_CERT,
    key: process.env.PG_CLIENT_KEY,
    ca: process.env.PG_CA_CERT,
    passphrase: process.env.PG_KEY_PASSPHRASE,
  },
});
```

**Neo4j:**
```typescript
import { RemoteGraphStore } from '@agentix-e/causality-analyzer-storage-remote';
const store = new RemoteGraphStore({
  uri: process.env.NEO4J_BOLT_URI,
  mtls: {
    cert: process.env.NEO4J_CLIENT_CERT,
    key: process.env.NEO4J_CLIENT_KEY,
  },
});
```

### Audit Trail

Enable the audit trail for tamper-evident logging:

```typescript
import { AuditTrail } from '@agentix-e/causality-analyzer-pipeline';
const audit = new AuditTrail({ hmacKey: process.env.AUDIT_HMAC_KEY });
audit.append('analysis.start', { graphId: 'g1', timestamp: Date.now() });
// ... run analysis ...
audit.append('analysis.complete', { result });
const { valid } = audit.verify(); // true = untampered
```

### Encryption at Rest

```typescript
import { EncryptedStore } from '@agentix-e/causality-analyzer-pipeline';
const encrypted = new EncryptedStore({
  key: Buffer.from(process.env.ENCRYPTION_KEY, 'hex'), // 32 bytes
  aad: 'causality-store',
});
// encrypt/decrypt sensitive data before storage
```

## Capacity Planning

| Graph Size | Memory | CPU | Storage (PG) |
|------------|--------|-----|--------------|
| ≤ 20 nodes, 5000 samples | 256MB | 0.5 core | 10MB |
| ≤ 50 nodes, 10000 samples | 512MB | 1 core | 50MB |
| ≤ 100 nodes, 50000 samples | 1GB | 2 cores | 200MB |
| 100+ nodes | 2GB+ | 4+ cores | 500MB+ |

Graph storage (Neo4j): ~1KB per node + ~0.5KB per edge.

## Monitoring Setup

### Prometheus Metrics

The `/metrics` endpoint exposes:
- `ca_requests_total` (counter) — total HTTP requests
- `ca_uptime_seconds` (gauge) — server uptime
- `ca_memory_heap_used_bytes` (gauge) — heap memory usage

### OpenTelemetry

Inject a real tracer for distributed tracing:

```typescript
import { Telemetry } from '@agentix-e/causality-analyzer-core';
import { trace } from '@opentelemetry/api';

Telemetry.init({
  tracer: {
    startSpan: (name, opts) => {
      const span = trace.getTracer('causality-analyzer').startSpan(name, opts);
      return {
        name, end: () => span.end(),
        setAttribute: (k, v) => span.setAttribute(k, v),
        addEvent: (n, a) => span.addEvent(n, a),
        recordException: (e) => span.recordException(e),
      };
    },
  },
});
```

### Health Checks

Kubernetes probes:
```yaml
livenessProbe:
  httpGet: { path: /live, port: 3000 }
readinessProbe:
  httpGet: { path: /ready, port: 3000 }
```

### Rate Limiting

Default: 100 requests/minute per IP. Configure:
```typescript
new CausalityServer({ rateLimitMax: 500, rateLimitWindowMs: 60000 });
```

## Backup & Disaster Recovery

### PostgreSQL Backup
```bash
pg_dump $DATABASE_URL > causality-backup-$(date +%Y%m%d).sql
```

### Neo4j Backup
```bash
neo4j-admin database dump --to-path=/backups neo4j
```

### Disaster Recovery Checklist
1. Restore PostgreSQL from backup
2. Restore Neo4j from backup
3. Verify with `curl http://localhost:3000/health`
4. Run benchmark parity: `pnpm test -- src/__tests__/i68-benchmark-parity.test.ts`

## Zero-Downtime Rolling Update

1. Start new instance on different port
2. Wait for `/ready` to return 200
3. Update load balancer to point to new instance
4. Drain old instance (wait 30s for in-flight requests)
5. Stop old instance

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `CAUSALITY_API_TOKEN` | No | Bearer token for API auth |
| `DATABASE_URL` | No | PostgreSQL connection string |
| `NEO4J_BOLT_URI` | No | Neo4j Bolt URI |
| `PG_CLIENT_CERT` | No | mTLS client certificate (PEM) |
| `PG_CLIENT_KEY` | No | mTLS client key (PEM) |
| `PG_CA_CERT` | No | mTLS CA certificate (PEM) |
| `AUDIT_HMAC_KEY` | No | HMAC key for audit trail signing |
| `ENCRYPTION_KEY` | No | 32-byte AES-256 key (hex encoded) |
