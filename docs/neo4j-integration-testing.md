# Neo4j Integration Testing Guide

## Overview

The `@agentix-e/causality-analyzer-storage-remote` package includes integration
tests against a real Neo4j instance. Two transport modes are supported:

| Mode | URI Scheme | Authentication | When to use |
|------|-----------|---------------|-------------|
| **Plain** | `bolt://` | Username/password | Local dev, CI without TLS |
| **mTLS** | `bolt+ssc://` | Client certificate | Production-like environments |

## Quick Start — Fully Automated

```bash
# Run all Neo4j tests (plain + mTLS) with Docker auto-management:
pnpm neo4j:test

# Plain TLS only:
pnpm neo4j:test:plain

# mTLS only (generates certs automatically):
pnpm neo4j:test:mtls
```

The script `scripts/run-neo4j-tests.sh` handles the complete lifecycle:
1. Generates self-signed TLS certificates (for mTLS mode)
2. Starts a Neo4j 5 Docker container
3. Waits for Neo4j to become ready
4. Runs the integration test suite
5. Tears down the container and cleans up

## Manual Setup

### Prerequisites

- Docker
- OpenSSL (for mTLS cert generation)
- pnpm

### Plain Mode (No TLS)

```bash
# Start Neo4j
docker run -d --rm --name neo4j-test \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5-community

# Run tests
NEO4J_BOLT_URI=bolt://localhost:7687 \
NEO4J_USER=neo4j \
NEO4J_PASSWORD=password \
pnpm --filter @agentix-e/causality-analyzer-storage-remote test

# Clean up
docker rm -f neo4j-test
```

### mTLS Mode

```bash
# 1. Generate certificates
bash packages/storage-remote/scripts/generate-neo4j-certs.sh
# Certificates are created in /tmp/neo4j-certs/

# 2. Start Neo4j with TLS
docker run -d --rm --name neo4j-test \
  -p 7687:7687 \
  -v /tmp/neo4j-certs/neo4j/https:/var/lib/neo4j/certificates/https \
  -v /tmp/neo4j-certs/neo4j/https:/var/lib/neo4j/certificates/bolt \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_dbms_connector_bolt_tls__level=OPTIONAL \
  -e NEO4J_dbms_ssl_policy_bolt_client__auth=NONE \
  -e NEO4J_dbms_ssl_policy_bolt_enabled=true \
  -e NEO4J_dbms_ssl_policy_bolt_base__directory=/var/lib/neo4j/certificates/bolt \
  -e NEO4J_dbms_ssl_policy_bolt_trusted__dir=/var/lib/neo4j/certificates/bolt/trusted \
  -e NEO4J_dbms_ssl_policy_bolt_private__key=/var/lib/neo4j/certificates/bolt/private.key \
  -e NEO4J_dbms_ssl_policy_bolt_public__certificate=/var/lib/neo4j/certificates/bolt/public.crt \
  neo4j:5-community

# 3. Run tests with mTLS
NEO4J_BOLT_URI=bolt+ssc://localhost:7687 \
NEO4J_USER=neo4j \
NEO4J_PASSWORD=password \
NEO4J_MTLS_CA_FILE=/tmp/neo4j-certs/ca.pem \
NEO4J_MTLS_CERT_FILE=/tmp/neo4j-certs/client.pem \
NEO4J_MTLS_KEY_FILE=/tmp/neo4j-certs/client.key \
pnpm --filter @agentix-e/causality-analyzer-storage-remote test

# 4. Clean up
docker rm -f neo4j-test
```

## Certificate Generation

The `scripts/generate-neo4j-certs.sh` script creates a complete PKI:

```
/tmp/neo4j-certs/
├── ca.key              # CA private key
├── ca.pem              # CA certificate (self-signed root)
├── server.key          # Neo4j server private key
├── server.pem          # Neo4j server certificate (signed by CA)
├── server.csr          # Server certificate signing request
├── client.key          # Client private key
├── client.pem          # Client certificate (signed by CA)
├── client.csr          # Client certificate signing request
└── neo4j/
    └── https/
        ├── public.crt   # Server cert + CA cert (concatenated)
        ├── private.key  # Server private key
        └── trusted/
            └── root.crt # CA cert (trusted for client auth)
```

- All certificates valid for 365 days
- Server cert has SAN: DNS:localhost, DNS:neo4j-test, IP:127.0.0.1
- Client cert has SAN: DNS:causality-analyzer
- CA has basicConstraints CA:TRUE for proper chain validation

## Environment Variables

| Variable | Required | Description |
|----------|:---:|-------------|
| `NEO4J_BOLT_URI` | Yes | Neo4j Bolt URI (`bolt://` or `bolt+ssc://`) |
| `NEO4J_USER` | No | Username (default: `neo4j`) |
| `NEO4J_PASSWORD` | No | Password (default: `password`) |
| `NEO4J_MTLS_CA_FILE` | mTLS only | Path to CA certificate PEM file |
| `NEO4J_MTLS_CERT_FILE` | mTLS only | Path to client certificate PEM file |
| `NEO4J_MTLS_KEY_FILE` | mTLS only | Path to client private key PEM file |

## Test Suite

The test file `src/__tests__/remote-graph-store-neo4j.test.ts` covers:

| Test | Category | Description |
|------|----------|-------------|
| connect | Basic | Verifies connection to Neo4j |
| round-trip | CRUD | saveGraph → loadGraph preserves nodes and edges |
| edge fidelity | CRUD | Edge weight and direction are preserved |
| unknown ID | Edge case | `loadGraph` returns null for nonexistent IDs |
| versioned storage | History | Multiple graph versions accessible |
| list versions | History | Version count and monotonic ordering |
| latest version | History | `loadGraph` returns the latest version |
| similarity search | Query | `findSimilarGraphs` with structural fingerprinting |
| large graph | Scale | 50-node, 90+ edge graph round-trip |
| empty graph | Edge case | Graph with zero nodes and edges |

All tests run in both plain and mTLS mode.

## CI Integration

For GitHub Actions, use the service container pattern:

```yaml
jobs:
  neo4j-test:
    runs-on: ubuntu-latest
    services:
      neo4j:
        image: neo4j:5-community
        env:
          NEO4J_AUTH: neo4j/testpassword
        ports:
          - 7687:7687
    steps:
      - run: pnpm neo4j:test
        env:
          NEO4J_BOLT_URI: bolt://localhost:7687
```

For mTLS in CI, add cert generation before starting the service container.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ECONNREFUSED` | Ensure Neo4j Docker container is running and port 7687 is mapped |
| `UNABLE_TO_VERIFY_LEAF_SIGNATURE` | Check that `NEO4J_MTLS_CA_FILE` points to the correct CA cert |
| `certificate has expired` | Re-run `bash scripts/generate-neo4j-certs.sh` |
| `Neo4j is running` timeout | Check Docker logs: `docker logs neo4j-test` |
| mTLS tests skipped | Verify all mTLS env vars are set and certificate files exist |
