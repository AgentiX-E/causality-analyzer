# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x.x   | ✅ Security fixes  |
| 0.x.x   | ❌ End of life     |

## Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

Instead, report vulnerabilities via email to security@agentix-e.dev.

### What to Expect

1. **Acknowledgment**: Within 48 hours of receipt, you will receive a confirmation with a tracking ID.
2. **Triage**: Within 5 business days, we will assess severity and determine a remediation timeline.
3. **Resolution**: Critical vulnerabilities are targeted for fix within 7 days. Non-critical within 30 days.
4. **Disclosure**: A security advisory will be published on GitHub after the fix is released (max 90 days from report). Credit will be given to reporters who request it.

### Scope

Causality Analyzer packages:
- `@agentix-e/causality-analyzer-core`
- `@agentix-e/causality-analyzer-pipeline`
- `@agentix-e/causality-analyzer-storage-embed`
- `@agentix-e/causality-analyzer-storage-remote`
- `@agentix-e/causality-analyzer-visual`

### Out of Scope

- Issues in transitive dependencies (report to the upstream project)
- Denial of Service attacks requiring unrealistic resource consumption
- Issues requiring physical access to the host machine

## Security Features

Causality Analyzer provides these built-in security capabilities:

- **mTLS**: Full mTLS support on both Bolt (Neo4j) and PG-wire (PostgreSQL) via PEM-string configuration
- **Audit Trail**: SHA-256 hash-chained tamper-evident audit log (Merkle-Damgård, RFC 6962 pattern)
- **Encrypted Store**: AES-256-GCM encryption for persisted causal analysis results
- **Rate Limiter**: Overflow protection with drop_oldest/drop_newest/block strategies

## Responsible Development

- All changes pass `pnpm audit --audit-level high` in CI
- CodeQL security scanning runs on every push to main
- No hardcoded credentials, tokens, or API keys in the repository
- All dependencies are pinned via `pnpm-lock.yaml` with integrity hashes
