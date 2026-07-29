#!/usr/bin/env bash
# run-neo4j-tests.sh — Fully automated Neo4j integration test runner.
#
# Handles the complete lifecycle:
#   1. Generate self-signed TLS certificates (for mTLS mode)
#   2. Start Neo4j Docker container (plain mode, then mTLS mode)
#   3. Run both plain and mTLS test suites
#   4. Tear down container and clean up certs
#
# Usage:
#   pnpm neo4j:test         — run all Neo4j tests (plain + mTLS)
#   pnpm neo4j:test:plain   — run plain Neo4j tests only
#   pnpm neo4j:test:mtls    — run mTLS Neo4j tests only
#
# Requires: docker, openssl, pnpm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DIR="${SCRIPT_DIR}/../../scripts/generate-neo4j-certs.sh"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NEO4J_PORT=7687
NEO4J_USER=neo4j
NEO4J_PASS=password
CONTAINER_NAME="ca-neo4j-test-$$"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

cleanup() {
    echo -e "${YELLOW}=== Cleaning up Neo4j container ${CONTAINER_NAME} ===${NC}"
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    echo -e "${GREEN}=== Cleanup complete ===${NC}"
}
trap cleanup EXIT INT TERM

wait_neo4j() {
    local uri="${1:-bolt://localhost:${NEO4J_PORT}}"
    echo -n "Waiting for Neo4j at ${uri}..."
    for i in $(seq 1 60); do
        if docker exec "${CONTAINER_NAME}" neo4j status 2>/dev/null | grep -q "Neo4j is running"; then
            echo -e " ${GREEN}ready after ${i}s${NC}"
            return 0
        fi
        sleep 1
    done
    echo -e " ${RED}timeout${NC}"
    docker logs --tail=30 "${CONTAINER_NAME}"
    return 1
}

run_plain_tests() {
    echo -e "\n${GREEN}=== Running plain (no-TLS) Neo4j tests ===${NC}"
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    docker run -d --rm --name "${CONTAINER_NAME}" \
        -p "${NEO4J_PORT}:7687" \
        -e "NEO4J_AUTH=${NEO4J_USER}/${NEO4J_PASS}" \
        neo4j:5-community 2>&1 | tail -1

    wait_neo4j

    cd "${PROJECT_ROOT}"
    NEO4J_BOLT_URI="bolt://localhost:${NEO4J_PORT}" \
    NEO4J_USER="${NEO4J_USER}" \
    NEO4J_PASSWORD="${NEO4J_PASS}" \
    pnpm run --filter @agentix-e/causality-analyzer-storage-remote test \
        -- --testPathPattern='remote-graph-store-neo4j' 2>&1 | tail -20
    echo -e "${GREEN}=== Plain tests complete ===${NC}"
}

run_mtls_tests() {
    echo -e "\n${GREEN}=== Generating mTLS certificates ===${NC}"
    bash "${CERT_DIR}"
    CERT_PATH="/tmp/neo4j-certs"

    echo -e "${GREEN}=== Starting Neo4j with mTLS ===${NC}"
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    docker run -d --rm --name "${CONTAINER_NAME}" \
        -p "${NEO4J_PORT}:7687" \
        -v "${CERT_PATH}/neo4j/https:/var/lib/neo4j/certificates/https" \
        -v "${CERT_PATH}/neo4j/https:/var/lib/neo4j/certificates/bolt" \
        -e "NEO4J_AUTH=${NEO4J_USER}/${NEO4J_PASS}" \
        -e NEO4J_dbms_connector_bolt_tls__level=OPTIONAL \
        -e NEO4J_dbms_ssl_policy_bolt_client__auth=NONE \
        -e NEO4J_dbms_ssl_policy_bolt_enabled=true \
        -e NEO4J_dbms_ssl_policy_bolt_base__directory=/var/lib/neo4j/certificates/bolt \
        -e NEO4J_dbms_ssl_policy_bolt_trusted__dir=/var/lib/neo4j/certificates/bolt/trusted \
        -e NEO4J_dbms_ssl_policy_bolt_revoked__dir=/var/lib/neo4j/certificates/bolt/revoked \
        -e NEO4J_dbms_ssl_policy_bolt_private__key=/var/lib/neo4j/certificates/bolt/private.key \
        -e NEO4J_dbms_ssl_policy_bolt_public__certificate=/var/lib/neo4j/certificates/bolt/public.crt \
        neo4j:5-community 2>&1 | tail -1

    wait_neo4j

    echo -e "${GREEN}=== Running mTLS Neo4j tests ===${NC}"
    cd "${PROJECT_ROOT}"
    NEO4J_BOLT_URI="bolt+ssc://localhost:${NEO4J_PORT}" \
    NEO4J_USER="${NEO4J_USER}" \
    NEO4J_PASSWORD="${NEO4J_PASS}" \
    NEO4J_MTLS_CA_FILE="${CERT_PATH}/ca.pem" \
    NEO4J_MTLS_CERT_FILE="${CERT_PATH}/client.pem" \
    NEO4J_MTLS_KEY_FILE="${CERT_PATH}/client.key" \
    pnpm run --filter @agentix-e/causality-analyzer-storage-remote test \
        -- --testPathPattern='remote-graph-store-neo4j' 2>&1 | tail -20
    echo -e "${GREEN}=== mTLS tests complete ===${NC}"
}

# Parse command
case "${1:-all}" in
    plain)
        run_plain_tests
        ;;
    mtls)
        run_mtls_tests
        ;;
    all)
        run_plain_tests
        run_mtls_tests
        ;;
    *)
        echo "Usage: $0 {plain|mtls|all}"
        echo ""
        echo "  plain  — Run Neo4j tests without TLS"
        echo "  mtls   — Run Neo4j tests with mutual TLS"
        echo "  all    — Run both (default)"
        exit 1
        ;;
esac

echo -e "\n${GREEN}=== All Neo4j integration tests passed ===${NC}"
