#!/usr/bin/env bash
# generate-neo4j-certs.sh — Generate self-signed TLS certificates for Neo4j mTLS testing.
#
# Creates a complete PKI in /tmp/neo4j-certs/:
#   ca.pem          — Certificate Authority (self-signed root)
#   ca.key          — CA private key
#   server.pem      — Server certificate (for Neo4j, CN=localhost)
#   server.key      — Server private key
#   client.pem      — Client certificate (for causality-analyzer)
#   client.key      — Client private key
#   neo4j/          — Certificates in Neo4j-expected directory layout
#
# Usage:
#   bash packages/storage-remote/scripts/generate-neo4j-certs.sh
#
# After generation, start Neo4j with:
#   docker run -d --rm --name neo4j-test \
#     -p 7687:7687 \
#     -v /tmp/neo4j-certs/neo4j:/var/lib/neo4j/certificates \
#     -e NEO4J_AUTH=neo4j/password \
#     -e NEO4J_dbms_connector_bolt_tls__level=OPTIONAL \
#     neo4j:5-community
#
# Run mTLS tests with:
#   NEO4J_BOLT_URI=bolt+ssc://localhost:7687 \
#   NEO4J_MTLS_CA_FILE=/tmp/neo4j-certs/ca.pem \
#   NEO4J_MTLS_CERT_FILE=/tmp/neo4j-certs/client.pem \
#   NEO4J_MTLS_KEY_FILE=/tmp/neo4j-certs/client.key \
#   pnpm --filter @agentix-e/causality-analyzer-storage-remote test

set -euo pipefail

CERT_DIR="${CERT_DIR:-/tmp/neo4j-certs}"
CA_KEY="${CERT_DIR}/ca.key"
CA_CERT="${CERT_DIR}/ca.pem"
SERVER_KEY="${CERT_DIR}/server.key"
SERVER_CERT="${CERT_DIR}/server.pem"
SERVER_CSR="${CERT_DIR}/server.csr"
CLIENT_KEY="${CERT_DIR}/client.key"
CLIENT_CERT="${CERT_DIR}/client.pem"
CLIENT_CSR="${CERT_DIR}/client.csr"
NEO4J_CERT_DIR="${CERT_DIR}/neo4j/https"
DAYS=365

mkdir -p "${CERT_DIR}" "${NEO4J_CERT_DIR}"

echo "=== Generating CA certificate ==="
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "${CA_KEY}" -out "${CA_CERT}" -days "${DAYS}" \
  -subj "/CN=Neo4j Test CA/O=Causality Analyzer/C=US" \
  -addext "basicConstraints=critical,CA:TRUE" \
  -addext "keyUsage=critical,keyCertSign,cRLSign"

echo "=== Generating server certificate for Neo4j ==="
openssl req -newkey rsa:2048 -nodes \
  -keyout "${SERVER_KEY}" -out "${SERVER_CSR}" \
  -subj "/CN=localhost/O=Causality Analyzer/C=US"

openssl x509 -req -in "${SERVER_CSR}" \
  -CA "${CA_CERT}" -CAkey "${CA_KEY}" -CAcreateserial \
  -out "${SERVER_CERT}" -days "${DAYS}" \
  -extfile <(printf "subjectAltName=DNS:localhost,DNS:neo4j-test,IP:127.0.0.1\nextendedKeyUsage=serverAuth")

echo "=== Generating client certificate for causality-analyzer ==="
openssl req -newkey rsa:2048 -nodes \
  -keyout "${CLIENT_KEY}" -out "${CLIENT_CSR}" \
  -subj "/CN=causality-analyzer/O=Causality Analyzer/C=US"

openssl x509 -req -in "${CLIENT_CSR}" \
  -CA "${CA_CERT}" -CAkey "${CA_KEY}" -CAcreateserial \
  -out "${CLIENT_CERT}" -days "${DAYS}" \
  -extfile <(printf "subjectAltName=DNS:causality-analyzer\nextendedKeyUsage=clientAuth")

echo "=== Preparing Neo4j certificate directory ==="
# Neo4j expects certificates in /var/lib/neo4j/certificates/https/
#   public.crt — server certificate + CA certificate (concatenated chain)
#   private.key — server private key
#   trusted/ — CA certificate (client auth trust)
mkdir -p "${NEO4J_CERT_DIR}/trusted"
cat "${SERVER_CERT}" "${CA_CERT}" > "${NEO4J_CERT_DIR}/public.crt"
cp "${SERVER_KEY}" "${NEO4J_CERT_DIR}/private.key"
cp "${CA_CERT}" "${NEO4J_CERT_DIR}/trusted/root.crt"

# Set restrictive permissions on private keys
chmod 600 "${CA_KEY}" "${SERVER_KEY}" "${CLIENT_KEY}" \
  "${NEO4J_CERT_DIR}/private.key"

echo ""
echo "=== Certificates generated in ${CERT_DIR}/ ==="
echo "Files:"
ls -la "${CERT_DIR}"/
echo ""
echo "Neo4j certificate directory: ${NEO4J_CERT_DIR}/"
ls -la "${NEO4J_CERT_DIR}/"
echo ""
echo "=== Environment variables for tests ==="
echo "export NEO4J_BOLT_URI=bolt+ssc://localhost:7687"
echo "export NEO4J_USER=neo4j"
echo "export NEO4J_PASSWORD=password"
echo "export NEO4J_MTLS_CA_FILE=${CA_CERT}"
echo "export NEO4J_MTLS_CERT_FILE=${CLIENT_CERT}"
echo "export NEO4J_MTLS_KEY_FILE=${CLIENT_KEY}"
echo ""
echo "=== To verify server certificate ==="
echo "openssl s_client -connect localhost:7687 -CAfile ${CA_CERT} -cert ${CLIENT_CERT} -key ${CLIENT_KEY}"
