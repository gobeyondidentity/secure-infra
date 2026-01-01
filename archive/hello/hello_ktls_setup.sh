#!/bin/bash
#
# kTLS Setup Script
# Generates self-signed certificates for testing mTLS
#

set -e

echo "================================="
echo "  kTLS Certificate Setup"
echo "================================="
echo ""

# Create certificates directory
mkdir -p certs
cd certs

echo "[1/5] Generating CA private key..."
openssl genrsa -out ca-key.pem 2048 2>/dev/null

echo "[2/5] Generating CA certificate..."
openssl req -new -x509 -days 365 -key ca-key.pem -out ca-cert.pem \
    -subj "/C=US/ST=CA/L=SantaClara/O=TestCA/CN=Test CA" 2>/dev/null

echo "[3/5] Generating server private key..."
openssl genrsa -out server-key.pem 2048 2>/dev/null

echo "[4/5] Generating server certificate..."
openssl req -new -key server-key.pem -out server-req.pem \
    -subj "/C=US/ST=CA/L=SantaClara/O=TestOrg/CN=localhost" 2>/dev/null
openssl x509 -req -days 365 -in server-req.pem -CA ca-cert.pem -CAkey ca-key.pem \
    -CAcreateserial -out server-cert.pem 2>/dev/null

echo "[5/5] Generating client certificate..."
openssl genrsa -out client-key.pem 2048 2>/dev/null
openssl req -new -key client-key.pem -out client-req.pem \
    -subj "/C=US/ST=CA/L=SantaClara/O=TestOrg/CN=Test Client" 2>/dev/null
openssl x509 -req -days 365 -in client-req.pem -CA ca-cert.pem -CAkey ca-key.pem \
    -CAcreateserial -out client-cert.pem 2>/dev/null

# Copy certificates to parent directory for convenience
cp ca-cert.pem ../
cp server-cert.pem ../
cp server-key.pem ../
cp client-cert.pem ../
cp client-key.pem ../

cd ..

echo ""
echo "✓ Certificates generated successfully!"
echo ""
echo "Generated files:"
echo "  ca-cert.pem       - CA certificate (trust anchor)"
echo "  server-cert.pem   - Server certificate"
echo "  server-key.pem    - Server private key"
echo "  client-cert.pem   - Client certificate"
echo "  client-key.pem    - Client private key"
echo ""
echo "All certificates are in ./certs/ (backup) and ./ (for programs)"
echo ""
echo "================================="
echo "Next steps:"
echo "  1. make -f Makefile.ktls"
echo "  2. Terminal 1: ./hello_ktls_server"
echo "  3. Terminal 2: ./hello_ktls_client"
echo "================================="
