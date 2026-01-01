# kTLS + OpenSSL Hello World

## Overview

This example demonstrates **kernel TLS (kTLS) offload** with **mTLS client certificate authentication** on NVIDIA ConnectX-7 NICs. After the TLS handshake completes, session keys can be passed to the kernel for hardware crypto acceleration.

## What is kTLS?

**Kernel TLS (kTLS)** is a Linux kernel feature that offloads TLS encryption/decryption to the network interface card. Benefits include:

- **Hardware crypto acceleration**: ConnectX-7 performs AES-GCM encryption/decryption at line rate
- **Zero CPU overhead**: Crypto operations offloaded from host CPU
- **Standard socket API**: Applications use normal `read()`/`write()` calls after handshake
- **TLS 1.2 and 1.3 support**: Modern TLS protocols supported

## Hardware Requirements

**Supported on:**
- ✅ NVIDIA ConnectX-6 Dx and newer (including ConnectX-7)
- ✅ NVIDIA Bluefield-2 and Bluefield-3 DPUs

**DGX Spark Configuration:**
- 4x ConnectX-7 logical ports (2x dual-port NICs)
- kTLS TX offload: **ENABLED** by default
- kTLS RX offload: Can be enabled via `ethtool`

## Architecture

### Traditional TLS Stack
```
Application
    ↓
OpenSSL (TLS handshake + crypto)
    ↓
TCP Socket
    ↓
Network Driver
    ↓
NIC (data only)
```

### kTLS Stack
```
Application
    ↓
OpenSSL (TLS handshake only)
    ↓ (pass session keys)
Kernel TLS Layer ← crypto offload
    ↓
TCP Socket
    ↓
Network Driver
    ↓
ConnectX-7 NIC (hardware crypto)
```

## Implementation Flow

### 1. TLS Handshake (OpenSSL in Userspace)

**Server:**
```c
SSL_CTX *ctx = SSL_CTX_new(TLS_server_method());
SSL_CTX_use_certificate_file(ctx, "server-cert.pem", SSL_FILETYPE_PEM);
SSL_CTX_use_PrivateKey_file(ctx, "server-key.pem", SSL_FILETYPE_PEM);

// Require client certificate
SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);
SSL_CTX_load_verify_locations(ctx, "ca-cert.pem", NULL);

// Accept connection
SSL *ssl = SSL_new(ctx);
SSL_set_fd(ssl, client_fd);
SSL_accept(ssl);  // TLS handshake
```

**Client:**
```c
SSL_CTX *ctx = SSL_CTX_new(TLS_client_method());

// Load client certificate for mTLS
SSL_CTX_use_certificate_file(ctx, "client-cert.pem", SSL_FILETYPE_PEM);
SSL_CTX_use_PrivateKey_file(ctx, "client-key.pem", SSL_FILETYPE_PEM);

// Verify server certificate
SSL_CTX_load_verify_locations(ctx, "ca-cert.pem", NULL);
SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, NULL);

// Connect
SSL *ssl = SSL_new(ctx);
SSL_set_fd(ssl, sock_fd);
SSL_connect(ssl);  // TLS handshake
```

### 2. Enable kTLS Offload (After Handshake)

**Extract session keys from OpenSSL:**
```c
#include <linux/tls.h>

struct tls12_crypto_info_aes_gcm_128 crypto_info;
memset(&crypto_info, 0, sizeof(crypto_info));

crypto_info.info.version = TLS_1_2_VERSION;
crypto_info.info.cipher_type = TLS_CIPHER_AES_GCM_128;

// Extract IV, keys, sequence number from SSL session
// (Implementation varies by OpenSSL version)

// Pass to kernel for TX offload
setsockopt(sock_fd, SOL_TLS, TLS_TX, &crypto_info, sizeof(crypto_info));

// Enable RX offload
setsockopt(sock_fd, SOL_TLS, TLS_RX, &crypto_info, sizeof(crypto_info));
```

### 3. Use Standard Socket I/O (Crypto Offloaded)

After kTLS is enabled, use standard socket calls:
```c
// Write data (encrypted by ConnectX-7 hardware)
write(sock_fd, data, len);

// Read data (decrypted by ConnectX-7 hardware)
read(sock_fd, buffer, len);

// Or use sendfile() for zero-copy
sendfile(sock_fd, file_fd, NULL, file_size);
```

## Running the Example

### Build Programs
```bash
make -f Makefile.ktls
```

### Generate Test Certificates
```bash
bash hello_ktls_setup.sh
```

This creates:
- `ca-cert.pem` - Certificate Authority (trust anchor)
- `server-cert.pem` / `server-key.pem` - Server credentials
- `client-cert.pem` / `client-key.pem` - Client credentials

### Run Server (Terminal 1)
```bash
./hello_ktls_server
```

Expected output:
```
=================================
  kTLS + OpenSSL Server
  mTLS with Hardware Offload
=================================

Loading server certificate: server-cert.pem
Loading server key: server-key.pem
Configuring client certificate verification

Server listening on port 8443
Waiting for client connection...
```

### Run Client (Terminal 2)
```bash
./hello_ktls_client
```

Expected output:
```
=================================
  kTLS + OpenSSL Client
  mTLS with Hardware Offload
=================================

Loading client certificate: client-cert.pem
Loading client key: client-key.pem
Loading CA certificate: ca-cert.pem

Connecting to 127.0.0.1:8443...
TCP connection established
Performing TLS handshake...
TLS handshake completed successfully
Protocol: TLSv1.3
Cipher: TLS_AES_256_GCM_SHA384

=== Server Certificate ===
Subject: /C=US/ST=CA/L=SantaClara/O=TestOrg/CN=localhost
Issuer: /C=US/ST=CA/L=SantaClara/O=TestCA/CN=Test CA
Certificate verified: YES

Sending message to server: Hello from kTLS client! I have a valid certificate.
Sent 51 bytes

Received from server: Hello from kTLS server! Your certificate was validated.
```

### Server Output (After Client Connects)
```
Client connected from 127.0.0.1:53554
Performing TLS handshake...
TLS handshake completed successfully
Protocol: TLSv1.3

=== Client Certificate ===
Subject: /C=US/ST=CA/L=SantaClara/O=TestOrg/CN=Test Client
Issuer: /C=US/ST=CA/L=SantaClara/O=TestCA/CN=Test CA
Certificate verified: YES

Received from client: Hello from kTLS client! I have a valid certificate.
Sent response to client
```

## Verifying kTLS Offload

### Check Interface Support
```bash
# Check current kTLS settings
ethtool -k enp1s0f0np0 | grep tls

# Output on ConnectX-7:
# tls-hw-tx-offload: on       ← Hardware TX offload enabled
# tls-hw-rx-offload: off      ← Can be enabled
# tls-hw-record: off [fixed]
```

### Enable RX Offload
```bash
sudo ethtool -K enp1s0f0np0 tls-hw-rx-offload on
```

### Monitor Offload Statistics (During Active Connection)
```bash
# Run while TLS connection is active
ethtool -S enp1s0f0np0 | grep tls

# Expected output (example):
# tls_tx_packets: 142
# tls_tx_bytes: 7234
# tls_rx_packets: 0
# tls_rx_bytes: 0
```

### Verify Hardware Acceleration
```bash
# Monitor CPU usage during large transfer
# With kTLS: CPU usage should be minimal
# Without kTLS: CPU performs encryption (high usage)

# Example test:
# 1. Start server with kTLS enabled
# 2. Send large file (1GB+)
# 3. Monitor: htop or mpstat
# 4. Compare CPU usage with/without kTLS
```

## Beyond Identity Integration Use Case

### mTLS Client Certificate Enforcement

**Flow:**
1. **User authentication**: User authenticates to Beyond Identity using passwordless passkey
2. **Certificate issuance**: Beyond Identity issues short-lived X.509 client certificate (API call)
3. **Key binding**: Private key bound to device TPM/secure enclave (never exported)
4. **Connection**: Client connects to service with client certificate
5. **kTLS offload**: After handshake, ConnectX-7 performs hardware crypto
6. **Certificate validation**: Server validates certificate, checks OCSP/CRL status
7. **Policy enforcement**: Map certificate claims (subject, OU, device ID) to RBAC policies
8. **Revocation**: Short cert lifetime (minutes–hours) + automated OCSP/CRL

**Benefits:**
- Hardware crypto offload (zero CPU overhead)
- Phishing-resistant authentication (device-bound keys in TPM)
- Line-rate enforcement (200+ Gbps)
- Short-lived certificates reduce blast radius

**Limitations (ConnectX-7 vs Bluefield DPU):**
- ⚠️ Control plane runs on host CPU (not isolated)
- ⚠️ No hardware attestation (DICE/SPDM requires Bluefield)
- ✅ Data plane crypto offload works (kTLS)
- ✅ Certificate validation in application layer

## Implementation Notes

### Current Example Status

**✅ Working:**
- mTLS handshake with client certificate
- Certificate verification (server ↔ client)
- TLS 1.2 and 1.3 support
- Message exchange over secure channel

**⚠️ Simplified (Not Production-Ready):**
- kTLS offload **prepared but not fully enabled**
- Session key extraction from OpenSSL requires additional code
- Example uses self-signed certificates (not Beyond Identity-issued)

**Why kTLS offload is simplified:**
- Session key extraction is OpenSSL version-dependent
- Requires accessing internal SSL session state
- TLS 1.3 has different key derivation than TLS 1.2
- Full implementation needs 100+ lines of key material handling

### Production Implementation Requirements

**For full kTLS offload:**
1. Extract crypto info from SSL session (IV, keys, sequence)
2. Call `setsockopt(SOL_TLS, TLS_TX, ...)` after handshake
3. Call `setsockopt(SOL_TLS, TLS_RX, ...)` for bidirectional offload
4. Handle key updates (TLS 1.3 rekey events)
5. Monitor offload statistics via `ethtool -S`

**For Beyond Identity integration:**
1. Replace certificate generation with Beyond Identity API calls
2. Implement certificate lifecycle management (issuance, renewal, revocation)
3. Add OCSP/CRL checking during certificate validation
4. Map certificate claims to RBAC policies
5. Integrate with centralized audit logging (SIEM)

## Performance Characteristics

### kTLS Offload Benefits

**Throughput:**
- Without kTLS: Limited by CPU crypto performance (~10-20 Gbps per core)
- With kTLS: Line rate (200 Gbps on ConnectX-7, limited by NIC bandwidth)

**Latency:**
- Hardware crypto adds <1μs overhead
- Eliminates context switches to userspace for crypto operations

**CPU Utilization:**
- Without kTLS: 100% CPU for sustained crypto workload
- With kTLS: <5% CPU (only socket I/O, no crypto)

**Power Efficiency:**
- Dedicated crypto hardware more efficient than general-purpose CPU
- Allows CPU to sleep or handle other workloads

## Supported Ciphers

**kTLS requires specific cipher suites:**

**TLS 1.2:**
- `TLS_CIPHER_AES_GCM_128` ✅ Widely supported
- `TLS_CIPHER_AES_GCM_256` ✅ Supported on newer kernels
- `TLS_CIPHER_AES_CCM_128` ⚠️ Limited support

**TLS 1.3:**
- `TLS_AES_128_GCM_SHA256` ✅ Supported
- `TLS_AES_256_GCM_SHA384` ✅ Supported

**Not supported:**
- ChaCha20-Poly1305 (no hardware acceleration)
- AES-CBC (deprecated for TLS)

## Troubleshooting

### "Failed to load client certificate"
**Solution:** Run `bash hello_ktls_setup.sh` to generate certificates

### "TLS handshake failed"
**Check:**
- Server is running: `ps aux | grep hello_ktls_server`
- Port is accessible: `netstat -an | grep 8443`
- Certificates are valid: `openssl x509 -in client-cert.pem -text -noout`

### "Certificate verified: NO"
**Causes:**
- CA certificate mismatch (client/server using different CAs)
- Certificate expired
- Hostname mismatch (CN doesn't match server hostname)

**Solution:**
- Regenerate certificates: `make -f Makefile.ktls distclean && bash hello_ktls_setup.sh`
- Check certificate validity: `openssl verify -CAfile ca-cert.pem client-cert.pem`

### "kTLS offload not working"
**Check:**
1. Interface supports kTLS: `ethtool -k enp1s0f0np0 | grep tls`
2. Correct cipher negotiated: Must be AES-GCM
3. Kernel version: kTLS requires Linux 4.13+ (TX), 4.17+ (RX)
4. Offload enabled: `sudo ethtool -K enp1s0f0np0 tls-hw-tx-offload on`

## References

- [Linux Kernel TLS Documentation](https://www.kernel.org/doc/html/latest/networking/tls-offload.html)
- [NVIDIA kTLS Offload Guide](https://docs.nvidia.com/doca/sdk/ktls+offloads/index.html)
- [OpenSSL Documentation](https://www.openssl.org/docs/)
- [Beyond Identity Developer Docs](https://developer.beyondidentity.com/)

## Next Steps

**For production deployment:**
1. **Implement full kTLS offload** (session key extraction)
2. **Integrate Beyond Identity API** for certificate issuance
3. **Add OCSP/CRL checking** for real-time revocation
4. **Deploy on real network** (not localhost)
5. **Load test** at line rate to verify hardware offload
6. **Monitor statistics** with `ethtool -S` during production traffic

**For Bluefield DPU deployment:**
- Migrate control plane to DPU ARM cores (isolated from host)
- Add DICE/SPDM attestation for hardware root of trust
- Use DOCA TLS library instead of kernel kTLS
- Implement inline policy enforcement at DPU data plane
