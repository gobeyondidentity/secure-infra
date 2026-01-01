# DOCA Hardware Capabilities: ConnectX-7 vs Bluefield DPU

## Executive Summary

NVIDIA DOCA SDK libraries have different support levels across hardware platforms. This document clarifies which capabilities are available on **ConnectX-7 NICs** (DGX Spark) vs **Bluefield DPUs**.

## ConnectX-7 NICs (Available on Spark)

### Supported DOCA Capabilities

**✅ DOCA Core Libraries:**
- `doca_dev` - Device enumeration and management
- `doca_buf` - Buffer management
- `doca_mmap` - Memory mapping
- `doca_ctx` - Context management
- `doca_pe` - Progress engine

**✅ DOCA Flow (with limitations):**
- Hardware-accelerated packet processing
- **Requires**: DPDK environment (EAL initialization, huge pages, driver binding)
- **Use case**: Packet matching, filtering, forwarding at line rate

**✅ DOCA RDMA:**
- Remote Direct Memory Access
- GPUDirect RDMA support

**✅ Kernel-Level Crypto Offload (not DOCA SDK):**
- kTLS offload (Tx and Rx)
- IPsec inline encryption/decryption
- MACsec
- **Configuration**: Via `ethtool`, kernel sockets, not DOCA API

### NOT Supported on ConnectX-7

**❌ DOCA Crypto Libraries:**
- `doca_sha` - Requires Bluefield-2 DPU SHA acceleration engine
- `doca_aes_gcm` - Requires DPU crypto engine
- Other DOCA crypto primitives

**❌ DPU-Specific Libraries:**
- `doca_dpa` - Data Path Accelerator (DPU only)
- DOCA Services (Firefly, DMS, Comch)
- App Shield (host introspection from DPU)
- Network infrastructure services (BGP, EVPN, VXLAN offload)

**❌ Isolated ARM Control Plane:**
- ConnectX-7 has no ARM cores
- All processing happens on host CPU (not hardware-isolated)

## Bluefield-2 DPU

### All ConnectX-7 Capabilities PLUS:

**✅ DOCA Crypto:**
- `doca_sha` - Hardware SHA-1, SHA-256, SHA-512 acceleration
- `doca_aes_gcm` - Hardware AES-GCM encryption/decryption
- Dedicated crypto acceleration engine

**✅ ARM Compute Cores:**
- 8x ARM A72 cores @ 2.5 GHz
- Isolated control plane (separate from host OS)
- Run DOCA applications directly on DPU

**✅ DOCA Services:**
- Firefly - Service mesh enforcement
- DMS - DOCA Management Service
- Comch - Host-DPU communication channel

**✅ Network Infrastructure:**
- Hardware-accelerated VXLAN, BGP, EVPN
- OVS offload

**✅ App Shield:**
- Host memory introspection from DPU
- Security telemetry and enforcement

**✅ Attestation:**
- DICE/SPDM firmware attestation
- Customer certificate slots for trust anchors

## Bluefield-3 DPU

### All Bluefield-2 Capabilities EXCEPT:

**❌ DOCA SHA:**
- SHA acceleration engine **removed** in Bluefield-3
- Official NVIDIA documentation: "NVIDIA BlueField-3 does not support this library because it has no SHA acceleration engine"

**✅ Other Crypto:**
- AES-GCM and other crypto primitives still supported
- TLS offload still available

**✅ Enhanced Compute:**
- 16x ARM A78 cores @ 3.3 GHz (vs 8x A72 in BF-2)
- More memory and PCIe bandwidth

## Beyond Identity Integration Implications

### Pattern A: mTLS Client Certificate Enforcement

**On ConnectX-7 (Spark):**
- ✅ Can use **kTLS** for mTLS offload
  - Configuration: Kernel sockets + `setsockopt(SOL_TLS, ...)`
  - Requires: OpenSSL/TLS library for handshake, then pass keys to kernel
  - NOT using DOCA SDK, using kernel TLS offload
- ✅ Can use **DOCA Flow** for policy enforcement (requires DPDK)
- ❌ Cannot use DOCA TLS library (designed for DPU control plane)

**On Bluefield DPU:**
- ✅ Full DOCA TLS library support
- ✅ Isolated ARM cores run TLS validation (immune to host compromise)
- ✅ Line-rate certificate validation with hardware crypto acceleration

### Pattern B: OIDC/JWT + Attestation

**On ConnectX-7 (Spark):**
- ✅ DOCA Flow for data plane enforcement (requires DPDK)
- ❌ No DPU attestation (no ARM cores, no DICE/SPDM)
- ⚠️ Control plane runs on host CPU (not isolated)

**On Bluefield DPU:**
- ✅ Full attestation support (DICE/SPDM, customer cert slots)
- ✅ Isolated control plane for token validation
- ✅ Hardware root of trust

## Recommended Development Path for Spark

### Phase 1: DOCA Core Exploration (Completed)
- ✅ Device enumeration (`hello_doca.c`) - **WORKING**

### Phase 2: Kernel TLS Offload (kTLS)
- **Goal**: Demonstrate mTLS with hardware crypto offload
- **Approach**: Use OpenSSL + kernel kTLS (not DOCA SDK)
- **Steps**:
  1. Create TLS client/server with OpenSSL
  2. Enable kTLS via `setsockopt(SOL_TLS, ...)`
  3. Pass session keys to kernel after handshake
  4. Verify crypto offload with `ethtool -S <interface> | grep tls`

### Phase 3: DOCA Flow Packet Processing
- **Goal**: Demonstrate packet matching and policy enforcement
- **Requires**: DPDK setup (huge pages, driver binding, EAL init)
- **Complexity**: High (full DPDK environment)
- **Alternative**: Document concepts, defer to Bluefield DPU testing

### Phase 4: Beyond Identity API Integration
- **Goal**: Certificate issuance and lifecycle management
- **Platform**: Can develop on any system (API calls, not hardware-dependent)

## Hardware Verification Commands

### Check kTLS Offload Support
```bash
# Enable Tx offload
sudo ethtool -K <interface> tls-hw-tx-offload on

# Enable Rx offload
sudo ethtool -K <interface> tls-hw-rx-offload on

# Verify offload is active
ethtool -k <interface> | grep tls

# Check offload statistics (during TLS connection)
ethtool -S <interface> | grep tls
```

### Check DOCA Device Capabilities
```bash
# Run device enumeration
./hello_doca

# Expected output: 4 devices on Spark (2x ConnectX-7 dual-port)
# - 0000:01:00.0 (enp1s0f0np0)
# - 0000:01:00.1 (enp1s0f1np1)
# - 0002:01:00.0 (enp129s0f0np0)
# - 0002:01:00.1 (enp129s0f1np1)
```

### Check for SHA Acceleration (will fail on ConnectX-7)
```bash
# This will fail on ConnectX-7
./hello_crypto

# Expected error: "No device found with SHA-256 support"
# Reason: ConnectX-7 has no SHA acceleration engine
```

## References

- [NVIDIA DOCA SHA Documentation](https://docs.nvidia.com/doca/sdk/doca+sha/index.html) - "NVIDIA BlueField-3 does not support this library because it has no SHA acceleration engine"
- [NVIDIA kTLS Offload Guide](https://docs.nvidia.com/doca/sdk/ktls+offloads/index.html) - "This feature is supported on NVIDIA ConnectX-6 Dx and NVIDIA BlueField-2 crypto devices onwards"
- [NVIDIA DOCA Profiles](https://docs.nvidia.com/doca/sdk/doca+profiles/index.html) - Hardware capability matrix

## Conclusion

**For Beyond Identity integration on DGX Spark (ConnectX-7):**
- Use **kTLS** for mTLS certificate validation (kernel-based, not DOCA SDK)
- Use **DOCA Flow** if packet-level policy enforcement is needed (requires DPDK)
- Plan for **Bluefield DPU** deployment to get:
  - Hardware-isolated control plane (ARM cores)
  - Full DOCA crypto library support
  - Device attestation (DICE/SPDM)
  - True zero-trust architecture (enforcement layer immune to host compromise)

**Current Spark system is suitable for:**
- DOCA API learning and prototyping
- kTLS development and testing
- DOCA Flow exploration (with DPDK setup)
- Beyond Identity API integration
- Performance baseline measurements

**Bluefield DPU required for:**
- DOCA SHA/crypto library usage
- Isolated enforcement plane (critical for production zero-trust)
- DPU attestation and hardware root of trust
- Full pattern A and pattern B implementations as designed
