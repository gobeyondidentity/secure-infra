# DICE/SPDM Attestation: Technical Overview

**Purpose:** Cryptographically prove that a device's firmware and hardware are genuine and haven't been tampered with.

---

## DICE (Device Identifier Composition Engine)

### What It Is

A hardware root of trust that creates a unique, unforgeable identity for a device based on its firmware.

### How It Works

#### 1. Unique Device Secret (UDS)
- Factory-burned secret key in device hardware
- Never leaves the chip, can't be read out
- Unique per device (like a hardware fingerprint)

#### 2. Firmware Measurement
- During boot, device measures (hashes) its own firmware
- Creates a "measurement" = cryptographic hash of firmware code

#### 3. Compound Device Identity (CDI)
- Combines UDS + firmware measurement
- `CDI = KDF(UDS, Hash(Firmware))`
- Result: Identity that proves "this specific hardware running this specific firmware"

#### 4. Attestation Key Pair
- CDI used to derive public/private key pair
- Private key proves device identity
- Public key goes into certificate (Device ID Certificate)

### Key Insight

If firmware changes (malware, backdoor, update), the measurement changes, so the CDI changes, so the attestation key changes. **You can't fake the identity without the UDS.**

---

## SPDM (Security Protocol and Data Model)

### What It Is

A protocol for securely requesting and verifying attestation from a device.

### How It Works

#### Phase 1: Get Device Measurements

```
Client → Device: "What firmware are you running?"
Device → Client: [Signed measurement chain]
```

The device returns:
- **Firmware measurements** (hashes of bootloader, OS, applications)
- **Device ID Certificate** (proves public key belongs to this device)
- **Signature** over measurements using Device ID private key

#### Phase 2: Verify Certificate Chain

```
Client: Check Device ID Certificate → Intermediate CA → Vendor Root CA
        Verify: Is this a genuine NVIDIA BlueField-3?
```

Client validates:
- Certificate chain is valid
- Root CA is trusted (NVIDIA, Intel, AMD, etc.)
- Device ID cert is genuine

#### Phase 3: Verify Measurements

```
Client: Check firmware measurements against known-good values
        - Bootloader hash = expected value?
        - DOCA version = approved version?
        - No unsigned code loaded?
```

Client compares measurements to:
- **Reference values** (golden hashes from vendor)
- **Policy** ("only approve DOCA 2.5.0 or newer")

#### Phase 4: Establish Trust

If all checks pass:
- ✅ Device is genuine hardware (cert chain validated)
- ✅ Device is running known-good firmware (measurements match)
- ✅ Device hasn't been compromised (signature valid)

**Result:** Client now trusts the device and can proceed with secure communication.

---

## BlueField-3 DPU Implementation

### Hardware Root of Trust

- UDS burned into BlueField-3 silicon at manufacture
- ARM TrustZone secure world manages attestation

### Firmware Layers Measured

1. **UEFI/Bootloader** - first thing that runs
2. **BlueField OS** (Linux kernel)
3. **DOCA runtime**
4. **Customer DOCA application** (our Beyond Identity integration)

### Attestation Flow

```
Client → BlueField DPU: "Prove you're genuine and unmodified"

BlueField DPU:
  1. Measure firmware stack (hash each layer)
  2. Sign measurements with Device ID key (derived from UDS + measurements)
  3. Return:
     - Firmware hashes
     - Device ID certificate (signed by NVIDIA CA)
     - Signature

Client:
  1. Verify NVIDIA CA signature → confirms genuine BlueField-3
  2. Verify measurements match known-good values
  3. Trust established
```

---

## Application to Beyond Identity + DPU Project

### Pattern A (mTLS Only) - No DPU Attestation

**Flow:**
```
1. Data scientist laptop → Beyond Identity: Authenticate
2. Beyond Identity → Laptop: Here's your certificate (TPM-bound)
3. Laptop → DPU: Present certificate
4. DPU: Validates cert, enforces policy, allows access
```

**Risk:** Compromised DPU could steal certificates, log traffic, bypass policies.

### Pattern B (mTLS + DPU Attestation) - Full Mutual Authentication

**Flow:**
```
1. Data scientist laptop → Beyond Identity: Authenticate
2. Beyond Identity → Laptop: Here's your OIDC token
3. Laptop → DPU: "Attest yourself first"
4. DPU → Laptop: [DICE measurements + Device ID cert + signature]
5. Laptop: Verify measurements match approved DOCA version
6. Laptop: Verify Device ID cert chains to NVIDIA root CA
7. Laptop: Attestation valid → send OIDC token
8. DPU: Validates token, enforces policy, allows access
```

**Benefit:** Mutual authentication. Both client and DPU prove they're trustworthy.

---

## Use Cases

### 1. Zero-Trust Architecture
- Don't trust infrastructure by default
- Every component (client, DPU, backend) must attest before communicating
- Defense in depth: hardware-verified trust at every layer

### 2. Compliance & Audit
- Prove infrastructure is running approved firmware versions
- Audit trail: "Access was granted by verified, unmodified DPU"
- Meet regulatory requirements for infrastructure integrity

### 3. Rogue DPU Detection
- Attacker compromises DPU firmware
- Firmware measurement changes
- Attestation fails → client refuses to connect
- Prevents compromised infrastructure from intercepting credentials

### 4. Supply Chain Security
- Certificate chain proves device is genuine NVIDIA hardware
- Not counterfeit, not tampered with during shipping
- Protects against hardware backdoors

---

## BlueField-2 vs BlueField-3

### BlueField-2 Limitations
- Has secure boot
- Has ARM TrustZone
- **Does NOT have standardized DICE/SPDM attestation**
- Custom attestation possible but not standardized
- Cannot be used for Pattern B validation

### BlueField-3 Capabilities
- **Adds DICE/SPDM support**
- Standardized attestation protocol
- Certificate chain to NVIDIA root CA
- Interoperable with industry standards (TCG DICE, DMTF SPDM)
- Required for Pattern B implementation

**Implication for Project:**
- Stage 1 (LaunchPad with BlueField-2): Can only validate Pattern A (mTLS)
- Stage 2 (Physical BlueField-3): Required for Pattern B (mTLS + attestation)

---

## Security Benefits

### Without DICE/SPDM Attestation
- Client trusts DPU by default
- No verification of DPU firmware integrity
- Compromised DPU can intercept all traffic
- Single point of failure

### With DICE/SPDM Attestation
- Client verifies DPU before sending credentials
- DPU firmware tampering detected immediately
- Compromised DPU rejected by clients
- Defense in depth: hardware-backed trust chain

---

## Integration Architecture

### Three Layers of Hardware-Backed Security

#### 1. Client Device (TPM)
- Private keys bound to laptop/workstation TPM
- Non-exportable credentials
- Proves: "This specific user on this specific device"

#### 2. DPU Layer (ARM TrustZone + DICE/SPDM)
- DPU attests firmware integrity
- Validates client credentials
- Enforces network policies
- Proves: "Genuine NVIDIA DPU running approved firmware"

#### 3. Protected Resources (Optional TPM)
- DGX nodes may have TPMs for secure boot
- Used for infrastructure security (separate from access control)
- Proves: "Genuine server hardware with verified boot chain"

### End-to-End Trust Chain

```
Client TPM → Beyond Identity Certificate → DPU Validation → DPU DICE/SPDM Attestation → Backend Verification
     ↓                                              ↓                                              ↓
Device-bound                            Hardware-enforced                           Infrastructure
credentials                              network access                               integrity
```

---

## Implementation Considerations

### Pattern A (Simpler, Stage 1 Compatible)
- **Pros:** Works on BlueField-2 (LaunchPad), faster implementation
- **Cons:** No DPU attestation, must trust DPU by default
- **Timeline:** Can validate in Stage 1 (2-3 weeks)

### Pattern B (More Secure, Requires BlueField-3)
- **Pros:** Mutual authentication, detects rogue DPUs, zero-trust compliant
- **Cons:** Requires BlueField-3 hardware, more complex implementation
- **Timeline:** Deferred to Stage 2 (requires physical BF3)

### Recommended Approach
1. **Stage 1:** Validate Pattern A on LaunchPad (BlueField-2)
2. **Stage 2:** Implement Pattern B on physical BlueField-3
3. **Production:** Offer both patterns based on customer security requirements

---

## Reference Materials

- **TCG DICE Specification:** Trusted Computing Group Device Identifier Composition Engine
- **DMTF SPDM Specification:** Security Protocol and Data Model for device attestation
- **NVIDIA BlueField-3 Documentation:** DICE/SPDM attestation support and certificate management
- **ARM Platform Security Architecture:** TrustZone and secure boot implementation

---

## Glossary

- **UDS (Unique Device Secret):** Factory-burned secret key unique to each device
- **CDI (Compound Device Identity):** Derived identity combining hardware secret and firmware measurement
- **DICE (Device Identifier Composition Engine):** Hardware-based device identity standard
- **SPDM (Security Protocol and Data Model):** Protocol for requesting and verifying attestation
- **Device ID Certificate:** Certificate containing public key derived from device identity
- **Reference Values:** Known-good firmware measurements (golden hashes)
- **Measurement:** Cryptographic hash of firmware or software component
- **Attestation:** Process of proving device identity and firmware integrity
