# Security Review Checklist

**Purpose**: Human review of security-critical code before design partner pilots.
**Reviewer**: Nelson
**Status**: Pending

---

## 1. SSH CA (`pkg/sshca/`)

### ca.go - Key Generation

- [ ] Ed25519 key generation uses `crypto/ed25519` (not custom implementation)
- [ ] Private keys are never logged or printed
- [ ] Key material is zeroed after use where possible
- [ ] No hardcoded keys or seeds

### certificate.go - Certificate Signing

- [ ] Certificate validity is enforced (cannot issue certs > max validity)
- [ ] Principals are validated (no injection of unexpected principals)
- [ ] Extensions are explicit (no unexpected capabilities granted)
- [ ] Certificate serial numbers are unique/random
- [ ] Signing uses proper SSH certificate format (`golang.org/x/crypto/ssh`)

### Questions to Answer

- [ ] Can an attacker craft a principal that grants unexpected access?
- [ ] Can validity be bypassed or extended beyond intended window?
- [ ] Are there any timing attacks on signature verification?

---

## 2. Attestation (`pkg/attestation/`)

### redfish.go - DICE/SPDM Retrieval

- [ ] TLS verification enabled for BMC connections (no InsecureSkipVerify in prod)
- [ ] BMC credentials not hardcoded (read from config/env)
- [ ] Nonce is random and unpredictable for each attestation request
- [ ] Response signature is verified before trusting measurements

### measurements.go - Measurement Validation

- [ ] Measurements are compared against known-good values (not just accepted)
- [ ] Hash comparison is constant-time (prevents timing attacks)
- [ ] Partial matches are rejected (all-or-nothing validation)

### gate.go - Distribution Gate

- [ ] Unknown attestation status = blocked (fail-secure)
- [ ] Failed attestation cannot be forced (only stale can be forced)
- [ ] Freshness window is enforced correctly (time comparison)
- [ ] Clock skew considered (what if system clock is wrong?)

### Questions to Answer

- [ ] Can an attacker replay old attestation responses?
- [ ] Can measurements be spoofed if BMC is compromised?
- [ ] Is the DICE chain validated end-to-end or just the leaf?

---

## 3. Storage Encryption (`pkg/store/`)

### encryption.go - Key Encryption at Rest

- [ ] Uses standard algorithm (AES-256-GCM or similar)
- [ ] Key derivation uses proper KDF (not raw password)
- [ ] IV/nonce is random and never reused
- [ ] Authentication tag is verified before decryption
- [ ] Encryption key source is secure (SECURE_INFRA_KEY env var - is this sufficient?)

### sqlite.go - Database Security

- [ ] Database file permissions are restrictive (0600)
- [ ] No SQL injection vulnerabilities (parameterized queries)
- [ ] Sensitive data is encrypted before storage
- [ ] Backup considerations (encrypted backups?)

### Questions to Answer

- [ ] What happens if SECURE_INFRA_KEY is not set? Plaintext storage?
- [ ] Is the encryption key rotatable?
- [ ] Are there any fields that should be encrypted but aren't?

---

## 4. Agent Communication (`internal/agent/`, `pkg/grpcclient/`)

### Authentication

- [ ] Pre-shared token is validated on every request
- [ ] Token comparison is constant-time
- [ ] Token is not logged
- [ ] Token transmission is encrypted (TLS)

### gRPC Security

- [ ] TLS enabled for agent connections
- [ ] Certificate validation enabled (no InsecureSkipVerify in prod)
- [ ] Request size limits enforced (prevent DoS)
- [ ] Rate limiting considered

### Distribution Flow

- [ ] CA public key is validated before installation
- [ ] sshd config changes are atomic (no partial writes)
- [ ] Rollback on failure (don't leave broken config)
- [ ] Host SSH connection uses known host verification

### Questions to Answer

- [ ] What if agent is compromised? Blast radius?
- [ ] Can an attacker inject malicious CA public key?
- [ ] What permissions does the agent run with?

---

## 5. Audit Logging (`pkg/audit/`)

### logger.go - Audit Trail Integrity

- [ ] Audit logs cannot be modified after creation (append-only)
- [ ] Timestamps are from trusted source
- [ ] All security-relevant events are logged
- [ ] Sensitive data is redacted from logs (no keys in logs)

### Questions to Answer

- [ ] Can an attacker delete audit entries?
- [ ] Are logs signed or tamper-evident?
- [ ] Is there log rotation that might lose evidence?

---

## 6. CLI Security (`cmd/bluectl/`, `cmd/keymaker/`)

### Input Validation

- [ ] All user inputs are validated
- [ ] File paths are sanitized (no path traversal)
- [ ] No shell injection vulnerabilities

### Credential Handling

- [ ] Private keys not printed to stdout
- [ ] Passwords/tokens not visible in process list
- [ ] Temp files with keys have restrictive permissions and are cleaned up

---

## 7. General Security

### Dependencies

- [ ] Run `go mod verify` - all dependencies match checksums
- [ ] Check for known vulnerabilities: `govulncheck ./...`
- [ ] Review direct dependencies for security issues

### Error Handling

- [ ] Errors don't leak sensitive information
- [ ] Stack traces not exposed to users
- [ ] Crypto errors don't reveal timing information

### Configuration

- [ ] No secrets in code or config files checked into git
- [ ] Default configuration is secure (not permissive)
- [ ] Secure defaults for TLS, timeouts, etc.

---

## Review Sign-Off

| Section | Reviewed | Issues Found | Resolved |
|---------|----------|--------------|----------|
| SSH CA | [ ] | | |
| Attestation | [ ] | | |
| Storage Encryption | [ ] | | |
| Agent Communication | [ ] | | |
| Audit Logging | [ ] | | |
| CLI Security | [ ] | | |
| General Security | [ ] | | |

**Reviewer**: _______________
**Date**: _______________
**Verdict**: [ ] Approved for pilot  [ ] Issues must be fixed first

---

## Post-Review Actions

Issues found during review:

1.
2.
3.

Fixes applied:

1.
2.
3.

Re-review required: [ ] Yes  [ ] No
