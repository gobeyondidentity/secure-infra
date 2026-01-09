# Secure Infrastructure Roadmap

**Last Updated**: January 2026

---

## Shipped (v0.1–v0.3)

- Connect to BlueField DPUs, collect hardware attestation
- Create SSH CAs, sign certificates, push to hosts
- Attestation gate blocks credential distribution to unverified hardware

---

## Next

| Version | Focus | Key Deliverable |
|---------|-------|-----------------|
| v0.4 | Stability | Trust model fix, CLI polish, docs |
| v0.5 | Discovery | Scan hosts for existing SSH keys, flag sprawl/risk |
| v0.6 | Migration | Batch rollout from old keys to CA certs with rollback |

