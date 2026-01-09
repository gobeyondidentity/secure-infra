# Secure Infrastructure Roadmap

**Last Updated**: January 2026

---

## Shipped (v0.1–v0.5)

- Connect to BlueField DPUs, collect hardware attestation
- Create SSH CAs, sign certificates, push to hosts
- Attestation gate blocks credential distribution to unverified hardware
- Discover existing SSH keys across hosts before migration

---

## Next

| Version | Focus | Key Deliverable |
|---------|-------|-----------------|
| v0.6 | Migration | Batch rollout from old keys to CA certs with rollback |

---

## Backlog

| Item | Description |
|------|-------------|
| Production Deployment | Dockerfiles, compose, install scripts, systemd units |
| Web Dashboard | Next.js admin UI for tenant/DPU management |
