# Secure Infrastructure Roadmap

**Last Updated**: January 2026

---

## Shipped (v0.1–v0.6)

### v0.1–v0.5
- Connect to BlueField DPUs, collect hardware attestation
- Create SSH CAs, sign certificates, push to hosts
- Attestation gate blocks credential distribution to unverified hardware
- Discover existing SSH keys across hosts before migration

### v0.6 "Astro"
- DOCA ComCh transport (replaces tmfifo with production-grade PCIe channel)
- Binary packaging (Homebrew, apt/yum, Docker images)
- Automated release infrastructure (GitHub Actions, self-hosted runners)
- Version check in CLI (`--check` flag)

---

## Next

| Version | Codename | Focus | Key Deliverable |
|---------|----------|-------|-----------------|
| v0.7 | Bender | Migration | Batch rollout from SSH keys to CA certs with rollback |

---

## Backlog

| Item | Description |
|------|-------------|
| Web Dashboard | Next.js admin UI for tenant/DPU management |
| MUNGE Token Discovery | Extend discovery to MUNGE credentials (HPC workloads) |
| Multi-Tenant Isolation | Namespace isolation for shared infrastructure |
