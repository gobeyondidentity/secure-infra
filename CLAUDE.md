# CLAUDE.md

## Role

Engineering for Fabric Credentials. This folder contains the codebase for attestation-gated credential management.

Product requirements: `../product/docs/PRD.md`
Strategy and market validation: `../pmm/`

## Tech Stack

| Layer | Technology |
|-------|------------|
| API | Go 1.22+ |
| Policy | Cedar |
| Dashboard | Next.js 14 + Tailwind/shadcn |
| Communication | gRPC/protobuf |
| Storage | SQLite (encrypted) |

## Development

```bash
# Build
go build -o agent ./cmd/agent
go build -o bluectl ./cmd/bluectl

# Test
go test ./...

# Dashboard
cd web && npm install && npm run dev
```

## Lab Environment

**BlueField-3 DPU**:
- SSH: `ubuntu@192.168.1.204` (LAN) or `ubuntu@100.123.57.51` (Tailscale)
- Model: B3210E, DOCA 3.2.0

**BMC** (192.168.1.203):
- Redfish: `https://192.168.1.203/redfish/v1/`
- Credentials: `root` / `BluefieldBMC1`

**Workbench**: `192.168.1.235` (rshim host)

## Key Documents

- `docs/domain-model.md` - Entity definitions
- `reference/` - Research and SDK samples
- `../product/docs/PRD.md` - Requirements
- `../product/docs/plans/` - Implementation plans

## Cross-Domain Requests

For product requirements, competitive context, or market positioning, request through the supervisor agent at the project root.
