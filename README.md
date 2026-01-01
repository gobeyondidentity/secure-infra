# Secure Infrastructure

Attestation-gated credential management for AI infrastructure using NVIDIA BlueField DPUs.

## Overview

Secure Infrastructure provides hardware-enforced Zero Trust authentication for GPU clusters. It uses BlueField-3 DPUs as trust anchors, leveraging DICE/SPDM attestation to gate credential deployment to verified infrastructure.

## Components

| Component | Description |
|-----------|-------------|
| `cmd/agent` | DPU agent running on BlueField ARM cores |
| `cmd/api` | Control plane API server |
| `cmd/bluectl` | CLI for managing DPUs, attestation, and flows |
| `cmd/host-agent` | Host-side agent for inventory collection |
| `web/` | Next.js dashboard |
| `dpuemu/` | DPU emulator for local development |

## Tech Stack

- **API/Agent**: Go 1.22+
- **Policy**: Cedar (AWS policy language)
- **Dashboard**: Next.js 14, Tailwind, shadcn/ui
- **Communication**: gRPC/protobuf
- **Storage**: SQLite (encrypted)

## Development

```bash
# Build binaries
go build -o bin/agent ./cmd/agent
go build -o bin/bluectl ./cmd/bluectl
go build -o bin/api ./cmd/api

# Run tests
go test ./...

# Dashboard
cd web && npm install && npm run dev
```

## Project Structure

```
eng/
├── cmd/           # CLI entrypoints
├── internal/      # Private application code
├── pkg/           # Public libraries
├── proto/         # Protobuf definitions
├── gen/           # Generated gRPC code
├── web/           # Dashboard (Next.js)
├── docs/          # Documentation
│   ├── reference/ # Technical reference material
│   └── research/  # Technical feasibility studies
├── deploy/        # Deployment configs
└── archive/       # Legacy prototypes
```

## License

Proprietary - Beyond Identity, Inc.
