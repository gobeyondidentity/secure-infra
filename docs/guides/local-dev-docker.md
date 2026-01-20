# Local Development with Docker Compose

Run Secure Infrastructure locally for development and demos using Docker Compose.

## 1. Prerequisites

- Docker 20.10+
- Docker Compose v2

Verify installation:

```bash
docker --version
# Expected: Docker version 20.10.x or higher

docker compose version
# Expected: Docker Compose version v2.x.x
```

## 2. Quick Start

```bash
cd secure-infra
docker compose up
```

This starts:
- **control-plane**: API server with health endpoint
- **host-agent**: Connects to control-plane automatically

Wait for health checks to pass:

```
secureinfra-control-plane  | level=info msg="Server started" http_port=18080
secureinfra-host-agent     | level=info msg="Connected to control plane" address=control-plane:18080
```

## 3. Services

| Service | Container | Ports |
|---------|-----------|-------|
| control-plane | `secureinfra-control-plane` | 18080 (HTTP API) |
| host-agent | `secureinfra-host-agent` | Internal only |

### Verify Services

```bash
# Health check
curl http://localhost:18080/health
# Expected: {"status":"ok"}
```

## 4. Environment Variables

### Control Plane

| Variable | Default | Description |
|----------|---------|-------------|
| `SECUREINFRA_LOG_LEVEL` | `debug` | Log verbosity (debug, info, warn, error) |
| `SECUREINFRA_LOG_FORMAT` | `text` | Log format (text, json) |

### Host Agent

The host agent connects to the control plane via the `--dpu-agent` flag specified in docker-compose.yml.

### Override Environment Variables

Create a `.env` file or pass directly:

```bash
# Via .env file
echo "SECUREINFRA_LOG_LEVEL=info" > .env
docker compose up

# Via command line
SECUREINFRA_LOG_LEVEL=info docker compose up
```

## 5. Common Operations

### Start in Background

```bash
docker compose up -d
```

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f control-plane
docker compose logs -f host-agent
```

### Stop Services

```bash
docker compose down
```

### Reset Data

Remove persistent volume to start fresh:

```bash
docker compose down -v
```

### Rebuild Images

Pull latest images:

```bash
docker compose pull
docker compose up -d
```

## 6. Using with CLI Tools

The CLI tools (`bluectl`, `km`) can connect to the local control plane:

```bash
# Point CLI to local control plane
export SECUREINFRA_SERVER=http://localhost:18080

# Or use flag
bluectl --server http://localhost:18080 tenant list
```

## 7. Health Checks

The control-plane has a built-in health check:

| Endpoint | Purpose |
|----------|---------|
| `/health` | Liveness and readiness probe |

Docker Compose waits for control-plane to be healthy before starting host-agent:

```yaml
depends_on:
  control-plane:
    condition: service_healthy
```

## 8. Persistent Data

Control plane data is stored in a Docker volume:

```bash
# List volumes
docker volume ls | grep secureinfra

# Inspect volume
docker volume inspect secure-infra_control-plane-data
```

Data persists across restarts. Use `docker compose down -v` to reset.

## 9. Troubleshooting

### Connection Refused

```
Error: connection refused localhost:18080
```

**Cause:** Service not running or not ready

**Fix:**
```bash
# Check service status
docker compose ps

# Check logs for errors
docker compose logs control-plane
```

### Port Already in Use

```
Error: bind: address already in use
```

**Fix:** Stop conflicting service or change port in docker-compose.yml:

```yaml
ports:
  - "28080:18080"  # Use different host port
```

### Host Agent Can't Connect

```
level=error msg="Failed to connect to control plane"
```

**Cause:** Control plane not healthy yet

**Fix:** The host-agent should wait automatically. If persistent, check:
```bash
docker compose logs control-plane | grep -i error
```

### Reset Everything

```bash
docker compose down -v
docker compose pull
docker compose up
```

## 10. Production Warning

This configuration disables TLS for local development simplicity.

**DO NOT use in production.** For production:
- Enable TLS (`SECUREINFRA_TLS_ENABLED=true`)
- Provide certificates
- Use proper secrets management
- Deploy behind a load balancer

See [Hardware Setup](setup-hardware.md) for production deployment.
