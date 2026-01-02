# Network Requirements

Firewall and port configuration for Secure Infrastructure components.

## Port Requirements

| Component | Port | Protocol | Direction | Purpose |
|-----------|------|----------|-----------|---------|
| DPU Agent | 50051 | TCP | Inbound | gRPC API for control plane communication |
| Host Agent | 50052 | TCP | Inbound | gRPC API for DPU communication |
| API Server | 8080 | TCP | Inbound | REST API (health, management) |
| API Server | 443 | TCP | Outbound | External services (attestation verification, updates) |
| SSH | 22 | TCP | DPU to Host | Credential distribution (DPU configures host sshd) |

## Firewall Configuration

### UFW (Ubuntu/Debian)

```bash
# DPU Agent (run on DPU)
sudo ufw allow 50051/tcp comment "Secure Infra DPU Agent gRPC"

# Host Agent (run on host server)
sudo ufw allow 50052/tcp comment "Secure Infra Host Agent gRPC"

# API Server (run on control plane)
sudo ufw allow 8080/tcp comment "Secure Infra API Server"

# Verify rules
sudo ufw status verbose
```

### firewalld (RHEL/CentOS/Rocky)

```bash
# DPU Agent
sudo firewall-cmd --permanent --add-port=50051/tcp
sudo firewall-cmd --permanent --add-service-description=50051/tcp="Secure Infra DPU Agent"

# Host Agent
sudo firewall-cmd --permanent --add-port=50052/tcp

# API Server
sudo firewall-cmd --permanent --add-port=8080/tcp

# Apply changes
sudo firewall-cmd --reload

# Verify
sudo firewall-cmd --list-ports
```

### iptables (Direct)

```bash
# DPU Agent
sudo iptables -A INPUT -p tcp --dport 50051 -j ACCEPT -m comment --comment "Secure Infra DPU Agent"

# Host Agent
sudo iptables -A INPUT -p tcp --dport 50052 -j ACCEPT -m comment --comment "Secure Infra Host Agent"

# API Server
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT -m comment --comment "Secure Infra API"

# Save rules (Debian/Ubuntu)
sudo iptables-save > /etc/iptables/rules.v4

# Save rules (RHEL/CentOS)
sudo service iptables save
```

## BlueField-Specific Notes

The BlueField DPU runs its own ARM Linux instance with an independent network stack. Firewall rules must be configured on the DPU itself, not the host.

**Key considerations:**

1. **Separate network namespace**: The DPU has its own IP address and firewall. Rules on the host do not affect the DPU.

2. **Management interface**: DOCA typically exposes a management interface (e.g., `tmfifo_net0` or `oob_net0`). Ensure this interface is accessible from your control plane.

3. **Default restrictions**: BlueField ships with restrictive iptables rules. Check existing rules before adding new ones:
   ```bash
   ssh ubuntu@<DPU_IP>
   sudo iptables -L -n -v
   ```

4. **OVS/DOCA Flow**: If using OVS or DOCA Flow for datapath acceleration, ensure the management port is not affected by accelerated flow rules.

5. **rshim interface**: The rshim console interface (used for initial setup) does not require these network ports. It operates over USB/PCIe.

## Troubleshooting

### Test Port Connectivity

```bash
# From control plane to DPU Agent
nc -zv <DPU_IP> 50051

# From DPU to Host Agent
nc -zv <HOST_IP> 50052

# From anywhere to API Server
nc -zv <API_IP> 8080

# Verbose with timeout
nc -zv -w 5 <IP> <PORT>
```

### Common Issues

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| Connection refused | Service not running | Check if agent/server process is running: `ps aux \| grep agent` |
| Connection timeout | Firewall blocking | Check iptables rules: `sudo iptables -L -n` |
| No route to host | Network misconfiguration | Verify IP routing and interface config |
| gRPC handshake fails | TLS mismatch or version | Check gRPC/TLS configuration on both ends |

### Verify Service is Listening

```bash
# Check what's listening on a port
sudo ss -tlnp | grep 50051

# Alternative with netstat
sudo netstat -tlnp | grep 50051

# Check from DPU
ssh ubuntu@<DPU_IP> "sudo ss -tlnp | grep 50051"
```

### Debug Network Path

```bash
# Trace route to target
traceroute <DPU_IP>

# Check if host can reach DPU
ping -c 3 <DPU_IP>

# Test TCP connection with verbose output
curl -v telnet://<DPU_IP>:50051
```

### BlueField-Specific Debugging

```bash
# Check DPU interfaces
ssh ubuntu@<DPU_IP> "ip addr show"

# Check DPU routing table
ssh ubuntu@<DPU_IP> "ip route"

# List iptables rules on DPU
ssh ubuntu@<DPU_IP> "sudo iptables -L -n --line-numbers"

# Check if agent is bound correctly
ssh ubuntu@<DPU_IP> "sudo ss -tlnp | grep agent"
```
