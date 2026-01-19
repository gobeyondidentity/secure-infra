# Hardware Setup: BlueField DPU

Deploy Secure Infrastructure with real BlueField DPUs. This guide covers the full production path including trust relationships between hosts.

## Prerequisites

- Go 1.22+
- Make
- NVIDIA BlueField-3 DPU with network access
- SSH access to the DPU (default: `ubuntu@<DPU_IP>`)
- Linux host with the BlueField DPU installed
- rshim driver on host (for tmfifo communication)

## Clone and Build

```bash
git clone git@github.com:gobeyondidentity/secure-infra.git
cd secure-infra
make agent
# Expected:
# Building agent...
#   bin/agent
# Cross-compiling agent for BlueField (linux/arm64)...
#   bin/agent-arm64

make host-agent
# Expected:
# Building host-agent...
#   bin/host-agent
# Cross-compiling host-agent for Linux (amd64)...
#   bin/host-agent-amd64
# Cross-compiling host-agent for Linux (arm64)...
#   bin/host-agent-arm64

make
# Expected:
# Building all binaries...
#   bin/agent
#   bin/bluectl
#   bin/km
#   bin/server
#   bin/host-agent
#   bin/dpuemu
# Done.
```

This builds:
- `bin/agent-arm64` for the BlueField DPU
- `bin/host-agent-amd64` for x86_64 hosts (or `host-agent-arm64` for ARM hosts)
- Control plane tools (`bluectl`, `km`, `server`)

## Step 1: Start the Server

The server tracks your DPU inventory, attestation state, and authorization policies. Run this on your control plane host.

```bash
bin/server
```

The server listens on port 18080 by default. Verify it's running:

```bash
curl http://localhost:18080/api/health
# Expected: {"status":"ok","version":"0.4.1"}
```

---

## Step 2: Create a Tenant

Tenants are organizational boundaries that group DPUs, operators, and policies. Use them to separate environments (dev/staging/prod) or teams.

```bash
bin/bluectl tenant add gpu-prod --description "GPU Production Cluster"
# Expected: Created tenant 'gpu-prod'.

bin/bluectl tenant list
# Expected:
# NAME      DESCRIPTION             CONTACT  DPUs  TAGS
# gpu-prod  GPU Production Cluster  -        0     -
```

---

## Step 3: Copy Agent to DPU

The DPU agent runs on the BlueField and serves as the hardware trust anchor. It exposes a gRPC interface that the control plane uses to query hardware identity, and a local HTTP API that the host agent uses to receive credentials.

```bash
scp bin/agent-arm64 ubuntu@<DPU_IP>:~/agent
```

---

## Step 4: Start DPU Agent

SSH into the DPU and start the agent with local API enabled:

```bash
ssh ubuntu@<DPU_IP>
chmod +x ~/agent
~/agent -local-api -control-plane http://<CONTROL_PLANE_IP>:18080 -dpu-name bf3-prod-01
```

Replace `<CONTROL_PLANE_IP>` with the IP of the machine running the server. The `-dpu-name` should match what you'll use in Step 5.

```
# Expected:
# Fabric Console Agent v0.4.1 starting...
# gRPC server listening on :18051
# Starting local API for Host Agent communication...
# Local API listening on localhost:9443
# Local API enabled: localhost:9443
# Control Plane: http://<CONTROL_PLANE_IP>:18080
# DPU Name: bf3-prod-01
# tmfifo: device not available, using HTTP API only
```

The "tmfifo: device not available" message is normal if you're not using the hardware FIFO channel. The agent will fall back to HTTP for host communication.

Leave this terminal open. The agent must be running for registration and credential distribution.

---

## Step 5: Register DPU

Back on your control plane, register the DPU:

```bash
bin/bluectl dpu add <DPU_IP> --name bf3-prod-01
# Expected:
# Checking connectivity to <DPU_IP>:18051...
# Connected to DPU:
#   Hostname: <hostname>
#   Serial:   <serial>
#   Model:    BlueField-3
# Connection verified: agent is healthy
# Added DPU 'bf3-prod-01' at <DPU_IP>:18051
#
# Next: Assign to a tenant with 'bluectl tenant assign <tenant> bf3-prod-01'
```

---

## Step 6: Assign DPU to Tenant

```bash
bin/bluectl tenant assign gpu-prod bf3-prod-01
# Expected: Assigned DPU 'bf3-prod-01' to tenant 'gpu-prod'

bin/bluectl dpu list
# Expected:
# NAME          HOST        PORT   STATUS*  LAST SEEN
# bf3-prod-01   <DPU_IP>    18051  healthy  <timestamp>
#
# * Status reflects last known state. Use 'bluectl dpu health <name>' for live status.
```

---

## Step 7: Create Operator Invitation

Admins manage infrastructure (bluectl). Operators push credentials (km). This separation creates an audit trail where credential distribution is always tied to an authenticated operator.

```bash
bin/bluectl operator invite operator@example.com gpu-prod
# Expected:
# Invite created for operator@example.com
# Code: <CODE>
# Expires: <timestamp>
#
# Share this code with the operator. They will need to:
#   1. Run: km init
#   2. Enter the invite code when prompted
```

Save the invite code.

---

## Step 8: Accept Operator Invitation

```bash
bin/km init
# Expected:
# KeyMaker v0.4.1
# Platform: <platform> (<arch>)
# Secure Element: <type>
#
# Enter invite code:
```

Enter the code when prompted:

```
# Expected after entering code:
# Generating keypair...
# Binding to server...
#
# Bound successfully.
#   Operator: operator@example.com
#   Tenant: gpu-prod (operator)
#   KeyMaker: <keymaker-id>
#
# Config saved to ~/.km/config.json
#
# Next steps:
#   Run 'km whoami' to verify your identity.
#
# You have access to 0 CA(s) and 0 device(s).
# Ask your admin to grant access: bluectl operator grant operator@example.com <tenant> <ca> <devices>
```

Verify your identity:

```bash
bin/km whoami
# Expected:
# Operator: operator@example.com
# Server:   http://localhost:18080
#
# Authorizations: none
```

---

## Step 9: Create SSH CA

An SSH CA signs short-lived certificates instead of distributing static keys across servers. Certificates expire automatically, so revocation is rarely needed.

```bash
bin/km ssh-ca create prod-ca
# Expected: SSH CA 'prod-ca' created.
```

---

## Step 10: Grant CA Access

Link the operator, CA, and DPU together. This authorizes the operator to push this CA to this DPU:

```bash
bin/bluectl operator grant operator@example.com gpu-prod prod-ca bf3-prod-01
# Expected:
# Authorization granted:
#   Operator: operator@example.com
#   Tenant:   gpu-prod
#   CA:       prod-ca
#   Devices:  bf3-prod-01
```

---

## Step 11: Submit Attestation

Attestation is the core security mechanism. The DPU proves it's running trusted firmware by providing cryptographic evidence from its hardware root of trust (TPM/DICE).

```bash
bin/bluectl attestation bf3-prod-01
```

**If attestation succeeds** (DOCA properly configured):

```
# Expected:
# Attestation Status: ATTESTATION_STATUS_SUCCESS
#
# Certificates:
#   ...certificate chain...
#
# Attestation saved: status=success, last_validated=<timestamp>
```

**If attestation is unavailable** (common during initial setup):

```
# Expected:
# Attestation Status: ATTESTATION_STATUS_UNAVAILABLE
#
# No certificates available
#
# Attestation saved: status=unavailable, last_validated=<timestamp>
```

Attestation may be unavailable if DOCA is not configured or the BlueField firmware doesn't support DICE attestation. You can still proceed with `--force` in Step 15, but this bypasses the security guarantee.

---

## Step 12: Verify tmfifo on Host

The host agent communicates with the DPU agent through tmfifo, a hardware FIFO channel over PCIe. This is more secure than network communication because identity is physical: bytes arriving via tmfifo on the DPU can only come from the physically-attached host.

SSH to the host server (not the DPU) and verify tmfifo is available:

```bash
ssh <user>@<HOST_IP>

# Check rshim driver is loaded
lsmod | grep rshim
# Expected: rshim  <size>  0

# Check tmfifo device exists
ls -la /dev/rshim0/
# Expected:
# crw-rw---- 1 root root ... /dev/rshim0/boot
# crw-rw---- 1 root root ... /dev/rshim0/console
# crw-rw---- 1 root root ... /dev/rshim0/misc
# crw-rw---- 1 root root ... /dev/rshim0/rshim
```

If rshim is not loaded:

```bash
sudo modprobe rshim
sudo systemctl enable rshim
sudo systemctl start rshim
```

**Why tmfifo matters:** Network-based enrollment (HTTP fallback) works but is less secure. A compromised host could potentially spoof network traffic. With tmfifo, the DPU knows with hardware certainty which physical host is communicating.

---

## Step 13: Pair Host with DPU

Before the host agent can enroll, the admin must authorize the host-DPU pairing. This prevents rogue hosts from enrolling.

On the control plane:

```bash
bin/bluectl host pair bf3-prod-01
# Expected:
# Host pairing enabled for DPU 'bf3-prod-01'
# The next host agent to connect via tmfifo will be registered.
```

---

## Step 14: Copy and Run Host Agent

Copy the host agent binary:

```bash
scp bin/host-agent-amd64 <user>@<HOST_IP>:~/host-agent
```

SSH to the host and start the agent:

```bash
ssh <user>@<HOST_IP>
chmod +x ~/host-agent
~/host-agent
```

With tmfifo available, the agent enrolls through the hardware channel:

```
# Expected (tmfifo path):
# Host Agent v0.4.1 starting...
# Initial posture collected: hash=<hash>
# tmfifo detected: /dev/rshim0/misc
# Enrolling via tmfifo...
# Hostname: <hostname>
# Paired with DPU: bf3-prod-01
# Registered as host <host_id>
```

**Fallback (no tmfifo):** If tmfifo is unavailable, use the network path:

```bash
~/host-agent --dpu-agent http://localhost:9443
# Expected:
# Host Agent v0.4.1 starting...
# No tmfifo detected. Using network enrollment.
# DPU Agent: http://localhost:9443
# ...
```

Verify registration on the control plane:

```bash
bin/bluectl host list
# Expected:
# NAME          DPU           STATUS   LAST SEEN    CHANNEL
# <hostname>    bf3-prod-01   online   <timestamp>  tmfifo
```

Options:
- `--oneshot`: Register once and exit (useful for testing)
- `--dpu-agent <url>`: Force network enrollment to specified URL

---

## Step 15: Distribute Credentials

With the host agent running and registered, push the SSH CA through the DPU to the host:

```bash
bin/km push ssh-ca prod-ca bf3-prod-01
```

**If attestation succeeded** in Step 11:

```
# Expected:
# Pushing CA 'prod-ca' to bf3-prod-01...
# CA installed.
```

**If attestation was unavailable**, force distribution (logs a warning):

```bash
bin/km push ssh-ca prod-ca bf3-prod-01 --force
# Expected:
# Pushing CA 'prod-ca' to bf3-prod-01...
#   Attestation refresh attempted but unavailable.
# ! Attestation stale (0s ago)
# Warning: Forcing push despite attestation unavailable (logged)
#
# CA installed.
```

On success, the CA public key is installed at `/etc/ssh/trusted-user-ca-keys.d/prod-ca.pub` on the host, and sshd is automatically reloaded to trust the new CA.

---

## Step 16: Sign and Use Certificates

This is the payoff. Your host now trusts the CA. Sign a certificate and SSH in.

Sign your SSH key (or generate a new one):

```bash
bin/km ssh-ca sign prod-ca --principal ubuntu --pubkey ~/.ssh/id_ed25519.pub > ~/.ssh/id_ed25519-cert.pub
```

The certificate is valid for 8 hours by default. Use `--validity 24h` or `--validity 7d` for longer durations.

SSH to your host using the certificate:

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@<HOST_IP>
# Expected: successful login
```

The host verifies:
1. The certificate was signed by a CA it trusts (prod-ca)
2. The certificate hasn't expired
3. The principal (ubuntu) matches an allowed user

No authorized_keys. No public key distribution. The credential chain is hardware-attested end to end.

Inspect your certificate anytime:

```bash
ssh-keygen -L -f ~/.ssh/id_ed25519-cert.pub
```

---

## Appendix A: Trust Relationships (Optional)

Trust relationships let hosts authenticate each other for SSH or mTLS connections. This is useful for distributed training clusters or data pipelines where servers need to communicate securely.

### Prerequisites

You need two hosts, each with:
- A running host-agent (Steps 12-14)
- A paired DPU with attestation (Step 11)

Check your registered hosts:

```bash
bin/bluectl host list
# Expected:
# DPU           HOSTNAME      LAST SEEN  SECURE BOOT  DISK ENC
# bf3-prod-01   compute-01    2m ago     enabled      LUKS
# bf3-prod-02   compute-02    1m ago     enabled      LUKS
```

### Add a second DPU and host

Repeat Steps 3-6 and 11-15 for the second host/DPU pair:

```bash
# Register second DPU
bin/bluectl dpu add <DPU2_IP> --name bf3-prod-02
bin/bluectl tenant assign gpu-prod bf3-prod-02

# Grant operator access
bin/bluectl operator grant operator@example.com gpu-prod prod-ca bf3-prod-02

# Submit attestation
bin/bluectl attestation bf3-prod-02
```

Deploy and run host-agent on the second host (Steps 12-14).

### Create trust relationship

Trust is directional: the source host accepts connections from the target host. The target initiates connections and receives a CA-signed certificate.

```bash
bin/bluectl trust create compute-01 compute-02
# Expected: Trust relationship created: compute-01 <- compute-02
```

For bidirectional trust (both hosts can initiate connections to each other):

```bash
bin/bluectl trust create compute-01 compute-02 --bidirectional
# Expected:
# Trust relationship created: compute-01 <- compute-02
# Trust relationship created: compute-02 <- compute-01
```

Options:
- `--type ssh_host` (default): SSH host key trust
- `--type mtls`: Mutual TLS trust
- `--bidirectional`: Create trust in both directions
- `--force`: Bypass attestation checks (use with caution)

### Verify trust relationships

```bash
bin/bluectl trust list
# Expected:
# SOURCE       TARGET       TYPE      CREATED
# compute-01   compute-02   ssh_host  <timestamp>
# compute-02   compute-01   ssh_host  <timestamp>
```

---

## Appendix B: Shell Completion

```bash
# Zsh
echo 'source <(bin/bluectl completion zsh)' >> ~/.zshrc
echo 'source <(bin/km completion zsh)' >> ~/.zshrc
source ~/.zshrc

# Bash
echo 'source <(bin/bluectl completion bash)' >> ~/.bashrc
echo 'source <(bin/km completion bash)' >> ~/.bashrc
source ~/.bashrc
```
