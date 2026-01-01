# BCM BFB Provisioning for BlueField-3 DPU

## Overview

This document covers the BFB (BlueField Bootstream) provisioning workflow for managing the BlueField-3 DPU via BCM (Base Command Manager). BFB images contain the complete DPU software stack: bootloader, OS, DOCA SDK, and drivers.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  BCM Head Node (10.141.255.254)                                 │
│  └── Stores BFB images in /cm/shared/dpu/                       │
│      └── Triggers provisioning via cmsh                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │ NFS/SSH
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Workbench (10.141.0.10) - BCM Diskless Node                    │
│  └── rshim service (FUSE-based, userspace)                      │
│      └── /dev/rshim0/{boot,console,misc,rshim}                  │
│          └── PCIe connection to DPU                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ PCIe (rshim)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  BlueField-3 DPU                                                │
│  ├── BMC (10.141.0.100) - Redfish API for power/health          │
│  ├── DPU ARM (10.141.0.3) - Ubuntu 24.04 + DOCA                 │
│  └── eMMC Boot: boot0 (backup) / boot1 (primary)                │
└─────────────────────────────────────────────────────────────────┘
```

## Current DPU State (as of 2024-12-22)

| Property | Value |
|----------|-------|
| BFB Version | bf-bundle-3.2.0-113_25.10_ubuntu-24.04_64k_prod |
| DOCA SDK | 3.2.0118-1 |
| OS | Ubuntu 24.04.3 LTS |
| Kernel | 5.15.0-1033-bluefield (ARM64) |
| Lifecycle State | GA Secured (production secure boot) |
| Boot Mode | eMMC (primary: boot1, backup: boot0) |
| Storage | 38.9GB eMMC + 119.2GB NVMe |

## rshim Configuration

### Service Status

The rshim driver runs as a FUSE-based userspace service (not kernel module):

```bash
# Check rshim status
ssh root@192.168.1.232 "ssh 10.141.0.10 'systemctl status rshim'"

# Expected output:
# ● rshim.service - rshim driver for BlueField SoC
#      Active: active (running)
#      rshim0 attached
```

### Device Files

| Device | Purpose |
|--------|---------|
| `/dev/rshim0/boot` | BFB image upload (write BFB here to flash) |
| `/dev/rshim0/console` | DPU serial console access |
| `/dev/rshim0/misc` | Status and configuration |
| `/dev/rshim0/rshim` | Low-level rshim control |

### Reading DPU Status

```bash
# Get DPU info via rshim
ssh root@192.168.1.232 "ssh 10.141.0.10 'cat /dev/rshim0/misc'"

# Output:
# DISPLAY_LEVEL   0 (0:basic, 1:advanced, 2:log)
# BOOT_MODE       1 (0:rshim, 1:emmc, 2:emmc-boot-swap)
# DEV_INFO        BlueField-3(Rev 1)
# UP_TIME         153429(s)
```

## Recovery Mechanisms

### 1. rshim (Primary Recovery)

rshim provides hardware-level access via PCIe, independent of DPU OS state:

| Scenario | Recovery Action |
|----------|-----------------|
| OS corrupted | Push BFB via `/dev/rshim0/boot` |
| Boot failure | rshim still accessible, reflash BFB |
| Network down | rshim uses PCIe, no network needed |

```bash
# Force BFB push even if DPU unresponsive
cat bf-bundle.bfb > /dev/rshim0/boot
```

### 2. Boot Partition Swap

DPU has dual eMMC boot partitions:

```bash
# Check current boot config
ssh ubuntu@100.123.57.51 "sudo mlxbf-bootctl"

# Output:
# primary: /dev/mmcblk0boot1
# backup: /dev/mmcblk0boot0
```

If primary fails, DPU can boot from backup partition.

### 3. BMC/Redfish

Out-of-band management via BMC at 10.141.0.100:

```bash
# Power cycle DPU
curl -sk -u root:BluefieldBMC1 -X POST \
  https://10.141.0.100/redfish/v1/Systems/Bluefield/Actions/ComputerSystem.Reset \
  -H "Content-Type: application/json" \
  -d '{"ResetType": "ForceRestart"}'

# Check power state
curl -sk -u root:BluefieldBMC1 \
  https://10.141.0.100/redfish/v1/Systems/Bluefield | jq .PowerState
```

### 4. BCM Power Control

```bash
# Via cmsh
cmsh -c 'device; use bluefield3; power status'
cmsh -c 'device; use bluefield3; power off'
cmsh -c 'device; use bluefield3; power on'
```

## BFB Provisioning Workflow

### Step 1: Download Official BFB

Download from NVIDIA's official repository:

```bash
# DOCA 3.2 BFB for Ubuntu 24.04
wget https://linux.mellanox.com/public/BlueField/BFBs/Ubuntu24.04/DOCA_3.2.0/bf-bundle-3.2.0-113_25.10_ubuntu-24.04_prod.bfb

# Verify checksum
sha256sum bf-bundle-3.2.0-113_25.10_ubuntu-24.04_prod.bfb
```

### Step 2: Transfer to Workbench

```bash
# Copy BFB to workbench via BCM head node
scp bf-bundle*.bfb root@192.168.1.232:/cm/shared/dpu/
ssh root@192.168.1.232 "scp /cm/shared/dpu/bf-bundle*.bfb 10.141.0.10:/tmp/"
```

### Step 3: Flash via rshim

```bash
# On workbench, flash the BFB
ssh root@192.168.1.232 "ssh 10.141.0.10 'cat /tmp/bf-bundle*.bfb > /dev/rshim0/boot'"

# Monitor progress via console
ssh root@192.168.1.232 "ssh 10.141.0.10 'screen /dev/rshim0/console'"
```

### Step 4: Verify Boot

```bash
# Wait for DPU to reboot (10-15 minutes)
# Then verify via Tailscale or BCM
ssh ubuntu@100.123.57.51 "cat /etc/mlnx-release"
```

## BCM DPU Commands

### Device Registration

```bash
# Add DPU to BCM
cmsh -c 'device; add dpu bluefield3 10.141.0.3; set mac 8C:91:3A:F4:7C:EE; commit'

# Associate with host
cmsh -c 'device; use bluefield3; set hostnode workbench; commit'

# Add BMC interface
cmsh -c 'device; use bluefield3; interfaces; add bmc rf0 10.141.0.100 internalnet; commit'
```

### DPU Management

```bash
# Status
cmsh -c 'device; use bluefield3; status'

# Power control
cmsh -c 'device; use bluefield3; power status'

# Health overview
cmsh -c 'device; use bluefield3; healthoverview'

# BMC event log
cmsh -c 'device; use bluefield3; bmceventlog'
```

### BFB Push via BCM (when fully configured)

```bash
# Push BFB image
cmsh -c 'device; use bluefield3; dpu push-bfb /cm/shared/dpu/bf-bundle.bfb'

# Apply configuration
cmsh -c 'device; use bluefield3; dpu apply'
```

## Network Topology

| Interface | IP | Network | Purpose |
|-----------|-----|---------|---------|
| DPU ARM (oob_net0) | 10.141.0.3 | internalnet | BCM management |
| DPU BMC | 10.141.0.100 | internalnet | Redfish/IPMI |
| DPU ARM (Tailscale) | 100.123.57.51 | Tailscale | Direct access |
| Workbench | 10.141.0.10 | internalnet | BCM diskless node |
| BCM Head | 10.141.255.254 | internalnet | Cluster manager |

## Troubleshooting

### rshim Not Detecting DPU

```bash
# Check if rshim service is running
systemctl status rshim

# Restart rshim
systemctl restart rshim

# Check PCIe device
lspci | grep -i mellanox
# Should show: Mellanox Technologies BlueField-3
```

### BFB Flash Stuck

```bash
# Check rshim misc for status
cat /dev/rshim0/misc

# If BOOT_MODE shows 0 (rshim), DPU is waiting for BFB
# If stuck, power cycle via BMC and retry
```

### DPU Not Booting After Flash

1. Wait full 15 minutes (BFB install takes time)
2. Check console via `screen /dev/rshim0/console`
3. If no output, verify rshim connection: `cat /dev/rshim0/misc`
4. Power cycle via BMC if needed
5. Reflash BFB if still failing

## Security Considerations

### GA Secured Lifecycle

The DPU is in "GA Secured" lifecycle state:
- Only NVIDIA-signed BFB images can be installed
- Secure boot is enforced
- Unsigned or modified BFB images will be rejected

### BMC Credentials

| Setting | Value |
|---------|-------|
| Username | root |
| Password | BluefieldBMC1 |
| Access | Redfish API over HTTPS |

## cm-dpu-setup Wizard (BCM 11)

### Overview

BCM provides `cm-dpu-setup` for configuring DPU infrastructure. This is the production method for integrating DPUs into a BCM cluster.

### Prerequisites

1. DPU physically installed in host with rshim connection
2. Host registered as BCM node (e.g., diskless category)
3. BFB file downloaded to head node

### Running the Wizard

```bash
# Interactive mode (recommended)
/cm/local/apps/cm-setup/bin/cm-dpu-setup

# Config-driven mode
cm-dpu-setup -c /root/dpu-setup.yaml
```

### Key Wizard Screens

| Screen | Recommended Selection |
|--------|----------------------|
| Host category | Select category containing DPU hosts (e.g., `diskless`) |
| Port mode | `eth` for Ethernet, `ib` for InfiniBand |
| Performance network | Use existing `internalnet` for lab |
| OOB network | `internalnet` (for management traffic) |
| BMC/Redfish | Skip if `ipminet` doesn't exist (configure manually) |
| BFB selection | Skip, push BFB afterwards |
| Host image source | `default-image` |

### Post-Wizard: rshim Package Issue

**CRITICAL**: The cm-dpu-setup wizard may reboot nodes with a new image that lacks rshim.

**Symptom**: After wizard completion, `/dev/rshim0` is missing on host.

**Fix**:
```bash
# 1. Install rshim into the software image
cm-chroot-sw-img /cm/images/default-image apt-get update
cm-chroot-sw-img /cm/images/default-image apt-get install -y rshim

# 2. Update the node
cmsh -c 'device; imageupdate -n workbench -w'

# 3. Start rshim service
ssh 10.141.0.10 'systemctl start rshim'

# 4. Verify
ssh 10.141.0.10 'cat /dev/rshim0/misc'
```

### BFB File Location

BCM expects BFB files in `/cm/shared/dpu/bfb/`:

```bash
# Copy BFB to standard location
cp /path/to/bf-bundle.bfb /cm/shared/dpu/bfb/

# List available BFBs
/cm/local/apps/cmd/scripts/cm-dpu-manage --list-bfb /cm/shared/dpu/bfb
```

### Pushing BFB via cm-dpu-manage

The `cmsh dpu push-bfb` command uses `cm-dpu-manage` under the hood:

```bash
# Direct cm-dpu-manage (runs on head node, executes on host via rshim)
/cm/local/apps/cmd/scripts/cm-dpu-manage \
  --push-bfb /cm/shared/dpu/bfb/bf-bundle.bfb \
  --rshim /dev/rshim0

# Via cmsh (recommended)
cmsh -c 'device; dpu push-bfb -n bluefield3 -f bf-bundle.bfb'
```

### Wizard Configuration Created

After successful cm-dpu-setup, BCM creates:

| Entity | Purpose |
|--------|---------|
| `dpu` category | Category for DPU nodes |
| `dpu-host` image | Software image for DPU host nodes |
| `dpunet` network | Data plane network (172.0.0.0/8) |
| `tmfifo_net` network | rshim communication (192.168.100.0/30) |
| DPU settings | Boot order, interface modes, OVS offload |

### Known Issues (BCM 11)

| Issue | Workaround |
|-------|------------|
| `ipminet not found` | Uncheck BMC in wizard, configure manually |
| `dpu_distro not set` | Skip error, set after BFB push |
| `mlnx-ofed58 not found` | Skip, not required for rshim provisioning |
| `Invalid IP addresses` | Skip, configure DPU IP manually |
| rshim missing after reboot | Install rshim into software image |

## Current BFB: DOCA 3.2.1

**File**: `bf-bundle-3.2.1-34_25.11_ubuntu-24.04_64k_prod.bfb`
**Location**: `/cm/shared/dpu/bfb/`
**Size**: 1.4 GB
**Source**: https://content.mellanox.com/BlueField/BFBs/Ubuntu24.04/

## Fresh BFB First Login

After flashing a fresh BFB, the DPU requires password change on first login:

| Credential | Value |
|------------|-------|
| Default User | `ubuntu` |
| Default Password | `ubuntu` |
| Lab Password | `BluefieldBMC1` |
| Requirement | Must change password on first login (12+ chars) |

### First Login via tmfifo (with expect)

```bash
# From workbench, use expect for non-interactive password change
cat > /tmp/set_password.exp << 'EOF'
#!/usr/bin/expect -f
set timeout 30
spawn ssh -o StrictHostKeyChecking=no ubuntu@192.168.100.2
expect "password:"
send "ubuntu\r"
expect "Current password:"
send "ubuntu\r"
expect "New password:"
send "BluefieldBMC1\r"
expect "Retype new password:"
send "BluefieldBMC1\r"
expect eof
EOF
expect /tmp/set_password.exp
```

### Verify After Flash

```bash
# Verify DOCA version
sshpass -p BluefieldBMC1 ssh ubuntu@192.168.100.2 "cat /etc/mlnx-release"
# Expected: bf-bundle-3.2.1-34_25.11_ubuntu-24.04_64k_prod
```

## References

- [NVIDIA DOCA Installation Guide](https://docs.nvidia.com/doca/sdk/installation-guide/index.html)
- [BlueField BMC Documentation](https://docs.nvidia.com/networking/display/bluefieldbmcv2501/)
- [BCM DPU Management](https://docs.nvidia.com/base-command-manager/)
- [rshim User Guide](https://docs.nvidia.com/networking/display/BlueFieldDPUOSv420/rshim)
- [BCM Admin Manual - Chapter 3.8: Configuring BlueField DPUs](bcm%20manuals/admin-manual.txt)
