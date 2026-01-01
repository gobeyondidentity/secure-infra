# BCM Node Provisioning on TrueNAS

This guide documents how to provision full BCM (Base Command Manager) compute nodes as VMs on TrueNAS SCALE. These are real provisioned nodes with local disk installation, not LiteNodes.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TrueNAS SCALE (192.168.1.223)                                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  br0 (Isolated Bridge - NO physical interface)                  │    │
│  │  Purpose: BCM internalnet - DHCP isolated from home network     │    │
│  │                                                                 │    │
│  │   ┌─────────────────┐      ┌─────────────────┐                  │    │
│  │   │ BCM Head Node   │      │ bcmnode1        │                  │    │
│  │   │ (VM ID: 1)      │      │ (VM ID: 5)      │                  │    │
│  │   │                 │      │                 │                  │    │
│  │   │ ens3: 10.141.255.254   │ ens3: 10.141.0.1│                  │    │
│  │   │ (internalnet)   │      │ (internalnet)   │                  │    │
│  │   │                 │      │                 │                  │    │
│  │   │ DHCP Server     │      │ PXE Boot Client │                  │    │
│  │   │ TFTP Server     │      │ Slurm Client    │                  │    │
│  │   │ Provisioner     │      │                 │                  │    │
│  │   └────────┬────────┘      └────────┬────────┘                  │    │
│  │            │                        │                           │    │
│  │            └────────────────────────┘                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  eno1 (Physical NIC - Home LAN 192.168.1.0/24)                  │    │
│  │                                                                 │    │
│  │   BCM Head Node ens4: 192.168.1.232 (management access)         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**
- `br0` bridge has NO physical interface attached, isolating BCM's DHCP from the home network
- BCM head node has two NICs: ens3 (internalnet/br0) and ens4 (externalnet/eno1)
- Compute nodes only need one NIC on br0 for provisioning and cluster communication

## Prerequisites

### 1. iPXE ISO for PXE Boot Workaround

TrueNAS forces `bootindex=1` on disk devices, preventing direct PXE boot. The workaround is to use an iPXE ISO as a CDROM that chainloads to network boot.

```bash
# Download iPXE ISO to TrueNAS (one-time setup)
ssh root@192.168.1.223 "mkdir -p /mnt/Data/isos && curl -L -o /mnt/Data/isos/ipxe.iso https://boot.ipxe.org/ipxe.iso"
```

### 2. BCM License Capacity

Check available node slots before adding nodes:

```bash
ssh root@192.168.1.232 'cmsh -c "device; list"'
```

If at license limit, remove unused nodes (see Cleanup section).

### 3. Network Requirements

- br0 bridge must exist on TrueNAS (already configured)
- BCM head node must be running with DHCP active

Verify:
```bash
# Check br0 exists
ssh root@192.168.1.223 "ip link show br0"

# Check BCM DHCP is running
ssh root@192.168.1.232 "systemctl status dhcpd"
```

## Quick Reference: Adding a New Node

For experienced users, here's the condensed checklist:

```bash
# Variables - CHANGE THESE
NODE_NAME="bcmnode2"
NODE_IP="10.141.0.2"
NODE_MAC="00:a0:98:00:00:02"  # Will be auto-generated if not specified
DISK_SIZE="50G"
RAM_MB=8192
VCPUS=4
VNC_PORT=5914
VNC_PASSWORD="bcm123"  # Same password for all nodes

# 1. Create zvol for disk
ssh root@192.168.1.223 "zfs create -V ${DISK_SIZE} Data/${NODE_NAME}-disk"

# 2. Create VM with NIC, CDROM (iPXE), DISK, and VNC
ssh root@192.168.1.223 "midclt call vm.create '{
  \"name\": \"${NODE_NAME}\",
  \"vcpus\": ${VCPUS},
  \"memory\": ${RAM_MB},
  \"bootloader\": \"UEFI_CSM\",
  \"autostart\": false
}'"

# Get VM ID (use the returned ID in subsequent commands)
VM_ID=$(ssh root@192.168.1.223 "midclt call vm.query | jq '.[] | select(.name==\"${NODE_NAME}\") | .id'")

# 3. Add devices (NIC with static MAC, CDROM, DISK, VNC)
ssh root@192.168.1.223 "midclt call vm.device.create '{\"vm\": ${VM_ID}, \"attributes\": {\"dtype\": \"NIC\", \"type\": \"E1000\", \"nic_attach\": \"br0\", \"mac\": \"${NODE_MAC}\"}, \"order\": 1003}'"
ssh root@192.168.1.223 "midclt call vm.device.create '{\"vm\": ${VM_ID}, \"attributes\": {\"dtype\": \"CDROM\", \"path\": \"/mnt/Data/isos/ipxe.iso\"}, \"order\": 1000}'"
ssh root@192.168.1.223 "midclt call vm.device.create '{\"vm\": ${VM_ID}, \"attributes\": {\"dtype\": \"DISK\", \"path\": \"/dev/zvol/Data/${NODE_NAME}-disk\", \"type\": \"VIRTIO\"}, \"order\": 1001}'"
ssh root@192.168.1.223 "midclt call vm.device.create '{\"vm\": ${VM_ID}, \"attributes\": {\"dtype\": \"DISPLAY\", \"type\": \"VNC\", \"port\": ${VNC_PORT}, \"bind\": \"0.0.0.0\", \"password\": \"${VNC_PASSWORD}\", \"web\": false}, \"order\": 1002}'"

# 4. Register node in BCM
ssh root@192.168.1.232 "cmsh -c \"device; add physicalnode ${NODE_NAME} ${NODE_IP} ens3; set mac ${NODE_MAC}; set category default; commit\""

# 5. Configure interface settings
ssh root@192.168.1.232 "cmsh -c \"device; use ${NODE_NAME}; interfaces; use ens3; set bootable yes; set bringupduringinstall yes; commit\""

# 6. Start VM
ssh root@192.168.1.223 "midclt call vm.start ${VM_ID}"

# 7. Connect VNC and trigger PXE boot (REQUIRED)
open vnc://192.168.1.223:${VNC_PORT}  # Password: bcm123
# Press Ctrl+Alt+Delete in VNC to trigger PXE boot

# 8. Monitor provisioning
watch -n5 "ssh root@192.168.1.232 'cmsh -c \"device; status ${NODE_NAME}\"'"
```

## Detailed Steps

### Step 1: Plan Node Configuration

Decide on:
- **Node name**: Must be unique (e.g., bcmnode2, bcmnode3)
- **IP address**: From 10.141.0.0/16 range, avoid .254 (head node) and existing nodes
- **MAC address**: Use format `00:a0:98:XX:XX:XX` for consistency
- **Resources**: vCPUs, RAM, disk size based on workload

Current allocations:
| Node | IP | MAC | VNC Port |
|------|-----|-----|----------|
| bcm11-headnode | 10.141.255.254 | 00:A0:98:51:16:7F | 5900 |
| bcmnode1 | 10.141.0.1 | 00:A0:98:74:94:07 | 5910 |
| bcmnode2 | 10.141.0.2 | 00:A0:98:00:00:02 | 5914 |

### Step 2: Create Storage (zvol)

```bash
NODE_NAME="bcmnode2"
DISK_SIZE="50G"

ssh root@192.168.1.223 "zfs create -V ${DISK_SIZE} Data/${NODE_NAME}-disk"

# Verify
ssh root@192.168.1.223 "zfs list Data/${NODE_NAME}-disk"
```

### Step 3: Create VM

```bash
NODE_NAME="bcmnode2"
RAM_MB=8192
VCPUS=4

ssh root@192.168.1.223 "midclt call vm.create '{
  \"name\": \"${NODE_NAME}\",
  \"vcpus\": ${VCPUS},
  \"memory\": ${RAM_MB},
  \"bootloader\": \"UEFI_CSM\",
  \"autostart\": false,
  \"description\": \"BCM compute node\"
}'"
```

**Important settings:**
- `bootloader: UEFI_CSM` - Uses SeaBIOS which supports iPXE/PXE boot
- `autostart: false` - Don't auto-start until fully configured

Get the VM ID:
```bash
VM_ID=$(ssh root@192.168.1.223 "midclt call vm.query | jq '.[] | select(.name==\"${NODE_NAME}\") | .id'")
echo "VM ID: ${VM_ID}"
```

### Step 4: Add VM Devices

**Order matters for boot sequence:**
- Order 1000: CDROM (iPXE ISO) - boots first, chainloads to PXE
- Order 1001: DISK - available for OS installation
- Order 1002: DISPLAY - VNC for console access
- Order 1003: NIC - network interface

```bash
NODE_NAME="bcmnode2"
NODE_MAC="00:a0:98:00:00:02"
VNC_PORT=5911
VM_ID=6  # Use actual VM ID from step 3

# NIC - E1000 type for PXE compatibility, static MAC required
ssh root@192.168.1.223 "midclt call vm.device.create '{
  \"vm\": ${VM_ID},
  \"attributes\": {
    \"dtype\": \"NIC\",
    \"type\": \"E1000\",
    \"nic_attach\": \"br0\",
    \"mac\": \"${NODE_MAC}\"
  },
  \"order\": 1003
}'"

# CDROM with iPXE ISO - boots first due to lower order number
ssh root@192.168.1.223 "midclt call vm.device.create '{
  \"vm\": ${VM_ID},
  \"attributes\": {
    \"dtype\": \"CDROM\",
    \"path\": \"/mnt/Data/isos/ipxe.iso\"
  },
  \"order\": 1000
}'"

# DISK - will have bootindex=2 (after CDROM)
ssh root@192.168.1.223 "midclt call vm.device.create '{
  \"vm\": ${VM_ID},
  \"attributes\": {
    \"dtype\": \"DISK\",
    \"path\": \"/dev/zvol/Data/${NODE_NAME}-disk\",
    \"type\": \"VIRTIO\"
  },
  \"order\": 1001
}'"

# VNC Display
ssh root@192.168.1.223 "midclt call vm.device.create '{
  \"vm\": ${VM_ID},
  \"attributes\": {
    \"dtype\": \"DISPLAY\",
    \"type\": \"VNC\",
    \"port\": ${VNC_PORT},
    \"bind\": \"0.0.0.0\",
    \"password\": \"${NODE_NAME}\",
    \"web\": false
  },
  \"order\": 1002
}'"
```

**Why E1000 NIC type?**
- VirtIO doesn't have PXE boot ROM in SeaBIOS
- E1000 includes iPXE ROM for network boot
- After OS install, performance difference is negligible for our use case

### Step 5: Register Node in BCM

```bash
NODE_NAME="bcmnode2"
NODE_IP="10.141.0.2"
NODE_MAC="00:a0:98:00:00:02"

# Add node with IP and interface name in one command
ssh root@192.168.1.232 "cmsh -c \"device; add physicalnode ${NODE_NAME} ${NODE_IP} ens3; set mac ${NODE_MAC}; set category default; commit\""
```

**Important:** The interface name `ens3` must match what the Linux kernel assigns to the E1000 NIC. This is predictable for our VM configuration.

### Step 6: Configure Interface Settings

Critical settings that must be enabled:

```bash
NODE_NAME="bcmnode2"

ssh root@192.168.1.232 "cmsh -c \"device; use ${NODE_NAME}; interfaces; use ens3; set bootable yes; set bringupduringinstall yes; commit\""
```

| Setting | Purpose |
|---------|---------|
| `bootable yes` | Marks interface as valid boot interface |
| `bringupduringinstall yes` | Keeps interface active during provisioning |

**Verify configuration:**
```bash
ssh root@192.168.1.232 "cmsh -c \"device; use ${NODE_NAME}; interfaces; show ens3\""
```

Expected output should show:
```
Bootable                         yes
Bring up during install          yes
```

### Step 7: Start VM and Trigger PXE Boot

```bash
VM_ID=6  # Use actual VM ID
NODE_NAME="bcmnode2"
VNC_PORT=5914  # Use port from Step 4

# Start the VM
ssh root@192.168.1.223 "midclt call vm.start ${VM_ID}"
```

**IMPORTANT: VNC Console Required**

After starting the VM, you must connect via VNC and trigger PXE boot manually:

1. Connect to VNC: `open vnc://192.168.1.223:${VNC_PORT}` (password: `bcm123`)
2. Press **Ctrl+Alt+Delete** in the VNC window to trigger a reboot
3. iPXE will catch the boot and chainload to BCM's PXE server
4. Watch the console as the node installer loads

This is required because UEFI_CSM (SeaBIOS) sometimes halts at the initial boot and needs a manual reboot trigger.

```bash
# Monitor provisioning status (run in separate terminal)
watch -n5 "ssh root@192.168.1.232 'cmsh -c \"device; status ${NODE_NAME}\"'"
```

**Expected status progression:**
1. `[ DOWN ]` - VM starting
2. `[ DOWN ], pingable` - Got IP via DHCP
3. `[ INSTALLING ]` - Node installer running
4. `[ INSTALLING ] (provisioning started)` - rsync copying OS image
5. `[ INSTALLING ] (provisioning completed)` - Image copied
6. `[ INSTALLER_CALLINGINIT ]` - Switching to local root
7. `[ UP ]` - Node fully provisioned and running

**Typical provisioning time:** 5-10 minutes (13GB image over virtual bridge)

### Step 8: Verify Node

```bash
NODE_NAME="bcmnode2"

# Check status
ssh root@192.168.1.232 "cmsh -c \"device; status ${NODE_NAME}\""

# SSH to node
ssh root@192.168.1.232 "ssh ${NODE_NAME} 'hostname; uptime; df -h /'"

# List all nodes
ssh root@192.168.1.232 "cmsh -c \"device; list\""
```

## Monitoring and Logs

### DHCP Activity
```bash
# Watch DHCP in real-time
ssh root@192.168.1.232 "journalctl -u dhcpd -f"

# Check recent DHCP activity
ssh root@192.168.1.232 "journalctl -u dhcpd --since '5 minutes ago'"
```

### Node Installer Log
```bash
# Tail the installer log
ssh root@192.168.1.232 "tail -f /var/log/node-installer"

# Check for errors
ssh root@192.168.1.232 "grep -i error /var/log/node-installer | tail -20"
```

### Provisioning Progress
```bash
# Check if rsync is still running
ssh root@192.168.1.232 "ps aux | grep 'rsync.*default-image' | grep -v grep"

# Image size (to estimate progress)
ssh root@192.168.1.232 "du -sh /cm/images/default-image/"
```

### VNC Console Access

VNC is required to trigger PXE boot and monitor the installation console.

```bash
# VNC connection info
# Host: 192.168.1.223
# Ports: 5910 (bcmnode1), 5914 (bcmnode2), etc.
# Password: bcm123 (same for all nodes)

# Connect on macOS
open vnc://192.168.1.223:5910  # bcmnode1
open vnc://192.168.1.223:5914  # bcmnode2

# Trigger PXE boot: Press Ctrl+Alt+Delete after connecting
```

## Troubleshooting

### "no free leases" in DHCP Log

**Cause:** Node MAC not registered in BCM, or node not added yet.

**Fix:**
```bash
# Verify node exists in BCM
ssh root@192.168.1.232 "cmsh -c \"device; list\""

# Check MAC matches
ssh root@192.168.1.232 "cmsh -c \"device; use ${NODE_NAME}; get mac\""
```

### "interface not defined in node object"

**Cause:** Interface not marked as bootable or not configured for install.

**Fix:**
```bash
ssh root@192.168.1.232 "cmsh -c \"device; use ${NODE_NAME}; interfaces; use ens3; set bootable yes; set bringupduringinstall yes; commit\""
```

### "missing device assert"

**Cause:** No disk attached to VM.

**Fix:** Ensure disk device is added with order > 1000 (after CDROM):
```bash
ssh root@192.168.1.223 "midclt call vm.device.query | jq '.[] | select(.vm==${VM_ID})'"
```

### VM Boots to Disk Instead of PXE

**Cause:** CDROM not attached or has higher order number than disk.

**Fix:** Verify device order:
```bash
ssh root@192.168.1.223 "midclt call vm.device.query | jq '.[] | select(.vm==${VM_ID}) | {dtype: .attributes.dtype, order: .order}'"
```

CDROM should have order 1000, DISK should have order 1001.

### "License exceeded" When Adding Node

**Cause:** BCM license node limit reached.

**Fix:** Remove unused nodes:
```bash
# Close then remove
ssh root@192.168.1.232 "cmsh -c \"device; close ${OLD_NODE}; remove ${OLD_NODE}; commit\""
```

### Node Stuck in INSTALLING State

**Check provisioning:**
```bash
# Is rsync still running?
ssh root@192.168.1.232 "ps aux | grep rsync | grep -v grep"

# Check installer log for errors
ssh root@192.168.1.232 "tail -50 /var/log/node-installer"
```

### Cannot Remove Node ("being used")

**Cause:** Node must be in CLOSED state before removal.

**Fix:**
```bash
ssh root@192.168.1.232 "cmsh -c \"device; close ${NODE_NAME}\""
# Wait a moment
ssh root@192.168.1.232 "cmsh -c \"device; remove ${NODE_NAME}; commit\""
```

## Cleanup

### Remove a Node

```bash
NODE_NAME="bcmnode2"
VM_ID=6

# 1. Close and remove from BCM
ssh root@192.168.1.232 "cmsh -c \"device; close ${NODE_NAME}\""
sleep 5
ssh root@192.168.1.232 "cmsh -c \"device; remove ${NODE_NAME}; commit\""

# 2. Stop and delete VM
ssh root@192.168.1.223 "midclt call vm.stop ${VM_ID} '{\"force\": true}'"
ssh root@192.168.1.223 "midclt call vm.delete ${VM_ID}"

# 3. Delete zvol
ssh root@192.168.1.223 "zfs destroy Data/${NODE_NAME}-disk"
```

### Re-provision Existing Node

To wipe and re-provision a node:

```bash
NODE_NAME="bcmnode1"

# Set next install mode to FULL
ssh root@192.168.1.232 "cmsh -c \"device; use ${NODE_NAME}; set nextinstallmode FULL; commit\""

# Reboot the node
ssh root@192.168.1.232 "cmsh -c \"device; reboot ${NODE_NAME}\""
```

## Reference

### TrueNAS VM API

```bash
# List VMs
midclt call vm.query

# Get VM details
midclt call vm.get_instance ${VM_ID}

# List VM devices
midclt call vm.device.query | jq '.[] | select(.vm==${VM_ID})'

# Start/Stop/Restart
midclt call vm.start ${VM_ID}
midclt call vm.stop ${VM_ID} '{"force": true}'
midclt call vm.restart ${VM_ID}
```

### BCM cmsh Commands

```bash
# Device mode
cmsh -c "device; list"
cmsh -c "device; status ${NODE_NAME}"
cmsh -c "device; use ${NODE_NAME}; show"

# Add node (syntax)
cmsh -c "device; add physicalnode <hostname> [ip] [interface]"

# Interface configuration
cmsh -c "device; use ${NODE_NAME}; interfaces; list"
cmsh -c "device; use ${NODE_NAME}; interfaces; use ens3; show"

# Node lifecycle
cmsh -c "device; close ${NODE_NAME}"
cmsh -c "device; open ${NODE_NAME}"
cmsh -c "device; reboot ${NODE_NAME}"
cmsh -c "device; remove ${NODE_NAME}; commit"
```

### Network Configuration

| Network | CIDR | Purpose |
|---------|------|---------|
| internalnet | 10.141.0.0/16 | BCM cluster network (br0) |
| externalnet | 192.168.1.0/24 | Management access (eno1) |

| Service | IP | Port |
|---------|-----|------|
| BCM Head Node (DHCP/TFTP) | 10.141.255.254 | 67, 69 |
| BCM Head Node (CMDaemon) | 10.141.255.254 | 8081 |
| BCM Head Node (HTTP/TFTP) | 10.141.255.254 | 8080 |
| BCM Web UI | 192.168.1.232 | 80 |
