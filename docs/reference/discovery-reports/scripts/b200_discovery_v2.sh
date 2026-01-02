#!/bin/bash
#===============================================================================
# B200 GPU Instance Comprehensive Discovery Script - VERSION 2 (ENHANCED)
# Target: NVIDIA Blackwell instances (Lambda Labs, Brev.dev, DGX Spark)
# Runtime: Up to 15 minutes of comprehensive discovery
#
# VERSION 2 ADDITIONS:
#   - OEM/Vendor identification (dmidecode system/bios)
#   - GPU security posture and firmware inventory
#   - Hardware topology visualization (lstopo)
#   - Thermal sensors (all system sensors)
#   - Enhanced VM probes (steal time, hypervisor details)
#   - Bare metal probes (IPMI, device tree for ARM)
#   - Mellanox/ConnectX firmware and configuration
#   - Enhanced DCGM diagnostics
#   - PCIe error and capability analysis
#   - RDMA performance readiness
#
# Verified against:
#   - Lambda Labs B200 (x86_64, KVM virtualized)
#   - Brev B200 (x86_64, KVM virtualized)
#   - DGX Spark GB10 (aarch64, bare metal)
#===============================================================================

set -o pipefail
LOGFILE="$HOME/b200_discovery_v2_$(hostname)_$(date +%Y%m%d_%H%M%S).log"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

section() {
    local title="$1"
    echo ""
    echo "==============================================================================="
    echo "  $title"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==============================================================================="
    echo ""
}

subsection() {
    echo ""
    echo "--- $1 ---"
    echo ""
}

run_cmd() {
    local desc="$1"
    shift
    echo "[CMD] $desc: $*"
    if command -v "${1}" &>/dev/null || [[ -f "$1" ]]; then
        "$@" 2>&1 || echo "[WARN] Command returned non-zero exit code"
    else
        echo "[SKIP] Command not found: $1"
    fi
    echo ""
}

run_sudo_cmd() {
    local desc="$1"
    shift
    echo "[SUDO] $desc: sudo $*"
    if sudo -n true 2>/dev/null; then
        sudo "$@" 2>&1 || echo "[WARN] Command returned non-zero exit code"
    else
        echo "[SKIP] Sudo not available without password"
    fi
    echo ""
}

check_file() {
    local file="$1"
    if [[ -f "$file" ]]; then
        echo "[FILE] $file:"
        cat "$file" 2>&1
    elif [[ -d "$file" ]]; then
        echo "[DIR] $file:"
        ls -la "$file" 2>&1
    else
        echo "[SKIP] File/dir not found: $file"
    fi
    echo ""
}

check_service() {
    local svc="$1"
    echo "[SERVICE] $svc:"
    systemctl is-active "$svc" 2>/dev/null || echo "inactive/not found"
    systemctl status "$svc" 2>/dev/null | head -20 || echo "[SKIP] Service not found"
    echo ""
}

# Redirect all output to log file AND terminal
exec > >(tee -a "$LOGFILE") 2>&1

echo -e "${GREEN}Starting B200 Discovery Script v2 (Enhanced Edition)${NC}"
echo "Log file: $LOGFILE"
echo "Started at: $(date)"
echo "Running as: $(whoami)"
echo ""

#===============================================================================
section "PHASE 1: SYSTEM IDENTIFICATION & OEM DISCOVERY"
#===============================================================================

subsection "Basic System Info"
run_cmd "Hostname" hostname
run_cmd "Kernel" uname -a
run_cmd "OS Release" cat /etc/os-release
run_cmd "Architecture" uname -m
run_cmd "Uptime" uptime
run_cmd "Current user" whoami
run_cmd "User groups" groups

subsection "OEM/Vendor Identification (Critical for Supply Chain)"
echo "[INFO] Identifying hardware vendor - reveals Lambda's OEM supplier"
run_sudo_cmd "System manufacturer" dmidecode -t system
run_sudo_cmd "BIOS/UEFI info" dmidecode -t bios
run_sudo_cmd "Baseboard info" dmidecode -t baseboard
run_sudo_cmd "Chassis info" dmidecode -t chassis

# Quick OEM summary
echo "[SUMMARY] Quick OEM identification:"
run_cmd "Product name" cat /sys/class/dmi/id/product_name 2>/dev/null
run_cmd "Board vendor" cat /sys/class/dmi/id/board_vendor 2>/dev/null
run_cmd "Board name" cat /sys/class/dmi/id/board_name 2>/dev/null
run_cmd "BIOS vendor" cat /sys/class/dmi/id/bios_vendor 2>/dev/null
run_cmd "BIOS version" cat /sys/class/dmi/id/bios_version 2>/dev/null
run_cmd "BIOS date" cat /sys/class/dmi/id/bios_date 2>/dev/null

subsection "Memory Module Details"
echo "[INFO] Identifying RAM manufacturer, speed, and configuration"
run_sudo_cmd "Memory modules" dmidecode -t memory

subsection "Kernel Type Analysis"
KERNEL_VER=$(uname -r)
if [[ "$KERNEL_VER" == *"nvidia"* ]]; then
    echo "[OK] NVIDIA-optimized kernel detected: $KERNEL_VER"
elif [[ "$KERNEL_VER" == *"generic"* ]]; then
    echo "[INFO] Generic Ubuntu kernel: $KERNEL_VER"
else
    echo "[INFO] Kernel: $KERNEL_VER"
fi

#===============================================================================
section "PHASE 2: VIRTUALIZATION & HYPERVISOR ANALYSIS"
#===============================================================================

subsection "Virtualization Detection"
run_cmd "Virtualization type" systemd-detect-virt
run_cmd "virt-what" virt-what 2>/dev/null
check_file /sys/hypervisor/type
run_cmd "Hypervisor properties" cat /sys/hypervisor/properties/* 2>/dev/null

subsection "VM Performance Metrics"
echo "[INFO] Checking for CPU steal time (VM overhead indicator)"
run_cmd "vmstat (5 samples)" vmstat 1 5
echo "[NOTE] 'st' column shows steal time - CPU cycles taken by hypervisor"

run_cmd "CPU steal from /proc/stat" grep -E "^cpu " /proc/stat
echo "[INFO] Large steal time indicates noisy neighbor or oversold host"

subsection "IOMMU/Passthrough Configuration"
run_cmd "Kernel cmdline" cat /proc/cmdline
run_cmd "IOMMU dmesg" dmesg 2>/dev/null | grep -i iommu | head -30
run_cmd "IOMMU groups count" find /sys/kernel/iommu_groups -maxdepth 1 -type d 2>/dev/null | wc -l
run_cmd "VFIO modules" lsmod | grep -E "vfio|iommu"

# Check GPU IOMMU group
echo "[INFO] GPU IOMMU group (passthrough isolation):"
for gpu in /sys/bus/pci/devices/*/class; do
    if [[ "$(cat $gpu 2>/dev/null)" == "0x030000" ]] || [[ "$(cat $gpu 2>/dev/null)" == "0x030200" ]]; then
        dev=$(dirname $gpu)
        dev_id=$(basename $dev)
        iommu_group=$(readlink $dev/iommu_group 2>/dev/null | xargs basename)
        echo "  GPU $dev_id in IOMMU group $iommu_group"
    fi
done
echo ""

subsection "Clock Source (VM Timing)"
run_cmd "Current clocksource" cat /sys/devices/system/clocksource/clocksource0/current_clocksource
run_cmd "Available clocksources" cat /sys/devices/system/clocksource/clocksource0/available_clocksource

#===============================================================================
section "PHASE 3: BARE METAL SPECIFIC (DGX Spark / Physical Servers)"
#===============================================================================

subsection "IPMI/BMC Access (Bare Metal Only)"
echo "[INFO] IPMI available on physical servers, not in VMs"
run_cmd "IPMI MC info" ipmitool mc info 2>/dev/null
run_cmd "IPMI sensor list" ipmitool sensor list 2>/dev/null | head -50
run_cmd "IPMI SEL list" ipmitool sel list 2>/dev/null | tail -20

subsection "ARM Device Tree (aarch64 Only)"
if [[ "$(uname -m)" == "aarch64" ]]; then
    echo "[INFO] ARM64 system - checking device tree"
    check_file /sys/firmware/devicetree/base/model
    check_file /sys/firmware/devicetree/base/compatible
    run_cmd "Device tree dirs" ls /sys/firmware/devicetree/base/ 2>/dev/null | head -30
    run_cmd "ARM CPU info" cat /proc/cpuinfo | grep -E "^CPU|^Features|^model" | head -20
else
    echo "[SKIP] Not ARM64 architecture"
fi

subsection "Firmware Update Status"
run_cmd "fwupdmgr devices" fwupdmgr get-devices 2>/dev/null
run_cmd "fwupdmgr updates" fwupdmgr get-updates 2>/dev/null

subsection "DGX-Specific Checks"
check_file /etc/dgx-release
run_cmd "DGX packages" dpkg -l 2>/dev/null | grep -i dgx
run_cmd "NVIDIA health check" nvidia-healthcheck 2>/dev/null

#===============================================================================
section "PHASE 4: GPU DEEP ANALYSIS"
#===============================================================================

subsection "Quick GPU Overview"
run_cmd "nvidia-smi overview" nvidia-smi
run_cmd "GPU list" nvidia-smi -L

subsection "Driver Verification (B200 REQUIRES nvidia-open)"
check_file /proc/driver/nvidia/version
run_cmd "Kernel modules" lsmod | grep -i nvidia
run_cmd "Driver module info" modinfo nvidia 2>/dev/null | grep -E "^filename|^version|^license|^srcversion" | head -10

if grep -q "open" /proc/driver/nvidia/version 2>/dev/null; then
    echo "[OK] nvidia-open kernel modules detected (required for B200)"
else
    echo "[WARN] Could not confirm nvidia-open"
fi

subsection "GPU Security Posture"
echo "[INFO] Checking GPU security features (confidential computing, attestation)"
run_cmd "Security info" nvidia-smi -q -d SECURITY 2>/dev/null
run_cmd "Confidential compute" nvidia-smi conf-compute -grs 2>/dev/null

subsection "GPU Firmware Inventory"
echo "[INFO] GPU firmware versions (VBIOS, InfoROM, etc.)"
run_cmd "Full GPU query" nvidia-smi -q | grep -iE "vbios|inforom|firmware|image|oem|ecc|product"
run_cmd "InfoROM detailed" nvidia-smi -q -d INFOROM 2>/dev/null

subsection "ECC Memory Health (Critical)"
echo "[INFO] ECC errors indicate GPU memory health issues"
run_cmd "ECC status" nvidia-smi -q -d ECC
echo ""
echo "[INFO] Retired pages = memory cells that failed and were remapped"
echo "[INFO] High retired page count suggests GPU wear"
run_cmd "Retired pages" nvidia-smi -q -d PAGE_RETIREMENT

subsection "GPU Clocks and Throttling"
run_cmd "Clock info" nvidia-smi -q -d CLOCK
run_cmd "Performance state" nvidia-smi -q -d PERFORMANCE
run_cmd "Throttle reasons" nvidia-smi -q | grep -iE "throttle|violation|slowdown"

subsection "NVLink Status and Errors"
echo "[INFO] NVLink 5.0: 18 links per B200, 50GB/s per link"
run_cmd "NVLink status" nvidia-smi nvlink -s
run_cmd "NVLink capabilities" nvidia-smi nvlink -c
echo ""
echo "[CRITICAL] NVLink error counters - any non-zero indicates hardware issues"
run_cmd "NVLink ERRORS" nvidia-smi nvlink -e

subsection "MIG Configuration"
run_cmd "MIG mode" nvidia-smi -q -d MIG
run_cmd "MIG GPU instances" nvidia-smi mig -lgi 2>/dev/null
run_cmd "MIG compute instances" nvidia-smi mig -lci 2>/dev/null

subsection "Full GPU Query"
run_cmd "Complete nvidia-smi -q" nvidia-smi -q

#===============================================================================
section "PHASE 5: PCIe DEEP ANALYSIS"
#===============================================================================

subsection "PCIe Device Overview"
run_cmd "All PCIe devices" lspci
run_cmd "PCIe tree" lspci -tv

subsection "GPU PCIe Details"
echo "[INFO] Checking PCIe link negotiation (Gen5 x16 expected for B200)"
run_cmd "NVIDIA PCIe info" nvidia-smi -q -d PCIE
run_cmd "PCIe current link" nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv
run_cmd "PCIe max link" nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max --format=csv

subsection "PCIe Capabilities and Errors (Detailed)"
echo "[INFO] Detailed PCIe analysis from lspci"
run_sudo_cmd "GPU PCIe verbose" lspci -vvv -d 10de: 2>/dev/null | head -500

echo "[INFO] Extracting PCIe link status:"
sudo lspci -vvv -d 10de: 2>/dev/null | grep -E "LnkCap|LnkSta|LnkCtl|Width|Speed|DevSta|UESta|CESta" | head -40
echo ""

subsection "PCIe ACS (Access Control Services)"
echo "[INFO] ACS affects GPU passthrough isolation"
run_cmd "ACS status" sudo lspci -vvv 2>/dev/null | grep -i "access control" | head -10

subsection "NVIDIA Interrupt Assignment"
run_cmd "GPU interrupts" cat /proc/interrupts | grep -i nvidia
run_cmd "MSI-X assignment" sudo lspci -vvv -d 10de: 2>/dev/null | grep -iE "MSI|Capabilities.*MSI" | head -10

#===============================================================================
section "PHASE 6: THERMAL SENSORS & TOPOLOGY"
#===============================================================================

subsection "Hardware Topology Visualization"
echo "[INFO] lstopo provides visual hardware topology (CPU, memory, PCIe, GPU)"
run_cmd "lstopo text" lstopo-no-graphics --of txt 2>/dev/null
run_cmd "lstopo summary" lstopo-no-graphics --of console 2>/dev/null | head -50

subsection "NUMA Topology"
run_cmd "numactl hardware" numactl --hardware 2>/dev/null
run_cmd "numactl show" numactl --show 2>/dev/null
run_cmd "NUMA from lscpu" lscpu | grep -i numa

subsection "All Thermal Sensors"
echo "[INFO] Temperature readings from all system sensors"
run_cmd "sensors (all)" sensors 2>/dev/null

echo "[INFO] Thermal zones from sysfs:"
for zone in /sys/class/thermal/thermal_zone*/; do
    if [[ -d "$zone" ]]; then
        type=$(cat ${zone}type 2>/dev/null)
        temp=$(cat ${zone}temp 2>/dev/null)
        if [[ -n "$temp" ]]; then
            temp_c=$((temp / 1000))
            echo "  $(basename $zone): $type = ${temp_c}C"
        fi
    fi
done
echo ""

subsection "GPU Temperature History"
run_cmd "GPU temp current" nvidia-smi --query-gpu=temperature.gpu --format=csv
run_cmd "GPU memory temp" nvidia-smi --query-gpu=temperature.memory --format=csv 2>/dev/null

#===============================================================================
section "PHASE 7: DCGM COMPREHENSIVE DIAGNOSTICS"
#===============================================================================

if command -v dcgmi &>/dev/null; then
    subsection "DCGM Discovery"
    echo "[INFO] DCGM (Data Center GPU Manager) is THE standard for datacenter monitoring"
    run_cmd "DCGM discovery list" dcgmi discovery -l

    subsection "DCGM Group Configuration"
    run_cmd "DCGM groups" dcgmi group -l

    subsection "DCGM Health Monitoring"
    run_cmd "DCGM health config" dcgmi health -c -g 0 2>/dev/null
    run_cmd "DCGM health check" dcgmi health -c 2>/dev/null

    subsection "DCGM Diagnostics"
    echo "[INFO] DCGM diagnostic levels:"
    echo "  Level 1: Quick health check (~10 seconds)"
    echo "  Level 2: Memory tests (~2 minutes)"
    echo "  Level 3: Full diagnostic suite (~15 minutes)"
    echo ""
    echo "[RUNNING] Level 1 diagnostic (quick)..."
    run_cmd "DCGM diag level 1" dcgmi diag -r 1

    subsection "DCGM Field Values"
    run_cmd "DCGM field sample" dcgmi dmon -e 150,155,156,203,204,1001,1002,1003,1004,1005 -c 3 2>/dev/null

    subsection "DCGM Introspection"
    run_cmd "DCGM hostengine status" systemctl status nv-hostengine 2>/dev/null | head -10
else
    echo "[SKIP] DCGM not installed"
    echo "[INFO] Install with: apt install datacenter-gpu-manager"
fi

#===============================================================================
section "PHASE 8: MELLANOX/CONNECTX NETWORK DEEP DIVE"
#===============================================================================

subsection "Mellanox Software Tools (MST)"
run_cmd "MST status" mst status 2>/dev/null
run_cmd "MST start" sudo mst start 2>/dev/null

subsection "ConnectX Firmware Versions"
echo "[INFO] NIC firmware critical for RDMA performance"
run_cmd "mlxfwmanager query" mlxfwmanager --query 2>/dev/null

subsection "ConnectX Configuration"
for dev in /dev/mst/*; do
    if [[ -e "$dev" ]]; then
        echo "[INFO] Querying $dev configuration..."
        run_cmd "mlxconfig $dev" mlxconfig -d "$dev" q 2>/dev/null | head -50
    fi
done

subsection "Mellanox Link Diagnostics"
run_cmd "mlxlink" mlxlink 2>/dev/null | head -50

subsection "InfiniBand Status"
run_cmd "OFED version" ofed_info -s 2>/dev/null
run_cmd "IB status" ibstat 2>/dev/null
run_cmd "IB device info verbose" ibv_devinfo -v 2>/dev/null

subsection "Network Interface Details"
for iface in $(ip -o link show | awk -F': ' '{print $2}' | grep -vE "^lo$|^docker|^veth|^br-"); do
    echo "[INFO] Interface: $iface"
    run_cmd "ethtool $iface driver" ethtool -i "$iface" 2>/dev/null
    run_cmd "ethtool $iface settings" ethtool "$iface" 2>/dev/null | head -20
    run_cmd "ethtool $iface stats" ethtool -S "$iface" 2>/dev/null | head -30
    echo ""
done

subsection "GPUDirect RDMA Status"
run_cmd "nvidia-peermem module" lsmod | grep -E "nvidia_peermem|nv_peer_mem"
run_cmd "Peer memory version" cat /sys/kernel/mm/memory_peers/nv_mem/version 2>/dev/null
run_cmd "GPUDirect Storage" nvidia-fs -V 2>/dev/null

subsection "RDMA Performance Readiness"
echo "[INFO] Checking RDMA kernel modules"
run_cmd "RDMA modules" lsmod | grep -E "rdma|ib_|mlx"
run_cmd "RDMA devices" rdma link show 2>/dev/null
run_cmd "RDMA system config" rdma system show 2>/dev/null

#===============================================================================
section "PHASE 9: WIRELESS (DGX Spark WiFi 7)"
#===============================================================================

if command -v iw &>/dev/null && iw dev 2>/dev/null | grep -q Interface; then
    subsection "WiFi Configuration"
    run_cmd "iw dev" iw dev 2>/dev/null
    run_cmd "iw phy capabilities" iw phy 2>/dev/null | head -100
    run_cmd "rfkill status" rfkill list 2>/dev/null
    run_cmd "WiFi regulatory" iw reg get 2>/dev/null
else
    echo "[SKIP] No WiFi interfaces detected"
fi

#===============================================================================
section "PHASE 10: CUDA AND SOFTWARE STACK"
#===============================================================================

subsection "CUDA Installation"
run_cmd "nvcc version" nvcc --version 2>/dev/null
check_file /usr/local/cuda/version.txt
check_file /usr/local/cuda/version.json
run_cmd "CUDA directories" ls -la /usr/local/ 2>/dev/null | grep cuda

subsection "NVIDIA Libraries"
run_cmd "NVIDIA libs" ldconfig -p 2>/dev/null | grep -E "nvidia|cuda|cudnn|nccl|tensorrt|cublas" | head -40
run_cmd "cuDNN version" grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" /usr/include/cudnn_version.h 2>/dev/null
run_cmd "NCCL version" grep -E "NCCL_MAJOR|NCCL_MINOR|NCCL_PATCH" /usr/include/nccl.h 2>/dev/null

subsection "Fabric Manager Status"
check_service nvidia-fabricmanager
check_service nvidia-nvlsm
check_service nvidia-persistenced
run_cmd "FM version" nv-fabricmanager --version 2>/dev/null

subsection "NVIDIA Packages"
run_cmd "NVIDIA dpkg" dpkg -l 2>/dev/null | grep -i nvidia | head -50
run_cmd "CUDA dpkg" dpkg -l 2>/dev/null | grep -i cuda | head -30

#===============================================================================
section "PHASE 11: ML FRAMEWORKS - BLACKWELL COMPATIBILITY"
#===============================================================================

subsection "Python Environment"
run_cmd "Python version" python3 --version
run_cmd "Python path" which python3
run_cmd "Pip version" pip3 --version 2>/dev/null

subsection "PyTorch SM_100 Check"
python3 << 'PYEOF' 2>/dev/null || echo "[SKIP] PyTorch not available"
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (torch): {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

cuda_arch = torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else []
print(f"CUDA arch list: {cuda_arch}")
if 'sm_100' in str(cuda_arch) or 'compute_100' in str(cuda_arch):
    print("[OK] SM_100 (Blackwell) support detected in PyTorch")
else:
    print("[WARN] SM_100 NOT in arch list")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}, SM {props.major}.{props.minor}, {props.total_memory/1024**3:.1f}GB")
PYEOF

subsection "Other ML Frameworks"
python3 << 'PYEOF' 2>/dev/null || echo "[SKIP] ML check failed"
import importlib
libs = ['tensorflow', 'jax', 'transformers', 'vllm', 'triton', 'flash_attn', 'deepspeed', 'accelerate']
for lib in libs:
    try:
        mod = importlib.import_module(lib.replace('-', '_'))
        print(f"{lib}: {getattr(mod, '__version__', 'installed')}")
    except:
        pass
PYEOF

#===============================================================================
section "PHASE 12: CONTAINER RUNTIME"
#===============================================================================

subsection "Docker Configuration"
run_cmd "Docker version" docker version 2>/dev/null
run_cmd "Docker info" docker info 2>/dev/null | head -60
check_file /etc/docker/daemon.json
run_cmd "nvidia-container-cli" nvidia-container-cli info 2>/dev/null

#===============================================================================
section "PHASE 13: PERFORMANCE BASELINES"
#===============================================================================

subsection "Official NVIDIA Bandwidth Test"
echo "[INFO] nvbandwidth is the official NVIDIA memory bandwidth tool"
if command -v nvbandwidth &>/dev/null; then
    run_cmd "nvbandwidth" nvbandwidth 2>/dev/null | head -100
elif [[ -f /usr/local/cuda/extras/demo_suite/bandwidthTest ]]; then
    run_cmd "CUDA bandwidthTest" /usr/local/cuda/extras/demo_suite/bandwidthTest 2>/dev/null
else
    echo "[SKIP] nvbandwidth not found"
fi

subsection "GPU Utilization Baseline"
run_cmd "GPU dmon 5 samples" nvidia-smi dmon -c 5 -s pucvmet 2>/dev/null

subsection "GPU Memory Bandwidth (PyTorch)"
python3 << 'PYEOF' 2>/dev/null || echo "[SKIP] Memory bandwidth test failed"
import torch, time
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    size = 256 * 1024 * 1024
    try:
        a = torch.randn(size, device=device, dtype=torch.float32)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            b = a.clone()
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        bw = (size * 4 * 10 * 2) / elapsed / 1e9
        print(f"Memory bandwidth: ~{bw:.1f} GB/s")
        del a, b
    except Exception as e:
        print(f"Failed: {e}")
PYEOF

subsection "FP16 Compute Benchmark"
python3 << 'PYEOF' 2>/dev/null || echo "[SKIP] Compute test failed"
import torch, time
if torch.cuda.is_available():
    N = 8192
    try:
        a = torch.randn(N, N, device='cuda', dtype=torch.float16)
        b = torch.randn(N, N, device='cuda', dtype=torch.float16)
        for _ in range(3): c = torch.matmul(a, b)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20): c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tflops = (2 * N**3 * 20) / elapsed / 1e12
        print(f"FP16 MatMul ({N}x{N}): {tflops:.1f} TFLOPS")
        del a, b, c
    except Exception as e:
        print(f"Failed: {e}")
PYEOF

subsection "Disk I/O Baseline"
run_cmd "Disk write" dd if=/dev/zero of=/tmp/disktest bs=1M count=1024 conv=fdatasync 2>&1 | tail -1
run_cmd "Disk read" dd if=/tmp/disktest of=/dev/null bs=1M 2>&1 | tail -1
rm -f /tmp/disktest 2>/dev/null

#===============================================================================
section "PHASE 14: SECURITY POSTURE"
#===============================================================================

subsection "System Security Configuration"
run_cmd "Boot parameters" cat /proc/cmdline
run_cmd "AppArmor status" aa-status 2>/dev/null | head -20
run_cmd "SELinux status" sestatus 2>/dev/null

subsection "Kernel Security Settings"
run_cmd "ASLR" cat /proc/sys/kernel/randomize_va_space
run_cmd "dmesg restrict" cat /proc/sys/kernel/dmesg_restrict
run_cmd "kptr restrict" cat /proc/sys/kernel/kptr_restrict
run_cmd "Secure boot" mokutil --sb-state 2>/dev/null

subsection "System Limits"
run_cmd "ulimit" ulimit -a
run_cmd "Max open files" cat /proc/sys/fs/file-max

#===============================================================================
section "PHASE 15: PROVIDER-SPECIFIC DISCOVERY"
#===============================================================================

subsection "Lambda Labs Detection"
run_cmd "Lambda packages" dpkg -l 2>/dev/null | grep -i lambda
check_file /etc/lambda-stack-version
run_cmd "Lambda metadata" curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null

subsection "Brev Detection"
run_cmd "Brev CLI" brev version 2>/dev/null
check_file ~/.brev
run_cmd "Brev processes" pgrep -a brev 2>/dev/null

subsection "DGX Detection"
check_file /etc/dgx-release
run_cmd "DGX packages" dpkg -l 2>/dev/null | grep -iE "dgx|nvidia-dgx"

#===============================================================================
section "DISCOVERY COMPLETE"
#===============================================================================

echo ""
echo "==============================================================================="
echo "  DISCOVERY v2 COMPLETE"
echo "  Finished at: $(date)"
echo "  Log file: $LOGFILE"
echo "==============================================================================="
echo ""

# Quick Summary
echo "=== QUICK SUMMARY ==="
echo "System: $(hostname) - $(uname -r) - $(uname -m)"
echo "OS: $(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d'"' -f2)"
echo "OEM: $(cat /sys/class/dmi/id/product_name 2>/dev/null || echo 'Unknown')"
echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null
echo ""
echo "Virtualization: $(systemd-detect-virt 2>/dev/null || echo 'Unknown')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',')"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo ""

# B200 Validation
echo "=== B200 BLACKWELL VALIDATION ==="
if grep -q "open" /proc/driver/nvidia/version 2>/dev/null; then
    echo "[OK] nvidia-open driver"
else
    echo "[CHECK] Verify nvidia-open driver"
fi

if nvidia-smi nvlink -e 2>/dev/null | grep -qE "Replay|Recovery|CRC" | grep -v " 0$"; then
    echo "[WARN] NVLink errors detected - check hardware"
else
    echo "[OK] No NVLink errors"
fi

retired=$(nvidia-smi -q -d PAGE_RETIREMENT 2>/dev/null | grep -E "Retired pages" | head -1)
if [[ "$retired" == *": 0"* ]] || [[ -z "$retired" ]]; then
    echo "[OK] No retired GPU memory pages"
else
    echo "[WARN] Retired pages detected: $retired"
fi

echo ""
echo "Log saved to: $LOGFILE"
