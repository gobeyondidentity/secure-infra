#!/bin/bash
#===============================================================================
# B200 GPU Instance Comprehensive Discovery Script
# Target: NVIDIA Blackwell B200 instances (Lambda Labs, Brev.dev)
# Runtime: Up to 10 minutes of continuous discovery
#
# Verified against:
#   - Lambda Labs: Ubuntu 22.04, Lambda Stack, username "ubuntu"
#   - NVIDIA Brev: Ubuntu 22.04, various launchables
#   - B200 requirements: CUDA 12.8+, Driver 570+, nvidia-open kernel modules
#
# Sources:
#   - https://docs.lambda.ai/public-cloud/on-demand/
#   - https://docs.nvidia.com/brev/latest/quick-start.html
#   - https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
#===============================================================================

set -o pipefail
LOGFILE="$HOME/b200_discovery_$(hostname)_$(date +%Y%m%d_%H%M%S).log"

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

echo -e "${GREEN}Starting B200 Discovery Script (Verified Edition)${NC}"
echo "Log file: $LOGFILE"
echo "Started at: $(date)"
echo "Running as: $(whoami)"
echo ""

#===============================================================================
section "PHASE 1: QUICK SYSTEM SNAPSHOT"
#===============================================================================

subsection "Basic System Info"
run_cmd "Hostname" hostname
run_cmd "Kernel" uname -a
run_cmd "OS Release" cat /etc/os-release
run_cmd "Architecture" uname -m
run_cmd "Uptime" uptime
run_cmd "Current user" whoami
run_cmd "User groups" groups

subsection "Kernel Type Analysis"
echo "[INFO] Checking if NVIDIA-optimized kernel vs generic kernel"
KERNEL_VER=$(uname -r)
if [[ "$KERNEL_VER" == *"nvidia"* ]]; then
    echo "[OK] NVIDIA-optimized kernel detected: $KERNEL_VER"
    echo "     Benefits: GPU-aware scheduling, optimized memory management"
elif [[ "$KERNEL_VER" == *"generic"* ]]; then
    echo "[INFO] Generic Ubuntu kernel: $KERNEL_VER"
    echo "     Note: Should work fine but lacks NVIDIA-specific optimizations"
else
    echo "[INFO] Kernel: $KERNEL_VER (type unknown)"
fi
echo ""

subsection "Virtualization Detection"
run_cmd "Virtualization" systemd-detect-virt 2>/dev/null || echo "Unable to detect"
run_cmd "Hypervisor info" cat /sys/hypervisor/type 2>/dev/null || echo "No hypervisor info"
run_cmd "DMI product" cat /sys/class/dmi/id/product_name 2>/dev/null || echo "N/A"
run_cmd "CPU flags (vmx/svm)" grep -E "vmx|svm" /proc/cpuinfo | head -1 || echo "No virtualization flags"

subsection "Quick GPU Check"
run_cmd "nvidia-smi overview" nvidia-smi
run_cmd "GPU count and names" nvidia-smi -L

subsection "Memory Overview"
run_cmd "Memory" free -h
run_cmd "Swap" swapon --show 2>/dev/null || echo "No swap configured"

#===============================================================================
section "PHASE 2: B200 BLACKWELL CRITICAL CHECKS"
#===============================================================================

subsection "Driver Type Verification (B200 REQUIRES nvidia-open)"
echo "[CRITICAL] B200/Blackwell ONLY works with nvidia-open kernel modules"
echo "[CRITICAL] Proprietary driver does NOT support Blackwell architecture"
echo ""
check_file /proc/driver/nvidia/version
run_cmd "Kernel modules (nvidia)" lsmod | grep -i nvidia
run_cmd "Driver module info" modinfo nvidia 2>/dev/null | grep -E "^filename|^version|^license" | head -10
# Check if using open kernel modules
if grep -q "open" /proc/driver/nvidia/version 2>/dev/null; then
    echo "[OK] nvidia-open kernel modules detected (required for B200)"
else
    echo "[WARN] Could not confirm nvidia-open - B200 requires open kernel modules"
fi
echo ""

subsection "DKMS Status (Kernel Module Build System)"
echo "[INFO] DKMS manages kernel module compilation for different kernels"
run_cmd "DKMS status" dkms status 2>/dev/null
run_cmd "DKMS nvidia modules" dkms status 2>/dev/null | grep -i nvidia

subsection "Kernel Logs (NVIDIA driver initialization)"
echo "[INFO] Checking kernel logs for NVIDIA driver messages"
run_cmd "dmesg nvidia (last 30)" dmesg 2>/dev/null | grep -i nvidia | tail -30 || echo "[SKIP] Need sudo for dmesg"
run_cmd "journalctl nvidia" journalctl -k 2>/dev/null | grep -i nvidia | tail -20 || echo "[SKIP] journalctl not available"

subsection "Fabric Manager Status (REQUIRED for NVSwitch on B200)"
echo "[INFO] B200 HGX systems require Fabric Manager + NVLSM for NVSwitch operation"
echo ""
check_service nvidia-fabricmanager
check_service nvidia-nvlsm
check_service nvidia-persistenced
run_cmd "Fabric Manager version" nv-fabricmanager --version 2>/dev/null
check_file /usr/share/nvidia/nvswitch/fabricmanager.cfg

subsection "Driver and CUDA Version Check"
echo "[INFO] B200 requires: Driver 570.133.20+, CUDA 12.8+ for SM_100"
echo ""
run_cmd "nvidia-smi driver info" nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
run_cmd "CUDA version from nvidia-smi" nvidia-smi --query-gpu=name,driver_version --format=csv | head -2
run_cmd "nvcc version" nvcc --version 2>/dev/null

#===============================================================================
section "PHASE 3: DEEP GPU ANALYSIS"
#===============================================================================

subsection "Full GPU Query"
run_cmd "nvidia-smi full query" nvidia-smi -q

subsection "GPU Topology (NVLink 5.0 / NVSwitch 4.0)"
echo "[INFO] B200 NVLink 5.0: 18 links per GPU, 50GB/s per link, 1.8TB/s total"
echo "[INFO] HGX B200: 2 NVSwitch chips + 8 GPUs, 9 NVLinks per GPU to each switch"
echo ""
run_cmd "Topology matrix" nvidia-smi topo -m
run_cmd "Topology with PCIe" nvidia-smi topo -mp

subsection "NVLink Status (Critical for B200 Performance)"
run_cmd "NVLink status" nvidia-smi nvlink -s
run_cmd "NVLink capabilities" nvidia-smi nvlink -c
run_cmd "NVLink error counters" nvidia-smi nvlink -e

subsection "GPU Clocks and Power"
run_cmd "Clock speeds" nvidia-smi -q -d CLOCK
run_cmd "Power readings" nvidia-smi -q -d POWER
run_cmd "Performance state" nvidia-smi -q -d PERFORMANCE

subsection "GPU Memory (B200: 180GB HBM3e expected)"
run_cmd "Memory info" nvidia-smi -q -d MEMORY
run_cmd "ECC status" nvidia-smi -q -d ECC
run_cmd "Retired pages" nvidia-smi -q -d PAGE_RETIREMENT

subsection "MIG Configuration"
run_cmd "MIG mode" nvidia-smi -q -d MIG
run_cmd "MIG devices" nvidia-smi mig -lgi 2>/dev/null || echo "MIG not enabled"

subsection "Compute Mode and Processes"
run_cmd "Compute mode" nvidia-smi -q -d COMPUTE
run_cmd "Running processes" nvidia-smi pmon -c 1 2>/dev/null

subsection "PCIe Configuration (B200: Gen5 x16 expected)"
run_cmd "GPU PCIe info" nvidia-smi -q -d PCIE
run_cmd "PCIe link gen/width" nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv

subsection "All GPU Properties"
run_cmd "GPU properties CSV" nvidia-smi --query-gpu=name,driver_version,pstate,memory.total,memory.used,memory.free,compute_mode,gpu_bus_id,gpu_uuid --format=csv

subsection "DCGM (Data Center GPU Manager)"
if command -v dcgmi &>/dev/null; then
    run_cmd "DCGM discovery" dcgmi discovery -l
    run_cmd "DCGM health" dcgmi health -c
    run_cmd "DCGM diag quick" dcgmi diag -r 1
else
    echo "[SKIP] DCGM not installed (optional for monitoring)"
fi

#===============================================================================
section "PHASE 4: INFINIBAND / RDMA NETWORKING"
#===============================================================================

subsection "OFED Version (Lambda includes MLNX_OFED)"
echo "[INFO] Lambda Labs includes OFED for InfiniBand/RDMA support"
echo "[INFO] MLNX_OFED transitioning to DOCA-OFED (Jan 2025+)"
echo ""
run_cmd "OFED version" ofed_info -s 2>/dev/null
run_cmd "OFED full info" ofed_info 2>/dev/null | head -30
run_cmd "DOCA version" doca_version 2>/dev/null

subsection "InfiniBand Device Status"
run_cmd "IB status" ibstat 2>/dev/null
run_cmd "IB device info" ibv_devinfo 2>/dev/null
run_cmd "IB devices list" ibv_devices 2>/dev/null
run_cmd "IB link info" iblinkinfo 2>/dev/null | head -50

subsection "RDMA Configuration"
run_cmd "RDMA devices" rdma link show 2>/dev/null
run_cmd "RDMA system" rdma system show 2>/dev/null
check_file /sys/class/infiniband

subsection "GPUDirect RDMA (nvidia-peermem)"
echo "[INFO] nvidia-peermem enables direct GPU-to-NIC transfers for InfiniBand"
run_cmd "nvidia-peermem module" lsmod | grep -E "nvidia_peermem|nv_peer_mem"
run_cmd "Peer memory status" cat /sys/kernel/mm/memory_peers/nv_mem/version 2>/dev/null || echo "[SKIP] nvidia-peermem not loaded"

subsection "Network Interfaces"
run_cmd "IP addresses" ip addr
run_cmd "Network interfaces" ip link show
run_cmd "Routing table" ip route
run_cmd "Interface MTU values" ip link show | grep -E "mtu [0-9]+"
# Check ethernet link speed with ethtool
for iface in $(ip -o link show | awk -F': ' '{print $2}' | grep -E '^(eth|ens|eno|enp)'); do
    run_cmd "Ethtool $iface" ethtool "$iface" 2>/dev/null | grep -E "Speed|Duplex|Link detected" || echo "[SKIP] ethtool not available for $iface"
done

#===============================================================================
section "PHASE 5: SYSTEM HARDWARE"
#===============================================================================

subsection "CPU Information"
run_cmd "lscpu" lscpu
run_cmd "CPU model" cat /proc/cpuinfo | grep "model name" | head -1

subsection "NUMA Topology (GPU-CPU affinity)"
run_cmd "numactl hardware" numactl --hardware 2>/dev/null
run_cmd "numactl show" numactl --show 2>/dev/null
run_cmd "NUMA info from lscpu" lscpu | grep -i numa

subsection "Memory Hardware"
run_cmd "dmidecode memory" sudo dmidecode -t memory 2>/dev/null | head -100 || echo "[SKIP] Need sudo for dmidecode"
check_file /proc/meminfo

subsection "PCIe Devices"
run_cmd "All PCIe devices" lspci
run_cmd "NVIDIA PCIe detailed" lspci -d 10de: -vvv 2>/dev/null | head -300
run_cmd "PCIe tree" lspci -tv 2>/dev/null

subsection "Storage"
run_cmd "Block devices" lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL
run_cmd "Disk usage" df -h
run_cmd "NVMe devices" nvme list 2>/dev/null

subsection "IOMMU Configuration"
run_cmd "IOMMU groups" find /sys/kernel/iommu_groups -type l 2>/dev/null | wc -l
run_cmd "Kernel cmdline" cat /proc/cmdline
run_cmd "VFIO modules" lsmod | grep -E "vfio|iommu"

subsection "Huge Pages (GPU Memory Performance)"
echo "[INFO] Huge pages can improve GPU memory mapping performance"
run_cmd "Huge pages config" grep -i huge /proc/meminfo
run_cmd "Transparent huge pages" cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null
run_cmd "THP defrag" cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null

subsection "Cgroups Configuration"
echo "[INFO] cgroups v2 is preferred for modern container runtimes"
run_cmd "Cgroups version" stat -fc %T /sys/fs/cgroup 2>/dev/null
run_cmd "Cgroups mount" mount | grep cgroup
check_file /sys/fs/cgroup/cgroup.controllers

#===============================================================================
section "PHASE 6: CUDA AND NVIDIA SOFTWARE STACK"
#===============================================================================

subsection "CUDA Installations"
run_cmd "CUDA version (nvcc)" nvcc --version 2>/dev/null
check_file /usr/local/cuda/version.txt
check_file /usr/local/cuda/version.json
run_cmd "CUDA directories" ls -la /usr/local/ 2>/dev/null | grep cuda

subsection "NVIDIA Libraries"
run_cmd "NVIDIA shared libs" ldconfig -p 2>/dev/null | grep -E "nvidia|cuda|cudnn|nccl|tensorrt|cublas" | head -30
run_cmd "cuDNN version header" grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" /usr/include/cudnn_version.h 2>/dev/null
run_cmd "NCCL version header" grep -E "NCCL_MAJOR|NCCL_MINOR|NCCL_PATCH" /usr/include/nccl.h 2>/dev/null

subsection "NVIDIA Packages"
run_cmd "NVIDIA dpkg packages" dpkg -l 2>/dev/null | grep -i nvidia | head -40
run_cmd "CUDA dpkg packages" dpkg -l 2>/dev/null | grep -i cuda | head -20

subsection "NVIDIA Container Toolkit"
run_cmd "nvidia-container-cli" nvidia-container-cli info 2>/dev/null
check_file /etc/nvidia-container-runtime/config.toml

subsection "Environment Variables"
run_cmd "CUDA env vars" env | grep -iE "cuda|nvidia|nccl|ld_library|path" | sort

#===============================================================================
section "PHASE 7: ML FRAMEWORKS - BLACKWELL COMPATIBILITY"
#===============================================================================

subsection "Python Environments"
run_cmd "Python version" python3 --version
run_cmd "Python path" which python3
run_cmd "Pip version" pip3 --version 2>/dev/null
run_cmd "Conda info" conda info 2>/dev/null
run_cmd "Conda envs" conda env list 2>/dev/null

subsection "PyTorch Blackwell/SM_100 Compatibility Check"
echo "[CRITICAL] B200 requires PyTorch with SM_100 (compute capability 10.0) support"
echo "[CRITICAL] Standard PyTorch releases do NOT support B200 - need nightly or NGC builds"
echo ""
python3 << 'PYEOF' 2>/dev/null || echo "[SKIP] PyTorch not available or check failed"
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (torch): {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Check CUDA arch list for SM_100 support
cuda_arch = torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else []
print(f"\nCUDA arch list: {cuda_arch}")
if 'sm_100' in str(cuda_arch) or 'compute_100' in str(cuda_arch):
    print("[OK] SM_100 (Blackwell) support detected in PyTorch")
else:
    print("[WARN] SM_100 NOT in arch list - this PyTorch may not work with B200")
    print("[WARN] Consider: pip install torch --pre --index-url https://download.pytorch.org/whl/nightly")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"\nGPU {i}: {props.name}")
    print(f"  Compute capability: {props.major}.{props.minor}")
    if props.major == 10:
        print(f"  [OK] Blackwell architecture confirmed (SM {props.major}.{props.minor})")
    print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Multi-processor count: {props.multi_processor_count}")
PYEOF

subsection "TensorFlow"
python3 << 'PYEOF' 2>/dev/null || echo "[SKIP] TensorFlow not available"
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
PYEOF

subsection "JAX"
python3 << 'PYEOF' 2>/dev/null || echo "[SKIP] JAX not available"
import jax
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
PYEOF

subsection "vLLM Blackwell Support"
echo "[INFO] vLLM on B200 requires NGC PyTorch or nightly builds"
echo "[INFO] Recent vLLM includes CUTLASS attention (3.6x faster on B200)"
echo ""
python3 << 'PYEOF' 2>/dev/null || echo "[SKIP] vLLM not available"
import vllm
print(f"vLLM version: {vllm.__version__}")
# Check for Blackwell-optimized attention backends
try:
    from vllm.attention.backends import flash_attn
    print("FlashAttention backend available")
except:
    pass
PYEOF

subsection "TensorRT / TensorRT-LLM"
python3 << 'PYEOF' 2>/dev/null || echo "[SKIP] TensorRT not available"
try:
    import tensorrt as trt
    print(f"TensorRT version: {trt.__version__}")
except Exception as e:
    print(f"TensorRT import failed: {e}")
try:
    import tensorrt_llm
    print(f"TensorRT-LLM version: {tensorrt_llm.__version__}")
except Exception as e:
    print(f"TensorRT-LLM not found")
PYEOF

subsection "FlashAttention / FlashInfer (Blackwell Optimized)"
python3 << 'PYEOF' 2>/dev/null || echo "[SKIP] Flash attention check failed"
try:
    import flash_attn
    print(f"flash-attn version: {flash_attn.__version__}")
except ImportError:
    print("flash-attn not installed")

try:
    import flashinfer
    print(f"flashinfer version: {flashinfer.__version__}")
except ImportError:
    print("flashinfer not installed (provides Blackwell-optimized attention)")
PYEOF

subsection "Other ML Libraries"
python3 << 'PYEOF' 2>/dev/null || echo "[SKIP] Python ML check failed"
import importlib
libs = [
    'transformers', 'accelerate', 'deepspeed', 'bitsandbytes',
    'triton', 'xformers', 'safetensors', 'sentencepiece',
    'datasets', 'tokenizers', 'peft', 'trl', 'einops',
    'numpy', 'scipy', 'pandas', 'huggingface_hub',
    'wandb', 'mlflow', 'tensorboard', 'lightning', 'ray'
]
print("Installed ML libraries:")
for lib in libs:
    try:
        mod = importlib.import_module(lib.replace('-', '_'))
        ver = getattr(mod, '__version__', 'unknown')
        print(f"  {lib}: {ver}")
    except ImportError:
        pass
PYEOF

subsection "Full Pip Package List"
run_cmd "Pip list" pip3 list 2>/dev/null | head -100
run_cmd "Pip list (filtered ML)" pip3 list 2>/dev/null | grep -iE "torch|tensor|cuda|nvidia|transformer|vllm|flash|triton|deep"

#===============================================================================
section "PHASE 8: CONTAINER AND ORCHESTRATION"
#===============================================================================

subsection "Docker"
run_cmd "Docker version" docker version 2>/dev/null
run_cmd "Docker info" docker info 2>/dev/null | head -50
check_file /etc/docker/daemon.json
echo "[CHECK] Docker default runtime:"
if [[ -f /etc/docker/daemon.json ]]; then
    grep -E "default-runtime|nvidia" /etc/docker/daemon.json 2>/dev/null || echo "No nvidia runtime config found"
fi
run_cmd "Docker images" docker images 2>/dev/null | head -20
run_cmd "Running containers" docker ps 2>/dev/null

subsection "Podman (Alternative Container Runtime)"
run_cmd "Podman version" podman version 2>/dev/null
run_cmd "Podman info" podman info 2>/dev/null | head -30

subsection "Kubernetes"
run_cmd "kubectl version" kubectl version --client 2>/dev/null

subsection "Slurm"
run_cmd "Slurm version" sinfo --version 2>/dev/null
run_cmd "Slurm nodes" sinfo 2>/dev/null

#===============================================================================
section "PHASE 9: PERFORMANCE BASELINE TESTS"
#===============================================================================

subsection "Disk I/O Baseline (Model Loading Speed)"
echo "[INFO] Testing disk write/read speed (affects model loading time)"
run_cmd "Disk write test" dd if=/dev/zero of=/tmp/disktest bs=1M count=1024 conv=fdatasync 2>&1 | tail -1
run_cmd "Disk read test" dd if=/tmp/disktest of=/dev/null bs=1M 2>&1 | tail -1
rm -f /tmp/disktest 2>/dev/null

subsection "GPU Utilization Baseline"
run_cmd "GPU stats 5 samples" nvidia-smi dmon -c 5 -s pucvmet 2>/dev/null

subsection "GPU Memory Bandwidth Test"
python3 << 'PYEOF' 2>/dev/null || echo "[SKIP] GPU bandwidth test failed"
import torch
import time

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"Testing on: {torch.cuda.get_device_name(0)}")

    # 1GB tensor test
    size = 256 * 1024 * 1024  # 1GB in float32
    try:
        a = torch.randn(size, device=device, dtype=torch.float32)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(10):
            b = a.clone()
            torch.cuda.synchronize()
        end = time.perf_counter()

        bytes_moved = size * 4 * 10 * 2
        bw = bytes_moved / (end - start) / 1e9
        print(f"Memory bandwidth (1GB clone): ~{bw:.1f} GB/s")
        print(f"[INFO] B200 HBM3e theoretical: ~8000 GB/s")
        del a, b
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Test failed: {e}")
PYEOF

subsection "FP16 Matrix Multiplication (Transformer Workload)"
python3 << 'PYEOF' 2>/dev/null || echo "[SKIP] MatMul test failed"
import torch
import time

if torch.cuda.is_available():
    device = torch.device('cuda:0')

    N = 8192
    try:
        a = torch.randn(N, N, device=device, dtype=torch.float16)
        b = torch.randn(N, N, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(3):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()

        start = time.perf_counter()
        iterations = 20
        for _ in range(iterations):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.perf_counter()

        flops = 2 * N**3 * iterations
        tflops = flops / (end - start) / 1e12
        print(f"FP16 MatMul ({N}x{N}): {tflops:.1f} TFLOPS")
        print(f"[INFO] B200 theoretical FP16: ~2250 TFLOPS (with sparsity)")

        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Test failed: {e}")
PYEOF

subsection "NVLink Bandwidth Test (Multi-GPU)"
python3 << 'PYEOF' 2>/dev/null || echo "[SKIP] NVLink test failed"
import torch
import time

if torch.cuda.device_count() >= 2:
    print(f"Testing NVLink between {torch.cuda.device_count()} GPUs")

    size = 256 * 1024 * 1024  # 1GB
    try:
        a = torch.randn(size, device='cuda:0')

        # Warmup
        for _ in range(3):
            b = a.to('cuda:1')
            torch.cuda.synchronize()

        start = time.perf_counter()
        iterations = 10
        for _ in range(iterations):
            b = a.to('cuda:1')
            torch.cuda.synchronize()
        end = time.perf_counter()

        bytes_moved = size * 4 * iterations
        bw = bytes_moved / (end - start) / 1e9
        print(f"GPU0 -> GPU1 transfer: {bw:.1f} GB/s")
        print(f"[INFO] B200 NVLink 5.0 theoretical: ~900 GB/s bidirectional per link")

        del a, b
    except Exception as e:
        print(f"Test failed: {e}")
else:
    print(f"Only {torch.cuda.device_count()} GPU(s), need 2+ for NVLink test")
PYEOF

#===============================================================================
section "PHASE 10: ADDITIONAL DISCOVERY"
#===============================================================================

subsection "Pre-loaded Models"
run_cmd "Hugging Face cache" ls -la ~/.cache/huggingface/ 2>/dev/null | head -10
run_cmd "HF hub models" ls ~/.cache/huggingface/hub/ 2>/dev/null | head -20
run_cmd "Torch hub cache" ls -la ~/.cache/torch/ 2>/dev/null | head -10

subsection "Lambda Stack Specific"
run_cmd "Lambda Stack packages" dpkg -l 2>/dev/null | grep -i lambda
check_file /etc/lambda-stack-version

subsection "Brev Specific"
echo "[INFO] Brev (acquired by NVIDIA mid-2024) acts as multi-cloud GPU aggregator"
run_cmd "Brev CLI" brev version 2>/dev/null
run_cmd "Brev workspace info" brev ls 2>/dev/null
check_file ~/.brev/config.yaml
check_file ~/.brev
run_cmd "Brev agent processes" pgrep -a brev 2>/dev/null || echo "No Brev agent process found"
run_cmd "Brev services" systemctl list-units 2>/dev/null | grep -i brev
run_cmd "Brev directories" ls -la /opt/brev* /etc/brev* ~/.brev* 2>/dev/null || echo "No Brev directories found"
run_cmd "SSH authorized keys (Brev injected)" cat ~/.ssh/authorized_keys 2>/dev/null | head -5 | cut -c1-100

subsection "Cloud Provider Backend Detection"
echo "[INFO] Detecting underlying cloud provider (Lambda, CoreWeave, etc.)"
# Check for Lambda Labs identifiers
if dpkg -l 2>/dev/null | grep -qi lambda || [[ -f /etc/lambda-stack-version ]]; then
    echo "[DETECTED] Lambda Labs backend"
fi
# Check cloud metadata
run_cmd "Cloud metadata check" curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "No AWS/Lambda metadata"
# Check hostname patterns
if hostname | grep -qE "^[0-9]+-[0-9]+-[0-9]+-[0-9]+$"; then
    echo "[INFO] Hostname pattern suggests Lambda Labs IP-based naming"
fi

subsection "Jupyter Environment"
run_cmd "JupyterLab" jupyter lab --version 2>/dev/null
run_cmd "Jupyter kernels" jupyter kernelspec list 2>/dev/null

subsection "System Services (GPU-related)"
run_cmd "GPU services" systemctl list-units --type=service 2>/dev/null | grep -iE "nvidia|gpu|cuda|docker|fabric"

subsection "Kernel Modules"
run_cmd "Loaded nvidia modules" lsmod | grep -i nvidia
run_cmd "Module count" lsmod | wc -l

subsection "System Limits"
run_cmd "ulimit -a" ulimit -a
run_cmd "Max open files" cat /proc/sys/fs/file-max

subsection "Instance Metadata (Cloud)"
run_cmd "Cloud provider metadata" curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/ 2>/dev/null | head -20 || echo "No cloud metadata endpoint"

#===============================================================================
section "DISCOVERY COMPLETE"
#===============================================================================

echo ""
echo "==============================================================================="
echo "  DISCOVERY COMPLETE"
echo "  Finished at: $(date)"
echo "  Log file: $LOGFILE"
echo "==============================================================================="
echo ""

# Summary
echo "=== QUICK SUMMARY ==="
echo ""
echo "System: $(hostname) - $(uname -r)"
echo "OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2)"
echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null
echo ""
echo "CUDA Toolkit: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',')"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "Python: $(python3 --version 2>/dev/null | awk '{print $2}')"
echo ""

# B200-specific validation
echo "=== B200 BLACKWELL VALIDATION ==="
if grep -q "open" /proc/driver/nvidia/version 2>/dev/null; then
    echo "[OK] nvidia-open driver (required for B200)"
else
    echo "[CHECK] Verify nvidia-open driver"
fi

if systemctl is-active nvidia-fabricmanager &>/dev/null; then
    echo "[OK] Fabric Manager running"
else
    echo "[CHECK] Fabric Manager status"
fi

python3 -c "import torch; print('[OK] PyTorch SM_100 support' if 'sm_100' in str(torch.cuda.get_arch_list()) else '[WARN] PyTorch may lack B200 support')" 2>/dev/null || echo "[CHECK] PyTorch installation"

echo ""
echo "Log saved to: $LOGFILE"
echo "To compare instances: diff <log1> <log2>"
