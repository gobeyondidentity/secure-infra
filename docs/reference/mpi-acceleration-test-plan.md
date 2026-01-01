# MPI Acceleration Test Plan for BlueField-3

> **Purpose**: Test MPI collective offload to BlueField DPU ARM cores
> **Created**: 2024-12-23
> **Status**: Planning

## Lab Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BEDROOM DESK                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    QSFP112     ┌─────────────────────────────────────┐ │
│  │   DGX Spark     │◄──────────────►│         Workbench PC                │ │
│  │   (ARM64)       │    100GbE      │         (x86_64)                    │ │
│  │                 │    (TODO)      │  ┌───────────────────────────────┐  │ │
│  │ enp1s0f0np0 (D) │                │  │      BlueField-3 DPU          │  │ │
│  │ enp1s0f1np1 (U) │                │  │      (ARM64)                  │  │ │
│  │                 │                │  │                               │  │ │
│  │ Tailscale:      │                │  │  p0 ──► ovsbr1 ──► pf0hpf     │  │ │
│  │ 100.85.204.118  │                │  │  p1 ──► ovsbr2 ──► pf1hpf     │  │ │
│  └─────────────────┘                │  │         │                     │  │ │
│                                     │  │         ▼                     │  │ │
│  ┌─────────────────┐                │  │  SF: enp3s0f0s0 (to host)     │  │ │
│  │   DGX Spark 2   │                │  │  Tailscale: 100.106.76.67     │  │ │
│  │   (ARM64)       │                │  └───────────────────────────────┘  │ │
│  │                 │                │                                     │ │
│  │ Tailscale:      │                │  Tailscale: 100.99.121.42           │ │
│  │ 100.94.50.24    │                │  LAN: 192.168.1.236                 │ │
│  └─────────────────┘                └─────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Current Hardware State

| Node | Arch | RDMA Status | MPI | QSFP Status |
|------|------|-------------|-----|-------------|
| Workbench | x86_64 | Unknown | Need to check | N/A (via BF3) |
| BF3 DPU | ARM64 | mlx5_2/3 Active | OpenMPI 4.1.9 | p0/p1 need check |
| Spark | ARM64 | mlx5_0 Down | Need to check | port1 UP, port0 DOWN |
| Spark-2 | ARM64 | Unknown | Need to check | Unknown |

## Architecture Compatibility Note

**Q: Will MPI work between x86_64 (Workbench) and ARM64 (Spark/BF3)?**

**A: Yes.** MPI is architecture-agnostic. Key points:

- MPI defines a wire protocol, not a binary format
- Data serialization handles endianness (both are little-endian anyway)
- Heterogeneous clusters (x86 + ARM + POWER) are common in HPC
- UCX/RDMA transport works across architectures
- Only requirement: compatible MPI libraries on both ends

**Caveats**:
- Must use same MPI implementation (OpenMPI ↔ OpenMPI, not OpenMPI ↔ MPICH)
- Application binaries must be compiled for each architecture
- Some collective optimizations may differ per-arch

---

## Test Phases

### Phase 1: Verify Prerequisites

**Goal**: Confirm all components can communicate

```bash
# 1. Check BF3 QSFP port status
ssh ubuntu@100.106.76.67 "ip link show p0 p1"

# 2. Check BF3 RDMA devices
ssh ubuntu@100.106.76.67 "ibstat"

# 3. Check Workbench RDMA (need to install rdma-core if missing)
ssh 100.99.121.42 "ibstat || sudo apt install rdma-core && ibstat"

# 4. Check Spark RDMA
ssh nmelo@100.85.204.118 "ibstat"

# 5. Verify MPI versions match
ssh ubuntu@100.106.76.67 "mpirun --version"
ssh 100.99.121.42 "mpirun --version"
ssh nmelo@100.85.204.118 "mpirun --version"
```

### Phase 2: PCIe Loopback Test (No Cable Required)

**Goal**: Test MPI between Workbench host and BF3 ARM via PCIe/SF path

```bash
# On Workbench host - find the SF interface
ip link show | grep enp3

# Assign IP to SF interface (Workbench side)
sudo ip addr add 10.10.10.1/24 dev enp3s0f0s0
sudo ip link set enp3s0f0s0 up

# On BF3 - assign IP to corresponding interface
ssh ubuntu@100.106.76.67 "sudo ip addr add 10.10.10.2/24 dev enp3s0f0s0"
ssh ubuntu@100.106.76.67 "sudo ip link set enp3s0f0s0 up"

# Test basic connectivity
ping 10.10.10.2

# Test RDMA bandwidth (install perftest if needed)
# On BF3:
ssh ubuntu@100.106.76.67 "ib_send_bw -d mlx5_2"
# On Workbench:
ib_send_bw -d mlx5_X 10.10.10.2

# Run simple MPI test
mpirun -np 2 --host 10.10.10.1,10.10.10.2 hostname
```

### Phase 3: QSFP Network Test (Cable Required)

**Goal**: Test MPI over 100GbE through DPU

**Physical Setup**:
1. Locate Amphenol NJAAKK-N911 QSFP cable
2. Connect Spark QSFP port 0 → BF3 p0
3. Verify link comes up on both ends

```bash
# After connecting cable, on BF3:
ssh ubuntu@100.106.76.67 "ip link show p0"
# Should show: state UP

# On Spark:
ssh nmelo@100.85.204.118 "ip link show enp1s0f0np0"
# Should show: state UP

# Configure IPs
# Spark:
ssh nmelo@100.85.204.118 "sudo ip addr add 10.20.20.1/24 dev enp1s0f0np0"

# BF3 p0 (or configure via OVS):
ssh ubuntu@100.106.76.67 "sudo ip addr add 10.20.20.2/24 dev p0"

# Test connectivity
ssh nmelo@100.85.204.118 "ping 10.20.20.2"

# Test RDMA
# On BF3:
ssh ubuntu@100.106.76.67 "ib_send_bw -d mlx5_0"
# On Spark:
ssh nmelo@100.85.204.118 "ib_send_bw 10.20.20.2"
```

### Phase 4: MPI Collective Offload Test

**Goal**: Test actual DPU offload of MPI collectives

**Requirements**:
- MVAPICH2-DPU or HPC-X with DPU offload enabled
- OSU Micro-Benchmarks

**Install HPC-X** (if not present):
```bash
# Download from NVIDIA (requires account)
# https://developer.nvidia.com/networking/hpc-x

# Or check if available in DOCA repos
ssh ubuntu@100.106.76.67 "apt search hpc-x"
```

**Install OSU Benchmarks**:
```bash
# On each node
wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.3.tar.gz
tar xzf osu-micro-benchmarks-7.3.tar.gz
cd osu-micro-benchmarks-7.3
./configure CC=mpicc CXX=mpicxx
make
```

**Run Benchmark**:
```bash
# Blocking alltoall (baseline)
mpirun -np 4 --host node1,node2,node3,node4 ./osu_alltoall

# Non-blocking alltoall (should show DPU offload benefit)
mpirun -np 4 --host node1,node2,node3,node4 ./osu_ialltoall

# With overlap measurement
mpirun -np 4 --host node1,node2,node3,node4 ./osu_ialltoall -t 1000
```

**Expected Results with DPU Offload**:
- ~100% overlap between computation and communication
- Lower CPU utilization during collectives
- Similar or better latency vs non-offload

---

## Software Installation Checklist

| Package | BF3 | Workbench | Spark | Notes |
|---------|-----|-----------|-------|-------|
| OpenMPI | Installed | Check | Check | Base MPI |
| UCX | Installed | Check | Check | RDMA transport |
| rdma-core | Installed | Check | Check | RDMA userspace |
| perftest | Check | Check | Check | ib_send_bw, etc. |
| HPC-X | NOT installed | NOT installed | Check | For DPU offload |
| OSU Benchmarks | NOT installed | NOT installed | NOT installed | For testing |

---

## Success Criteria

### Phase 2 (Loopback)
- [ ] Ping works between Workbench (10.10.10.1) and BF3 (10.10.10.2)
- [ ] ib_send_bw shows RDMA working over SF path
- [ ] mpirun hostname works across both nodes

### Phase 3 (QSFP)
- [ ] QSFP link UP on both Spark and BF3
- [ ] Ping works over 100GbE path
- [ ] ib_send_bw shows high bandwidth (>10 GB/s)
- [ ] mpirun works across Spark and Workbench

### Phase 4 (Offload)
- [ ] HPC-X or MVAPICH2-DPU installed
- [ ] osu_ialltoall shows >90% overlap
- [ ] CPU utilization lower during collectives vs baseline

---

## Security Testing Integration

Once MPI is working, test identity enforcement:

1. **Baseline**: MPI works without authentication
2. **Add OVS ACL**: Block MPI traffic by MAC/IP
3. **Verify block**: MPI fails with ACL in place
4. **Add identity rule**: Allow only authenticated nodes
5. **Test enforcement**: Authorized nodes can MPI, others blocked

This validates our identity fabric value proposition at the MPI layer.

---

## References

- [MVAPICH2-DPU](https://mvapich.cse.ohio-state.edu/userguide/dpu/)
- [HPC-X Download](https://developer.nvidia.com/networking/hpc-x)
- [OSU Benchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/)
- [UCX Documentation](https://openucx.readthedocs.io/)
