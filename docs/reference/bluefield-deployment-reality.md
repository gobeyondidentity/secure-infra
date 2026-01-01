# Bluefield Deployment in AI Infrastructure: Reference vs. Reality

**Created**: 2025-12-19
**Purpose**: Understand how Bluefield DPUs are actually deployed in AI factories to inform Beyond Identity's product strategy.

---

## Executive Summary

NVIDIA's reference designs for GB200/NVL72 AI infrastructure include Bluefield-3 DPUs as a core component. However, **most customers are skipping Bluefield deployment entirely** to reduce cost and complexity. This creates both a challenge (smaller immediate TAM) and an opportunity (Beyond Identity can articulate the security value proposition that drives BF3 adoption).

**Key Insight**: The firms skipping BF3 are losing the hardware-enforced security enforcement point. They're relying on host-based security which doesn't work for AI workloads that "talk at the network level."

---

## NVIDIA Reference Architecture

### GB200 NVL36 Compute Tray (Per Diagrams)

| Component | Reference Design | Actual Deployments |
|-----------|------------------|-------------------|
| Bluefield-3 DPUs | 2 per compute tray (one per Grace CPU) | **Most firms skip entirely** |
| ConnectX-7 NICs | 4 per tray for backend fabric | Typically deployed |
| Frontend bandwidth | 800 Gbps per tray | Depopulated to ~200 Gbps |
| Backend bandwidth | 400-800G per GPU | Maintained (critical path) |

### NVL72 Reference Design

| Component | Specification |
|-----------|---------------|
| GPUs | 72 Blackwell GPUs |
| CPUs | 36 Grace CPUs |
| NVSwitches | 18 |
| Backend NICs (InfiniBand) | 72 (1:1 with GPUs) |
| Frontend NICs (Bluefield-3) | 36 (2:1 with GPUs) |
| NVLink cables | 5,000+ cables, 2 miles of copper |

### Network Segmentation in Reference Design

The architecture defines six distinct networks:

| Network | Purpose | Bluefield Role |
|---------|---------|----------------|
| **External-net** | North-South (NAS, corp, internet) | BF3 controls via OVS, 400Gbps |
| **Internal-net** | In-band mgmt (SSH, provisioning, telemetry) | BF3 controls, 200Gbps |
| **IPMI-net** | OOB management (BMC) | Separate, not through BF3 |
| **Fabric-net** | East-West (GPU-to-switch) | ConnectX-7, not BF3 |
| **NVLink-net** | GPU-to-GPU within rack | Direct copper, no NIC |
| **NVLink-sw** | NVLink switch management | 1G ports |

**Critical Observation**: Bluefield-3 controls the External-net and Internal-net through its OVS Switch + Rules engine. This is the identity enforcement point.

---

## Reality: Who Actually Deploys Bluefield-3?

### Current Adoption

| Customer | Bluefield-3 Usage | Source | Notes |
|----------|-------------------|--------|-------|
| **Oracle** | Yes - DPU mode | GTC DC Oct 2025, SemiAnalysis | Frontend networking virtualization for OCI; publicly confirmed |
| **CoreWeave** | Yes | GTC DC Oct 2025 | GPU cloud multi-tenancy (VFs, virtio) |
| **xAI** | Yes - NIC mode only | SemiAnalysis | Workaround for early Spectrum-X Ethernet limitations |
| **Most hyperscalers** | No | SemiAnalysis | Cost reduction, no compelling use case |
| **Enterprise** | No | SemiAnalysis | Following hyperscaler patterns |

### Oracle BF3 Confirmation (GTC DC October 2025)

Jensen's keynote showed Oracle's NVIDIA partnership slide explicitly listing **BF3** as part of their networking stack:

![Oracle NVIDIA BF3](oracle-nvidia-bf3-gtc-oct2025.png)

**Oracle Infrastructure Stack (from slide)**:
- **Networking**: Spectrum-X, InfiniBand, CX7, **BF3**
- **Compute**: Blackwell, Hopper, Grace Hopper
- **Scale**: Zettascale Supercluster (800,000 NVIDIA GPUs, 16 ZettaFLOPS)
- **Software**: NeMo, NIM, TensorRT, DGX Cloud

**Implication**: Oracle is a validated target customer with publicly confirmed BF3 deployment. They use BF3 for OCI virtualization (SR-IOV, virtio). Beyond Identity can approach Oracle about adding identity enforcement to their existing DPU infrastructure.

### Why Firms Skip Bluefield-3

1. **Cost**: BF3 adds significant cost per compute tray with unclear ROI
2. **Complexity**: DPU mode requires additional management and orchestration
3. **No compelling use case**: Basic networking works without DPU offload
4. **ConnectX-8 coming**: Mid-2025, 800Gb/s per GPU, further reduces BF3 dependency

> "Most customers will not opt for this additional cost... With Spectrum-X800 Ultra and ConnectX-8, Bluefield dependency disappears entirely."
> -- SemiAnalysis, GB200 Hardware Architecture

### Depopulation Economics

Firms are aggressively depopulating reference designs:

| Component | Reference | Typical Deployment | Savings |
|-----------|-----------|-------------------|---------|
| Frontend NICs | 800 Gbps/tray | 200 Gbps/tray | $3.5k/system (transceivers) |
| Backend switches | 4-rail optics | ToR with DAC/ACC | $32k/system |
| Bluefield-3 | 2 per tray | 0 | Significant (TBD) |

---

## NVL72 vs. NVL36x2: What's Actually Deployed?

### NVL72 (Single 120kW Rack)
- **Status**: Largely theoretical
- **Challenge**: 120kW per-rack power density exceeds most datacenter infrastructure
- **Adoption**: Only one hyperscaler plans primary deployment

### NVL36x2 (Two Racks)
- **Status**: Actual production standard
- **Advantage**: Fits existing datacenter power/cooling infrastructure
- **Trade-off**: Extra NVLink hop between racks (negligible impact for training)

**Implication**: Even customers buying GB200 are choosing the more conservative two-rack configuration, suggesting risk aversion and cost sensitivity.

---

## Gigawatt AI Factory: DPU Market Sizing

Jensen Huang stated at GTC DC October 2025 that a 1GW AI factory would have **16 wide x 500 deep racks** of compute infrastructure.

### Validation of Jensen's Claim

| Configuration | Power/Rack | Total Racks | Total Power |
|--------------|------------|-------------|-------------|
| NVL72 (dense, 1 rack) | 120 kW | 8,000 | 960 MW ≈ **1 GW** ✓ |
| NVL36x2 (production standard) | 60 kW | 8,000 | 480 MW |

Jensen assumes the dense NVL72 configuration (120kW/rack). Purpose-built AI factories (xAI Colossus, hyperscaler new builds) can handle this density. Retrofitted datacenters typically use NVL36x2.

### DPU Count per 1GW Facility

From the NVL72 reference architecture:
- 72 GPUs, 36 Grace CPUs, 18 compute trays per rack
- Reference design: 2 BF3 per compute tray = 36 BF3 per NVL72
- Depopulated (Supermicro SC25 pattern): 1 BF3 per tray = 18 BF3 per NVL72

| Config | BF3/Tray | BF3/Rack | 8,000 Racks (1GW) |
|--------|----------|----------|-------------------|
| **Reference** | 2 | 36 | **288,000 DPUs** |
| **Depopulated** | 1 | 18 | **144,000 DPUs** |
| **Current Reality** | 0 | 0 | **0 DPUs** |

### Market Sizing Implications

**Per Gigawatt Facility:**
- Reference TAM: 288,000 DPUs needing identity enforcement
- Realistic TAM: 144,000 DPUs (if following Supermicro depopulation pattern)
- Current TAM: Near zero (most firms skip BF3 entirely)

**The Opportunity Gap**: Most firms deploy 0 BF3s today, relying on ConnectX-7 for backend fabric only. Beyond Identity's value proposition is providing the security justification for deploying 144K-288K DPUs per gigawatt facility.

**Scale Context**: xAI's Colossus is reportedly 150MW today, scaling to 1GW. At full scale with reference BF3 deployment, that's 288,000 enforcement points for identity-based access control.

---

## Networking Deep Dive

### ConnectX-7 Architecture

Each GPU connects directly to its ConnectX-7 NIC through a Broadcom PCIe switch:
- Direct RDMA access without CPU or PCIe root complex involvement
- Enables GPU-to-GPU communication at line rate
- Backend scale-out maintains 1:1 NIC-to-GPU ratio (critical)

### Cedar Fever Module (DGX H100)

Consolidates 4x ConnectX-7 into a single module:
- 33% reduction in transceiver count
- $900k savings per 256-GPU scalable unit
- Adopted by Microsoft Azure, NVIDIA DGX
- Traditional PCIe architecture used by Meta, HPE, Dell, Supermicro

### InfiniBand vs. Ethernet

| Aspect | InfiniBand | RoCE Ethernet |
|--------|------------|---------------|
| Backend scale-out | Traditional choice | Emerging (Spectrum-X) |
| Performance | Proven at scale | Catching up |
| BF3 dependency | Lower | Higher (early Spectrum-X) |

---

## Implications for Beyond Identity

### The Opportunity

1. **Security as the reason to deploy BF3**: If Beyond Identity can articulate why hardware-enforced identity requires BF3, we create demand for a component customers currently skip.

2. **Oracle as reference customer**: Oracle deploys BF3 for virtualization. Security is a natural extension of their use case.

3. **Federal/defense mandate potential**: Government buyers may require hardware-enforced security, forcing BF3 adoption regardless of cost.

4. **AI cloud providers need differentiation**: CoreWeave, Lambda Labs could use "hardware-secured AI infrastructure" as competitive positioning.

### The Challenge

1. **Timing**: ConnectX-8 (mid-2025) further reduces BF3 dependency for basic networking
2. **Cost sensitivity**: Customers are aggressively depopulating; adding BF3 for security is a hard sell without clear ROI
3. **Alternative enforcement points**: Could identity be enforced elsewhere (host, switch, gateway)?

### Strategic Questions

1. **Can identity enforcement be the "killer app" that drives BF3 adoption?**
2. **Should we target Oracle (already deploying BF3) vs. trying to convince non-deployers?**
3. **Is there a path through NVIDIA (bundle security with DGX/HGX reference)?**
4. **Can we enforce identity without BF3 (ConnectX-7 in NIC mode, host-based)?**

---

## Bluefield-4 Announcement (GTC DC October 2025)

Jensen Huang announced Bluefield-4 at GTC Washington D.C. in October 2025:

![Bluefield-4 Announcement](bluefield-4-announcement-gtc-oct2025.png)

| Spec | Bluefield-3 | Bluefield-4 |
|------|-------------|-------------|
| Speed | 400Gbps | 800Gbps |
| CPU | 16-core ARM Cortex-A78 | 64-core Grace CPU |
| NIC | ConnectX-7 | ConnectX-9 |
| Transistors | ~22B | 126B |
| Target | AI Infrastructure | AI Factories |

**Key Features:**
- 800G SmartNIC for AI Factories
- 64-core Grace CPU (same architecture as DGX Grace Hopper)
- ConnectX-9 networking
- AI Data Storage Acceleration
- 126 billion transistors

**Strategic Implication**: BF4 addresses the ConnectX-8 concern. While ConnectX-8 (800Gbps NIC-only) reduces BF3 dependency for basic networking, BF4 keeps pace with 800Gbps speeds while adding massive compute (64 Grace cores vs 16 ARM cores). The DPU remains the programmable datapath for security/policy enforcement, now with 4x the CPU power.

---

## Sources

1. [GB200 Hardware Architecture and Component Analysis](https://newsletter.semianalysis.com/p/gb200-hardware-architecture-and-component) - SemiAnalysis
2. [NVIDIA's Optical Boogeyman: NVL72, InfiniBand](https://newsletter.semianalysis.com/p/nvidias-optical-boogeyman-nvl72-infiniband) - SemiAnalysis
3. [NVIDIA ConnectX-7: Cedar Fever](https://pytorchtoatoms.substack.com/p/nvidia-connectx-7-16tbits-cedar-fever) - PyTorch to Atoms
4. Architecture diagrams from Nelson's research (Identity Fabric for the AI Factory.zip)
5. GTC Washington D.C. October 2025 Keynote - Bluefield-4 announcement, 1GW datacenter sizing (16x500 racks)
6. GTC Washington D.C. October 2025 Keynote - Oracle/NVIDIA partnership slide showing BF3 in networking stack

---

## Appendix: Network Definitions (From Architecture Diagram)

| Network | Description | Bandwidth | Via |
|---------|-------------|-----------|-----|
| External-net | "North South network" to NAS, corp network, internet | 400Gbps | 2x BF3 ConnectX-7 |
| Internal-net | "In-band mgmt network" for SSH, provisioning, orchestration, telemetry | 200Gbps | 2x BF3 or ConnectX-7 |
| IPMI-net | OOB Management, connecting BMCs | Low | Dedicated |
| Fabric-net | "East West network" connecting CX7 to compute IB/Ethernet switch | High | 4x CX7 per tray |
| NVLink-net | GPU-to-GPU communication | 900GB/s/GPU | Direct copper |
| NVLink-sw | NVLink switch management | 2x 1G | Dedicated |
