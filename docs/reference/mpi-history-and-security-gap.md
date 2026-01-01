# MPI: History and the Security Gap

> **Purpose**: Background context on why MPI lacks authentication
> **Created**: 2024-12-23

## Executive Summary

MPI (Message Passing Interface) is the dominant parallel programming model for HPC,
used by virtually every supercomputer and AI cluster. Its security model was designed
in 1992 for closed, trusted environments. Thirty years later, MPI still has no
standard authentication mechanism, creating a significant vulnerability in modern
multi-tenant AI infrastructure.

---

## The Pre-MPI Era (1980s)

Before MPI, parallel computing was fragmented. Every vendor shipped proprietary
message passing libraries:

| System | Library | Vendor Lock-in |
|--------|---------|----------------|
| Intel iPSC | NX | Intel-only |
| nCUBE | Vertex | nCUBE-only |
| Thinking Machines CM-5 | CMMD | CM-only |
| IBM SP | EUI | IBM-only |
| Cray T3D | SHMEM | Cray-only |
| Meiko CS-2 | CSTools | Meiko-only |

**The problem**: Scientists wrote code for one machine. Moving to another meant
rewriting everything. More time was spent porting than computing.

---

## PVM: The First Portable Solution (1989)

**Parallel Virtual Machine** emerged from Oak Ridge National Laboratory.

**Innovation**: Abstract the network, let scientists focus on algorithms.

```c
// PVM code could run on heterogeneous workstation clusters
pvm_spawn("worker", NULL, PvmTaskDefault, "", nprocs, tids);
pvm_send(tid, msgtag);
pvm_recv(-1, msgtag);
```

**Limitations**:
- Inconsistent semantics across operations
- Performance issues on tightly-coupled systems
- No formal specification (implementation was the standard)
- Difficult to optimize for vendor hardware

PVM proved portability was possible but showed the need for a real standard.

---

## The MPI Forum (1992-1994)

### The Founding Meeting

**Supercomputing '92**, Minneapolis, Minnesota.

40+ representatives from vendors, national labs, and universities gathered with
a radical idea: create a *specification*, not an implementation. Let vendors
compete on performance while guaranteeing portability.

### Key Participants

**National Labs**:
- Argonne National Laboratory
- Oak Ridge National Laboratory
- Sandia National Laboratories

**Vendors**:
- IBM
- Intel
- Cray Research
- Convex
- Meiko
- nCUBE
- Thinking Machines

**Universities**:
- University of Tennessee, Knoxville
- University of Edinburgh
- Syracuse University
- Mississippi State

**Notable Individuals**:
- Jack Dongarra (UTK/ORNL) - Lead organizer
- Tony Hey (Southampton)
- David Walker (ORNL)
- Bill Gropp (Argonne)
- Ewing Lusk (Argonne)
- Marc Snir (IBM)

### Design Principles

1. **Portability**: Same code runs everywhere
2. **Performance**: Don't prevent vendor optimizations
3. **Functionality**: Cover common parallel patterns
4. **Practicality**: Based on existing successful systems

### What They Explicitly Did NOT Address

- **Security**: Networks were assumed trusted
- **Fault tolerance**: Jobs died if any process died
- **Dynamic resources**: Fixed process count at launch
- **Authentication**: If you could log in, you were authorized

---

## MPI-1.0 (June 1994)

The first standard defined 127 functions covering:

### Point-to-Point Communication
```c
MPI_Send(buf, count, datatype, dest, tag, comm);
MPI_Recv(buf, count, datatype, source, tag, comm, status);
```

### Collective Operations
```c
MPI_Bcast(buf, count, datatype, root, comm);      // One to all
MPI_Reduce(sendbuf, recvbuf, count, datatype,     // All to one
           op, root, comm);
MPI_Allreduce(sendbuf, recvbuf, count, datatype,  // All to all
              op, comm);
MPI_Alltoall(sendbuf, sendcount, sendtype,        // Transpose
             recvbuf, recvcount, recvtype, comm);
```

### Core Concepts
```c
MPI_Init(&argc, &argv);     // Initialize MPI
MPI_Comm_rank(comm, &rank); // My process ID
MPI_Comm_size(comm, &size); // Total processes
MPI_Finalize();             // Clean shutdown
```

### The Launch Model

MPI-1.0 assumed an external launcher would start all processes:

```bash
# The "standard" way to launch MPI (not part of MPI spec!)
mpirun -np 16 -hostfile nodes.txt ./my_application
```

**How mpirun worked in 1994**:
1. Read hostfile
2. Use `rsh` (remote shell) to start processes on each node
3. Processes connect back to each other
4. Application runs

**rsh security model**: Trust based on `.rhosts` file. No encryption. No authentication beyond "this IP is allowed."

---

## MPI-2.0 (1997)

Added features deferred from MPI-1.0:

### One-Sided Communication (RMA)
```c
MPI_Put(origin_buf, origin_count, origin_datatype,
        target_rank, target_disp, target_count, target_datatype, win);
MPI_Get(...);
```
Remote memory access without receiver participation.

### Dynamic Process Management
```c
MPI_Comm_spawn(command, argv, maxprocs, info, root, comm,
               intercomm, errcodes);
```
Finally could spawn new processes after MPI_Init.

### Parallel I/O (MPI-IO)
```c
MPI_File_open(comm, filename, amode, info, fh);
MPI_File_write_at(fh, offset, buf, count, datatype, status);
```
Coordinated file access for parallel applications.

### C++ Bindings
Added, then later deprecated and removed (MPI-3.0).

---

## The Challenge Years (2000s)

MPI faced criticism for being too complex, too low-level:

| Alternative | Promise | Outcome |
|-------------|---------|---------|
| **OpenMP** | Easy shared memory | Became complementary (MPI+OpenMP hybrid) |
| **UPC** | Unified Parallel C | Niche adoption, limited vendor support |
| **Co-Array Fortran** | Fortran-native parallelism | Added to Fortran standard |
| **Chapel** | Productive HPC language | Ongoing development at HPE |
| **X10** | Safe parallelism | IBM research project, ended |
| **Hadoop/MapReduce** | Big data at scale | Different domain (data, not HPC) |
| **Spark** | In-memory MapReduce | Different domain |

**MPI survived them all.** Reasons:
1. Enormous existing code investment
2. Performance still unmatched for tightly-coupled problems
3. Vendor support and optimization
4. Portability proven across decades of hardware

---

## MPI-3.0 (2012)

Major modernization addressing real-world needs:

### Non-Blocking Collectives (Key for DPU Offload)
```c
MPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);
// Returns immediately, collective happens in background
// ... do computation ...
MPI_Wait(request, status);
```

**This is what BlueField DPUs offload**: The ARM cores execute the collective
while the host CPU continues computing.

### Neighborhood Collectives
```c
MPI_Neighbor_alltoall(sendbuf, sendcount, sendtype,
                      recvbuf, recvcount, recvtype, comm);
```
Efficient sparse communication patterns (stencils, graphs).

### Improved RMA
Better semantics for one-sided operations, memory models.

### Matched Probe
```c
MPI_Mprobe(source, tag, comm, message, status);
MPI_Mrecv(buf, count, datatype, message, status);
```
Safely receive messages in multi-threaded code.

---

## MPI-4.0 (2021)

Latest major revision:

### Persistent Collectives
```c
MPI_Allreduce_init(sendbuf, recvbuf, count, datatype, op, comm,
                   info, request);
// Setup once, then:
MPI_Start(request);  // Very low overhead
MPI_Wait(request, status);
```
Amortize setup cost across many iterations.

### Partitioned Communication
```c
MPI_Psend_init(buf, partitions, count, datatype, dest, tag,
               comm, info, request);
MPI_Pready(partition, request);  // Signal partition ready
```
Fine-grained message construction for streaming.

### Session Model
```c
MPI_Session_init(info, errhandler, session);
MPI_Session_get_pset_info(session, pset_name, info);
```
Alternative to MPI_COMM_WORLD for modular applications.

### Large Count Support
Operations with more than 2 billion elements (finally!).

### Fault Tolerance Hooks
Basic building blocks (not full fault tolerance, but progress).

---

## Major Implementations

| Implementation | Base | Primary Users |
|----------------|------|---------------|
| **MPICH** | Original Argonne reference | Many derivatives |
| **Open MPI** | LAM + FT-MPI + LA-MPI merger | General HPC |
| **Intel MPI** | MPICH-derived | Intel clusters |
| **MVAPICH/MVAPICH2** | MPICH-derived | InfiniBand clusters |
| **MVAPICH2-DPU** | MVAPICH2 | BlueField offload |
| **Cray MPICH** | MPICH-derived | Cray/HPE systems |
| **IBM Spectrum MPI** | Custom | IBM POWER systems |
| **NVIDIA HPC-X** | Open MPI-based | NVIDIA/Mellanox hardware |
| **Microsoft MPI** | MPICH-derived | Windows HPC |

---

## The Security Model (Or Lack Thereof)

### What MPI Specifies About Security

**Nothing.**

The MPI standard does not mention:
- Authentication
- Authorization
- Encryption
- Integrity checking
- Access control

### Why?

**1992 Context**:
- Supercomputers were physically isolated
- Users had accounts managed by sysadmins
- Networks were private and trusted
- Multi-tenancy meant "time sharing," not "shared infrastructure"
- Security meant "don't let unauthorized people into the building"

### The Launch Security Evolution

```
1994: rsh (remote shell)
      └── .rhosts trust model
      └── No encryption
      └── "If your IP is listed, you're trusted"

2000s: SSH replaces rsh
      └── Encrypted channel
      └── Key-based authentication
      └── But: keys copied everywhere, never rotated

2010s: Job schedulers (Slurm, PBS, LSF)
      └── Scheduler handles process launch
      └── Munge tokens for authentication
      └── But: shared symmetric key, no hardware binding

2020s: Still no standard
      └── Each site does something different
      └── Multi-tenant clouds improvise
      └── No hardware-based trust
```

### What Implementations Actually Do

| Implementation | Launch Method | Auth Mechanism |
|----------------|---------------|----------------|
| Open MPI | SSH, Slurm, PBS | Whatever launcher provides |
| MPICH | SSH, Slurm, Hydra | Whatever launcher provides |
| Intel MPI | SSH, Slurm | Whatever launcher provides |
| MVAPICH2 | SSH, Slurm | Whatever launcher provides |
| Cray MPI | ALPS/Slurm | Scheduler-specific |

**Common thread**: MPI implementations delegate security to the launcher, and
launchers delegate to SSH or scheduler-specific tokens.

---

## The Modern Problem

### 1992 Assumptions vs. 2024 Reality

| Assumption (1992) | Reality (2024) |
|-------------------|----------------|
| Closed network | Cloud, multi-tenant |
| Trusted nodes | VMs, containers, shared hardware |
| Single organization | Multiple tenants, customers |
| Physical access control | API-based provisioning |
| Long-lived allocations | Spot instances, preemption |
| Homogeneous systems | Heterogeneous (x86, ARM, GPU) |

### The Attack Surface

```
1. SSH keys distributed to all nodes (static, unrotated)
2. Keys stored on shared filesystems (NFS, Lustre)
3. Keys baked into container images
4. Keys in VM snapshots and backups
5. No binding between job identity and network traffic
6. Compromised node can impersonate any MPI rank
7. No audit trail of MPI communication
```

### Real-World Implications

**AI Cloud Providers** (CoreWeave, Lambda Labs, Together AI):
- Offer GPU clusters to multiple customers
- Customers run arbitrary code (training jobs)
- Traditional MPI security model fails completely

**Enterprise GPU Clusters**:
- Data scientists SSH in with personal keys
- Keys exist on every node, every backup
- One compromised laptop = cluster-wide access

---

## The Opportunity

### What's Missing

| Requirement | Current State | Needed |
|-------------|---------------|--------|
| Hardware-bound identity | Keys on disk | TPM/DICE attestation |
| Short-lived credentials | Static SSH keys | Job-scoped certificates |
| Network enforcement | Trust the wire | DPU policy enforcement |
| Mutual authentication | One-way (client to server) | Both endpoints verified |
| Audit trail | SSH logs only | Full communication audit |

### Our Value Proposition

```
NVIDIA's Position: "We accelerate MPI" (DPU offloads collectives)
Our Position:      "We secure MPI" (DPU enforces identity)

Combined:          Fast AND secure MPI for multi-tenant AI
```

### Integration Point

BlueField DPU sits in the data path. It already handles MPI traffic for
acceleration. Adding identity verification at the same point means:

1. Zero additional latency (verification at line rate)
2. Zero CPU overhead (DPU handles it)
3. Cryptographic proof (TPM-bound certs)
4. Job-scoped access (certs expire with job)

---

## Timeline Summary

```
1989    PVM released (Oak Ridge) - first portable message passing
1992    MPI Forum formed at Supercomputing '92
1993    MPI draft circulated
1994    MPI-1.0 published (127 functions)
1995    MPICH 1.0 released (reference implementation)
1997    MPI-2.0 published (one-sided, dynamic, I/O)
2004    Open MPI project begins (merger of three projects)
2008    MPI-2.1, 2.2 (clarifications, bug fixes)
2012    MPI-3.0 (non-blocking collectives) ← DPU offload target
2015    MPI-3.1 (minor updates)
2021    MPI-4.0 (sessions, large counts, fault tolerance hooks)
2023    MPI-4.1 (in progress)

Security additions: NONE in 30 years of MPI standards
```

---

## References

### Official Sources
- [MPI Forum](https://www.mpi-forum.org/) - Official standards body
- [MPI-4.0 Standard](https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf)

### Implementations
- [MPICH](https://www.mpich.org/)
- [Open MPI](https://www.open-mpi.org/)
- [MVAPICH2](https://mvapich.cse.ohio-state.edu/)
- [MVAPICH2-DPU](https://mvapich.cse.ohio-state.edu/userguide/dpu/)
- [NVIDIA HPC-X](https://developer.nvidia.com/networking/hpc-x)

### Historical
- Gropp, Lusk, Skjellum. "Using MPI" (1994) - The original MPI book
- Dongarra et al. "A Message Passing Standard for MPP and Workstations"
- Walker, Dongarra. "MPI: A Standard Message Passing Interface"

### Security Research
- "Security Analysis of HPC Systems" (various NIST/NSF reports)
- "SSH Key Management in Scientific Computing" (SC conference papers)
