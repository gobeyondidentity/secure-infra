# DOCA Flow Overview

## Introduction

DOCA Flow is NVIDIA's hardware-accelerated packet processing framework for ConnectX and Bluefield devices. It enables line-rate (200+ Gbps) packet classification, modification, and forwarding using hardware offload capabilities.

**Key Concept**: DOCA Flow moves packet processing from CPU to hardware, enabling zero-overhead network policy enforcement at wire speed.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│         (Beyond Identity Integration, Policy Logic)      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    DOCA Flow API                         │
│  - Flow Initialization                                   │
│  - Port Management                                       │
│  - Pipe Creation (Match/Action Rules)                   │
│  - Entry Management                                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 Hardware Offload Engine                  │
│              (ConnectX-7 / Bluefield DPU)               │
│  - Line-rate packet matching (200+ Gbps)                │
│  - Hardware flow tables                                  │
│  - Zero CPU overhead                                     │
└─────────────────────────────────────────────────────────┘
```

### Flow Processing Pipeline

```
Incoming Packet
      │
      ▼
┌──────────┐     Match?     ┌──────────┐
│  Pipe 1  │────────────────▶│ Actions  │──┐
│ (Root)   │     YES         │ (modify) │  │
└──────────┘                 └──────────┘  │
      │                                      │
      │ NO                                   │
      ▼                                      ▼
┌──────────┐     Match?     ┌──────────┐  Forward
│  Pipe 2  │────────────────▶│ Actions  │──┐
└──────────┘     YES         └──────────┘  │
      │                                      │
      │ NO                                   │
      ▼                                      ▼
┌──────────┐                           ┌─────────┐
│   Drop   │                           │ Output  │
└──────────┘                           │  Port   │
                                       └─────────┘
```

## API Patterns (DOCA 3.1.0)

### 1. Initialization

DOCA Flow uses **opaque structures** that must be created through API functions:

```c
#include <doca_flow.h>

/* Initialize DOCA Flow (one-time setup) */
struct doca_flow_cfg *flow_cfg;

doca_flow_cfg_create(&flow_cfg);
doca_flow_cfg_set_pipe_queues(flow_cfg, nb_queues);
doca_flow_cfg_set_mode_args(flow_cfg, "vnf,hws");  // VNF mode, HW steering
doca_flow_cfg_set_nr_counters(flow_cfg, 1024);

doca_error_t result = doca_flow_init(flow_cfg);
doca_flow_cfg_destroy(flow_cfg);
```

**Key Points:**
- Must be called before any other DOCA Flow operations
- Allocates hardware resources
- Configures number of queues (for multi-threading)
- Sets operating mode (VNF, switch, etc.)

### 2. Port Creation

Ports represent physical or virtual network interfaces:

```c
struct doca_flow_port_cfg *port_cfg;
struct doca_flow_port *port;

/* Create port configuration */
doca_flow_port_cfg_create(&port_cfg);
doca_flow_port_cfg_set_dev(port_cfg, doca_dev);          // DOCA device
doca_flow_port_cfg_set_port_id(port_cfg, port_id);       // DPDK port ID
doca_flow_port_cfg_set_devargs(port_cfg, "port_args");

/* Start the port */
doca_flow_port_start(port_cfg, &port);
doca_flow_port_cfg_destroy(port_cfg);
```

**Requirements:**
- Requires DPDK initialization (DPDK port must exist)
- Each port maps to a physical/virtual NIC
- Supports multiple ports per application

### 3. Pipe Creation

Pipes define packet processing logic:

```c
struct doca_flow_pipe_cfg *pipe_cfg;
struct doca_flow_pipe *pipe;
struct doca_flow_match match = {0};
struct doca_flow_actions actions = {0};
struct doca_flow_fwd fwd = {0};

/* Define match criteria (which packets to process) */
match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
match.outer.ip4.dst_ip = 0xffffffff;  // Changeable field (set per entry)
match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
match.outer.tcp.l4_port.dst_port = 0xffff;  // Changeable

/* Define actions (what to do with matched packets) */
// Actions can modify packets, set metadata, count, etc.

/* Define forwarding behavior */
fwd.type = DOCA_FLOW_FWD_PORT;
fwd.port_id = 1;  // Forward to port 1

/* Create pipe configuration */
doca_flow_pipe_cfg_create(&pipe_cfg, port);
doca_flow_pipe_cfg_set_name(pipe_cfg, "MY_PIPE");
doca_flow_pipe_cfg_set_type(pipe_cfg, DOCA_FLOW_PIPE_BASIC);
doca_flow_pipe_cfg_set_is_root(pipe_cfg, true);  // Root pipe (first in chain)
doca_flow_pipe_cfg_set_match(pipe_cfg, &match, NULL);
doca_flow_pipe_cfg_set_actions(pipe_cfg, &actions, NULL, NULL, 1);

/* Create the pipe */
doca_flow_pipe_create(pipe_cfg, &fwd, NULL, &pipe);
doca_flow_pipe_cfg_destroy(pipe_cfg);
```

**Match Field Types:**
- **Ignore** (0x00000000): Match any value
- **Constant** (specific value): All entries use same value
- **Changeable** (0xffffffff): Value specified per entry

### 4. Entry Creation

Entries are specific match/action instances:

```c
struct doca_flow_match entry_match = {0};
struct doca_flow_actions entry_actions = {0};
struct doca_flow_pipe_entry *entry;

/* Set specific values for changeable fields */
entry_match.outer.ip4.dst_ip = BE_IPV4_ADDR(192, 168, 1, 100);
entry_match.outer.tcp.l4_port.dst_port = htons(443);

/* Add entry to pipe */
doca_flow_pipe_add_entry(
    0,                  // Queue ID
    pipe,               // Pipe to add to
    &entry_match,       // Match values
    &entry_actions,     // Action values
    NULL,               // Monitor
    NULL,               // FWD override
    0,                  // Flags
    NULL,               // User context
    &entry             // Output entry handle
);

/* Process entries (commit to hardware) */
doca_flow_entries_process(port, 0, 1000, 0);
```

### 5. Cleanup

```c
/* Stop port */
doca_flow_port_stop(port);

/* Destroy DOCA Flow */
doca_flow_destroy();
```

## Match Criteria Reference

### Layer 3 (IP)

```c
/* IPv4 */
match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
match.outer.ip4.src_ip = source_ip;
match.outer.ip4.dst_ip = dest_ip;
match.outer.ip4.ttl = ttl_value;

/* IPv6 */
match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP6;
SET_IPV6_ADDR(match.outer.ip6.src_ip, a, b, c, d);
SET_IPV6_ADDR(match.outer.ip6.dst_ip, a, b, c, d);
```

### Layer 4 (TCP/UDP)

```c
/* TCP */
match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
match.outer.tcp.l4_port.src_port = htons(src_port);
match.outer.tcp.l4_port.dst_port = htons(dst_port);
match.outer.tcp.flags = tcp_flags;

/* UDP */
match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
match.outer.udp.l4_port.src_port = htons(src_port);
match.outer.udp.l4_port.dst_port = htons(dst_port);
```

### Metadata

```c
/* Match on metadata set by previous pipe */
match.meta.pkt_meta = meta_value;
match.meta.mark = mark_value;
```

## Actions Reference

### Packet Modification

```c
/* Modify IP address */
actions.outer.ip4.src_ip = new_src_ip;
actions.outer.ip4.dst_ip = new_dst_ip;

/* Modify MAC address */
SET_MAC_ADDR(actions.outer.eth.src_mac, a, b, c, d, e, f);
SET_MAC_ADDR(actions.outer.eth.dst_mac, a, b, c, d, e, f);

/* Decrement TTL */
actions.outer.ip4.ttl = ttl - 1;
```

### Metadata Actions

```c
/* Set metadata for next pipe */
actions.meta.pkt_meta = metadata_value;
actions.meta.mark = mark_value;
```

### Counting

```c
/* Attach counter to entry */
struct doca_flow_monitor monitor = {0};
monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;

doca_flow_pipe_add_entry(queue_id, pipe, &match, &actions,
                          &monitor, NULL, 0, NULL, &entry);
```

## Forward Types

### Port Forwarding

```c
fwd.type = DOCA_FLOW_FWD_PORT;
fwd.port_id = target_port_id;
```

### Pipe Chaining

```c
fwd.type = DOCA_FLOW_FWD_PIPE;
fwd.next_pipe = next_pipe;
```

### RSS (Receive Side Scaling)

```c
uint16_t rss_queues[] = {0, 1, 2, 3};

fwd.type = DOCA_FLOW_FWD_RSS;
fwd.rss_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
fwd.num_of_queues = 4;
fwd.rss_queues = rss_queues;
```

### Drop

```c
fwd.type = DOCA_FLOW_FWD_DROP;
```

## Beyond Identity Integration Use Cases

### Use Case 1: Certificate-Based Access Control

**Scenario**: Allow only authenticated users (valid client certificates) to access GPU resources.

```c
/* Pipe 1: Check if TLS client cert is valid (via metadata from TLS offload) */
struct doca_flow_match match = {0};
match.meta.pkt_meta = 0xffffffff;  // Metadata set by TLS layer

/* Entry 1: Valid cert (meta = 1) → Forward to GPU network */
entry_match.meta.pkt_meta = CERT_VALID;
fwd.type = DOCA_FLOW_FWD_PORT;
fwd.port_id = GPU_PORT;

/* Entry 2: Invalid cert (meta = 0) → Drop */
entry_match.meta.pkt_meta = CERT_INVALID;
fwd.type = DOCA_FLOW_FWD_DROP;
```

### Use Case 2: Micro-Segmentation by Certificate Claims

**Scenario**: Route traffic based on certificate subject/OU fields.

```c
/* After TLS handshake, metadata contains certificate claims */

/* DevOps team (OU=DevOps) → Port 1 (production network) */
entry_match.meta.pkt_meta = OU_DEVOPS;
fwd.port_id = PROD_PORT;

/* Analytics team (OU=Analytics) → Port 2 (data network) */
entry_match.meta.pkt_meta = OU_ANALYTICS;
fwd.port_id = DATA_PORT;

/* Guests (OU=Guest) → Port 3 (restricted network) */
entry_match.meta.pkt_meta = OU_GUEST;
fwd.port_id = GUEST_PORT;
```

### Use Case 3: Rate Limiting Unauthenticated Traffic

**Scenario**: Allow authenticated users full speed, rate-limit unknown traffic.

```c
/* Authenticated users → No rate limit */
entry_match.meta.pkt_meta = AUTHENTICATED;
fwd.type = DOCA_FLOW_FWD_PORT;
fwd.port_id = UNRESTRICTED_PORT;

/* Unauthenticated → Rate-limited queue */
entry_match.meta.pkt_meta = UNAUTHENTICATED;
fwd.type = DOCA_FLOW_FWD_RSS;
fwd.rss_queues = rate_limited_queues;  // Smaller queue set
```

### Use Case 4: Expired Certificate Redirect

**Scenario**: Redirect users with expired certificates to renewal portal.

```c
/* Valid cert → Forward normally */
entry_match.meta.pkt_meta = CERT_VALID;
fwd.type = DOCA_FLOW_FWD_PORT;
fwd.port_id = NORMAL_PORT;

/* Expired cert → Modify destination IP to renewal portal */
entry_match.meta.pkt_meta = CERT_EXPIRED;
actions.outer.ip4.dst_ip = RENEWAL_PORTAL_IP;
fwd.type = DOCA_FLOW_FWD_PORT;
fwd.port_id = MGMT_PORT;
```

## Integration with TLS Offload

DOCA Flow works alongside kTLS for complete solution:

```
1. Packet arrives
        │
        ▼
2. TLS Handshake (OpenSSL + kTLS)
   - Client presents certificate
   - Server validates certificate
   - Extract certificate claims (subject, OU, device ID)
        │
        ▼
3. Set Packet Metadata based on cert status
   - metadata = CERT_VALID | OU_VALUE | DEVICE_ID
        │
        ▼
4. DOCA Flow matches on metadata
   - Forward to appropriate network segment
   - Apply rate limits
   - Count packets for audit
        │
        ▼
5. Packet delivered to destination
```

## DPDK Integration Requirements

DOCA Flow requires DPDK for packet I/O:

### 1. DPDK Installation

```bash
# Install DPDK
apt-get install dpdk dpdk-dev

# Or build from source
wget https://fast.dpdk.org/rel/dpdk-23.11.tar.xz
tar xf dpdk-23.11.tar.xz
cd dpdk-23.11
meson build
ninja -C build
ninja -C build install
```

### 2. Huge Pages Configuration

```bash
# Allocate 1GB huge pages
echo 512 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# Mount huge pages
mkdir -p /mnt/huge
mount -t hugetlbfs nodev /mnt/huge

# Verify
cat /proc/meminfo | grep Huge
```

### 3. Bind Network Interfaces to DPDK

```bash
# Check current driver
lspci -nn | grep ConnectX

# Unbind from kernel driver
dpdk-devbind.py --unbind 0000:01:00.0

# Bind to DPDK driver (vfio-pci recommended)
modprobe vfio-pci
dpdk-devbind.py --bind=vfio-pci 0000:01:00.0

# Verify
dpdk-devbind.py --status
```

### 4. DPDK EAL Initialization in Code

```c
#include <rte_eal.h>
#include <rte_ethdev.h>

int main(int argc, char **argv)
{
    int ret;

    /* Initialize DPDK EAL */
    ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "EAL initialization failed\n");

    argc -= ret;
    argv += ret;

    /* Configure ports */
    struct rte_eth_conf port_conf = {0};
    rte_eth_dev_configure(port_id, nb_rx_queues, nb_tx_queues, &port_conf);

    /* Setup RX/TX queues */
    rte_eth_rx_queue_setup(port_id, 0, nb_rxd, socket_id, NULL, mbuf_pool);
    rte_eth_tx_queue_setup(port_id, 0, nb_txd, socket_id, NULL);

    /* Start device */
    rte_eth_dev_start(port_id);

    /* Now initialize DOCA Flow */
    doca_flow_init(&flow_cfg);

    // ... rest of DOCA Flow code
}
```

## Performance Characteristics

### Hardware Offload Benefits

| Metric | Software (CPU) | DOCA Flow (Hardware) |
|--------|----------------|----------------------|
| Throughput | ~20-40 Gbps per core | 200+ Gbps |
| Latency | 100+ microseconds | <10 microseconds |
| CPU Usage | 100% (dedicated cores) | 0% (full offload) |
| Rules Capacity | Thousands | Millions |
| Scalability | Limited by CPU cores | Limited by NIC memory |

### Typical Performance Numbers

- **Flow table size**: 1M+ concurrent flows (hardware dependent)
- **Rule insertion rate**: 100K+ rules/second
- **Packet processing**: Line rate (200 Gbps for ConnectX-7)
- **Latency overhead**: <5 microseconds for flow lookup
- **Memory footprint**: Minimal host memory (offloaded to NIC)

## Limitations and Considerations

### ConnectX-7 vs Bluefield DPU

| Feature | ConnectX-7 (Spark) | Bluefield DPU |
|---------|-------------------|----------------|
| DOCA Flow | ✅ Full support | ✅ Full support |
| Packet processing | ✅ Hardware offload | ✅ Hardware offload |
| Control plane | ⚠️ Runs on host CPU | ✅ Isolated ARM cores |
| Fail-safe | ❌ Host compromise affects policy | ✅ DPU isolated from host |
| Attestation | ❌ No DPU attestation | ✅ DICE/SPDM attestation |
| Services | ❌ Must run on host | ✅ Run on DPU (BGP, EVPN) |

### API Limitations

- **Match field constraints**: L3/L4 types cannot be changeable
- **Action limitations**: Some fields cannot be modified in hardware
- **Pipe chaining depth**: Limited number of pipe stages
- **Entry capacity**: Hardware flow table size limits
- **DPDK dependency**: Cannot run without DPDK initialization

### Security Considerations

- **Metadata trust**: Metadata set by TLS layer must be protected
- **Flow table overflow**: What happens when hardware table is full?
- **Fail-open vs fail-closed**: Define behavior when hardware fails
- **Audit logging**: Flow statistics must be exported to SIEM
- **Certificate revocation**: How quickly can flow entries be updated?

## Development Workflow

### 1. Development (without DPU hardware)

- Use DOCA Host SDK on Spark system
- Develop and compile DOCA Flow code
- Unit test match/action logic
- Simulate packet flows
- Validate policy rules

### 2. Testing (with hardware)

- Deploy to system with ConnectX-7 or Bluefield
- Initialize DPDK and DOCA Flow
- Program flow tables
- Send test traffic
- Measure performance and validate behavior

### 3. Production Deployment

- Integrate with Beyond Identity certificate issuance
- Connect to SIEM for audit logging
- Set up monitoring and alerting
- Define incident response procedures
- Plan for certificate rotation and policy updates

## Sample Code Structure

```c
/* Complete DOCA Flow application structure */

#include <doca_flow.h>
#include <rte_eal.h>

/* 1. Initialize DPDK */
int init_dpdk(int argc, char **argv);

/* 2. Initialize DOCA Flow */
doca_error_t init_flow(int nb_queues);

/* 3. Create ports */
doca_error_t create_ports(struct doca_dev *dev,
                          struct doca_flow_port **ports,
                          int nb_ports);

/* 4. Create policy pipes */
doca_error_t create_cert_validation_pipe(struct doca_flow_port *port,
                                          struct doca_flow_pipe **pipe);

/* 5. Add flow entries */
doca_error_t add_cert_policy_entries(struct doca_flow_pipe *pipe,
                                      cert_policy_t *policies,
                                      int nb_policies);

/* 6. Packet processing loop */
void packet_processing_loop(struct doca_flow_port **ports, int nb_ports);

/* 7. Cleanup */
void cleanup(struct doca_flow_port **ports, int nb_ports);

int main(int argc, char **argv)
{
    struct doca_dev *dev;
    struct doca_flow_port *ports[MAX_PORTS];
    struct doca_flow_pipe *cert_pipe;

    /* Initialize */
    init_dpdk(argc, argv);
    init_flow(NUM_QUEUES);

    /* Setup */
    open_doca_device(&dev);
    create_ports(dev, ports, num_ports);
    create_cert_validation_pipe(ports[0], &cert_pipe);

    /* Load policies from Beyond Identity */
    load_and_program_policies(cert_pipe);

    /* Run */
    packet_processing_loop(ports, num_ports);

    /* Cleanup */
    cleanup(ports, num_ports);
    return 0;
}
```

## Debugging and Troubleshooting

### Common Issues

**Issue**: `Failed to initialize DOCA Flow`
- **Cause**: DPDK not initialized or ports not started
- **Fix**: Ensure `rte_eal_init()` and `rte_eth_dev_start()` called first

**Issue**: `Failed to create pipe: Resource temporarily unavailable`
- **Cause**: Hardware flow table full
- **Fix**: Reduce number of pipes or entries, check hardware limits

**Issue**: `Entries not processing`
- **Cause**: Forgot to call `doca_flow_entries_process()`
- **Fix**: Call after adding entries to commit to hardware

**Issue**: `Packets not matching`
- **Cause**: Match criteria mismatch or incorrect field types
- **Fix**: Verify match masks (0x00, specific value, or 0xff...ff)

### Debugging Tools

```bash
# Check DPDK ports
dpdk-devbind.py --status

# View flow tables (if supported)
dpdk-testpmd
> flow list 0

# DOCA Flow debugging (in code)
export DOCA_LOG_LEVEL=20  # Debug level
```

## References

- [DOCA Flow API Documentation](https://docs.nvidia.com/doca/sdk/doca-flow/index.html)
- [DOCA Flow Samples](https://github.com/NVIDIA-DOCA/doca-samples/tree/master/samples/doca_flow)
- [DPDK Documentation](https://doc.dpdk.org/)
- [ConnectX-7 Product Brief](https://www.nvidia.com/en-us/networking/ethernet/connectx-7/)

## Next Steps

1. ✅ Understand DOCA Flow concepts (this document)
2. ⬜ Set up DPDK environment on Spark
3. ⬜ Run DOCA Flow sample applications
4. ⬜ Design Beyond Identity + DOCA Flow integration
5. ⬜ Implement certificate-based flow policies
6. ⬜ Integrate with Beyond Identity API
7. ⬜ Test and validate performance
8. ⬜ Deploy to production

---

**Document Version**: 1.0
**Last Updated**: 2025-11-19
**Author**: Beyond Identity + DOCA Integration Project
