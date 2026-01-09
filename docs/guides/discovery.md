# SSH Key Discovery

Audit existing SSH keys across your infrastructure before migrating to certificate-based authentication.

## 1. Why Discovery?

Most environments have accumulated SSH keys over years: personal keys, service accounts, deployment keys, forgotten test keys. Before migrating to certificates, you need to know what exists.

Discovery answers:
- How many static keys are authorized across hosts?
- Which keys are duplicated across multiple hosts?
- Are there unknown or suspicious keys?
- Which service accounts need certificate migration?

## 2. Prerequisites

- `km` CLI initialized (see [Quickstart](quickstart-emulator.md) Steps 1-7)
- Plus one of:
  - SSH access to target hosts (for Bootstrap mode), OR
  - Host agents running on target hosts (for Agent mode)

## 3. Bootstrap Mode (No Agent)

Scan hosts directly via SSH without deploying host-agent first. Useful for initial audits and migration planning.

```bash
bin/km discover scan <HOST> --ssh --ssh-user <USER> --ssh-key <KEY_PATH>
# Expected:
# HOST           METHOD  USER    TYPE        FINGERPRINT                                   COMMENT
# 192.168.1.100  ssh     root    ssh-ed2...  SHA256:abc123...                              admin@company.com
# 192.168.1.100  ssh     ubuntu  ssh-rsa     SHA256:def456...                              deploy-key
# ...
#
# Found N keys on 1 host (ssh)
# Warning: Bootstrap mode. Install host-agent for production use.
```

Options:
- `--ssh`: Force SSH mode (required for bootstrap)
- `--ssh-user`: SSH username (default: current user)
- `--ssh-key`: Path to SSH private key
- `--timeout`: Per-host timeout in seconds (default: 30)

## 4. Agent Mode (Production)

Once host-agents are deployed, scan through the agent channel instead of SSH. No SSH key management required, and scans scale to hundreds of hosts with `--parallel`.

### Scan a single host

```bash
bin/km discover scan <HOSTNAME>
# Expected:
# HOST        METHOD  USER    TYPE        FINGERPRINT                                   COMMENT
# gpu-node-1  agent   root    ssh-ed2...  SHA256:abc123...                              admin@company.com
# gpu-node-1  agent   ubuntu  ssh-ed2...  SHA256:def456...                              deploy-bot
```

### Scan all registered hosts

```bash
bin/km discover scan --all
# Expected:
# HOST        METHOD  USER    TYPE        FINGERPRINT                                   COMMENT
# gpu-node-1  agent   root    ssh-ed2...  SHA256:abc123...                              admin@company.com
# gpu-node-2  agent   root    ssh-ed2...  SHA256:abc123...                              admin@company.com
# gpu-node-2  agent   ubuntu  ssh-rsa     SHA256:xyz789...                              jenkins
#
# Found 3 keys on 2 hosts (agent)
```

Options:
- `--all`: Scan all registered hosts
- `--parallel`: Max concurrent scans (default: 10)
- `--ssh-fallback`: Use SSH for hosts without running agents

## 5. Exporting Results

Export scan results to JSON for further analysis:

```bash
bin/km discover scan --all -o json > ssh-key-inventory.json
```

See [Discovery Reference](../reference/discovery.md) for jq recipes and analysis workflows.

## 6. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `SSH authentication failed` | Wrong key or user | Verify `--ssh-user` and `--ssh-key` |
| `connection refused` | SSH not running or firewall | Check SSH service and network |
| `no hosts registered` | No host-agents connected | Use `--ssh` bootstrap mode |
| `timeout` | Host unreachable or slow | Increase `--timeout` value |

## 7. What's Next?

After auditing existing keys:

1. **Deploy host-agents** - Switch from SSH bootstrap to agent-based scanning ([Hardware Setup](setup-hardware.md) Steps 12-13)
2. **Create SSH CA** - Set up certificate authority ([Quickstart](quickstart-emulator.md) Step 8)
3. **Migrate users** - Issue certificates to replace static keys ([Quickstart](quickstart-emulator.md) Step 13)
4. **Remove static keys** - Clean up authorized_keys after migration
