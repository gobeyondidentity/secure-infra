# Discovery Reference

Reference documentation for `km discover scan` output and analysis workflows.

## 1. JSON Schema

```json
{
  "scan_time": "2026-01-09T16:56:22Z",
  "hosts_scanned": 2,
  "hosts_succeeded": 2,
  "hosts_failed": 0,
  "total_keys": 15,
  "method_breakdown": {
    "agent": 1,
    "ssh": 1
  },
  "keys": [
    {
      "host": "gpu-node-1",
      "method": "agent",
      "user": "root",
      "key_type": "ssh-ed25519",
      "key_bits": 256,
      "fingerprint": "SHA256:abc123...",
      "comment": "admin@company.com",
      "file_path": "/root/.ssh/authorized_keys"
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `scan_time` | string | ISO 8601 timestamp of scan |
| `hosts_scanned` | int | Total hosts attempted |
| `hosts_succeeded` | int | Hosts successfully scanned |
| `hosts_failed` | int | Hosts that failed |
| `total_keys` | int | Total keys found |
| `method_breakdown` | object | Count by scan method (agent/ssh) |
| `keys` | array | List of discovered keys |

### Key Object

| Field | Type | Description |
|-------|------|-------------|
| `host` | string | Hostname or IP |
| `method` | string | `agent` or `ssh` |
| `user` | string | Unix user account |
| `key_type` | string | `ssh-ed25519`, `ssh-rsa`, etc. |
| `key_bits` | int | Key size in bits |
| `fingerprint` | string | SHA256 fingerprint |
| `comment` | string | Key comment field |
| `file_path` | string | Path to authorized_keys file |

## 2. Analysis Recipes

### Count keys by host

```bash
cat inventory.json | jq '.keys | group_by(.host) | map({host: .[0].host, count: length})'
# Expected:
# [
#   { "host": "gpu-node-1", "count": 8 },
#   { "host": "gpu-node-2", "count": 5 }
# ]
```

### Count keys by user

```bash
cat inventory.json | jq '.keys | group_by(.user) | map({user: .[0].user, count: length})'
# Expected:
# [
#   { "user": "root", "count": 5 },
#   { "user": "ubuntu", "count": 8 }
# ]
```

### Find duplicate keys

Keys with the same fingerprint across multiple hosts or users:

```bash
cat inventory.json | jq '
  .keys | group_by(.fingerprint) |
  map(select(length > 1)) |
  map({
    fingerprint: .[0].fingerprint,
    comment: .[0].comment,
    locations: map({host: .host, user: .user})
  })'
# Expected:
# [
#   {
#     "fingerprint": "SHA256:abc123...",
#     "comment": "shared-cluster-key",
#     "locations": [
#       { "host": "gpu-node-1", "user": "root" },
#       { "host": "gpu-node-2", "user": "root" }
#     ]
#   }
# ]
```

### List RSA keys (migration candidates)

Older RSA keys may be candidates for migration to Ed25519:

```bash
cat inventory.json | jq '.keys | map(select(.key_type == "ssh-rsa")) | map({host, user, comment, key_bits})'
# Expected:
# [
#   { "host": "gpu-node-1", "user": "ubuntu", "comment": "legacy-deploy", "key_bits": 2048 }
# ]
```

### Keys by comment pattern

Find keys matching a pattern (e.g., service accounts):

```bash
cat inventory.json | jq '.keys | map(select(.comment | test("deploy|jenkins|ci"))) | map({host, user, comment})'
# Expected:
# [
#   { "host": "gpu-node-1", "user": "ubuntu", "comment": "jenkins-deploy" },
#   { "host": "gpu-node-2", "user": "ci", "comment": "ci-runner" }
# ]
```

### Summary report

Generate a migration planning summary:

```bash
cat inventory.json | jq '{
  total_hosts: .hosts_scanned,
  total_keys: .total_keys,
  keys_by_type: (.keys | group_by(.key_type) | map({type: .[0].key_type, count: length})),
  duplicate_count: ([.keys | group_by(.fingerprint) | map(select(length > 1))] | length),
  users_with_keys: ([.keys[].user] | unique | length)
}'
# Expected:
# {
#   "total_hosts": 10,
#   "total_keys": 47,
#   "keys_by_type": [
#     { "type": "ssh-ed25519", "count": 35 },
#     { "type": "ssh-rsa", "count": 12 }
#   ],
#   "duplicate_count": 3,
#   "users_with_keys": 4
# }
```

## 3. Command Reference

```
km discover scan [HOST] [flags]

Flags:
      --all               Scan all registered hosts
      --ssh               Force SSH mode (bootstrap)
      --ssh-user string   SSH username (default: current user)
      --ssh-key string    SSH private key path
      --ssh-fallback      Use SSH for hosts without agent
      --parallel int      Max concurrent scans (default 10)
      --timeout int       Per-host timeout in seconds (default 30)
      --no-color          Disable colored output
  -o, --output string     Output format: table, json (default "table")
  -h, --help              Help for scan
```
