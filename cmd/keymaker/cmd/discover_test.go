package cmd

import (
	"encoding/json"
	"testing"

	"github.com/nmelo/secure-infra/pkg/sshscan"
)

func TestTruncate(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		maxLen   int
		expected string
	}{
		{
			name:     "shorter than max",
			input:    "hello",
			maxLen:   10,
			expected: "hello",
		},
		{
			name:     "exactly max",
			input:    "hello",
			maxLen:   5,
			expected: "hello",
		},
		{
			name:     "longer than max",
			input:    "hello world",
			maxLen:   8,
			expected: "hello...",
		},
		{
			name:     "very short max",
			input:    "hello",
			maxLen:   3,
			expected: "hel",
		},
		{
			name:     "empty string",
			input:    "",
			maxLen:   10,
			expected: "",
		},
		{
			name:     "fingerprint truncation",
			input:    "SHA256:abc123def456ghi789jkl012mno345pqr678stu901",
			maxLen:   44,
			expected: "SHA256:abc123def456ghi789jkl012mno345pqr6...",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := truncate(tt.input, tt.maxLen)
			if got != tt.expected {
				t.Errorf("truncate(%q, %d) = %q, want %q", tt.input, tt.maxLen, got, tt.expected)
			}
		})
	}
}

func TestScanSummaryJSONFormat(t *testing.T) {
	// Verify the JSON structure matches the spec
	summary := ScanSummary{
		ScanTime:       "2026-01-15T10:30:22Z",
		HostsScanned:   2,
		HostsSucceeded: 2,
		HostsFailed:    0,
		TotalKeys:      3,
		MethodBreakdown: map[string]int{
			"agent": 1,
			"ssh":   1,
		},
		Keys: []ScanResultKey{
			{
				Host:        "gpu-node-01",
				Method:      "agent",
				User:        "root",
				KeyType:     "ed25519",
				KeyBits:     256,
				Fingerprint: "SHA256:abc123def456ghi789jkl012mno345pqr678stu901",
				Comment:     "deploy-key",
				FilePath:    "/root/.ssh/authorized_keys",
			},
		},
		Failures: []ScanFailure{},
	}

	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		t.Fatalf("Failed to marshal ScanSummary: %v", err)
	}

	// Verify we can unmarshal it back
	var decoded ScanSummary
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal ScanSummary: %v", err)
	}

	if decoded.ScanTime != summary.ScanTime {
		t.Errorf("ScanTime mismatch: got %q, want %q", decoded.ScanTime, summary.ScanTime)
	}
	if decoded.HostsScanned != summary.HostsScanned {
		t.Errorf("HostsScanned mismatch: got %d, want %d", decoded.HostsScanned, summary.HostsScanned)
	}
	if decoded.TotalKeys != summary.TotalKeys {
		t.Errorf("TotalKeys mismatch: got %d, want %d", decoded.TotalKeys, summary.TotalKeys)
	}
	if len(decoded.Keys) != len(summary.Keys) {
		t.Errorf("Keys length mismatch: got %d, want %d", len(decoded.Keys), len(summary.Keys))
	}
}

func TestScanResultKeyFields(t *testing.T) {
	// Verify ScanResultKey has all required fields from the spec
	key := ScanResultKey{
		Host:        "gpu-node-01",
		Method:      "agent",
		User:        "root",
		KeyType:     "ed25519",
		KeyBits:     256,
		Fingerprint: "SHA256:abc123",
		Comment:     "deploy-key",
		FilePath:    "/root/.ssh/authorized_keys",
	}

	data, err := json.Marshal(key)
	if err != nil {
		t.Fatalf("Failed to marshal ScanResultKey: %v", err)
	}

	// Check all required fields are present
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("Failed to unmarshal to map: %v", err)
	}

	requiredFields := []string{
		"host", "method", "user", "key_type", "key_bits", "fingerprint", "comment", "file_path",
	}
	for _, field := range requiredFields {
		if _, ok := m[field]; !ok {
			t.Errorf("Missing required field: %s", field)
		}
	}
}

func TestHostStructure(t *testing.T) {
	// Verify Host struct matches the API response format
	hostJSON := `{
		"name": "gpu-node-01",
		"hostname": "gpu-node-01.cluster.local",
		"dpu_name": "bf3-gpu-01",
		"tenant": "gpu-prod",
		"has_agent": true,
		"last_seen": "2026-01-15T10:25:00Z"
	}`

	var host Host
	if err := json.Unmarshal([]byte(hostJSON), &host); err != nil {
		t.Fatalf("Failed to unmarshal Host: %v", err)
	}

	if host.Name != "gpu-node-01" {
		t.Errorf("Name mismatch: got %q, want %q", host.Name, "gpu-node-01")
	}
	if host.Hostname != "gpu-node-01.cluster.local" {
		t.Errorf("Hostname mismatch: got %q, want %q", host.Hostname, "gpu-node-01.cluster.local")
	}
	if host.DPUName != "bf3-gpu-01" {
		t.Errorf("DPUName mismatch: got %q, want %q", host.DPUName, "bf3-gpu-01")
	}
	if host.Tenant != "gpu-prod" {
		t.Errorf("Tenant mismatch: got %q, want %q", host.Tenant, "gpu-prod")
	}
	if !host.HasAgent {
		t.Error("HasAgent should be true")
	}
	if host.LastSeen != "2026-01-15T10:25:00Z" {
		t.Errorf("LastSeen mismatch: got %q, want %q", host.LastSeen, "2026-01-15T10:25:00Z")
	}
}

func TestScanResultConversion(t *testing.T) {
	// Test that ScanResult correctly holds sshscan.SSHKey values
	result := ScanResult{
		Host:      "test-host",
		Method:    "agent",
		ScannedAt: "2026-01-15T10:30:22Z",
		Keys: []sshscan.SSHKey{
			{
				User:        "root",
				KeyType:     "ssh-ed25519",
				KeyBits:     256,
				Fingerprint: "SHA256:test123",
				Comment:     "test-key",
				FilePath:    "/root/.ssh/authorized_keys",
			},
			{
				User:        "ubuntu",
				KeyType:     "ssh-rsa",
				KeyBits:     4096,
				Fingerprint: "SHA256:rsa456",
				Comment:     "ubuntu-key",
				FilePath:    "/home/ubuntu/.ssh/authorized_keys",
			},
		},
	}

	if len(result.Keys) != 2 {
		t.Errorf("Expected 2 keys, got %d", len(result.Keys))
	}
	if result.Keys[0].User != "root" {
		t.Errorf("First key user mismatch: got %q, want %q", result.Keys[0].User, "root")
	}
	if result.Keys[1].KeyBits != 4096 {
		t.Errorf("Second key bits mismatch: got %d, want %d", result.Keys[1].KeyBits, 4096)
	}
}

func TestScanFailureJSON(t *testing.T) {
	failure := ScanFailure{
		Host:  "gpu-node-02",
		Error: "connection refused (is host-agent running?)",
	}

	data, err := json.Marshal(failure)
	if err != nil {
		t.Fatalf("Failed to marshal ScanFailure: %v", err)
	}

	var decoded ScanFailure
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal ScanFailure: %v", err)
	}

	if decoded.Host != failure.Host {
		t.Errorf("Host mismatch: got %q, want %q", decoded.Host, failure.Host)
	}
	if decoded.Error != failure.Error {
		t.Errorf("Error mismatch: got %q, want %q", decoded.Error, failure.Error)
	}
}

func TestExitCodes(t *testing.T) {
	// Verify exit codes are defined correctly per spec
	if ExitDiscoverSuccess != 0 {
		t.Errorf("ExitDiscoverSuccess should be 0, got %d", ExitDiscoverSuccess)
	}
	if ExitDiscoverAllFailed != 1 {
		t.Errorf("ExitDiscoverAllFailed should be 1, got %d", ExitDiscoverAllFailed)
	}
	if ExitDiscoverPartialFailed != 2 {
		t.Errorf("ExitDiscoverPartialFailed should be 2, got %d", ExitDiscoverPartialFailed)
	}
	if ExitDiscoverConfigError != 3 {
		t.Errorf("ExitDiscoverConfigError should be 3, got %d", ExitDiscoverConfigError)
	}
}

func TestGetDefaultSSHUser(t *testing.T) {
	// Just verify it doesn't panic and returns something
	user := getDefaultSSHUser()
	// We can't assert much about the actual value since it depends on the running user
	// but it shouldn't be empty in normal circumstances
	t.Logf("Default SSH user: %q", user)
}

func TestMethodBreakdownMap(t *testing.T) {
	// Verify method breakdown counts work correctly
	methodCounts := map[string]int{"agent": 0, "ssh": 0}

	methodCounts["agent"]++
	methodCounts["agent"]++
	methodCounts["ssh"]++

	if methodCounts["agent"] != 2 {
		t.Errorf("Agent count should be 2, got %d", methodCounts["agent"])
	}
	if methodCounts["ssh"] != 1 {
		t.Errorf("SSH count should be 1, got %d", methodCounts["ssh"])
	}
}
