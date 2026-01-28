package attestation

import (
	"context"
	"testing"
	"time"
)

func TestRIMClient_ListRIMIDs(t *testing.T) {
	client := NewRIMClient()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	ids, err := client.ListRIMIDs(ctx)
	if err != nil {
		t.Fatalf("ListRIMIDs failed: %v", err)
	}

	if len(ids) == 0 {
		t.Error("Expected non-empty RIM ID list")
	}

	t.Logf("Found %d RIM IDs", len(ids))

	// Check for expected patterns (GPUs should be present)
	hasGPU := false
	for _, id := range ids {
		if len(id) > 10 && (id[0:2] == "GH" || id[0:2] == "GB") {
			hasGPU = true
			break
		}
	}
	if !hasGPU {
		t.Log("Warning: No GPU RIMs found (expected GH100, GB100, etc.)")
	}
}

func TestRIMClient_FindRIMForFirmware_NotFound(t *testing.T) {
	client := NewRIMClient()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// BlueField-3 firmware that won't be found (not available until April 2025)
	_, err := client.FindRIMForFirmware(ctx, "32.47.1088")
	if err == nil {
		t.Log("Unexpectedly found RIM for BF3 firmware - this may mean BF3 CoRIMs are now available")
	} else {
		t.Logf("Expected error for BF3 firmware: %v", err)
	}
}

func TestVerifyRIMIntegrity(t *testing.T) {
	// Test with nil entry
	valid, err := VerifyRIMIntegrity(nil)
	if err == nil {
		t.Error("Expected error for nil entry")
	}
	if valid {
		t.Error("Expected invalid for nil entry")
	}

	// Test with empty entry
	entry := &RIMEntry{}
	valid, err = VerifyRIMIntegrity(entry)
	if err == nil {
		t.Error("Expected error for empty entry")
	}
	if valid {
		t.Error("Expected invalid for empty entry")
	}
}

func TestDigestAlgorithmName(t *testing.T) {
	tests := []struct {
		algID    uint64
		expected string
	}{
		{1, "SHA-256"},
		{2, "SHA-384"},
		{3, "SHA-512"},
		{7, "SHA3-256"},
		{8, "SHA3-384"},
		{9, "SHA3-512"},
		{0, "ALG-0"},
		{4, "ALG-4"},
		{5, "ALG-5"},
		{6, "ALG-6"},
		{10, "ALG-10"},
		{100, "ALG-100"},
		{255, "ALG-255"},
		{1000, "ALG-1000"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			result := digestAlgorithmName(tt.algID)
			if result != tt.expected {
				t.Errorf("digestAlgorithmName(%d) = %q, want %q", tt.algID, result, tt.expected)
			}
		})
	}
}
