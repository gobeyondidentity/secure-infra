package attestation

import (
	"strings"
	"testing"
)

func TestParseSPDMMeasurements(t *testing.T) {
	// Test with empty data
	_, err := ParseSPDMMeasurements("", "TPM_ALG_SHA_512")
	if err == nil {
		t.Error("Expected error for empty data")
	}

	// Test with invalid base64
	_, err = ParseSPDMMeasurements("not-valid-base64!!!", "TPM_ALG_SHA_512")
	if err == nil {
		t.Error("Expected error for invalid base64")
	}
}

func TestValidateMeasurements(t *testing.T) {
	// Test with empty inputs
	result := ValidateMeasurements(nil, nil)
	if result == nil {
		t.Fatal("Expected non-nil result")
	}
	if !result.Valid {
		t.Error("Empty comparison should be valid")
	}
	if result.TotalChecked != 0 {
		t.Errorf("Expected 0 checked, got %d", result.TotalChecked)
	}

	// Test with matching measurements
	live := []SPDMMeasurement{
		{Index: 2, Description: "PSC firmware hash", Algorithm: "SHA2-512", Digest: "abc123"},
		{Index: 3, Description: "NIC firmware hash", Algorithm: "SHA2-512", Digest: "def456"},
	}
	ref := []CoRIMMeasurement{
		{Index: 2, Description: "PSC Firmware", Algorithm: "SHA2-512", Digest: "abc123"},
		{Index: 3, Description: "NIC Firmware", Algorithm: "SHA2-512", Digest: "def456"},
	}

	result = ValidateMeasurements(live, ref)
	if !result.Valid {
		t.Error("Expected valid result for matching measurements")
	}
	if result.Matched != 2 {
		t.Errorf("Expected 2 matched, got %d", result.Matched)
	}
	if result.Mismatched != 0 {
		t.Errorf("Expected 0 mismatched, got %d", result.Mismatched)
	}

	// Test with mismatched measurements
	live[1].Digest = "different"
	result = ValidateMeasurements(live, ref)
	if result.Valid {
		t.Error("Expected invalid result for mismatched measurements")
	}
	if result.Mismatched != 1 {
		t.Errorf("Expected 1 mismatched, got %d", result.Mismatched)
	}

	// Test with missing reference
	live = append(live, SPDMMeasurement{Index: 4, Description: "ARM firmware", Digest: "ghi789"})
	result = ValidateMeasurements(live, ref)
	if result.MissingRef != 1 {
		t.Errorf("Expected 1 missing ref, got %d", result.MissingRef)
	}
}

func TestMeasurementDescriptions(t *testing.T) {
	// Verify key measurement indices have descriptions
	expectedIndices := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}

	for _, idx := range expectedIndices {
		desc, exists := BlueField3MeasurementDescriptions[idx]
		if !exists {
			t.Errorf("Missing description for index %d", idx)
		}
		if desc == "" {
			t.Errorf("Empty description for index %d", idx)
		}
	}

	// Verify specific descriptions
	if desc := BlueField3MeasurementDescriptions[2]; !strings.Contains(desc, "PSC") {
		t.Errorf("Index 2 should be PSC firmware, got %q", desc)
	}
	if desc := BlueField3MeasurementDescriptions[3]; !strings.Contains(desc, "NIC") {
		t.Errorf("Index 3 should be NIC firmware, got %q", desc)
	}
}

func TestGetHashSize(t *testing.T) {
	tests := []struct {
		alg  string
		size int
	}{
		{"SHA-256", 32},
		{"TPM_ALG_SHA_256", 32},
		{"SHA-384", 48},
		{"TPM_ALG_SHA_384", 48},
		{"SHA-512", 64},
		{"TPM_ALG_SHA_512", 64},
		{"unknown", 0},
	}

	for _, tt := range tests {
		got := getHashSize(tt.alg)
		if got != tt.size {
			t.Errorf("getHashSize(%q) = %d, want %d", tt.alg, got, tt.size)
		}
	}
}

func TestParseSimpleMeasurements(t *testing.T) {
	tests := []struct {
		name           string
		data           []byte
		algorithm      string
		wantCount      int
		wantFirstIndex int
		wantErr        bool
	}{
		{
			name:           "SHA-256 single measurement",
			data:           make([]byte, 32), // All zeros will be skipped
			algorithm:      "SHA-256",
			wantCount:      0, // All-zero digests are skipped
			wantFirstIndex: 0,
			wantErr:        false,
		},
		{
			name: "SHA-256 non-zero measurement",
			data: func() []byte {
				d := make([]byte, 32)
				d[0] = 0x01 // Non-zero so it won't be skipped
				return d
			}(),
			algorithm:      "SHA-256",
			wantCount:      1,
			wantFirstIndex: 1, // Index starts at 1
			wantErr:        false,
		},
		{
			name: "SHA-512 two measurements",
			data: func() []byte {
				d := make([]byte, 128) // Two SHA-512 hashes
				d[0] = 0xAB            // First measurement non-zero
				d[64] = 0xCD           // Second measurement non-zero
				return d
			}(),
			algorithm:      "SHA-512",
			wantCount:      2,
			wantFirstIndex: 1,
			wantErr:        false,
		},
		{
			name: "SHA-384 measurement",
			data: func() []byte {
				d := make([]byte, 48)
				d[0] = 0xFF
				return d
			}(),
			algorithm:      "SHA-384",
			wantCount:      1,
			wantFirstIndex: 1,
			wantErr:        false,
		},
		{
			name: "Unknown algorithm defaults to SHA-512 size",
			data: func() []byte {
				d := make([]byte, 64)
				d[0] = 0x01
				return d
			}(),
			algorithm:      "UNKNOWN_ALG",
			wantCount:      1,
			wantFirstIndex: 1,
			wantErr:        false,
		},
		{
			name:      "Empty data",
			data:      []byte{},
			algorithm: "SHA-256",
			wantCount: 0,
			wantErr:   false,
		},
		{
			name:      "Data shorter than hash size",
			data:      []byte{0x01, 0x02, 0x03},
			algorithm: "SHA-256",
			wantCount: 0,
			wantErr:   false,
		},
		{
			name: "All zeros skipped",
			data: func() []byte {
				return make([]byte, 64) // 64 zeros
			}(),
			algorithm: "SHA-256",
			wantCount: 0, // Both 32-byte chunks are zeros, so skipped
			wantErr:   false,
		},
		{
			name: "Mixed zero and non-zero measurements",
			data: func() []byte {
				d := make([]byte, 96) // Three SHA-256 hashes
				// First 32 bytes: all zeros (skipped)
				// Second 32 bytes: non-zero
				d[32] = 0xAA
				// Third 32 bytes: all zeros (skipped)
				return d
			}(),
			algorithm:      "SHA-256",
			wantCount:      1,
			wantFirstIndex: 2, // Second measurement (index 2)
			wantErr:        false,
		},
		{
			name: "Max 11 measurements cap",
			data: func() []byte {
				// Create 12 SHA-256 measurements (384 bytes)
				d := make([]byte, 384)
				for i := 0; i < 12; i++ {
					d[i*32] = byte(i + 1) // Make each non-zero
				}
				return d
			}(),
			algorithm:      "SHA-256",
			wantCount:      11, // Function caps at 11 measurements
			wantFirstIndex: 1,
			wantErr:        false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			measurements, err := parseSimpleMeasurements(tt.data, tt.algorithm)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseSimpleMeasurements() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if len(measurements) != tt.wantCount {
				t.Errorf("parseSimpleMeasurements() returned %d measurements, want %d", len(measurements), tt.wantCount)
			}
			if tt.wantCount > 0 && measurements[0].Index != tt.wantFirstIndex {
				t.Errorf("parseSimpleMeasurements() first index = %d, want %d", measurements[0].Index, tt.wantFirstIndex)
			}
		})
	}
}

func TestParseSimpleMeasurements_DigestContent(t *testing.T) {
	// Test that the digest content is correctly hex-encoded
	data := make([]byte, 32)
	for i := 0; i < 32; i++ {
		data[i] = byte(i)
	}

	measurements, err := parseSimpleMeasurements(data, "SHA-256")
	if err != nil {
		t.Fatalf("parseSimpleMeasurements failed: %v", err)
	}
	if len(measurements) != 1 {
		t.Fatalf("Expected 1 measurement, got %d", len(measurements))
	}

	expectedDigest := "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
	if measurements[0].Digest != expectedDigest {
		t.Errorf("Digest mismatch:\ngot:  %s\nwant: %s", measurements[0].Digest, expectedDigest)
	}

	// Verify Algorithm is preserved
	if measurements[0].Algorithm != "SHA-256" {
		t.Errorf("Algorithm = %q, want %q", measurements[0].Algorithm, "SHA-256")
	}

	// Verify RawValue is set correctly
	if len(measurements[0].RawValue) != 32 {
		t.Errorf("RawValue length = %d, want 32", len(measurements[0].RawValue))
	}
}

func TestParseSimpleMeasurements_Description(t *testing.T) {
	// Test that descriptions are populated from BlueField3MeasurementDescriptions
	// Create data for indices 1-3 (3 SHA-256 measurements)
	data := make([]byte, 96)
	data[0] = 0x01  // Index 1
	data[32] = 0x02 // Index 2
	data[64] = 0x03 // Index 3

	measurements, err := parseSimpleMeasurements(data, "SHA-256")
	if err != nil {
		t.Fatalf("parseSimpleMeasurements failed: %v", err)
	}
	if len(measurements) != 3 {
		t.Fatalf("Expected 3 measurements, got %d", len(measurements))
	}

	// Check that descriptions match BlueField3MeasurementDescriptions
	for _, m := range measurements {
		expectedDesc, exists := BlueField3MeasurementDescriptions[m.Index]
		if exists && m.Description != expectedDesc {
			t.Errorf("Description for index %d = %q, want %q", m.Index, m.Description, expectedDesc)
		}
	}
}

func TestNormalizeDigest(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"ABC123", "abc123"},
		{"  abc123  ", "abc123"},
		{"ABC123DEF", "abc123def"},
		{"", ""},
		{"   ", ""},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := normalizeDigest(tt.input)
			if result != tt.expected {
				t.Errorf("normalizeDigest(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}
