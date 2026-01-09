package sshscan

import (
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"strings"
)

// GenerateFingerprint generates an SSH fingerprint from base64-encoded key data.
// Returns the fingerprint in the format "SHA256:base64..." matching ssh-keygen -lf output.
func GenerateFingerprint(base64Data string) (string, error) {
	if base64Data == "" {
		return "", fmt.Errorf("empty key data")
	}

	// Decode the base64 key data
	keyBytes, err := base64.StdEncoding.DecodeString(base64Data)
	if err != nil {
		return "", fmt.Errorf("invalid base64: %w", err)
	}

	// SHA256 hash of the raw key bytes
	hash := sha256.Sum256(keyBytes)

	// Encode as base64, strip padding to match ssh-keygen output
	fp := base64.StdEncoding.EncodeToString(hash[:])
	fp = strings.TrimRight(fp, "=")

	return "SHA256:" + fp, nil
}
