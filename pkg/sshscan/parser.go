package sshscan

import (
	"encoding/base64"
	"path/filepath"
	"strings"

	"golang.org/x/crypto/ssh"
)

// ParseLine parses a single line from an authorized_keys file.
// Returns nil for empty lines, comments, and malformed lines (skips gracefully).
// Returns the parsed SSHKey for valid key lines.
func ParseLine(line string) (*SSHKey, error) {
	line = strings.TrimSpace(line)

	// Skip empty lines
	if line == "" {
		return nil, nil
	}

	// Skip comment lines
	if strings.HasPrefix(line, "#") {
		return nil, nil
	}

	// Parse the key using golang.org/x/crypto/ssh
	pubKey, comment, _, _, err := ssh.ParseAuthorizedKey([]byte(line))
	if err != nil {
		// Malformed line, skip gracefully
		return nil, nil
	}

	// Extract key type
	keyType := pubKey.Type()

	// Calculate key bits
	keyBits := getKeyBits(pubKey)

	// Extract base64 data from the original line for fingerprint
	parts := strings.Fields(line)
	if len(parts) < 2 {
		return nil, nil
	}

	// Verify the base64 portion is valid
	if _, err := base64.StdEncoding.DecodeString(parts[1]); err != nil {
		return nil, nil
	}

	// Generate fingerprint
	fingerprint, err := GenerateFingerprint(parts[1])
	if err != nil {
		return nil, nil
	}

	return &SSHKey{
		KeyType:     keyType,
		KeyBits:     keyBits,
		Fingerprint: fingerprint,
		Comment:     comment,
	}, nil
}

// ParseAuthorizedKeys parses the contents of an authorized_keys file.
// Returns all valid keys found, skipping empty lines, comments, and malformed entries.
func ParseAuthorizedKeys(content, user, filePath string) ([]*SSHKey, error) {
	var keys []*SSHKey

	lines := strings.Split(content, "\n")
	for _, line := range lines {
		key, err := ParseLine(line)
		if err != nil {
			return nil, err
		}
		if key != nil {
			key.User = user
			key.FilePath = filePath
			keys = append(keys, key)
		}
	}

	return keys, nil
}

// ExtractUsername extracts the Unix username from an authorized_keys file path.
// Handles /home/USER/.ssh/authorized_keys and /root/.ssh/authorized_keys patterns.
func ExtractUsername(filePath string) string {
	if filePath == "" {
		return ""
	}

	// Handle /root/.ssh/authorized_keys
	if strings.HasPrefix(filePath, "/root/") {
		return "root"
	}

	// Look for /.ssh/authorized_keys pattern
	dir := filepath.Dir(filePath)
	if filepath.Base(dir) != ".ssh" {
		return ""
	}

	// Parent of .ssh is the user's home directory
	homeDir := filepath.Dir(dir)
	return filepath.Base(homeDir)
}

// getKeyBits returns the bit size for an SSH public key.
func getKeyBits(pubKey ssh.PublicKey) int {
	keyType := pubKey.Type()

	switch keyType {
	case "ssh-ed25519":
		return 256
	case "ssh-rsa":
		// RSA key size is encoded in the key data
		// We can estimate from the marshaled key size
		// A 2048-bit RSA key marshals to ~279 bytes, 4096-bit to ~535 bytes
		marshaledSize := len(pubKey.Marshal())
		if marshaledSize > 400 {
			return 4096
		}
		return 2048
	case "ecdsa-sha2-nistp256":
		return 256
	case "ecdsa-sha2-nistp384":
		return 384
	case "ecdsa-sha2-nistp521":
		return 521
	default:
		return 0
	}
}
