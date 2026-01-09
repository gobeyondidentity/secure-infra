// Package sshscan provides SSH authorized_keys parsing and fingerprint generation.
package sshscan

// SSHKey represents a parsed SSH public key from an authorized_keys file.
type SSHKey struct {
	User        string // Unix username (derived from file path)
	KeyType     string // ssh-ed25519, ssh-rsa, ecdsa-sha2-nistp256
	KeyBits     int    // Key size: 256 (ed25519), 2048/4096 (RSA), 256/384/521 (ECDSA)
	Fingerprint string // SHA256:base64... (same format as ssh-keygen -lf)
	Comment     string // Key comment (may be empty)
	FilePath    string // /home/user/.ssh/authorized_keys
}
