package cmd

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/knownhosts"
)

// UnknownHostKeyError is returned when a host's key is not in known_hosts
// and --accept-host-key was not specified.
type UnknownHostKeyError struct {
	Hostname    string
	Fingerprint string
}

func (e *UnknownHostKeyError) Error() string {
	return fmt.Sprintf(`SSH host key verification failed: host "%s" not in known_hosts.
Fingerprint: %s
Use --accept-host-key to trust this host`, e.Hostname, e.Fingerprint)
}

// getKnownHostsPath returns the default known_hosts file path (~/.ssh/known_hosts)
func getKnownHostsPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		// Fallback to current directory if home not available
		return ".ssh/known_hosts"
	}
	return filepath.Join(home, ".ssh", "known_hosts")
}

// createHostKeyCallback creates an SSH host key callback implementing TOFU (Trust On First Use).
// If acceptHostKey is true, unknown hosts will be added to known_hosts.
// If acceptHostKey is false, unknown hosts will return an error with the fingerprint.
func createHostKeyCallback(knownHostsPath string, acceptHostKey bool) (ssh.HostKeyCallback, error) {
	// Try to load existing known_hosts file
	var knownHostsCallback ssh.HostKeyCallback
	var loadErr error

	if _, err := os.Stat(knownHostsPath); err == nil {
		// File exists, load it
		knownHostsCallback, loadErr = knownhosts.New(knownHostsPath)
		if loadErr != nil {
			return nil, fmt.Errorf("failed to load known_hosts: %w", loadErr)
		}
	}

	return func(hostname string, remote net.Addr, key ssh.PublicKey) error {
		// Normalize hostname (remove :22 suffix for standard port)
		normalizedHost := normalizeHostname(hostname)

		// If we have a loaded known_hosts, check it first
		if knownHostsCallback != nil {
			err := knownHostsCallback(hostname, remote, key)
			if err == nil {
				// Host is known and key matches
				return nil
			}

			// Check if this is a key mismatch (potential MITM) vs unknown host
			var keyErr *knownhosts.KeyError
			if isKeyError(err, &keyErr) {
				if len(keyErr.Want) > 0 {
					// Key mismatch: host is in known_hosts but key is different
					// This is a security concern, do not allow even with --accept-host-key
					return fmt.Errorf("host key mismatch for %s (possible MITM attack): %w", normalizedHost, err)
				}
				// Host not in known_hosts, continue to TOFU logic below
			} else {
				// Some other error
				return err
			}
		}

		// Host not in known_hosts
		fingerprint := ssh.FingerprintSHA256(key)

		if !acceptHostKey {
			// Return descriptive error
			return &UnknownHostKeyError{
				Hostname:    normalizedHost,
				Fingerprint: fingerprint,
			}
		}

		// TOFU: Add host to known_hosts
		if err := addHostKey(knownHostsPath, hostname, key); err != nil {
			return fmt.Errorf("failed to add host key: %w", err)
		}

		return nil
	}, nil
}

// normalizeHostname removes the standard SSH port (:22) from hostname for display.
func normalizeHostname(hostname string) string {
	// hostname is in the format "host:port" from SSH
	host, port, err := net.SplitHostPort(hostname)
	if err != nil {
		// No port in hostname
		return hostname
	}
	if port == "22" {
		return host
	}
	// Non-standard port, return as-is
	return hostname
}

// addHostKey appends a host key to the known_hosts file.
func addHostKey(knownHostsPath, hostname string, key ssh.PublicKey) error {
	// Ensure the directory exists
	dir := filepath.Dir(knownHostsPath)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Format the known_hosts line
	line := formatHostKeyLine(hostname, key)

	// Open file in append mode, create if not exists
	f, err := os.OpenFile(knownHostsPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return fmt.Errorf("failed to open known_hosts: %w", err)
	}
	defer f.Close()

	if _, err := f.WriteString(line + "\n"); err != nil {
		return fmt.Errorf("failed to write to known_hosts: %w", err)
	}

	return nil
}

// formatHostKeyLine formats a hostname and key for the known_hosts file.
// For standard port 22, the hostname is stored as-is.
// For non-standard ports, the hostname is stored as [host]:port.
func formatHostKeyLine(hostname string, key ssh.PublicKey) string {
	host, port, err := net.SplitHostPort(hostname)
	if err != nil {
		// No port in hostname, use as-is
		host = hostname
		port = "22"
	}

	var hostEntry string
	if port == "22" {
		hostEntry = host
	} else {
		// Non-standard port uses bracketed format
		hostEntry = fmt.Sprintf("[%s]:%s", host, port)
	}

	// Marshal the key in authorized_keys format and strip trailing newline
	keyStr := strings.TrimSpace(string(ssh.MarshalAuthorizedKey(key)))

	return hostEntry + " " + keyStr
}

// isKeyError checks if err is a knownhosts.KeyError and populates target
func isKeyError(err error, target **knownhosts.KeyError) bool {
	if e, ok := err.(*knownhosts.KeyError); ok {
		*target = e
		return true
	}
	return false
}
