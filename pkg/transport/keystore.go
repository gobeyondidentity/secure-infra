package transport

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// DefaultKeyStorePath is the default location for the TOFU key store.
const DefaultKeyStorePath = "/var/lib/secureinfra/known_hosts.json"

// KeyStore manages trusted host public keys using a Trust On First Use (TOFU) model.
// Keys are persisted to a JSON file for durability across restarts.
type KeyStore struct {
	path string              // Path to key store file
	keys map[string][]byte   // Public key fingerprint -> raw public key bytes
	mu   sync.RWMutex
}

// NewKeyStore creates a key store at the given path.
// If the file exists, it loads existing keys. If not, it creates an empty store.
func NewKeyStore(path string) (*KeyStore, error) {
	ks := &KeyStore{
		path: path,
		keys: make(map[string][]byte),
	}

	// Try to load existing keys
	if _, err := os.Stat(path); err == nil {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read key store: %w", err)
		}
		if err := json.Unmarshal(data, &ks.keys); err != nil {
			return nil, fmt.Errorf("parse key store: %w", err)
		}
	}

	return ks, nil
}

// IsKnown checks if a public key is already registered.
func (k *KeyStore) IsKnown(pubKey ed25519.PublicKey) bool {
	k.mu.RLock()
	defer k.mu.RUnlock()

	fp := Fingerprint(pubKey)
	_, ok := k.keys[fp]
	return ok
}

// Register adds a new public key to the store (TOFU model).
// The key is persisted to disk immediately. Registering a duplicate key is idempotent.
func (k *KeyStore) Register(pubKey ed25519.PublicKey) error {
	k.mu.Lock()
	defer k.mu.Unlock()

	fp := Fingerprint(pubKey)

	// Check if already registered (idempotent)
	if _, ok := k.keys[fp]; ok {
		return nil
	}

	// Store the raw public key bytes
	k.keys[fp] = []byte(pubKey)

	return k.saveLocked()
}

// Verify checks if a public key matches a registered key.
// Returns true if the key is registered and matches, false otherwise.
func (k *KeyStore) Verify(pubKey ed25519.PublicKey) bool {
	k.mu.RLock()
	defer k.mu.RUnlock()

	fp := Fingerprint(pubKey)
	stored, ok := k.keys[fp]
	if !ok {
		return false
	}

	// Compare the stored key bytes with the provided key
	return ed25519.PublicKey(stored).Equal(pubKey)
}

// saveLocked persists the key store to disk. Caller must hold the lock.
func (k *KeyStore) saveLocked() error {
	// Create parent directory if needed
	dir := filepath.Dir(k.path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create key store directory: %w", err)
	}

	data, err := json.MarshalIndent(k.keys, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal key store: %w", err)
	}

	// Write with restrictive permissions
	if err := os.WriteFile(k.path, data, 0600); err != nil {
		return fmt.Errorf("write key store: %w", err)
	}

	return nil
}

// Fingerprint returns the SHA256 fingerprint of an Ed25519 public key as a hex string.
func Fingerprint(pubKey ed25519.PublicKey) string {
	hash := sha256.Sum256(pubKey)
	return hex.EncodeToString(hash[:])
}
