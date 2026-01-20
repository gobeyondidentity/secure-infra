package transport

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/pem"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// AuthClient handles ComCh authentication for host agents.
// It manages Ed25519 keypair generation/loading and performs the
// AUTH_CHALLENGE/AUTH_RESPONSE exchange with the DPU.
type AuthClient struct {
	transport  Transport
	keyPath    string
	privateKey ed25519.PrivateKey
	publicKey  ed25519.PublicKey
}

// NewAuthClient creates an auth client with the given transport and key path.
// The key path specifies where the Ed25519 keypair is stored (PEM format).
func NewAuthClient(t Transport, keyPath string) *AuthClient {
	return &AuthClient{
		transport: t,
		keyPath:   keyPath,
	}
}

// LoadOrGenerateKey loads an existing keypair or generates a new one.
// If the key file does not exist, a new Ed25519 keypair is generated
// and saved to the key path with mode 0600. Parent directories are
// created with mode 0755 if needed.
func (c *AuthClient) LoadOrGenerateKey() error {
	// Check if key file exists
	if _, err := os.Stat(c.keyPath); os.IsNotExist(err) {
		return c.generateAndSaveKey()
	} else if err != nil {
		return fmt.Errorf("stat key file: %w", err)
	}

	return c.loadKey()
}

// generateAndSaveKey creates a new Ed25519 keypair and saves it to disk.
func (c *AuthClient) generateAndSaveKey() error {
	// Generate keypair
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return fmt.Errorf("generate keypair: %w", err)
	}

	c.publicKey = pub
	c.privateKey = priv

	// Create parent directory if needed
	dir := filepath.Dir(c.keyPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create key directory: %w", err)
	}

	// Encode private key to PEM
	// We store the seed (32 bytes) which can be used to reconstruct the full key
	pemBlock := &pem.Block{
		Type:  "PRIVATE KEY",
		Bytes: priv.Seed(),
	}
	pemData := pem.EncodeToMemory(pemBlock)

	// Write with restrictive permissions
	if err := os.WriteFile(c.keyPath, pemData, 0600); err != nil {
		return fmt.Errorf("write key file: %w", err)
	}

	return nil
}

// loadKey reads an existing keypair from disk.
func (c *AuthClient) loadKey() error {
	data, err := os.ReadFile(c.keyPath)
	if err != nil {
		return fmt.Errorf("read key file: %w", err)
	}

	block, _ := pem.Decode(data)
	if block == nil {
		return errors.New("key file does not contain valid PEM")
	}

	// The seed is 32 bytes; reconstruct the full 64-byte private key
	if len(block.Bytes) != ed25519.SeedSize {
		return fmt.Errorf("invalid key size: got %d bytes, want %d", len(block.Bytes), ed25519.SeedSize)
	}

	c.privateKey = ed25519.NewKeyFromSeed(block.Bytes)
	c.publicKey = c.privateKey.Public().(ed25519.PublicKey)

	return nil
}

// Authenticate performs the AUTH_CHALLENGE/AUTH_RESPONSE exchange with the DPU.
// It waits for an AUTH_CHALLENGE, signs the nonce with the private key,
// sends AUTH_RESPONSE with the signature and public key, and waits for
// AUTH_OK or AUTH_FAIL.
//
// The key must be loaded before calling Authenticate.
func (c *AuthClient) Authenticate(ctx context.Context) error {
	if c.privateKey == nil {
		return errors.New("key not loaded: call LoadOrGenerateKey first")
	}

	// Wait for AUTH_CHALLENGE from DPU
	challengeMsg, err := c.transport.Recv()
	if err != nil {
		return fmt.Errorf("recv auth challenge: %w", err)
	}

	if challengeMsg.Type != MessageAuthChallenge {
		return fmt.Errorf("expected AUTH_CHALLENGE, got %s", challengeMsg.Type)
	}

	challenge, err := ParseAuthPayload[AuthChallengePayload](challengeMsg)
	if err != nil {
		return fmt.Errorf("parse auth challenge: %w", err)
	}

	// Sign the nonce
	nonceBytes, err := hex.DecodeString(challenge.Nonce)
	if err != nil {
		return fmt.Errorf("decode nonce: %w", err)
	}

	signature := ed25519.Sign(c.privateKey, nonceBytes)
	signatureB64 := base64.StdEncoding.EncodeToString(signature)

	// Build and send AUTH_RESPONSE
	responsePayload := AuthResponsePayload{
		Nonce:     challenge.Nonce,
		Signature: signatureB64,
		PublicKey: c.PublicKeyPEM(),
	}

	responseMsg, err := NewAuthMessage(MessageAuthResponse, responsePayload)
	if err != nil {
		return fmt.Errorf("create auth response: %w", err)
	}

	if err := c.transport.Send(responseMsg); err != nil {
		return fmt.Errorf("send auth response: %w", err)
	}

	// Wait for AUTH_OK or AUTH_FAIL
	resultMsg, err := c.transport.Recv()
	if err != nil {
		return fmt.Errorf("recv auth result: %w", err)
	}

	switch resultMsg.Type {
	case MessageAuthOK:
		return nil

	case MessageAuthFail:
		fail, err := ParseAuthPayload[AuthFailPayload](resultMsg)
		if err != nil {
			return fmt.Errorf("parse auth fail: %w", err)
		}
		return fmt.Errorf("authentication failed: %s", fail.Reason)

	default:
		return fmt.Errorf("expected AUTH_OK or AUTH_FAIL, got %s", resultMsg.Type)
	}
}

// PublicKeyPEM returns the public key in PEM format.
// Returns an empty string if the key has not been loaded.
func (c *AuthClient) PublicKeyPEM() string {
	if c.publicKey == nil {
		return ""
	}

	// Marshal to PKIX format for interoperability
	pkixBytes, err := x509.MarshalPKIXPublicKey(c.publicKey)
	if err != nil {
		return ""
	}

	pemBlock := &pem.Block{
		Type:  "PUBLIC KEY",
		Bytes: pkixBytes,
	}

	return string(pem.EncodeToMemory(pemBlock))
}
