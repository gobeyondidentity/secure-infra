package transport

import (
	"context"
	"crypto/ed25519"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/pem"
	"errors"
	"fmt"
	"sync"
	"time"
)

// DefaultNonceTimeout is the default expiry time for authentication nonces.
const DefaultNonceTimeout = 30 * time.Second

// AuthServer handles ComCh authentication for DPU agents.
// It implements the server side of the AUTH_CHALLENGE/AUTH_RESPONSE exchange
// and manages trusted host keys using a TOFU (Trust On First Use) model.
type AuthServer struct {
	keyStore      *KeyStore
	maxConns      int
	nonceTimeout  time.Duration
	pendingNonces map[string]time.Time // nonce -> expiry time
	mu            sync.Mutex
}

// NewAuthServer creates an auth server with the given key store.
// maxConns specifies the maximum number of concurrent host connections (for future use).
func NewAuthServer(keyStore *KeyStore, maxConns int) *AuthServer {
	return &AuthServer{
		keyStore:      keyStore,
		maxConns:      maxConns,
		nonceTimeout:  DefaultNonceTimeout,
		pendingNonces: make(map[string]time.Time),
	}
}

// Authenticate performs server-side AUTH_CHALLENGE/AUTH_RESPONSE exchange.
// Called after Accept() for each new connection.
//
// Protocol flow:
// 1. Server generates random 32-byte nonce
// 2. Server sends AUTH_CHALLENGE with hex-encoded nonce
// 3. Client signs nonce with private key
// 4. Client sends AUTH_RESPONSE with signature and public key
// 5. Server verifies signature
// 6. If new key (TOFU): register it
// 7. If known key: verify it matches stored key
// 8. Server sends AUTH_OK on success, AUTH_FAIL with reason on failure
//
// Returns the host's public key on success (for connection tracking).
func (s *AuthServer) Authenticate(ctx context.Context, t Transport) (ed25519.PublicKey, error) {
	// Generate and track nonce
	nonce := GenerateNonce()
	expiry := time.Now().Add(s.nonceTimeout)

	s.mu.Lock()
	s.pendingNonces[nonce] = expiry
	s.mu.Unlock()

	// Clean up nonce when done
	defer func() {
		s.mu.Lock()
		delete(s.pendingNonces, nonce)
		s.mu.Unlock()
	}()

	// Send AUTH_CHALLENGE
	challengePayload := AuthChallengePayload{Nonce: nonce}
	challengeMsg, err := NewAuthMessage(MessageAuthChallenge, challengePayload)
	if err != nil {
		return nil, fmt.Errorf("create auth challenge: %w", err)
	}

	if err := t.Send(challengeMsg); err != nil {
		return nil, fmt.Errorf("send auth challenge: %w", err)
	}

	// Wait for AUTH_RESPONSE
	responseMsg, err := t.Recv()
	if err != nil {
		return nil, fmt.Errorf("recv auth response: %w", err)
	}

	// Verify message type
	if responseMsg.Type != MessageAuthResponse {
		reason := fmt.Sprintf("expected AUTH_RESPONSE, got %s", responseMsg.Type)
		s.sendAuthFail(t, "protocol_error")
		return nil, errors.New(reason)
	}

	// Parse response
	response, err := ParseAuthPayload[AuthResponsePayload](responseMsg)
	if err != nil {
		s.sendAuthFail(t, "invalid_payload")
		return nil, fmt.Errorf("parse auth response: %w", err)
	}

	// Verify nonce is valid and not expired
	if err := s.verifyNonce(response.Nonce); err != nil {
		s.sendAuthFail(t, "expired_nonce")
		return nil, err
	}

	// Parse public key from response
	pubKey, err := s.parsePublicKey(response.PublicKey)
	if err != nil {
		s.sendAuthFail(t, "missing_public_key")
		return nil, err
	}

	// Verify signature
	if err := s.verifySignature(response.Nonce, response.Signature, pubKey); err != nil {
		s.sendAuthFail(t, "invalid_signature")
		return nil, err
	}

	// TOFU: register new key or verify existing key
	if !s.keyStore.IsKnown(pubKey) {
		// Trust on first use: register the new key
		if err := s.keyStore.Register(pubKey); err != nil {
			s.sendAuthFail(t, "registration_failed")
			return nil, fmt.Errorf("register key: %w", err)
		}
	} else {
		// Known key: verify it matches
		if !s.keyStore.Verify(pubKey) {
			s.sendAuthFail(t, "key_mismatch")
			return nil, errors.New("public key does not match stored key")
		}
	}

	// Send AUTH_OK
	okMsg, err := NewAuthMessage(MessageAuthOK, AuthOKPayload{})
	if err != nil {
		return nil, fmt.Errorf("create auth ok: %w", err)
	}

	if err := t.Send(okMsg); err != nil {
		return nil, fmt.Errorf("send auth ok: %w", err)
	}

	return pubKey, nil
}

// verifyNonce checks if the nonce is valid and not expired.
func (s *AuthServer) verifyNonce(nonce string) error {
	s.mu.Lock()
	expiry, ok := s.pendingNonces[nonce]
	s.mu.Unlock()

	if !ok {
		return errors.New("unknown nonce")
	}

	if time.Now().After(expiry) {
		return errors.New("nonce expired")
	}

	return nil
}

// parsePublicKey parses a PEM-encoded Ed25519 public key.
func (s *AuthServer) parsePublicKey(pemStr string) (ed25519.PublicKey, error) {
	if pemStr == "" {
		return nil, errors.New("public key is required")
	}

	block, _ := pem.Decode([]byte(pemStr))
	if block == nil {
		return nil, errors.New("invalid PEM encoding")
	}

	pubInterface, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("parse public key: %w", err)
	}

	pubKey, ok := pubInterface.(ed25519.PublicKey)
	if !ok {
		return nil, errors.New("not an Ed25519 public key")
	}

	return pubKey, nil
}

// verifySignature verifies the Ed25519 signature of the nonce.
func (s *AuthServer) verifySignature(nonce, signatureB64 string, pubKey ed25519.PublicKey) error {
	// Decode nonce from hex
	nonceBytes, err := hex.DecodeString(nonce)
	if err != nil {
		return fmt.Errorf("decode nonce: %w", err)
	}

	// Decode signature from base64
	signature, err := base64.StdEncoding.DecodeString(signatureB64)
	if err != nil {
		return fmt.Errorf("decode signature: %w", err)
	}

	// Verify signature
	if !ed25519.Verify(pubKey, nonceBytes, signature) {
		return errors.New("signature verification failed")
	}

	return nil
}

// sendAuthFail sends an AUTH_FAIL message with the given reason.
func (s *AuthServer) sendAuthFail(t Transport, reason string) {
	failPayload := AuthFailPayload{Reason: reason}
	failMsg, err := NewAuthMessage(MessageAuthFail, failPayload)
	if err != nil {
		return // Best effort, ignore error
	}
	t.Send(failMsg) // Best effort, ignore error
}
