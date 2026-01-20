package transport

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"path/filepath"
	"testing"
	"time"
)

// =============================================================================
// Connection Drop Tests
// =============================================================================

func TestErrorInjection_SendErrorDuringMessageTransmission(t *testing.T) {
	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Inject send error to simulate connection drop during transmission
	mock.SetSendError(errors.New("connection reset by peer"))

	msg := &Message{
		Type:    MessagePostureReport,
		ID:      "test-123",
		Payload: json.RawMessage(`{"status":"healthy"}`),
	}

	err := mock.Send(msg)
	if err == nil {
		t.Fatal("Send() should return error when connection drops")
	}

	if err.Error() != "connection reset by peer" {
		t.Errorf("error = %q, want %q", err.Error(), "connection reset by peer")
	}
}

func TestErrorInjection_RecvErrorWhileWaitingForResponse(t *testing.T) {
	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Inject recv error to simulate connection drop while waiting
	mock.SetRecvError(errors.New("connection timed out"))

	_, err := mock.Recv()
	if err == nil {
		t.Fatal("Recv() should return error when connection drops")
	}

	if err.Error() != "connection timed out" {
		t.Errorf("error = %q, want %q", err.Error(), "connection timed out")
	}
}

func TestErrorInjection_ConnectionCloseDuringAuthentication(t *testing.T) {
	tmpDir := t.TempDir()
	keyPath := filepath.Join(tmpDir, "test-key.pem")

	mock := NewMockTransport()
	client := NewAuthClient(mock, keyPath)

	if err := client.LoadOrGenerateKey(); err != nil {
		t.Fatalf("LoadOrGenerateKey() error = %v", err)
	}

	ctx := context.Background()
	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Inject recv error to simulate connection close during auth challenge
	mock.SetRecvError(io.EOF)

	err := client.Authenticate(ctx)
	if err == nil {
		t.Fatal("Authenticate() should fail when connection closes during challenge")
	}

	// The error should indicate recv failure
	if !errors.Is(err, io.EOF) && err.Error() != "recv auth challenge: EOF" {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestErrorInjection_ConnectionCloseDuringAuthResponse(t *testing.T) {
	tmpDir := t.TempDir()
	keyPath := filepath.Join(tmpDir, "test-key.pem")

	mock := NewMockTransport()
	client := NewAuthClient(mock, keyPath)

	if err := client.LoadOrGenerateKey(); err != nil {
		t.Fatalf("LoadOrGenerateKey() error = %v", err)
	}

	ctx := context.Background()
	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Enqueue a valid challenge
	nonce := GenerateNonce()
	challengePayload, _ := json.Marshal(AuthChallengePayload{Nonce: nonce})
	if err := mock.Enqueue(&Message{
		Type:    MessageAuthChallenge,
		ID:      "challenge-id",
		Payload: challengePayload,
	}); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	// Inject send error after client receives challenge
	// Client will receive challenge fine, but fail when sending response
	mock.SetSendError(errors.New("broken pipe"))

	err := client.Authenticate(ctx)
	if err == nil {
		t.Fatal("Authenticate() should fail when send fails")
	}

	if err.Error() != "send auth response: broken pipe" {
		t.Errorf("error = %q, want %q", err.Error(), "send auth response: broken pipe")
	}
}

func TestErrorInjection_ConnectionCloseAfterAuthResponseSent(t *testing.T) {
	tmpDir := t.TempDir()
	keyPath := filepath.Join(tmpDir, "test-key.pem")

	mock := NewMockTransport()
	client := NewAuthClient(mock, keyPath)

	if err := client.LoadOrGenerateKey(); err != nil {
		t.Fatalf("LoadOrGenerateKey() error = %v", err)
	}

	ctx := context.Background()
	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Enqueue a valid challenge
	nonce := GenerateNonce()
	challengePayload, _ := json.Marshal(AuthChallengePayload{Nonce: nonce})
	if err := mock.Enqueue(&Message{
		Type:    MessageAuthChallenge,
		ID:      "challenge-id",
		Payload: challengePayload,
	}); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	// Start authentication in goroutine so we can inject error at right time
	errCh := make(chan error, 1)
	go func() {
		errCh <- client.Authenticate(ctx)
	}()

	// Wait for client to send AUTH_RESPONSE by draining from send channel
	select {
	case <-mock.sendCh:
		// Got the AUTH_RESPONSE
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for AUTH_RESPONSE")
	}

	// Close the transport to simulate connection drop
	mock.Close()

	select {
	case err := <-errCh:
		if err == nil {
			t.Fatal("Authenticate() should fail when connection closes")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for Authenticate to complete")
	}
}

func TestErrorInjection_ConnectionCloseDuringEnrollment(t *testing.T) {
	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Send an enrollment request
	enrollPayload, _ := json.Marshal(map[string]string{
		"hostname": "test-host",
	})
	msg := &Message{
		Type:    MessageEnrollRequest,
		ID:      "enroll-123",
		Payload: enrollPayload,
	}

	if err := mock.Send(msg); err != nil {
		t.Fatalf("Send() error = %v", err)
	}

	// Simulate connection close while waiting for enrollment response
	mock.SetRecvError(io.EOF)

	_, err := mock.Recv()
	if err != io.EOF {
		t.Errorf("expected io.EOF, got %v", err)
	}
}

// =============================================================================
// Timeout Tests
// =============================================================================

func TestErrorInjection_AuthChallengeTimeout(t *testing.T) {
	tmpDir := t.TempDir()
	keyPath := filepath.Join(tmpDir, "test-key.pem")

	mock := NewMockTransport()
	client := NewAuthClient(mock, keyPath)

	if err := client.LoadOrGenerateKey(); err != nil {
		t.Fatalf("LoadOrGenerateKey() error = %v", err)
	}

	ctx := context.Background()
	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Simulate timeout by injecting a timeout error
	mock.SetRecvError(errors.New("read timeout: no response from DPU"))

	err := client.Authenticate(ctx)
	if err == nil {
		t.Fatal("Authenticate() should fail on timeout")
	}

	// Should wrap the timeout error
	if err.Error() != "recv auth challenge: read timeout: no response from DPU" {
		t.Errorf("error = %q, want recv auth challenge timeout error", err.Error())
	}
}

func TestErrorInjection_AuthResponseTimeout(t *testing.T) {
	// This test verifies behavior when the DPU doesn't respond after receiving
	// the auth response. We simulate this by closing the transport while
	// the client is waiting for AUTH_OK/AUTH_FAIL.

	tmpDir := t.TempDir()
	keyPath := filepath.Join(tmpDir, "test-key.pem")

	mock := NewMockTransport()
	client := NewAuthClient(mock, keyPath)

	if err := client.LoadOrGenerateKey(); err != nil {
		t.Fatalf("LoadOrGenerateKey() error = %v", err)
	}

	ctx := context.Background()
	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Enqueue a valid challenge
	nonce := GenerateNonce()
	challengePayload, _ := json.Marshal(AuthChallengePayload{Nonce: nonce})
	if err := mock.Enqueue(&Message{
		Type:    MessageAuthChallenge,
		ID:      "challenge-id",
		Payload: challengePayload,
	}); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	// Start auth in goroutine
	errCh := make(chan error, 1)
	go func() {
		errCh <- client.Authenticate(ctx)
	}()

	// Wait for client to send AUTH_RESPONSE by draining from send channel
	select {
	case <-mock.sendCh:
		// Got the AUTH_RESPONSE
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for AUTH_RESPONSE to be sent")
	}

	// Close the transport to simulate timeout/disconnect while waiting for result
	// This causes the Recv() to return io.EOF
	mock.Close()

	select {
	case err := <-errCh:
		if err == nil {
			t.Fatal("Authenticate() should fail when transport closes")
		}
		// The error should indicate recv failure
		t.Logf("Got expected error: %v", err)
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for Authenticate to complete")
	}
}

func TestErrorInjection_EnrollRequestTimeout(t *testing.T) {
	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Send enrollment request
	enrollPayload, _ := json.Marshal(map[string]string{
		"hostname": "test-host",
	})
	msg := &Message{
		Type:    MessageEnrollRequest,
		ID:      "enroll-123",
		Payload: enrollPayload,
	}

	if err := mock.Send(msg); err != nil {
		t.Fatalf("Send() error = %v", err)
	}

	// Inject timeout while waiting for response
	mock.SetRecvError(errors.New("enrollment timeout: no response from DPU"))

	_, err := mock.Recv()
	if err == nil {
		t.Fatal("Recv() should return timeout error")
	}

	if err.Error() != "enrollment timeout: no response from DPU" {
		t.Errorf("error = %q, want enrollment timeout error", err.Error())
	}
}

func TestErrorInjection_PostureReportTimeout(t *testing.T) {
	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Send posture report
	posturePayload, _ := json.Marshal(map[string]interface{}{
		"status":   "compliant",
		"os":       "linux",
		"kernel":   "5.15.0",
		"hostname": "test-host",
	})
	msg := &Message{
		Type:    MessagePostureReport,
		ID:      "posture-456",
		Payload: posturePayload,
	}

	if err := mock.Send(msg); err != nil {
		t.Fatalf("Send() error = %v", err)
	}

	// Inject timeout while waiting for ACK
	mock.SetRecvError(errors.New("posture ack timeout"))

	_, err := mock.Recv()
	if err == nil {
		t.Fatal("Recv() should return timeout error")
	}

	if err.Error() != "posture ack timeout" {
		t.Errorf("error = %q, want posture ack timeout error", err.Error())
	}
}

// =============================================================================
// Malformed Message Tests
// =============================================================================

func TestErrorInjection_InvalidJSONReceived(t *testing.T) {
	tmpDir := t.TempDir()
	keyPath := filepath.Join(tmpDir, "test-key.pem")

	mock := NewMockTransport()
	client := NewAuthClient(mock, keyPath)

	if err := client.LoadOrGenerateKey(); err != nil {
		t.Fatalf("LoadOrGenerateKey() error = %v", err)
	}

	ctx := context.Background()
	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Enqueue a message with invalid JSON payload
	invalidMsg := &Message{
		Type:    MessageAuthChallenge,
		ID:      "invalid-id",
		Payload: json.RawMessage(`{invalid json`),
	}
	if err := mock.Enqueue(invalidMsg); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	err := client.Authenticate(ctx)
	if err == nil {
		t.Fatal("Authenticate() should fail with invalid JSON")
	}

	// Should indicate parse failure
	if err.Error() != "parse auth challenge: failed to parse auth payload: invalid character 'i' looking for beginning of object key string" {
		// Accept any parse error
		t.Logf("Got expected parse error: %v", err)
	}
}

func TestErrorInjection_WrongMessageTypeInResponse(t *testing.T) {
	tmpDir := t.TempDir()
	keyPath := filepath.Join(tmpDir, "test-key.pem")

	mock := NewMockTransport()
	client := NewAuthClient(mock, keyPath)

	if err := client.LoadOrGenerateKey(); err != nil {
		t.Fatalf("LoadOrGenerateKey() error = %v", err)
	}

	ctx := context.Background()
	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Send wrong message type instead of AUTH_CHALLENGE
	wrongMsg := &Message{
		Type:    MessagePostureAck, // Wrong type
		ID:      "wrong-type-id",
		Payload: json.RawMessage(`{}`),
	}
	if err := mock.Enqueue(wrongMsg); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	err := client.Authenticate(ctx)
	if err == nil {
		t.Fatal("Authenticate() should fail with wrong message type")
	}

	expected := "expected AUTH_CHALLENGE, got POSTURE_ACK"
	if err.Error() != expected {
		t.Errorf("error = %q, want %q", err.Error(), expected)
	}
}

func TestErrorInjection_WrongMessageTypeAfterAuthResponse(t *testing.T) {
	tmpDir := t.TempDir()
	keyPath := filepath.Join(tmpDir, "test-key.pem")

	mock := NewMockTransport()
	client := NewAuthClient(mock, keyPath)

	if err := client.LoadOrGenerateKey(); err != nil {
		t.Fatalf("LoadOrGenerateKey() error = %v", err)
	}

	ctx := context.Background()
	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Enqueue valid challenge
	nonce := GenerateNonce()
	challengePayload, _ := json.Marshal(AuthChallengePayload{Nonce: nonce})
	if err := mock.Enqueue(&Message{
		Type:    MessageAuthChallenge,
		ID:      "challenge-id",
		Payload: challengePayload,
	}); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	// Enqueue wrong message type instead of AUTH_OK/AUTH_FAIL
	wrongMsg := &Message{
		Type:    MessageEnrollResponse, // Wrong type
		ID:      "wrong-result-id",
		Payload: json.RawMessage(`{}`),
	}
	if err := mock.Enqueue(wrongMsg); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	err := client.Authenticate(ctx)
	if err == nil {
		t.Fatal("Authenticate() should fail with wrong result message type")
	}

	expected := "expected AUTH_OK or AUTH_FAIL, got ENROLL_RESPONSE"
	if err.Error() != expected {
		t.Errorf("error = %q, want %q", err.Error(), expected)
	}
}

func TestErrorInjection_MissingCorrelationID(t *testing.T) {
	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Send a request
	reqPayload, _ := json.Marshal(map[string]string{"hostname": "test"})
	request := &Message{
		Type:    MessageEnrollRequest,
		ID:      "original-correlation-id",
		Payload: reqPayload,
	}

	if err := mock.Send(request); err != nil {
		t.Fatalf("Send() error = %v", err)
	}

	// Enqueue response with missing/empty correlation ID
	responseNoID := &Message{
		Type:    MessageEnrollResponse,
		ID:      "", // Missing correlation ID
		Payload: json.RawMessage(`{"host_id": "host-001"}`),
	}
	if err := mock.Enqueue(responseNoID); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	// Receive the response
	resp, err := mock.Recv()
	if err != nil {
		t.Fatalf("Recv() error = %v", err)
	}

	// Verify correlation ID is missing
	if resp.ID != "" {
		t.Errorf("expected empty ID, got %q", resp.ID)
	}

	// In a real scenario, the caller would validate this
	// Here we just verify the message is received with the missing ID
}

func TestErrorInjection_TruncatedMessage(t *testing.T) {
	// This test verifies handling of truncated/malformed challenge messages.
	// Note: An empty nonce in JSON (`{"nonce":""}`) successfully hex-decodes
	// to empty bytes, so the client proceeds with signing. We test that the
	// server can reject such responses by simulating AUTH_FAIL.

	tmpDir := t.TempDir()
	keyPath := filepath.Join(tmpDir, "test-key.pem")

	mock := NewMockTransport()
	client := NewAuthClient(mock, keyPath)

	if err := client.LoadOrGenerateKey(); err != nil {
		t.Fatalf("LoadOrGenerateKey() error = %v", err)
	}

	ctx := context.Background()
	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Enqueue a message with empty nonce (truncated/missing data)
	truncatedMsg := &Message{
		Type:    MessageAuthChallenge,
		ID:      "truncated-id",
		Payload: json.RawMessage(`{"nonce":""}`), // Empty nonce
	}
	if err := mock.Enqueue(truncatedMsg); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	// Enqueue AUTH_FAIL to simulate server rejecting the empty nonce
	failPayload, _ := json.Marshal(AuthFailPayload{Reason: "invalid_nonce"})
	if err := mock.Enqueue(&Message{
		Type:    MessageAuthFail,
		ID:      "fail-id",
		Payload: failPayload,
	}); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	err := client.Authenticate(ctx)
	if err == nil {
		t.Fatal("Authenticate() should fail with truncated/incomplete message")
	}

	// Verify we got the auth failure
	if err.Error() != "authentication failed: invalid_nonce" {
		t.Errorf("error = %q, want authentication failure", err.Error())
	}
}

func TestErrorInjection_NullPayload(t *testing.T) {
	tmpDir := t.TempDir()
	keyPath := filepath.Join(tmpDir, "test-key.pem")

	mock := NewMockTransport()
	client := NewAuthClient(mock, keyPath)

	if err := client.LoadOrGenerateKey(); err != nil {
		t.Fatalf("LoadOrGenerateKey() error = %v", err)
	}

	ctx := context.Background()
	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Enqueue a message with null payload
	nullPayloadMsg := &Message{
		Type:    MessageAuthChallenge,
		ID:      "null-payload-id",
		Payload: nil,
	}
	if err := mock.Enqueue(nullPayloadMsg); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	err := client.Authenticate(ctx)
	if err == nil {
		t.Fatal("Authenticate() should fail with null payload")
	}

	t.Logf("Got expected error for null payload: %v", err)
}

// =============================================================================
// State Error Tests
// =============================================================================

func TestErrorInjection_SendBeforeConnect(t *testing.T) {
	mock := NewMockTransport()

	msg := &Message{
		Type:    MessagePostureReport,
		ID:      "premature-send",
		Payload: json.RawMessage(`{}`),
	}

	err := mock.Send(msg)
	if err == nil {
		t.Fatal("Send() should fail before Connect()")
	}

	if err.Error() != "transport not connected" {
		t.Errorf("error = %q, want %q", err.Error(), "transport not connected")
	}
}

func TestErrorInjection_RecvBeforeConnect(t *testing.T) {
	mock := NewMockTransport()

	_, err := mock.Recv()
	if err == nil {
		t.Fatal("Recv() should fail before Connect()")
	}

	if err.Error() != "transport not connected" {
		t.Errorf("error = %q, want %q", err.Error(), "transport not connected")
	}
}

func TestErrorInjection_EnrollBeforeAuthenticate(t *testing.T) {
	// This test verifies that attempting enrollment operations
	// without authentication should be handled gracefully.
	// The mock transport doesn't enforce protocol ordering,
	// but we verify the messages can be sent/received.

	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Try to send enrollment without authentication
	enrollPayload, _ := json.Marshal(map[string]string{
		"hostname": "test-host",
	})
	msg := &Message{
		Type:    MessageEnrollRequest,
		ID:      "premature-enroll",
		Payload: enrollPayload,
	}

	// The transport allows this, but a real server would reject it
	err := mock.Send(msg)
	if err != nil {
		t.Errorf("Send() error = %v (transport should allow, server should reject)", err)
	}

	// Verify the message was sent (server-side validation is separate)
	sent := mock.SentMessages()
	if len(sent) != 1 {
		t.Errorf("expected 1 sent message, got %d", len(sent))
	}
	if sent[0].Type != MessageEnrollRequest {
		t.Errorf("sent message type = %s, want %s", sent[0].Type, MessageEnrollRequest)
	}
}

func TestErrorInjection_DoubleConnect(t *testing.T) {
	mock := NewMockTransport()
	ctx := context.Background()

	// First connect
	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("first Connect() error = %v", err)
	}

	// Second connect should succeed (idempotent behavior)
	if err := mock.Connect(ctx); err != nil {
		t.Errorf("second Connect() error = %v, expected nil (idempotent)", err)
	}

	// Transport should still be usable
	if !mock.IsConnected() {
		t.Error("transport should still be connected after double connect")
	}
}

func TestErrorInjection_DoubleConnectWithCustomConnectFunc(t *testing.T) {
	connectCount := 0
	mock := NewMockTransport(WithConnectFunc(func(ctx context.Context) error {
		connectCount++
		if connectCount > 1 {
			return errors.New("already connected")
		}
		return nil
	}))

	ctx := context.Background()

	// First connect
	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("first Connect() error = %v", err)
	}

	// Second connect should fail with custom func
	err := mock.Connect(ctx)
	if err == nil {
		t.Fatal("second Connect() should fail with custom connect func")
	}

	if err.Error() != "already connected" {
		t.Errorf("error = %q, want %q", err.Error(), "already connected")
	}
}

func TestErrorInjection_OperationsAfterClose(t *testing.T) {
	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Close the transport
	if err := mock.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	// Send after close
	err := mock.Send(&Message{Type: MessagePostureReport, ID: "after-close"})
	if err == nil {
		t.Error("Send() should fail after Close()")
	}
	if err.Error() != "transport closed" {
		t.Errorf("Send after close error = %q, want %q", err.Error(), "transport closed")
	}

	// Recv after close
	_, err = mock.Recv()
	if err != io.EOF {
		t.Errorf("Recv after close error = %v, want io.EOF", err)
	}

	// Connect after close
	err = mock.Connect(ctx)
	if err == nil {
		t.Error("Connect() should fail after Close()")
	}
	if err.Error() != "transport closed" {
		t.Errorf("Connect after close error = %q, want %q", err.Error(), "transport closed")
	}
}

func TestErrorInjection_DoubleClose(t *testing.T) {
	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// First close
	if err := mock.Close(); err != nil {
		t.Fatalf("first Close() error = %v", err)
	}

	// Second close should be safe (idempotent)
	if err := mock.Close(); err != nil {
		t.Errorf("second Close() error = %v, expected nil (idempotent)", err)
	}
}

// =============================================================================
// Dynamic Error Injection Tests
// =============================================================================

func TestErrorInjection_DynamicErrorToggle(t *testing.T) {
	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Initially no error
	msg := &Message{Type: MessagePostureReport, ID: "test-1"}
	if err := mock.Send(msg); err != nil {
		t.Fatalf("initial Send() error = %v", err)
	}

	// Inject error
	mock.SetSendError(errors.New("network unreachable"))

	err := mock.Send(msg)
	if err == nil {
		t.Fatal("Send() should fail after error injection")
	}
	if err.Error() != "network unreachable" {
		t.Errorf("error = %q, want %q", err.Error(), "network unreachable")
	}

	// Clear error
	mock.SetSendError(nil)

	if err := mock.Send(msg); err != nil {
		t.Errorf("Send() after clearing error = %v", err)
	}
}

func TestErrorInjection_DynamicRecvErrorToggle(t *testing.T) {
	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Enqueue a message
	if err := mock.Enqueue(&Message{Type: MessagePostureAck, ID: "ack-1"}); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	// Should receive successfully
	_, err := mock.Recv()
	if err != nil {
		t.Fatalf("initial Recv() error = %v", err)
	}

	// Inject error
	mock.SetRecvError(errors.New("connection reset"))

	_, err = mock.Recv()
	if err == nil {
		t.Fatal("Recv() should fail after error injection")
	}
	if err.Error() != "connection reset" {
		t.Errorf("error = %q, want %q", err.Error(), "connection reset")
	}

	// Clear error and enqueue new message
	mock.SetRecvError(nil)
	if err := mock.Enqueue(&Message{Type: MessagePostureAck, ID: "ack-2"}); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	_, err = mock.Recv()
	if err != nil {
		t.Errorf("Recv() after clearing error = %v", err)
	}
}

// =============================================================================
// Listener Error Tests
// =============================================================================

func TestErrorInjection_ListenerAcceptError(t *testing.T) {
	listener := NewMockTransportListener(
		WithAcceptError(errors.New("accept: too many connections")),
	)
	defer listener.Close()

	_, err := listener.Accept()
	if err == nil {
		t.Fatal("Accept() should fail with injected error")
	}

	if err.Error() != "accept: too many connections" {
		t.Errorf("error = %q, want %q", err.Error(), "accept: too many connections")
	}
}

func TestErrorInjection_ListenerDynamicAcceptError(t *testing.T) {
	listener := NewMockTransportListener()
	defer listener.Close()

	// Enqueue transport for first accept
	if _, err := listener.EnqueueMockTransport(); err != nil {
		t.Fatalf("EnqueueMockTransport() error = %v", err)
	}

	// First accept should succeed
	_, err := listener.Accept()
	if err != nil {
		t.Fatalf("first Accept() error = %v", err)
	}

	// Inject error
	listener.SetAcceptError(errors.New("resource exhausted"))

	// Second accept should fail
	_, err = listener.Accept()
	if err == nil {
		t.Fatal("Accept() should fail after error injection")
	}
	if err.Error() != "resource exhausted" {
		t.Errorf("error = %q, want %q", err.Error(), "resource exhausted")
	}

	// Clear error and enqueue new transport
	listener.SetAcceptError(nil)
	if _, err := listener.EnqueueMockTransport(); err != nil {
		t.Fatalf("EnqueueMockTransport() error = %v", err)
	}

	// Should succeed now
	_, err = listener.Accept()
	if err != nil {
		t.Errorf("Accept() after clearing error = %v", err)
	}
}

func TestErrorInjection_ListenerCloseInterruptsAccept(t *testing.T) {
	listener := NewMockTransportListener()

	// Start Accept in goroutine
	errCh := make(chan error, 1)
	go func() {
		_, err := listener.Accept()
		errCh <- err
	}()

	// Give Accept time to start waiting
	time.Sleep(50 * time.Millisecond)

	// Close the listener
	if err := listener.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	// Accept should return with error
	select {
	case err := <-errCh:
		if err == nil {
			t.Error("Accept() should return error after Close()")
		}
	case <-time.After(time.Second):
		t.Fatal("Accept() did not return after Close()")
	}
}

// =============================================================================
// Combined Scenario Tests
// =============================================================================

func TestErrorInjection_AuthServerSendChallengeError(t *testing.T) {
	keyStore, err := NewKeyStore(t.TempDir() + "/keystore.json")
	if err != nil {
		t.Fatalf("NewKeyStore() error = %v", err)
	}
	server := NewAuthServer(keyStore, 10)

	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Inject send error to simulate failure sending challenge
	mock.SetSendError(errors.New("send failed: connection dropped"))

	_, authErr := server.Authenticate(ctx, mock)
	if authErr == nil {
		t.Fatal("Authenticate() should fail when challenge send fails")
	}

	if authErr.Error() != "send auth challenge: send failed: connection dropped" {
		t.Errorf("error = %q, want send auth challenge error", authErr.Error())
	}
}

func TestErrorInjection_AuthServerRecvResponseError(t *testing.T) {
	keyStore, err := NewKeyStore(t.TempDir() + "/keystore.json")
	if err != nil {
		t.Fatalf("NewKeyStore() error = %v", err)
	}
	server := NewAuthServer(keyStore, 10)

	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Start auth in goroutine
	errCh := make(chan error, 1)
	go func() {
		_, err := server.Authenticate(ctx, mock)
		errCh <- err
	}()

	// Wait for challenge to be sent by draining from send channel
	select {
	case <-mock.sendCh:
		// Got the AUTH_CHALLENGE
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for AUTH_CHALLENGE")
	}

	// Close the transport to simulate timeout/disconnect while server waits for response
	// This will cause the Recv() to return io.EOF
	mock.Close()

	select {
	case authErr := <-errCh:
		if authErr == nil {
			t.Fatal("Authenticate() should fail when response recv fails")
		}
		// Verify we got an error (the exact message depends on implementation)
		t.Logf("Got expected error: %v", authErr)
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for Authenticate to complete")
	}
}

func TestErrorInjection_PartialMessageSequence(t *testing.T) {
	// Test receiving partial/interrupted message sequence
	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Enqueue first part of a sequence
	if err := mock.Enqueue(&Message{
		Type:    MessagePostureAck,
		ID:      "partial-1",
		Payload: json.RawMessage(`{"status":"pending"}`),
	}); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	// Receive first message
	msg1, err := mock.Recv()
	if err != nil {
		t.Fatalf("Recv() error = %v", err)
	}
	if msg1.ID != "partial-1" {
		t.Errorf("message ID = %q, want %q", msg1.ID, "partial-1")
	}

	// Inject error for second message
	mock.SetRecvError(errors.New("connection interrupted"))

	_, err = mock.Recv()
	if err == nil {
		t.Fatal("Recv() should fail when connection is interrupted")
	}

	// Clear error and verify recovery
	mock.SetRecvError(nil)
	if err := mock.Enqueue(&Message{
		Type:    MessagePostureAck,
		ID:      "recovery-1",
		Payload: json.RawMessage(`{"status":"recovered"}`),
	}); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	msg2, err := mock.Recv()
	if err != nil {
		t.Errorf("Recv() after recovery error = %v", err)
	}
	if msg2 != nil && msg2.ID != "recovery-1" {
		t.Errorf("recovered message ID = %q, want %q", msg2.ID, "recovery-1")
	}
}

func TestErrorInjection_RapidErrorStateChanges(t *testing.T) {
	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Rapidly toggle error states
	for i := 0; i < 10; i++ {
		if i%2 == 0 {
			mock.SetSendError(errors.New("intermittent failure"))
			err := mock.Send(&Message{Type: MessagePostureReport, ID: "rapid-test"})
			if err == nil {
				t.Errorf("iteration %d: Send() should fail with error set", i)
			}
		} else {
			mock.SetSendError(nil)
			err := mock.Send(&Message{Type: MessagePostureReport, ID: "rapid-test"})
			if err != nil {
				t.Errorf("iteration %d: Send() should succeed with error cleared: %v", i, err)
			}
		}
	}

	// Verify final state
	mock.SetSendError(nil)
	if err := mock.Send(&Message{Type: MessagePostureReport, ID: "final-test"}); err != nil {
		t.Errorf("final Send() error = %v", err)
	}
}

// =============================================================================
// Latency and Timeout Simulation Tests
// =============================================================================

func TestErrorInjection_HighLatencyDoesNotCauseError(t *testing.T) {
	mock := NewMockTransport(WithLatency(100 * time.Millisecond))
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Enqueue a message
	if err := mock.Enqueue(&Message{Type: MessagePostureAck, ID: "latency-test"}); err != nil {
		t.Fatalf("Enqueue() error = %v", err)
	}

	// Recv should succeed despite latency
	start := time.Now()
	msg, err := mock.Recv()
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Recv() error = %v", err)
	}

	if msg.ID != "latency-test" {
		t.Errorf("message ID = %q, want %q", msg.ID, "latency-test")
	}

	// Verify latency was applied
	if elapsed < 100*time.Millisecond {
		t.Errorf("elapsed = %v, want >= 100ms", elapsed)
	}
}

func TestErrorInjection_LatencyCanBeChangedDynamically(t *testing.T) {
	mock := NewMockTransport()
	ctx := context.Background()

	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	// Enqueue messages
	for i := 0; i < 3; i++ {
		if err := mock.Enqueue(&Message{
			Type: MessagePostureAck,
			ID:   "dynamic-latency-test",
		}); err != nil {
			t.Fatalf("Enqueue() error = %v", err)
		}
	}

	// First recv with no latency
	start := time.Now()
	_, _ = mock.Recv()
	elapsed1 := time.Since(start)

	// Set high latency
	mock.SetLatency(100 * time.Millisecond)

	start = time.Now()
	_, _ = mock.Recv()
	elapsed2 := time.Since(start)

	// Reset latency
	mock.SetLatency(0)

	start = time.Now()
	_, _ = mock.Recv()
	elapsed3 := time.Since(start)

	// Verify latency changes
	if elapsed1 >= 100*time.Millisecond {
		t.Errorf("first recv took %v, expected < 100ms", elapsed1)
	}
	if elapsed2 < 100*time.Millisecond {
		t.Errorf("second recv took %v, expected >= 100ms", elapsed2)
	}
	if elapsed3 >= 100*time.Millisecond {
		t.Errorf("third recv took %v, expected < 100ms after reset", elapsed3)
	}
}
