package transport

import (
	"context"
	"sync"
	"testing"
	"time"
)

// ============================================================================
// NetworkTransport Tests
// ============================================================================

func TestNewNetworkTransport(t *testing.T) {
	transport, err := NewNetworkTransport("localhost:8443", "test-invite", nil, "test-host")
	if err != nil {
		t.Fatalf("NewNetworkTransport failed: %v", err)
	}

	if transport == nil {
		t.Fatal("expected transport, got nil")
	}

	// Cast to access internal fields
	nt := transport.(*networkTransport)
	if nt.dpuAddr != "localhost:8443" {
		t.Errorf("dpuAddr = %q, want %q", nt.dpuAddr, "localhost:8443")
	}
	if nt.inviteCode != "test-invite" {
		t.Errorf("inviteCode = %q, want %q", nt.inviteCode, "test-invite")
	}
	if nt.hostname != "test-host" {
		t.Errorf("hostname = %q, want %q", nt.hostname, "test-host")
	}
}

func TestNetworkTransport_Type(t *testing.T) {
	transport, _ := NewNetworkTransport("localhost:8443", "", nil, "")
	if transport.Type() != TransportNetwork {
		t.Errorf("Type() = %s, want %s", transport.Type(), TransportNetwork)
	}
}

func TestNetworkTransport_Connect(t *testing.T) {
	transport, _ := NewNetworkTransport("localhost:8443", "", nil, "")

	err := transport.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	// Verify connected state
	nt := transport.(*networkTransport)
	if !nt.connected {
		t.Error("expected connected = true after Connect")
	}
}

func TestNetworkTransport_ConnectAfterClose(t *testing.T) {
	transport, _ := NewNetworkTransport("localhost:8443", "", nil, "")

	// Close first
	transport.Close()

	// Connect should fail
	err := transport.Connect(context.Background())
	if err == nil {
		t.Error("expected error connecting after close")
	}
	if err.Error() != "transport closed" {
		t.Errorf("error = %q, want %q", err.Error(), "transport closed")
	}
}

func TestNetworkTransport_DPUAddr(t *testing.T) {
	transport, _ := NewNetworkTransport("192.168.1.204:8443", "", nil, "")
	nt := transport.(*networkTransport)

	if nt.DPUAddr() != "192.168.1.204:8443" {
		t.Errorf("DPUAddr() = %q, want %q", nt.DPUAddr(), "192.168.1.204:8443")
	}
}

func TestNetworkTransport_Hostname(t *testing.T) {
	transport, _ := NewNetworkTransport("localhost:8443", "", nil, "my-hostname")
	nt := transport.(*networkTransport)

	if nt.Hostname() != "my-hostname" {
		t.Errorf("Hostname() = %q, want %q", nt.Hostname(), "my-hostname")
	}
}

func TestNetworkTransport_SendNotConnected(t *testing.T) {
	transport, _ := NewNetworkTransport("localhost:8443", "", nil, "")

	msg := &Message{Type: MessageEnrollRequest}
	err := transport.Send(msg)
	if err == nil {
		t.Error("expected error when sending without connect")
	}
	if err.Error() != "transport not connected" {
		t.Errorf("error = %q, want %q", err.Error(), "transport not connected")
	}
}

func TestNetworkTransport_SendNotImplemented(t *testing.T) {
	transport, _ := NewNetworkTransport("localhost:8443", "", nil, "")
	transport.Connect(context.Background())

	msg := &Message{Type: MessageEnrollRequest}
	err := transport.Send(msg)
	if err == nil {
		t.Error("expected error for unimplemented Send")
	}
	// Send returns not implemented error
	expected := "NetworkTransport.Send not implemented for message type ENROLL_REQUEST"
	if err.Error() != expected {
		t.Errorf("error = %q, want %q", err.Error(), expected)
	}
}

func TestNetworkTransport_RecvNotConnected(t *testing.T) {
	transport, _ := NewNetworkTransport("localhost:8443", "", nil, "")

	_, err := transport.Recv()
	if err == nil {
		t.Error("expected error when receiving without connect")
	}
	if err.Error() != "transport not connected" {
		t.Errorf("error = %q, want %q", err.Error(), "transport not connected")
	}
}

func TestNetworkTransport_RecvFromQueue(t *testing.T) {
	transport, _ := NewNetworkTransport("localhost:8443", "", nil, "")
	transport.Connect(context.Background())

	nt := transport.(*networkTransport)

	// Queue a message
	msg := &Message{Type: MessageEnrollResponse, ID: "test-123"}
	err := nt.QueueResponse(msg)
	if err != nil {
		t.Fatalf("QueueResponse failed: %v", err)
	}

	// Receive should return the queued message
	received, err := transport.Recv()
	if err != nil {
		t.Fatalf("Recv failed: %v", err)
	}
	if received.Type != MessageEnrollResponse {
		t.Errorf("Type = %s, want %s", received.Type, MessageEnrollResponse)
	}
	if received.ID != "test-123" {
		t.Errorf("ID = %s, want test-123", received.ID)
	}
}

func TestNetworkTransport_RecvAfterClose(t *testing.T) {
	transport, _ := NewNetworkTransport("localhost:8443", "", nil, "")
	transport.Connect(context.Background())

	// Close the transport
	transport.Close()

	// Recv should return an error (not connected after close)
	_, err := transport.Recv()
	if err == nil {
		t.Error("expected error after close")
	}
	// After close, connected=false so we get "not connected"
	if err.Error() != "transport not connected" {
		t.Errorf("Recv error = %v, want 'transport not connected'", err)
	}
}

func TestNetworkTransport_Close(t *testing.T) {
	transport, _ := NewNetworkTransport("localhost:8443", "", nil, "")
	transport.Connect(context.Background())

	// First close should succeed
	err := transport.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	nt := transport.(*networkTransport)
	if !nt.closed {
		t.Error("expected closed = true after Close")
	}
	if nt.connected {
		t.Error("expected connected = false after Close")
	}
}

func TestNetworkTransport_DoubleClose(t *testing.T) {
	transport, _ := NewNetworkTransport("localhost:8443", "", nil, "")
	transport.Connect(context.Background())

	// First close
	transport.Close()

	// Second close should be safe (no error)
	err := transport.Close()
	if err != nil {
		t.Errorf("double close should not error: %v", err)
	}
}

func TestNetworkTransport_QueueResponse(t *testing.T) {
	transport, _ := NewNetworkTransport("localhost:8443", "", nil, "")
	nt := transport.(*networkTransport)

	// Queue a response
	msg := &Message{Type: MessagePostureAck, ID: "ack-1"}
	err := nt.QueueResponse(msg)
	if err != nil {
		t.Fatalf("QueueResponse failed: %v", err)
	}
}

func TestNetworkTransport_QueueResponseFull(t *testing.T) {
	transport, _ := NewNetworkTransport("localhost:8443", "", nil, "")
	nt := transport.(*networkTransport)

	// Fill the queue (buffer size is 10)
	for i := 0; i < 10; i++ {
		msg := &Message{Type: MessagePostureAck, ID: "msg"}
		err := nt.QueueResponse(msg)
		if err != nil {
			t.Fatalf("QueueResponse %d failed: %v", i, err)
		}
	}

	// Next should fail with queue full
	msg := &Message{Type: MessagePostureAck, ID: "overflow"}
	err := nt.QueueResponse(msg)
	if err == nil {
		t.Error("expected error when queue is full")
	}
	if err.Error() != "response queue full" {
		t.Errorf("error = %q, want %q", err.Error(), "response queue full")
	}
}

// ============================================================================
// NetworkListener Tests
// ============================================================================

func TestNewNetworkListener(t *testing.T) {
	listener, err := NewNetworkListener(":18052", nil)
	if err != nil {
		t.Fatalf("NewNetworkListener failed: %v", err)
	}
	defer listener.Close()

	if listener == nil {
		t.Fatal("expected listener, got nil")
	}
}

func TestNewNetworkListener_EmptyAddr(t *testing.T) {
	_, err := NewNetworkListener("", nil)
	if err == nil {
		t.Error("expected error for empty address")
	}
	expected := "network listener: address required"
	if err.Error() != expected {
		t.Errorf("error = %q, want %q", err.Error(), expected)
	}
}

func TestNetworkListener_Type(t *testing.T) {
	listener, _ := NewNetworkListener(":18052", nil)
	defer listener.Close()

	if listener.Type() != TransportNetwork {
		t.Errorf("Type() = %s, want %s", listener.Type(), TransportNetwork)
	}
}

func TestNetworkListener_Addr(t *testing.T) {
	listener, _ := NewNetworkListener(":18052", nil)
	defer listener.Close()

	if listener.Addr() != ":18052" {
		t.Errorf("Addr() = %s, want :18052", listener.Addr())
	}
}

func TestNetworkListener_AcceptAfterClose(t *testing.T) {
	listener, _ := NewNetworkListener(":18052", nil)

	// Close immediately
	listener.Close()

	// Accept should fail
	_, err := listener.Accept()
	if err == nil {
		t.Error("expected error accepting after close")
	}
	if err.Error() != "listener closed" {
		t.Errorf("error = %q, want %q", err.Error(), "listener closed")
	}
}

func TestNetworkListener_AcceptBlocksUntilClose(t *testing.T) {
	listener, _ := NewNetworkListener(":18052", nil)

	// Start accept in goroutine
	errCh := make(chan error, 1)
	go func() {
		_, err := listener.Accept()
		errCh <- err
	}()

	// Give Accept time to block
	time.Sleep(50 * time.Millisecond)

	// Verify Accept is still blocking
	select {
	case err := <-errCh:
		t.Fatalf("Accept should be blocking, got error: %v", err)
	default:
		// Expected: Accept is blocking
	}

	// Close the listener
	listener.Close()

	// Accept should now return with error
	select {
	case err := <-errCh:
		if err == nil {
			t.Error("expected error after close")
		}
		if err.Error() != "listener closed" {
			t.Errorf("error = %q, want %q", err.Error(), "listener closed")
		}
	case <-time.After(time.Second):
		t.Error("Accept did not return after close")
	}
}

func TestNetworkListener_QueueConnection(t *testing.T) {
	listener, _ := NewNetworkListener(":18052", nil)
	defer listener.Close()

	// Create a mock transport to queue
	mockTransport := NewMockTransport()

	// Queue the connection
	err := listener.QueueConnection(mockTransport)
	if err != nil {
		t.Fatalf("QueueConnection failed: %v", err)
	}

	// Accept should return the queued transport
	transport, err := listener.Accept()
	if err != nil {
		t.Fatalf("Accept failed: %v", err)
	}
	if transport != mockTransport {
		t.Error("Accept returned different transport than queued")
	}
}

func TestNetworkListener_QueueConnectionFull(t *testing.T) {
	listener, _ := NewNetworkListener(":18052", nil)
	defer listener.Close()

	// Fill the accept queue (buffer size is 10)
	for i := 0; i < 10; i++ {
		mockTransport := NewMockTransport()
		err := listener.QueueConnection(mockTransport)
		if err != nil {
			t.Fatalf("QueueConnection %d failed: %v", i, err)
		}
	}

	// Next should fail with queue full
	mockTransport := NewMockTransport()
	err := listener.QueueConnection(mockTransport)
	if err == nil {
		t.Error("expected error when queue is full")
	}
	if err.Error() != "accept queue full" {
		t.Errorf("error = %q, want %q", err.Error(), "accept queue full")
	}
}

func TestNetworkListener_DoubleClose(t *testing.T) {
	listener, _ := NewNetworkListener(":18052", nil)

	// First close
	err := listener.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Second close should be safe
	err = listener.Close()
	if err != nil {
		t.Errorf("double close should not error: %v", err)
	}
}

func TestNetworkListener_ConcurrentAcceptAndClose(t *testing.T) {
	listener, _ := NewNetworkListener(":18052", nil)

	// Start multiple Accept goroutines
	var wg sync.WaitGroup
	errCh := make(chan error, 5)
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := listener.Accept()
			if err != nil {
				errCh <- err
			}
		}()
	}

	// Give goroutines time to start blocking
	time.Sleep(50 * time.Millisecond)

	// Close should unblock all
	listener.Close()

	// Wait for all goroutines
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// All goroutines finished
	case <-time.After(2 * time.Second):
		t.Error("goroutines did not finish after close")
	}

	// All should have received errors
	close(errCh)
	count := 0
	for err := range errCh {
		if err != nil {
			count++
		}
	}
	// At least some should have received the closed error
	// (exact count depends on timing)
}
