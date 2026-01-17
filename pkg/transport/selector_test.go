package transport

import (
	"context"
	"testing"
)

// testTransport is a minimal Transport implementation for testing.
type testTransport struct {
	transportType TransportType
}

func (t *testTransport) Connect(ctx context.Context) error { return nil }
func (t *testTransport) Send(msg *Message) error          { return nil }
func (t *testTransport) Recv() (*Message, error)          { return nil, nil }
func (t *testTransport) Close() error                     { return nil }
func (t *testTransport) Type() TransportType              { return t.transportType }

func TestNewHostTransport_MockPriority(t *testing.T) {
	mock := &testTransport{transportType: TransportMock}
	cfg := &Config{
		MockTransport: mock,
	}

	transport, err := NewHostTransport(cfg)
	if err != nil {
		t.Fatalf("NewHostTransport with mock failed: %v", err)
	}

	if transport != mock {
		t.Error("Expected mock transport to be returned")
	}

	if transport.Type() != TransportMock {
		t.Errorf("Expected type %s, got %s", TransportMock, transport.Type())
	}
}

func TestNewHostTransport_MockOverridesHardware(t *testing.T) {
	// Even if tmfifo path is set, mock should take priority
	mock := &testTransport{transportType: TransportMock}
	cfg := &Config{
		MockTransport: mock,
		TmfifoPath:    "/dev/tmfifo_net0",
		InviteCode:    "test-invite",
		DPUAddr:       "localhost:8443",
	}

	transport, err := NewHostTransport(cfg)
	if err != nil {
		t.Fatalf("NewHostTransport failed: %v", err)
	}

	if transport != mock {
		t.Error("Mock transport should override all other options")
	}
}

func TestNewHostTransport_NoTransportAvailable(t *testing.T) {
	// No mock, no tmfifo device, no invite code
	cfg := &Config{
		TmfifoPath: "/nonexistent/device",
	}

	transport, err := NewHostTransport(cfg)
	if err == nil {
		t.Fatal("Expected error when no transport available")
	}

	if transport != nil {
		t.Error("Expected nil transport when error returned")
	}
}

func TestNewHostTransport_NilConfig(t *testing.T) {
	// Should handle nil config gracefully
	transport, err := NewHostTransport(nil)
	if err == nil {
		t.Fatal("Expected error with nil config (no transport available)")
	}

	if transport != nil {
		t.Error("Expected nil transport")
	}
}

func TestNewHostTransport_NetworkRequiresDPUAddr(t *testing.T) {
	// Only invite code, no DPU address - should error
	cfg := &Config{
		InviteCode: "test-invite",
		TmfifoPath: "/nonexistent/device",
	}

	_, err := NewHostTransport(cfg)
	if err == nil {
		t.Error("Expected error when DPUAddr missing")
	}

	// DPU address provided, no invite code - should succeed (legacy HTTP mode)
	cfg = &Config{
		DPUAddr:    "localhost:8443",
		TmfifoPath: "/nonexistent/device",
	}

	transport, err := NewHostTransport(cfg)
	if err != nil {
		t.Errorf("Should not error when only DPUAddr provided (legacy HTTP mode): %v", err)
	}
	if transport == nil {
		t.Error("Expected transport to be returned")
	}
	if transport != nil && transport.Type() != TransportNetwork {
		t.Errorf("Expected network transport, got %s", transport.Type())
	}
}

func TestNewHostTransport_ForceTmfifo(t *testing.T) {
	// ForceTmfifo with nonexistent device should error
	cfg := &Config{
		ForceTmfifo: true,
		TmfifoPath:  "/nonexistent/device",
		DPUAddr:     "localhost:8443", // Should be ignored
	}

	_, err := NewHostTransport(cfg)
	if err == nil {
		t.Error("Expected error when ForceTmfifo set but device doesn't exist")
	}
}

func TestNewHostTransport_ForceNetwork(t *testing.T) {
	// ForceNetwork skips hardware detection
	cfg := &Config{
		ForceNetwork: true,
		DPUAddr:      "localhost:8443",
		TmfifoPath:   "/dev/tmfifo_net0", // Would normally be checked first
	}

	transport, err := NewHostTransport(cfg)
	if err != nil {
		t.Fatalf("ForceNetwork should not error when DPUAddr provided: %v", err)
	}
	if transport.Type() != TransportNetwork {
		t.Errorf("Expected network transport, got %s", transport.Type())
	}
}

func TestNewHostTransport_ForceNetworkRequiresDPUAddr(t *testing.T) {
	// ForceNetwork without DPUAddr should error
	cfg := &Config{
		ForceNetwork: true,
	}

	_, err := NewHostTransport(cfg)
	if err == nil {
		t.Error("Expected error when ForceNetwork set but DPUAddr missing")
	}
}

func TestConfig_DefaultTmfifoPath(t *testing.T) {
	if DefaultTmfifoPath != "/dev/tmfifo_net0" {
		t.Errorf("Expected default tmfifo path /dev/tmfifo_net0, got %s", DefaultTmfifoPath)
	}
}

func TestDOCAComchAvailable_Stub(t *testing.T) {
	// On non-DOCA systems (like this test environment), should return false
	if DOCAComchAvailable() {
		t.Skip("DOCA Comch is available; skipping stub test")
	}
	// If we get here, the stub is working correctly
}
