package transport

import (
	"crypto/tls"
	"errors"
	"fmt"
	"os"
)

// DefaultTmfifoPath is the standard location for the tmfifo_net device on hosts.
const DefaultTmfifoPath = "/dev/tmfifo_net0"

// Config contains options for transport selection and initialization.
type Config struct {
	// MockTransport, if non-nil, is returned directly by NewHostTransport.
	// Used for test injection to bypass hardware detection.
	MockTransport Transport

	// TmfifoPath overrides the default tmfifo device path.
	// If empty, DefaultTmfifoPath is used for device detection.
	TmfifoPath string

	// InviteCode is the one-time code for network transport authentication.
	// Required when using NetworkTransport as a fallback.
	InviteCode string

	// DPUAddr is the network address of the DPU agent (host:port).
	// Required for NetworkTransport.
	DPUAddr string

	// TLSConfig provides TLS settings for NetworkTransport.
	// If nil, a default configuration with system roots is used.
	TLSConfig *tls.Config

	// Hostname is the host's name, used for identification in network transport.
	Hostname string

	// ForceTmfifo, if true, requires tmfifo transport and fails if unavailable.
	// When set, network fallback is disabled.
	ForceTmfifo bool

	// ForceNetwork, if true, skips hardware transport detection and uses network.
	// Requires InviteCode and DPUAddr to be set.
	ForceNetwork bool
}

// NewHostTransport creates a transport for the Host Agent to communicate with the DPU.
// Transport selection follows this priority (unless force flags are set):
//  1. MockTransport from config (test injection)
//  2. DOCA Comch if available (BlueField production)
//  3. TmfifoNet if device exists (BlueField legacy/emulator)
//  4. Network if invite code provided (non-BlueField fallback)
//
// If ForceTmfifo is set, only tmfifo is tried and an error is returned if unavailable.
// If ForceNetwork is set, hardware detection is skipped and network transport is used.
//
// Returns an error if no suitable transport is available.
func NewHostTransport(cfg *Config) (Transport, error) {
	if cfg == nil {
		cfg = &Config{}
	}

	// Priority 1: Mock transport for testing
	if cfg.MockTransport != nil {
		return cfg.MockTransport, nil
	}

	// Handle ForceNetwork: skip hardware detection
	if cfg.ForceNetwork {
		if cfg.DPUAddr == "" {
			return nil, errors.New("ForceNetwork requires DPUAddr to be set")
		}
		return NewNetworkTransport(cfg.DPUAddr, cfg.InviteCode, cfg.TLSConfig, cfg.Hostname)
	}

	// Determine tmfifo path
	tmfifoPath := cfg.TmfifoPath
	if tmfifoPath == "" {
		tmfifoPath = DefaultTmfifoPath
	}

	// Handle ForceTmfifo: only try tmfifo, fail if unavailable
	if cfg.ForceTmfifo {
		if _, err := os.Stat(tmfifoPath); err != nil {
			return nil, fmt.Errorf("tmfifo not available at %s (required by ForceTmfifo): %w", tmfifoPath, err)
		}
		return NewTmfifoNetTransport(tmfifoPath)
	}

	// Priority 2: DOCA Comch (preferred on BlueField)
	if DOCAComchAvailable() {
		return NewDOCAComchTransport()
	}

	// Priority 3: Tmfifo device (legacy BlueField or emulator)
	if _, err := os.Stat(tmfifoPath); err == nil {
		return NewTmfifoNetTransport(tmfifoPath)
	}

	// Priority 4: Network transport (non-BlueField fallback)
	// DPUAddr is required; InviteCode is optional (legacy HTTP mode doesn't use it)
	if cfg.DPUAddr != "" {
		return NewNetworkTransport(cfg.DPUAddr, cfg.InviteCode, cfg.TLSConfig, cfg.Hostname)
	}

	return nil, errors.New("no transport available: DOCA Comch not present, tmfifo device not found, and no DPU address provided")
}

// DOCAComchAvailable checks if the DOCA Comch transport can be used.
// Returns true only on systems with BlueField hardware and DOCA SDK.
// This is a placeholder; the actual implementation is in doca_comch.go.
func DOCAComchAvailable() bool {
	return docaComchAvailable()
}
