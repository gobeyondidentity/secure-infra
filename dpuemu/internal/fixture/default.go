// Package fixture handles loading and templating of DPU fixture data.
package fixture

import (
	agentv1 "github.com/nmelo/secure-infra/gen/go/agent/v1"
)

// DefaultFixture returns a minimal fixture with reasonable defaults for local development.
// This is used when no --fixture flag is provided, enabling quick startup without
// needing to specify a fixture file.
func DefaultFixture() *Fixture {
	return &Fixture{
		SystemInfo: &agentv1.GetSystemInfoResponse{
			Hostname:        "dpuemu-local",
			Model:           "Emulated BlueField-3",
			SerialNumber:    "EMU-00000001",
			FirmwareVersion: "emulated-1.0.0",
			DocaVersion:     "emulated",
			ArmCores:        16,
			MemoryGb:        32,
			UptimeSeconds:   0,
			OvsVersion:      "2.17.0",
			KernelVersion:   "emulated",
		},
		Bridges: []*agentv1.Bridge{},
		Flows:   make(map[string][]*agentv1.Flow),
		Attestation: &AttestationData{
			Status:       "ATTESTATION_STATUS_UNAVAILABLE",
			Certificates: []*CertData{},
			Measurements: make(map[string]string),
		},
		Health: &HealthData{
			Healthy:       true,
			Version:       "dpuemu-0.2.0",
			UptimeSeconds: 0,
			Components: map[string]*ComponentHealthData{
				"emulator": {
					Healthy: true,
					Message: "Emulator running with default fixture",
				},
			},
		},
		Metadata: map[string]string{
			"source":      "default",
			"description": "Auto-generated default fixture for local development",
		},
	}
}
