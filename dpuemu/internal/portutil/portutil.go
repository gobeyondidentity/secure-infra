// Package portutil provides utilities for parsing port specifications.
package portutil

import (
	"fmt"
	"strconv"
	"strings"
)

// DefaultPort is the default port for the dpuemu server.
const DefaultPort = 18051

// ParseListenAddr parses listen address specifications and returns a normalized address.
// Accepts formats:
//   - Port number only: "50052" -> ":50052"
//   - Address format: ":50052" -> ":50052"
//   - Full address: "localhost:50052" -> "localhost:50052"
//   - Full address with host: "0.0.0.0:50052" -> "0.0.0.0:50052"
//
// Returns an error if the port is invalid (not a number or out of range).
func ParseListenAddr(input string) (string, error) {
	if input == "" {
		return fmt.Sprintf(":%d", DefaultPort), nil
	}

	// If it looks like just a port number
	if !strings.Contains(input, ":") {
		port, err := strconv.Atoi(input)
		if err != nil {
			return "", fmt.Errorf("invalid port number: %s", input)
		}
		if port < 1 || port > 65535 {
			return "", fmt.Errorf("port out of range (1-65535): %d", port)
		}
		return fmt.Sprintf(":%d", port), nil
	}

	// It contains a colon, parse as host:port
	parts := strings.Split(input, ":")
	if len(parts) != 2 {
		return "", fmt.Errorf("invalid address format: %s", input)
	}

	portStr := parts[1]
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return "", fmt.Errorf("invalid port number: %s", portStr)
	}
	if port < 1 || port > 65535 {
		return "", fmt.Errorf("port out of range (1-65535): %d", port)
	}

	return input, nil
}

// ResolvePort determines the effective listen address from --port and --listen flags.
// Rules:
//   - If portFlag is set (non-zero), use it (takes precedence)
//   - If listenFlag is set (non-empty), parse and use it
//   - Otherwise, use default port 18051
func ResolvePort(portFlag int, listenFlag string) (string, error) {
	// --port takes precedence if set
	if portFlag != 0 {
		if portFlag < 1 || portFlag > 65535 {
			return "", fmt.Errorf("port out of range (1-65535): %d", portFlag)
		}
		return fmt.Sprintf(":%d", portFlag), nil
	}

	// --listen is second choice
	if listenFlag != "" {
		return ParseListenAddr(listenFlag)
	}

	// Default
	return fmt.Sprintf(":%d", DefaultPort), nil
}
