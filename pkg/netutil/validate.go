// Package netutil provides network address validation utilities for SSRF protection.
package netutil

import (
	"errors"
	"net"
	"net/url"
	"strings"
)

var (
	// ErrPrivateAddress indicates the address resolves to a private IP range (10/8, 172.16/12, 192.168/16).
	ErrPrivateAddress = errors.New("address resolves to private IP range")

	// ErrLoopbackAddress indicates the address resolves to loopback (127/8 or ::1).
	ErrLoopbackAddress = errors.New("address resolves to loopback")

	// ErrLinkLocalAddress indicates the address resolves to link-local range (169.254/16 or fe80::/10).
	ErrLinkLocalAddress = errors.New("address resolves to link-local range")

	// ErrInvalidEndpoint indicates the endpoint could not be parsed.
	ErrInvalidEndpoint = errors.New("invalid endpoint")
)

// CIDR blocks for private, loopback, and link-local addresses
var (
	privateBlocks = []string{
		"10.0.0.0/8",
		"172.16.0.0/12",
		"192.168.0.0/16",
	}

	loopbackBlocks = []string{
		"127.0.0.0/8",
	}

	linkLocalBlocks = []string{
		"169.254.0.0/16",
		"fe80::/10",
	}

	// Pre-parsed CIDRs
	privateCIDRs   []*net.IPNet
	loopbackCIDRs  []*net.IPNet
	linkLocalCIDRs []*net.IPNet
)

func init() {
	// Parse all CIDR blocks at init time
	for _, block := range privateBlocks {
		_, cidr, _ := net.ParseCIDR(block)
		privateCIDRs = append(privateCIDRs, cidr)
	}
	for _, block := range loopbackBlocks {
		_, cidr, _ := net.ParseCIDR(block)
		loopbackCIDRs = append(loopbackCIDRs, cidr)
	}
	for _, block := range linkLocalBlocks {
		_, cidr, _ := net.ParseCIDR(block)
		linkLocalCIDRs = append(linkLocalCIDRs, cidr)
	}
}

// ValidateEndpoint validates that an endpoint doesn't resolve to loopback or link-local ranges.
// Private ranges (10/8, 172.16/12, 192.168/16) are ALLOWED because BMCs typically use private IPs.
// Use ValidateEndpointStrict if you need to block private ranges as well.
//
// Blocks: 127.0.0.0/8, ::1, 169.254.0.0/16, fe80::/10
// Returns nil if endpoint is safe, error otherwise.
func ValidateEndpoint(endpoint string) error {
	return validateEndpoint(endpoint, false)
}

// ValidateEndpointStrict validates that an endpoint doesn't resolve to blocked ranges.
// This is the strict version that also blocks private IP ranges.
//
// Blocks: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8, ::1, 169.254.0.0/16, fe80::/10
// Returns nil if endpoint is safe, error otherwise.
func ValidateEndpointStrict(endpoint string) error {
	return validateEndpoint(endpoint, true)
}

// validateEndpoint is the internal implementation.
// If strict is true, private ranges are also blocked.
func validateEndpoint(endpoint string, strict bool) error {
	host := extractHost(endpoint)
	if host == "" {
		return ErrInvalidEndpoint
	}

	// Try to parse as IP first (no DNS lookup needed)
	ip := net.ParseIP(host)
	if ip != nil {
		return checkIP(ip, strict)
	}

	// Handle "localhost" specially since it may not resolve via DNS on all systems
	if strings.EqualFold(host, "localhost") {
		return ErrLoopbackAddress
	}

	// Resolve hostname to IPs
	ips, err := net.LookupIP(host)
	if err != nil {
		// If we can't resolve, we can't validate. Return error to be safe.
		return errors.New("cannot resolve hostname: " + err.Error())
	}

	// Check ALL resolved IPs (DNS may return multiple)
	for _, resolvedIP := range ips {
		if err := checkIP(resolvedIP, strict); err != nil {
			return err
		}
	}

	return nil
}

// checkIP validates a single IP address against blocked ranges.
func checkIP(ip net.IP, strict bool) error {
	// Check for loopback first (highest priority)
	if ip.IsLoopback() {
		return ErrLoopbackAddress
	}
	for _, cidr := range loopbackCIDRs {
		if cidr.Contains(ip) {
			return ErrLoopbackAddress
		}
	}

	// Check for link-local
	if ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() {
		return ErrLinkLocalAddress
	}
	for _, cidr := range linkLocalCIDRs {
		if cidr.Contains(ip) {
			return ErrLinkLocalAddress
		}
	}

	// Check for private ranges only in strict mode
	if strict {
		if ip.IsPrivate() {
			return ErrPrivateAddress
		}
		for _, cidr := range privateCIDRs {
			if cidr.Contains(ip) {
				return ErrPrivateAddress
			}
		}
	}

	return nil
}

// extractHost extracts the host from an endpoint string.
// Handles: "host", "host:port", "http://host:port/path", "[::1]:port"
func extractHost(endpoint string) string {
	if endpoint == "" {
		return ""
	}

	// If it looks like a URL (has scheme), parse it
	if strings.Contains(endpoint, "://") {
		u, err := url.Parse(endpoint)
		if err != nil {
			return ""
		}
		host := u.Hostname()
		if host == "" {
			return ""
		}
		return host
	}

	// Handle IPv6 with brackets: [::1] or [::1]:port
	if strings.HasPrefix(endpoint, "[") {
		closeBracket := strings.Index(endpoint, "]")
		if closeBracket == -1 {
			return ""
		}
		return endpoint[1:closeBracket]
	}

	// Handle host:port or bare host
	// For IPv4, we can use net.SplitHostPort
	host, _, err := net.SplitHostPort(endpoint)
	if err != nil {
		// No port, return as-is
		return endpoint
	}
	return host
}
