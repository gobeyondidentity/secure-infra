package transport

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"sync"
)

// networkTransport implements Transport using HTTP/mTLS for non-BlueField deployments.
// This transport is used when hardware transports (DOCA Comch, tmfifo) are unavailable.
type networkTransport struct {
	dpuAddr    string
	inviteCode string
	tlsConfig  *tls.Config
	hostname   string

	connected bool
	closed    bool
	mu        sync.Mutex

	// pendingRecv holds messages received from the server that haven't been read yet.
	// In a full implementation, this would be a bidirectional channel (e.g., WebSocket).
	// For now, the network transport is primarily request/response based.
	pendingRecv chan *Message
}

// NewNetworkTransport creates a transport using mTLS over TCP.
// The hostname is used for identification during enrollment.
func NewNetworkTransport(addr, inviteCode string, tlsConfig *tls.Config, hostname string) (Transport, error) {
	return &networkTransport{
		dpuAddr:     addr,
		inviteCode:  inviteCode,
		tlsConfig:   tlsConfig,
		hostname:    hostname,
		pendingRecv: make(chan *Message, 10),
	}, nil
}

// Connect establishes the network connection.
func (n *networkTransport) Connect(ctx context.Context) error {
	n.mu.Lock()
	defer n.mu.Unlock()

	if n.closed {
		return fmt.Errorf("transport closed")
	}

	// For HTTP-based transport, we don't maintain a persistent connection.
	// Each Send/Recv is a separate HTTP request.
	n.connected = true
	return nil
}

// Send transmits a message over the network.
// Note: The current network mode uses specific REST endpoints, not a generic message protocol.
// This implementation is a placeholder for future bidirectional protocol support.
func (n *networkTransport) Send(msg *Message) error {
	n.mu.Lock()
	if !n.connected {
		n.mu.Unlock()
		return fmt.Errorf("transport not connected")
	}
	n.mu.Unlock()

	// The network transport currently doesn't support arbitrary message sending
	// in the same way as tmfifo. The Host Agent uses specific HTTP endpoints
	// for register and posture when using network transport.
	// This will be fully implemented when the transport protocol is unified.
	return fmt.Errorf("NetworkTransport.Send not implemented for message type %s", msg.Type)
}

// Recv receives a message from the network.
// HTTP is request/response based, so Recv blocks waiting for queued responses.
func (n *networkTransport) Recv() (*Message, error) {
	n.mu.Lock()
	if !n.connected {
		n.mu.Unlock()
		return nil, fmt.Errorf("transport not connected")
	}
	n.mu.Unlock()

	msg, ok := <-n.pendingRecv
	if !ok {
		return nil, io.EOF
	}
	return msg, nil
}

// Close closes the network transport.
func (n *networkTransport) Close() error {
	n.mu.Lock()
	defer n.mu.Unlock()

	if n.closed {
		return nil
	}

	n.closed = true
	n.connected = false
	close(n.pendingRecv)
	return nil
}

// Type returns TransportNetwork.
func (n *networkTransport) Type() TransportType {
	return TransportNetwork
}

// DPUAddr returns the DPU Agent address.
func (n *networkTransport) DPUAddr() string {
	return n.dpuAddr
}

// Hostname returns the hostname.
func (n *networkTransport) Hostname() string {
	return n.hostname
}

// QueueResponse adds a response message to the pending receive queue.
// This bridges HTTP responses to the Transport interface.
func (n *networkTransport) QueueResponse(msg *Message) error {
	select {
	case n.pendingRecv <- msg:
		return nil
	default:
		return fmt.Errorf("response queue full")
	}
}
