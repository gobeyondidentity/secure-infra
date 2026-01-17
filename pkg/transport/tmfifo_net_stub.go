package transport

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
)

// tmfifoNetTransport implements Transport using the tmfifo_net0 device.
// This is the Host Agent side transport for communicating with the DPU Agent
// via the BlueField tmfifo character device.
type tmfifoNetTransport struct {
	devicePath string
	device     *os.File
	reader     *bufio.Reader
	mu         sync.Mutex
	connected  bool
	closed     bool
}

// NewTmfifoNetTransport creates a transport using the tmfifo_net0 device.
func NewTmfifoNetTransport(devicePath string) (Transport, error) {
	if devicePath == "" {
		devicePath = DefaultTmfifoPath
	}
	return &tmfifoNetTransport{
		devicePath: devicePath,
	}, nil
}

// Connect opens the tmfifo device.
func (t *tmfifoNetTransport) Connect(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed {
		return fmt.Errorf("transport closed")
	}
	if t.connected {
		return nil
	}

	f, err := os.OpenFile(t.devicePath, os.O_RDWR, 0600)
	if err != nil {
		return fmt.Errorf("open tmfifo device %s: %w", t.devicePath, err)
	}

	t.device = f
	t.reader = bufio.NewReader(f)
	t.connected = true
	return nil
}

// Send transmits a message over tmfifo.
func (t *tmfifoNetTransport) Send(msg *Message) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.connected || t.device == nil {
		return fmt.Errorf("transport not connected")
	}

	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshal message: %w", err)
	}

	// Append newline delimiter (tmfifo protocol uses newline-delimited JSON)
	data = append(data, '\n')

	if _, err := t.device.Write(data); err != nil {
		return fmt.Errorf("write to tmfifo: %w", err)
	}

	return nil
}

// Recv receives a message from tmfifo.
func (t *tmfifoNetTransport) Recv() (*Message, error) {
	t.mu.Lock()
	if !t.connected || t.reader == nil {
		t.mu.Unlock()
		return nil, fmt.Errorf("transport not connected")
	}
	reader := t.reader
	t.mu.Unlock()

	line, err := reader.ReadBytes('\n')
	if err != nil {
		if err == io.EOF {
			return nil, io.EOF
		}
		return nil, fmt.Errorf("read from tmfifo: %w", err)
	}

	var msg Message
	if err := json.Unmarshal(line, &msg); err != nil {
		return nil, fmt.Errorf("parse message: %w", err)
	}

	return &msg, nil
}

// Close closes the tmfifo device.
func (t *tmfifoNetTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed {
		return nil
	}

	t.closed = true
	t.connected = false

	if t.device != nil {
		err := t.device.Close()
		t.device = nil
		t.reader = nil
		return err
	}

	return nil
}

// Type returns TransportTmfifoNet.
func (t *tmfifoNetTransport) Type() TransportType {
	return TransportTmfifoNet
}
