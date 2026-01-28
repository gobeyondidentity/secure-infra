package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// NexusClient provides HTTP client access to the Nexus API.
type NexusClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewNexusClient creates a new client for the Nexus API.
func NewNexusClient(baseURL string) *NexusClient {
	return &NexusClient{
		baseURL: strings.TrimSuffix(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// addDPURequest is the request body for adding a DPU.
type addDPURequest struct {
	Name string `json:"name"`
	Host string `json:"host"`
	Port int    `json:"port"`
}

// dpuResponse matches the API response for DPU operations.
type dpuResponse struct {
	ID       string            `json:"id"`
	Name     string            `json:"name"`
	Host     string            `json:"host"`
	Port     int               `json:"port"`
	Status   string            `json:"status"`
	LastSeen *string           `json:"lastSeen,omitempty"`
	TenantID *string           `json:"tenantId,omitempty"`
	Labels   map[string]string `json:"labels,omitempty"`
}

// AddDPU registers a new DPU with the Nexus server.
func (c *NexusClient) AddDPU(ctx context.Context, name, host string, port int) (*dpuResponse, error) {
	reqBody := addDPURequest{
		Name: name,
		Host: host,
		Port: port,
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/dpus", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(body))
	}

	var dpu dpuResponse
	if err := json.NewDecoder(resp.Body).Decode(&dpu); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &dpu, nil
}

// ListDPUs retrieves all registered DPUs from the Nexus server.
func (c *NexusClient) ListDPUs(ctx context.Context) ([]dpuResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/dpus", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(body))
	}

	var dpus []dpuResponse
	if err := json.NewDecoder(resp.Body).Decode(&dpus); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return dpus, nil
}

// RemoveDPU deletes a DPU from the Nexus server.
func (c *NexusClient) RemoveDPU(ctx context.Context, id string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, c.baseURL+"/api/dpus/"+id, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNoContent {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("server returned %d: %s", resp.StatusCode, string(body))
	}

	return nil
}
