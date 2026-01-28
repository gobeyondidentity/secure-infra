package cmd

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNexusClient_AddDPU(t *testing.T) {
	tests := []struct {
		name       string
		dpuName    string
		host       string
		port       int
		serverResp dpuResponse
		serverCode int
		wantErr    bool
	}{
		{
			name:    "successful add",
			dpuName: "bf3-lab",
			host:    "192.168.1.204",
			port:    18051,
			serverResp: dpuResponse{
				ID:     "abc12345",
				Name:   "bf3-lab",
				Host:   "192.168.1.204",
				Port:   18051,
				Status: "healthy",
			},
			serverCode: http.StatusCreated,
			wantErr:    false,
		},
		{
			name:       "conflict - DPU exists",
			dpuName:    "bf3-lab",
			host:       "192.168.1.204",
			port:       18051,
			serverCode: http.StatusConflict,
			wantErr:    true,
		},
		{
			name:       "server error",
			dpuName:    "bf3-lab",
			host:       "192.168.1.204",
			port:       18051,
			serverCode: http.StatusInternalServerError,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost {
					t.Errorf("expected POST, got %s", r.Method)
				}
				if r.URL.Path != "/api/dpus" {
					t.Errorf("expected /api/dpus, got %s", r.URL.Path)
				}

				// Verify request body
				var req addDPURequest
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Errorf("failed to decode request body: %v", err)
				}
				if req.Name != tt.dpuName {
					t.Errorf("expected name %s, got %s", tt.dpuName, req.Name)
				}
				if req.Host != tt.host {
					t.Errorf("expected host %s, got %s", tt.host, req.Host)
				}
				if req.Port != tt.port {
					t.Errorf("expected port %d, got %d", tt.port, req.Port)
				}

				w.WriteHeader(tt.serverCode)
				if tt.serverCode == http.StatusCreated {
					json.NewEncoder(w).Encode(tt.serverResp)
				}
			}))
			defer server.Close()

			client := NewNexusClient(server.URL)
			resp, err := client.AddDPU(context.Background(), tt.dpuName, tt.host, tt.port)

			if (err != nil) != tt.wantErr {
				t.Errorf("AddDPU() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if resp.ID != tt.serverResp.ID {
					t.Errorf("expected ID %s, got %s", tt.serverResp.ID, resp.ID)
				}
				if resp.Name != tt.serverResp.Name {
					t.Errorf("expected Name %s, got %s", tt.serverResp.Name, resp.Name)
				}
			}
		})
	}
}

func TestNexusClient_ListDPUs(t *testing.T) {
	tests := []struct {
		name       string
		serverResp []dpuResponse
		serverCode int
		wantErr    bool
		wantCount  int
	}{
		{
			name: "successful list with DPUs",
			serverResp: []dpuResponse{
				{ID: "abc12345", Name: "bf3-lab", Host: "192.168.1.204", Port: 18051, Status: "healthy"},
				{ID: "def67890", Name: "bf3-dev", Host: "192.168.1.205", Port: 18051, Status: "offline"},
			},
			serverCode: http.StatusOK,
			wantErr:    false,
			wantCount:  2,
		},
		{
			name:       "empty list",
			serverResp: []dpuResponse{},
			serverCode: http.StatusOK,
			wantErr:    false,
			wantCount:  0,
		},
		{
			name:       "server error",
			serverCode: http.StatusInternalServerError,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet {
					t.Errorf("expected GET, got %s", r.Method)
				}
				if r.URL.Path != "/api/dpus" {
					t.Errorf("expected /api/dpus, got %s", r.URL.Path)
				}

				w.WriteHeader(tt.serverCode)
				if tt.serverCode == http.StatusOK {
					json.NewEncoder(w).Encode(tt.serverResp)
				}
			}))
			defer server.Close()

			client := NewNexusClient(server.URL)
			resp, err := client.ListDPUs(context.Background())

			if (err != nil) != tt.wantErr {
				t.Errorf("ListDPUs() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if len(resp) != tt.wantCount {
					t.Errorf("expected %d DPUs, got %d", tt.wantCount, len(resp))
				}
			}
		})
	}
}

func TestNexusClient_RemoveDPU(t *testing.T) {
	tests := []struct {
		name       string
		dpuID      string
		serverCode int
		wantErr    bool
	}{
		{
			name:       "successful remove",
			dpuID:      "abc12345",
			serverCode: http.StatusNoContent,
			wantErr:    false,
		},
		{
			name:       "not found",
			dpuID:      "nonexistent",
			serverCode: http.StatusNotFound,
			wantErr:    true,
		},
		{
			name:       "server error",
			dpuID:      "abc12345",
			serverCode: http.StatusInternalServerError,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodDelete {
					t.Errorf("expected DELETE, got %s", r.Method)
				}
				expectedPath := "/api/dpus/" + tt.dpuID
				if r.URL.Path != expectedPath {
					t.Errorf("expected %s, got %s", expectedPath, r.URL.Path)
				}

				w.WriteHeader(tt.serverCode)
			}))
			defer server.Close()

			client := NewNexusClient(server.URL)
			err := client.RemoveDPU(context.Background(), tt.dpuID)

			if (err != nil) != tt.wantErr {
				t.Errorf("RemoveDPU() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestNewNexusClient(t *testing.T) {
	// Test that trailing slash is removed
	client := NewNexusClient("http://localhost:8080/")
	if client.baseURL != "http://localhost:8080" {
		t.Errorf("expected trailing slash removed, got %s", client.baseURL)
	}

	// Test without trailing slash
	client = NewNexusClient("http://localhost:8080")
	if client.baseURL != "http://localhost:8080" {
		t.Errorf("expected base URL unchanged, got %s", client.baseURL)
	}
}
