package store

import (
	"os"
	"path/filepath"
	"testing"
)

// setupTestStore creates a temporary SQLite database for testing.
func setupTestStore(t *testing.T) *Store {
	t.Helper()
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := Open(dbPath)
	if err != nil {
		t.Fatalf("failed to open test store: %v", err)
	}

	t.Cleanup(func() {
		store.Close()
		os.Remove(dbPath)
	})

	return store
}

// TestTenantCRUD tests basic CRUD operations for tenants.
func TestTenantCRUD(t *testing.T) {
	store := setupTestStore(t)

	// Test AddTenant
	t.Run("AddTenant", func(t *testing.T) {
		err := store.AddTenant("t1", "Acme Corp", "Test tenant", "admin@acme.com", []string{"production", "us-east"})
		if err != nil {
			t.Fatalf("AddTenant failed: %v", err)
		}

		// Verify tenant was added
		tenant, err := store.GetTenant("t1")
		if err != nil {
			t.Fatalf("GetTenant failed: %v", err)
		}
		if tenant.Name != "Acme Corp" {
			t.Errorf("expected name 'Acme Corp', got '%s'", tenant.Name)
		}
		if tenant.Description != "Test tenant" {
			t.Errorf("expected description 'Test tenant', got '%s'", tenant.Description)
		}
		if tenant.Contact != "admin@acme.com" {
			t.Errorf("expected contact 'admin@acme.com', got '%s'", tenant.Contact)
		}
		if len(tenant.Tags) != 2 {
			t.Errorf("expected 2 tags, got %d", len(tenant.Tags))
		}
	})

	// Test AddTenant duplicate name
	t.Run("AddTenant_DuplicateName", func(t *testing.T) {
		err := store.AddTenant("t2", "Acme Corp", "Another", "", nil)
		if err == nil {
			t.Error("expected error for duplicate name, got nil")
		}
	})

	// Test GetTenant by name
	t.Run("GetTenant_ByName", func(t *testing.T) {
		tenant, err := store.GetTenant("Acme Corp")
		if err != nil {
			t.Fatalf("GetTenant by name failed: %v", err)
		}
		if tenant.ID != "t1" {
			t.Errorf("expected ID 't1', got '%s'", tenant.ID)
		}
	})

	// Test GetTenant not found
	t.Run("GetTenant_NotFound", func(t *testing.T) {
		_, err := store.GetTenant("nonexistent")
		if err == nil {
			t.Error("expected error for nonexistent tenant, got nil")
		}
	})

	// Test UpdateTenant
	t.Run("UpdateTenant", func(t *testing.T) {
		err := store.UpdateTenant("t1", "Acme Inc", "Updated description", "new@acme.com", []string{"staging"})
		if err != nil {
			t.Fatalf("UpdateTenant failed: %v", err)
		}

		tenant, _ := store.GetTenant("t1")
		if tenant.Name != "Acme Inc" {
			t.Errorf("expected name 'Acme Inc', got '%s'", tenant.Name)
		}
		if tenant.Description != "Updated description" {
			t.Errorf("expected description 'Updated description', got '%s'", tenant.Description)
		}
		if len(tenant.Tags) != 1 || tenant.Tags[0] != "staging" {
			t.Errorf("expected tags ['staging'], got %v", tenant.Tags)
		}
	})

	// Test ListTenants
	t.Run("ListTenants", func(t *testing.T) {
		// Add another tenant
		store.AddTenant("t3", "Beta Inc", "", "", nil)

		tenants, err := store.ListTenants()
		if err != nil {
			t.Fatalf("ListTenants failed: %v", err)
		}
		if len(tenants) != 2 {
			t.Errorf("expected 2 tenants, got %d", len(tenants))
		}
	})

	// Test RemoveTenant
	t.Run("RemoveTenant", func(t *testing.T) {
		err := store.RemoveTenant("t3")
		if err != nil {
			t.Fatalf("RemoveTenant failed: %v", err)
		}

		tenants, _ := store.ListTenants()
		if len(tenants) != 1 {
			t.Errorf("expected 1 tenant after removal, got %d", len(tenants))
		}
	})

	// Test RemoveTenant not found
	t.Run("RemoveTenant_NotFound", func(t *testing.T) {
		err := store.RemoveTenant("nonexistent")
		if err == nil {
			t.Error("expected error for removing nonexistent tenant, got nil")
		}
	})
}

// TestDPUTenantAssignment tests DPU-tenant assignment operations.
func TestDPUTenantAssignment(t *testing.T) {
	store := setupTestStore(t)

	// Setup: Create tenant and DPUs
	store.AddTenant("tenant1", "Test Tenant", "", "", nil)
	store.Add("dpu1", "bf3-lab-01", "192.168.1.204", 50051)
	store.Add("dpu2", "bf3-lab-02", "192.168.1.205", 50051)
	store.Add("dpu3", "bf3-prod-01", "192.168.1.206", 50051)

	// Test AssignDPUToTenant
	t.Run("AssignDPUToTenant", func(t *testing.T) {
		err := store.AssignDPUToTenant("dpu1", "tenant1")
		if err != nil {
			t.Fatalf("AssignDPUToTenant failed: %v", err)
		}

		dpu, _ := store.Get("dpu1")
		if dpu.TenantID == nil || *dpu.TenantID != "tenant1" {
			t.Errorf("expected tenant_id 'tenant1', got %v", dpu.TenantID)
		}
	})

	// Test AssignDPUToTenant by name
	t.Run("AssignDPUToTenant_ByName", func(t *testing.T) {
		err := store.AssignDPUToTenant("bf3-lab-02", "tenant1")
		if err != nil {
			t.Fatalf("AssignDPUToTenant by name failed: %v", err)
		}

		dpu, _ := store.Get("dpu2")
		if dpu.TenantID == nil || *dpu.TenantID != "tenant1" {
			t.Errorf("expected tenant_id 'tenant1', got %v", dpu.TenantID)
		}
	})

	// Test ListDPUsByTenant
	t.Run("ListDPUsByTenant", func(t *testing.T) {
		dpus, err := store.ListDPUsByTenant("tenant1")
		if err != nil {
			t.Fatalf("ListDPUsByTenant failed: %v", err)
		}
		if len(dpus) != 2 {
			t.Errorf("expected 2 DPUs for tenant, got %d", len(dpus))
		}
	})

	// Test GetTenantDPUCount
	t.Run("GetTenantDPUCount", func(t *testing.T) {
		count, err := store.GetTenantDPUCount("tenant1")
		if err != nil {
			t.Fatalf("GetTenantDPUCount failed: %v", err)
		}
		if count != 2 {
			t.Errorf("expected count 2, got %d", count)
		}
	})

	// Test UnassignDPUFromTenant
	t.Run("UnassignDPUFromTenant", func(t *testing.T) {
		err := store.UnassignDPUFromTenant("dpu1")
		if err != nil {
			t.Fatalf("UnassignDPUFromTenant failed: %v", err)
		}

		dpu, _ := store.Get("dpu1")
		if dpu.TenantID != nil {
			t.Errorf("expected nil tenant_id after unassign, got %v", dpu.TenantID)
		}

		count, _ := store.GetTenantDPUCount("tenant1")
		if count != 1 {
			t.Errorf("expected count 1 after unassign, got %d", count)
		}
	})

	// Test UnassignDPUFromTenant not found
	t.Run("UnassignDPUFromTenant_NotFound", func(t *testing.T) {
		err := store.UnassignDPUFromTenant("nonexistent")
		if err == nil {
			t.Error("expected error for unassigning nonexistent DPU, got nil")
		}
	})
}

// TestDPULabels tests the labels functionality on DPUs.
func TestDPULabels(t *testing.T) {
	store := setupTestStore(t)

	// Create a DPU
	store.Add("dpu1", "bf3-test", "192.168.1.204", 50051)

	t.Run("DPU_HasEmptyLabels", func(t *testing.T) {
		dpu, err := store.Get("dpu1")
		if err != nil {
			t.Fatalf("Get failed: %v", err)
		}
		if dpu.Labels == nil {
			t.Error("expected non-nil labels map, got nil")
		}
		if len(dpu.Labels) != 0 {
			t.Errorf("expected empty labels, got %v", dpu.Labels)
		}
	})
}

// TestTenantEmptyTags tests tenant with empty tags.
func TestTenantEmptyTags(t *testing.T) {
	store := setupTestStore(t)

	// Create tenant with nil tags
	err := store.AddTenant("t1", "No Tags Tenant", "", "", nil)
	if err != nil {
		t.Fatalf("AddTenant failed: %v", err)
	}

	tenant, err := store.GetTenant("t1")
	if err != nil {
		t.Fatalf("GetTenant failed: %v", err)
	}

	if tenant.Tags == nil {
		t.Error("expected non-nil tags slice, got nil")
	}
	if len(tenant.Tags) != 0 {
		t.Errorf("expected empty tags, got %v", tenant.Tags)
	}
}

// TestDPUPersistenceAcrossRestart tests that DPU state persists after store close and reopen.
// This simulates a Nexus server restart scenario.
func TestDPUPersistenceAcrossRestart(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "persistence_test.db")

	// Phase 1: Create store, add DPU and tenant, assign DPU to tenant
	t.Run("Phase1_Setup", func(t *testing.T) {
		s, err := Open(dbPath)
		if err != nil {
			t.Fatalf("failed to open store: %v", err)
		}

		// Add DPU
		err = s.Add("dpu1", "bf3-test", "192.168.1.100", 50051)
		if err != nil {
			t.Fatalf("failed to add DPU: %v", err)
		}

		// Update status
		err = s.UpdateStatus("dpu1", "healthy")
		if err != nil {
			t.Fatalf("failed to update status: %v", err)
		}

		// Add tenant
		err = s.AddTenant("t1", "Test Tenant", "Description", "contact@example.com", []string{"production"})
		if err != nil {
			t.Fatalf("failed to add tenant: %v", err)
		}

		// Assign DPU to tenant
		err = s.AssignDPUToTenant("dpu1", "t1")
		if err != nil {
			t.Fatalf("failed to assign DPU to tenant: %v", err)
		}

		// Verify state before close
		dpu, err := s.Get("dpu1")
		if err != nil {
			t.Fatalf("failed to get DPU: %v", err)
		}
		if dpu.Status != "healthy" {
			t.Errorf("expected status 'healthy', got '%s'", dpu.Status)
		}
		if dpu.TenantID == nil || *dpu.TenantID != "t1" {
			t.Errorf("expected tenant 't1', got %v", dpu.TenantID)
		}

		// Close store (simulate shutdown)
		s.Close()
	})

	// Phase 2: Reopen store and verify all state persisted
	t.Run("Phase2_VerifyPersistence", func(t *testing.T) {
		s, err := Open(dbPath)
		if err != nil {
			t.Fatalf("failed to reopen store: %v", err)
		}
		defer s.Close()

		// Verify DPU exists and has correct state
		dpu, err := s.Get("dpu1")
		if err != nil {
			t.Fatalf("DPU not found after restart: %v", err)
		}
		if dpu.Name != "bf3-test" {
			t.Errorf("expected name 'bf3-test', got '%s'", dpu.Name)
		}
		if dpu.Host != "192.168.1.100" {
			t.Errorf("expected host '192.168.1.100', got '%s'", dpu.Host)
		}
		if dpu.Port != 50051 {
			t.Errorf("expected port 50051, got %d", dpu.Port)
		}
		if dpu.Status != "healthy" {
			t.Errorf("expected status 'healthy', got '%s'", dpu.Status)
		}
		if dpu.TenantID == nil {
			t.Fatalf("DPU tenant assignment lost after restart")
		}
		if *dpu.TenantID != "t1" {
			t.Errorf("expected tenant 't1', got '%s'", *dpu.TenantID)
		}

		// Verify tenant exists
		tenant, err := s.GetTenant("t1")
		if err != nil {
			t.Fatalf("tenant not found after restart: %v", err)
		}
		if tenant.Name != "Test Tenant" {
			t.Errorf("expected tenant name 'Test Tenant', got '%s'", tenant.Name)
		}

		// Verify DPU count for tenant
		count, err := s.GetTenantDPUCount("t1")
		if err != nil {
			t.Fatalf("failed to get tenant DPU count: %v", err)
		}
		if count != 1 {
			t.Errorf("expected DPU count 1, got %d", count)
		}

		// Verify list operations work
		dpus, err := s.List()
		if err != nil {
			t.Fatalf("failed to list DPUs: %v", err)
		}
		if len(dpus) != 1 {
			t.Errorf("expected 1 DPU in list, got %d", len(dpus))
		}
	})
}

// TestAgentHostPersistenceAcrossRestart tests that agent host state persists after store close and reopen.
func TestAgentHostPersistenceAcrossRestart(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "agenthost_persistence_test.db")

	// Phase 1: Create store, add DPU and register agent host with posture
	t.Run("Phase1_Setup", func(t *testing.T) {
		s, err := Open(dbPath)
		if err != nil {
			t.Fatalf("failed to open store: %v", err)
		}

		// Add DPU first (agent hosts require DPU reference)
		err = s.Add("dpu1", "bf3-test", "192.168.1.100", 50051)
		if err != nil {
			t.Fatalf("failed to add DPU: %v", err)
		}

		// Add tenant and assign DPU
		err = s.AddTenant("t1", "Test Tenant", "", "", nil)
		if err != nil {
			t.Fatalf("failed to add tenant: %v", err)
		}
		s.AssignDPUToTenant("dpu1", "t1")

		// Register agent host
		host := &AgentHost{
			DPUName:  "bf3-test",
			DPUID:    "dpu1",
			Hostname: "gpu-node-01",
			TenantID: "t1",
		}
		err = s.RegisterAgentHost(host)
		if err != nil {
			t.Fatalf("failed to register agent host: %v", err)
		}

		// Update posture
		secureBoot := true
		tpmPresent := true
		posture := &AgentHostPosture{
			HostID:         host.ID,
			SecureBoot:     &secureBoot,
			DiskEncryption: "luks",
			OSVersion:      "Ubuntu 22.04",
			KernelVersion:  "5.15.0",
			TPMPresent:     &tpmPresent,
			PostureHash:    "abc123",
		}
		err = s.UpdateAgentHostPosture(posture)
		if err != nil {
			t.Fatalf("failed to update posture: %v", err)
		}

		s.Close()
	})

	// Phase 2: Reopen store and verify agent host state persisted
	t.Run("Phase2_VerifyPersistence", func(t *testing.T) {
		s, err := Open(dbPath)
		if err != nil {
			t.Fatalf("failed to reopen store: %v", err)
		}
		defer s.Close()

		// Verify agent host exists
		host, err := s.GetAgentHostByHostname("gpu-node-01")
		if err != nil {
			t.Fatalf("agent host not found after restart: %v", err)
		}
		if host.DPUName != "bf3-test" {
			t.Errorf("expected DPU name 'bf3-test', got '%s'", host.DPUName)
		}
		if host.TenantID != "t1" {
			t.Errorf("expected tenant 't1', got '%s'", host.TenantID)
		}

		// Verify posture persisted
		posture, err := s.GetAgentHostPosture(host.ID)
		if err != nil {
			t.Fatalf("posture not found after restart: %v", err)
		}
		if posture.SecureBoot == nil || !*posture.SecureBoot {
			t.Error("secure boot should be true")
		}
		if posture.DiskEncryption != "luks" {
			t.Errorf("expected disk encryption 'luks', got '%s'", posture.DiskEncryption)
		}
		if posture.OSVersion != "Ubuntu 22.04" {
			t.Errorf("expected OS version 'Ubuntu 22.04', got '%s'", posture.OSVersion)
		}

		// Verify list by tenant works
		hosts, err := s.ListAgentHosts("t1")
		if err != nil {
			t.Fatalf("failed to list agent hosts: %v", err)
		}
		if len(hosts) != 1 {
			t.Errorf("expected 1 agent host, got %d", len(hosts))
		}
	})
}
