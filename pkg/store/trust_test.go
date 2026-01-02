package store

import (
	"strings"
	"testing"
)

// TestTrustRelationshipCRUD tests basic CRUD operations for trust relationships.
func TestTrustRelationshipCRUD(t *testing.T) {
	store := setupTestStore(t)
	setupTrustTestData(t, store)

	// Test CreateTrustRelationship
	t.Run("CreateTrustRelationship", func(t *testing.T) {
		tr := &TrustRelationship{
			SourceDPUID:   "dpu1",
			SourceDPUName: "bf3-01",
			TargetDPUID:   "dpu2",
			TargetDPUName: "bf3-02",
			TenantID:      "tenant1",
			TrustType:     TrustTypeSSHHost,
			Bidirectional: false,
		}
		err := store.CreateTrustRelationship(tr)
		if err != nil {
			t.Fatalf("CreateTrustRelationship failed: %v", err)
		}

		// Verify ID was generated
		if tr.ID == "" {
			t.Error("expected ID to be generated")
		}
		if !strings.HasPrefix(tr.ID, "tr_") {
			t.Errorf("expected ID to start with 'tr_', got '%s'", tr.ID)
		}
		if len(tr.ID) != 11 { // "tr_" + 8 chars
			t.Errorf("expected ID length 11, got %d", len(tr.ID))
		}
	})

	// Test GetTrustRelationship
	t.Run("GetTrustRelationship", func(t *testing.T) {
		tr := &TrustRelationship{
			ID:            "tr_testget1",
			SourceDPUID:   "dpu1",
			SourceDPUName: "bf3-01",
			TargetDPUID:   "dpu2",
			TargetDPUName: "bf3-02",
			TenantID:      "tenant1",
			TrustType:     TrustTypeMTLS,
			Bidirectional: true,
		}
		err := store.CreateTrustRelationship(tr)
		if err != nil {
			t.Fatalf("CreateTrustRelationship failed: %v", err)
		}

		retrieved, err := store.GetTrustRelationship("tr_testget1")
		if err != nil {
			t.Fatalf("GetTrustRelationship failed: %v", err)
		}

		if retrieved.ID != "tr_testget1" {
			t.Errorf("expected ID 'tr_testget1', got '%s'", retrieved.ID)
		}
		if retrieved.SourceDPUID != "dpu1" {
			t.Errorf("expected SourceDPUID 'dpu1', got '%s'", retrieved.SourceDPUID)
		}
		if retrieved.SourceDPUName != "bf3-01" {
			t.Errorf("expected SourceDPUName 'bf3-01', got '%s'", retrieved.SourceDPUName)
		}
		if retrieved.TargetDPUID != "dpu2" {
			t.Errorf("expected TargetDPUID 'dpu2', got '%s'", retrieved.TargetDPUID)
		}
		if retrieved.TargetDPUName != "bf3-02" {
			t.Errorf("expected TargetDPUName 'bf3-02', got '%s'", retrieved.TargetDPUName)
		}
		if retrieved.TenantID != "tenant1" {
			t.Errorf("expected TenantID 'tenant1', got '%s'", retrieved.TenantID)
		}
		if retrieved.TrustType != TrustTypeMTLS {
			t.Errorf("expected TrustType 'mtls', got '%s'", retrieved.TrustType)
		}
		if !retrieved.Bidirectional {
			t.Error("expected Bidirectional to be true")
		}
		if retrieved.Status != TrustStatusActive {
			t.Errorf("expected Status 'active', got '%s'", retrieved.Status)
		}
		if retrieved.SuspendReason != nil {
			t.Errorf("expected nil SuspendReason, got '%s'", *retrieved.SuspendReason)
		}
		if retrieved.CreatedAt.IsZero() {
			t.Error("expected non-zero CreatedAt")
		}
		if retrieved.UpdatedAt.IsZero() {
			t.Error("expected non-zero UpdatedAt")
		}
	})

	// Test GetTrustRelationship not found
	t.Run("GetTrustRelationship_NotFound", func(t *testing.T) {
		_, err := store.GetTrustRelationship("nonexistent")
		if err == nil {
			t.Error("expected error for nonexistent trust relationship, got nil")
		}
	})

	// Test DeleteTrustRelationship
	t.Run("DeleteTrustRelationship", func(t *testing.T) {
		tr := &TrustRelationship{
			ID:            "tr_delete01",
			SourceDPUID:   "dpu1",
			SourceDPUName: "bf3-01",
			TargetDPUID:   "dpu3",
			TargetDPUName: "bf3-03",
			TenantID:      "tenant1",
			TrustType:     TrustTypeSSHHost,
		}
		store.CreateTrustRelationship(tr)

		err := store.DeleteTrustRelationship("tr_delete01")
		if err != nil {
			t.Fatalf("DeleteTrustRelationship failed: %v", err)
		}

		_, err = store.GetTrustRelationship("tr_delete01")
		if err == nil {
			t.Error("expected error for deleted trust relationship, got nil")
		}
	})

	// Test DeleteTrustRelationship not found
	t.Run("DeleteTrustRelationship_NotFound", func(t *testing.T) {
		err := store.DeleteTrustRelationship("nonexistent")
		if err == nil {
			t.Error("expected error for deleting nonexistent trust relationship, got nil")
		}
	})
}

// TestListTrustRelationshipsByTenant tests listing trust relationships by tenant.
func TestListTrustRelationshipsByTenant(t *testing.T) {
	store := setupTestStore(t)
	setupTrustTestData(t, store)

	// Create trust relationships for tenant1
	store.CreateTrustRelationship(&TrustRelationship{
		ID:            "tr_tenant1a",
		SourceDPUID:   "dpu1",
		SourceDPUName: "bf3-01",
		TargetDPUID:   "dpu2",
		TargetDPUName: "bf3-02",
		TenantID:      "tenant1",
		TrustType:     TrustTypeSSHHost,
	})
	store.CreateTrustRelationship(&TrustRelationship{
		ID:            "tr_tenant1b",
		SourceDPUID:   "dpu2",
		SourceDPUName: "bf3-02",
		TargetDPUID:   "dpu3",
		TargetDPUName: "bf3-03",
		TenantID:      "tenant1",
		TrustType:     TrustTypeMTLS,
	})

	// Create trust relationship for tenant2
	store.AddTenant("tenant2", "Tenant Two", "", "", nil)
	store.CreateTrustRelationship(&TrustRelationship{
		ID:            "tr_tenant2a",
		SourceDPUID:   "dpu4",
		SourceDPUName: "bf3-04",
		TargetDPUID:   "dpu5",
		TargetDPUName: "bf3-05",
		TenantID:      "tenant2",
		TrustType:     TrustTypeSSHHost,
	})

	// List for tenant1
	relationships, err := store.ListTrustRelationships("tenant1")
	if err != nil {
		t.Fatalf("ListTrustRelationships failed: %v", err)
	}

	if len(relationships) != 2 {
		t.Errorf("expected 2 trust relationships for tenant1, got %d", len(relationships))
	}

	for _, tr := range relationships {
		if tr.TenantID != "tenant1" {
			t.Errorf("expected TenantID 'tenant1', got '%s'", tr.TenantID)
		}
	}

	// List for tenant2
	relationships, err = store.ListTrustRelationships("tenant2")
	if err != nil {
		t.Fatalf("ListTrustRelationships failed: %v", err)
	}

	if len(relationships) != 1 {
		t.Errorf("expected 1 trust relationship for tenant2, got %d", len(relationships))
	}
}

// TestListAllTrustRelationships tests listing all trust relationships across all tenants.
func TestListAllTrustRelationships(t *testing.T) {
	store := setupTestStore(t)
	setupTrustTestData(t, store)

	// Create trust relationships for tenant1
	store.CreateTrustRelationship(&TrustRelationship{
		ID:            "tr_all_t1a",
		SourceDPUID:   "dpu1",
		SourceDPUName: "bf3-01",
		TargetDPUID:   "dpu2",
		TargetDPUName: "bf3-02",
		TenantID:      "tenant1",
		TrustType:     TrustTypeSSHHost,
	})
	store.CreateTrustRelationship(&TrustRelationship{
		ID:            "tr_all_t1b",
		SourceDPUID:   "dpu2",
		SourceDPUName: "bf3-02",
		TargetDPUID:   "dpu3",
		TargetDPUName: "bf3-03",
		TenantID:      "tenant1",
		TrustType:     TrustTypeMTLS,
	})

	// Create trust relationship for tenant2
	store.AddTenant("tenant2", "Tenant Two", "", "", nil)
	store.CreateTrustRelationship(&TrustRelationship{
		ID:            "tr_all_t2a",
		SourceDPUID:   "dpu4",
		SourceDPUName: "bf3-04",
		TargetDPUID:   "dpu5",
		TargetDPUName: "bf3-05",
		TenantID:      "tenant2",
		TrustType:     TrustTypeSSHHost,
	})

	// List all trust relationships
	relationships, err := store.ListAllTrustRelationships()
	if err != nil {
		t.Fatalf("ListAllTrustRelationships failed: %v", err)
	}

	if len(relationships) != 3 {
		t.Errorf("expected 3 trust relationships total, got %d", len(relationships))
	}

	// Verify we have relationships from both tenants
	tenantsSeen := make(map[string]bool)
	for _, tr := range relationships {
		tenantsSeen[tr.TenantID] = true
	}
	if !tenantsSeen["tenant1"] || !tenantsSeen["tenant2"] {
		t.Error("expected trust relationships from both tenant1 and tenant2")
	}
}

// TestListTrustRelationshipsByDPU tests listing trust relationships involving a specific DPU.
func TestListTrustRelationshipsByDPU(t *testing.T) {
	store := setupTestStore(t)
	setupTrustTestData(t, store)

	// Create trust relationships where dpu1 is source
	store.CreateTrustRelationship(&TrustRelationship{
		ID:            "tr_dpu1src",
		SourceDPUID:   "dpu1",
		SourceDPUName: "bf3-01",
		TargetDPUID:   "dpu2",
		TargetDPUName: "bf3-02",
		TenantID:      "tenant1",
		TrustType:     TrustTypeSSHHost,
	})

	// Create trust relationships where dpu1 is target
	store.CreateTrustRelationship(&TrustRelationship{
		ID:            "tr_dpu1tgt",
		SourceDPUID:   "dpu3",
		SourceDPUName: "bf3-03",
		TargetDPUID:   "dpu1",
		TargetDPUName: "bf3-01",
		TenantID:      "tenant1",
		TrustType:     TrustTypeMTLS,
	})

	// Create trust relationship not involving dpu1
	store.CreateTrustRelationship(&TrustRelationship{
		ID:            "tr_nodpu1",
		SourceDPUID:   "dpu2",
		SourceDPUName: "bf3-02",
		TargetDPUID:   "dpu3",
		TargetDPUName: "bf3-03",
		TenantID:      "tenant1",
		TrustType:     TrustTypeSSHHost,
	})

	// List for dpu1
	relationships, err := store.ListTrustRelationshipsByDPU("dpu1")
	if err != nil {
		t.Fatalf("ListTrustRelationshipsByDPU failed: %v", err)
	}

	if len(relationships) != 2 {
		t.Errorf("expected 2 trust relationships for dpu1, got %d", len(relationships))
	}

	for _, tr := range relationships {
		if tr.SourceDPUID != "dpu1" && tr.TargetDPUID != "dpu1" {
			t.Errorf("expected dpu1 to be source or target, got source=%s target=%s", tr.SourceDPUID, tr.TargetDPUID)
		}
	}

	// List for dpu2
	relationships, err = store.ListTrustRelationshipsByDPU("dpu2")
	if err != nil {
		t.Fatalf("ListTrustRelationshipsByDPU failed: %v", err)
	}

	if len(relationships) != 2 {
		t.Errorf("expected 2 trust relationships for dpu2, got %d", len(relationships))
	}

	// List for dpu with no relationships
	relationships, err = store.ListTrustRelationshipsByDPU("dpu-none")
	if err != nil {
		t.Fatalf("ListTrustRelationshipsByDPU failed: %v", err)
	}

	if len(relationships) != 0 {
		t.Errorf("expected 0 trust relationships for dpu-none, got %d", len(relationships))
	}
}

// TestUpdateTrustStatus tests updating the status of a trust relationship.
func TestUpdateTrustStatus(t *testing.T) {
	store := setupTestStore(t)
	setupTrustTestData(t, store)

	// Create a trust relationship
	tr := &TrustRelationship{
		ID:            "tr_status01",
		SourceDPUID:   "dpu1",
		SourceDPUName: "bf3-01",
		TargetDPUID:   "dpu2",
		TargetDPUName: "bf3-02",
		TenantID:      "tenant1",
		TrustType:     TrustTypeSSHHost,
	}
	store.CreateTrustRelationship(tr)

	// Verify initial status is active
	retrieved, _ := store.GetTrustRelationship("tr_status01")
	if retrieved.Status != TrustStatusActive {
		t.Errorf("expected initial status 'active', got '%s'", retrieved.Status)
	}

	// Suspend with reason
	reason := "bf3-02 attestation failed"
	err := store.UpdateTrustStatus("tr_status01", TrustStatusSuspended, &reason)
	if err != nil {
		t.Fatalf("UpdateTrustStatus failed: %v", err)
	}

	retrieved, _ = store.GetTrustRelationship("tr_status01")
	if retrieved.Status != TrustStatusSuspended {
		t.Errorf("expected status 'suspended', got '%s'", retrieved.Status)
	}
	if retrieved.SuspendReason == nil {
		t.Error("expected non-nil SuspendReason")
	} else if *retrieved.SuspendReason != reason {
		t.Errorf("expected SuspendReason '%s', got '%s'", reason, *retrieved.SuspendReason)
	}

	// Reactivate (clear reason)
	err = store.UpdateTrustStatus("tr_status01", TrustStatusActive, nil)
	if err != nil {
		t.Fatalf("UpdateTrustStatus failed: %v", err)
	}

	retrieved, _ = store.GetTrustRelationship("tr_status01")
	if retrieved.Status != TrustStatusActive {
		t.Errorf("expected status 'active', got '%s'", retrieved.Status)
	}
	if retrieved.SuspendReason != nil {
		t.Errorf("expected nil SuspendReason, got '%s'", *retrieved.SuspendReason)
	}

	// Test UpdateTrustStatus not found
	err = store.UpdateTrustStatus("nonexistent", TrustStatusSuspended, nil)
	if err == nil {
		t.Error("expected error for updating nonexistent trust relationship, got nil")
	}
}

// TestTrustRelationshipExists tests checking if a trust relationship exists.
func TestTrustRelationshipExists(t *testing.T) {
	store := setupTestStore(t)
	setupTrustTestData(t, store)

	// Create a trust relationship
	store.CreateTrustRelationship(&TrustRelationship{
		ID:            "tr_exists01",
		SourceDPUID:   "dpu1",
		SourceDPUName: "bf3-01",
		TargetDPUID:   "dpu2",
		TargetDPUName: "bf3-02",
		TenantID:      "tenant1",
		TrustType:     TrustTypeSSHHost,
	})

	// Test exists
	exists, err := store.TrustRelationshipExists("dpu1", "dpu2", TrustTypeSSHHost)
	if err != nil {
		t.Fatalf("TrustRelationshipExists failed: %v", err)
	}
	if !exists {
		t.Error("expected trust relationship to exist")
	}

	// Test does not exist (different trust type)
	exists, err = store.TrustRelationshipExists("dpu1", "dpu2", TrustTypeMTLS)
	if err != nil {
		t.Fatalf("TrustRelationshipExists failed: %v", err)
	}
	if exists {
		t.Error("expected trust relationship to NOT exist for different trust type")
	}

	// Test does not exist (reversed direction - not bidirectional check)
	exists, err = store.TrustRelationshipExists("dpu2", "dpu1", TrustTypeSSHHost)
	if err != nil {
		t.Fatalf("TrustRelationshipExists failed: %v", err)
	}
	if exists {
		t.Error("expected trust relationship to NOT exist for reversed direction")
	}

	// Test does not exist (different DPUs)
	exists, err = store.TrustRelationshipExists("dpu1", "dpu3", TrustTypeSSHHost)
	if err != nil {
		t.Fatalf("TrustRelationshipExists failed: %v", err)
	}
	if exists {
		t.Error("expected trust relationship to NOT exist for different DPUs")
	}
}

// setupTrustTestData creates prerequisite data for trust relationship tests.
func setupTrustTestData(t *testing.T, store *Store) {
	t.Helper()

	// Create tenant
	if err := store.AddTenant("tenant1", "Test Tenant", "", "", nil); err != nil {
		t.Fatalf("failed to create tenant: %v", err)
	}
}
