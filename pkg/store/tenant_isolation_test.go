package store

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMultiTenantDPUIsolation verifies that DPUs assigned to one tenant
// are not visible in another tenant's queries.
func TestMultiTenantDPUIsolation(t *testing.T) {
	SetInsecureMode(true)
	defer SetInsecureMode(false)

	s := setupTestStore(t)

	// Create two tenants
	err := s.AddTenant("tenant-a", "Tenant A", "First tenant", "", nil)
	require.NoError(t, err)
	err = s.AddTenant("tenant-b", "Tenant B", "Second tenant", "", nil)
	require.NoError(t, err)

	// Create three DPUs
	err = s.Add("dpu-a", "bf3-tenant-a", "192.168.1.1", 50051)
	require.NoError(t, err)
	err = s.Add("dpu-b", "bf3-tenant-b", "192.168.1.2", 50051)
	require.NoError(t, err)
	err = s.Add("dpu-unassigned", "bf3-unassigned", "192.168.1.3", 50051)
	require.NoError(t, err)

	// Assign DPUs to tenants
	err = s.AssignDPUToTenant("dpu-a", "tenant-a")
	require.NoError(t, err)
	err = s.AssignDPUToTenant("dpu-b", "tenant-b")
	require.NoError(t, err)

	t.Run("TenantA_OnlySeesOwnDPU", func(t *testing.T) {
		dpus, err := s.ListDPUsByTenant("tenant-a")
		require.NoError(t, err)
		require.Len(t, dpus, 1, "Tenant A should see exactly 1 DPU")
		assert.Equal(t, "dpu-a", dpus[0].ID)
		assert.Equal(t, "bf3-tenant-a", dpus[0].Name)
	})

	t.Run("TenantB_OnlySeesOwnDPU", func(t *testing.T) {
		dpus, err := s.ListDPUsByTenant("tenant-b")
		require.NoError(t, err)
		require.Len(t, dpus, 1, "Tenant B should see exactly 1 DPU")
		assert.Equal(t, "dpu-b", dpus[0].ID)
		assert.Equal(t, "bf3-tenant-b", dpus[0].Name)
	})

	t.Run("UnassignedDPU_NotInAnyTenantList", func(t *testing.T) {
		// Unassigned DPU should not appear in either tenant's list
		dpusA, err := s.ListDPUsByTenant("tenant-a")
		require.NoError(t, err)
		for _, dpu := range dpusA {
			assert.NotEqual(t, "dpu-unassigned", dpu.ID, "Unassigned DPU should not be in Tenant A's list")
		}

		dpusB, err := s.ListDPUsByTenant("tenant-b")
		require.NoError(t, err)
		for _, dpu := range dpusB {
			assert.NotEqual(t, "dpu-unassigned", dpu.ID, "Unassigned DPU should not be in Tenant B's list")
		}
	})

	t.Run("GetDPU_RevealsCorrectTenant", func(t *testing.T) {
		dpuA, err := s.Get("dpu-a")
		require.NoError(t, err)
		require.NotNil(t, dpuA.TenantID)
		assert.Equal(t, "tenant-a", *dpuA.TenantID)

		dpuB, err := s.Get("dpu-b")
		require.NoError(t, err)
		require.NotNil(t, dpuB.TenantID)
		assert.Equal(t, "tenant-b", *dpuB.TenantID)

		dpuUnassigned, err := s.Get("dpu-unassigned")
		require.NoError(t, err)
		assert.Nil(t, dpuUnassigned.TenantID, "Unassigned DPU should have nil TenantID")
	})
}

// TestMultiTenantHostIsolation verifies that agent hosts registered through
// a tenant's DPU are not visible to other tenants.
func TestMultiTenantHostIsolation(t *testing.T) {
	SetInsecureMode(true)
	defer SetInsecureMode(false)

	s := setupTestStore(t)

	// Create two tenants
	err := s.AddTenant("tenant-a", "Tenant A", "", "", nil)
	require.NoError(t, err)
	err = s.AddTenant("tenant-b", "Tenant B", "", "", nil)
	require.NoError(t, err)

	// Create DPUs and assign to tenants
	err = s.Add("dpu-a", "bf3-a", "192.168.1.1", 50051)
	require.NoError(t, err)
	err = s.Add("dpu-b", "bf3-b", "192.168.1.2", 50051)
	require.NoError(t, err)

	err = s.AssignDPUToTenant("dpu-a", "tenant-a")
	require.NoError(t, err)
	err = s.AssignDPUToTenant("dpu-b", "tenant-b")
	require.NoError(t, err)

	// Register hosts through their respective DPUs (inheriting tenant)
	hostA := &AgentHost{
		DPUName:  "bf3-a",
		DPUID:    "dpu-a",
		Hostname: "gpu-node-a",
		TenantID: "tenant-a",
	}
	err = s.RegisterAgentHost(hostA)
	require.NoError(t, err)

	hostB := &AgentHost{
		DPUName:  "bf3-b",
		DPUID:    "dpu-b",
		Hostname: "gpu-node-b",
		TenantID: "tenant-b",
	}
	err = s.RegisterAgentHost(hostB)
	require.NoError(t, err)

	t.Run("TenantA_OnlySeesOwnHost", func(t *testing.T) {
		hosts, err := s.ListAgentHosts("tenant-a")
		require.NoError(t, err)
		require.Len(t, hosts, 1, "Tenant A should see exactly 1 host")
		assert.Equal(t, "gpu-node-a", hosts[0].Hostname)
		assert.Equal(t, "tenant-a", hosts[0].TenantID)
	})

	t.Run("TenantB_OnlySeesOwnHost", func(t *testing.T) {
		hosts, err := s.ListAgentHosts("tenant-b")
		require.NoError(t, err)
		require.Len(t, hosts, 1, "Tenant B should see exactly 1 host")
		assert.Equal(t, "gpu-node-b", hosts[0].Hostname)
		assert.Equal(t, "tenant-b", hosts[0].TenantID)
	})

	t.Run("EmptyTenantFilter_ReturnsAllHosts", func(t *testing.T) {
		hosts, err := s.ListAgentHosts("")
		require.NoError(t, err)
		assert.Len(t, hosts, 2, "Empty filter should return all hosts")
	})
}

// TestMultiTenantCredentialIsolation verifies that SSH CAs assigned to one tenant
// are not visible in another tenant's credential queries.
func TestMultiTenantCredentialIsolation(t *testing.T) {
	SetInsecureMode(true)
	defer SetInsecureMode(false)

	s := setupTestStore(t)

	// Create two tenants
	err := s.AddTenant("tenant-a", "Tenant A", "", "", nil)
	require.NoError(t, err)
	err = s.AddTenant("tenant-b", "Tenant B", "", "", nil)
	require.NoError(t, err)

	tenantA := "tenant-a"
	tenantB := "tenant-b"

	// Create SSH CAs for each tenant
	err = s.CreateSSHCA("ca-a", "ssh-ca-tenant-a", []byte("pubkey-a"), []byte("privkey-a"), "ed25519", &tenantA)
	require.NoError(t, err)
	err = s.CreateSSHCA("ca-b", "ssh-ca-tenant-b", []byte("pubkey-b"), []byte("privkey-b"), "ed25519", &tenantB)
	require.NoError(t, err)

	// Create a global CA (no tenant)
	err = s.CreateSSHCA("ca-global", "ssh-ca-global", []byte("pubkey-global"), []byte("privkey-global"), "ed25519", nil)
	require.NoError(t, err)

	t.Run("TenantA_OnlySeesOwnCA", func(t *testing.T) {
		cas, err := s.GetSSHCAsByTenant("tenant-a")
		require.NoError(t, err)
		require.Len(t, cas, 1, "Tenant A should see exactly 1 CA")
		assert.Equal(t, "ca-a", cas[0].ID)
		assert.Equal(t, "ssh-ca-tenant-a", cas[0].Name)
	})

	t.Run("TenantB_OnlySeesOwnCA", func(t *testing.T) {
		cas, err := s.GetSSHCAsByTenant("tenant-b")
		require.NoError(t, err)
		require.Len(t, cas, 1, "Tenant B should see exactly 1 CA")
		assert.Equal(t, "ca-b", cas[0].ID)
		assert.Equal(t, "ssh-ca-tenant-b", cas[0].Name)
	})

	t.Run("GlobalCA_NotInTenantQueries", func(t *testing.T) {
		casA, err := s.GetSSHCAsByTenant("tenant-a")
		require.NoError(t, err)
		for _, ca := range casA {
			assert.NotEqual(t, "ca-global", ca.ID, "Global CA should not be in Tenant A's list")
		}

		casB, err := s.GetSSHCAsByTenant("tenant-b")
		require.NoError(t, err)
		for _, ca := range casB {
			assert.NotEqual(t, "ca-global", ca.ID, "Global CA should not be in Tenant B's list")
		}
	})

	t.Run("ListAllCAs_IncludesAllCAs", func(t *testing.T) {
		cas, err := s.ListSSHCAs()
		require.NoError(t, err)
		assert.Len(t, cas, 3, "ListSSHCAs should return all CAs")
	})
}

// TestTenantDeletionIsolation verifies that deleting one tenant does not affect
// another tenant's resources.
func TestTenantDeletionIsolation(t *testing.T) {
	SetInsecureMode(true)
	defer SetInsecureMode(false)

	s := setupTestStore(t)

	// Create two tenants
	err := s.AddTenant("tenant-a", "Tenant A", "", "", nil)
	require.NoError(t, err)
	err = s.AddTenant("tenant-b", "Tenant B", "", "", nil)
	require.NoError(t, err)

	// Create DPUs and assign to tenants
	err = s.Add("dpu-a", "bf3-a", "192.168.1.1", 50051)
	require.NoError(t, err)
	err = s.Add("dpu-b", "bf3-b", "192.168.1.2", 50051)
	require.NoError(t, err)

	err = s.AssignDPUToTenant("dpu-a", "tenant-a")
	require.NoError(t, err)
	err = s.AssignDPUToTenant("dpu-b", "tenant-b")
	require.NoError(t, err)

	// Register hosts for each tenant
	hostA := &AgentHost{
		DPUName:  "bf3-a",
		DPUID:    "dpu-a",
		Hostname: "gpu-node-a",
		TenantID: "tenant-a",
	}
	err = s.RegisterAgentHost(hostA)
	require.NoError(t, err)

	hostB := &AgentHost{
		DPUName:  "bf3-b",
		DPUID:    "dpu-b",
		Hostname: "gpu-node-b",
		TenantID: "tenant-b",
	}
	err = s.RegisterAgentHost(hostB)
	require.NoError(t, err)

	// Delete tenant A
	err = s.RemoveTenant("tenant-a")
	require.NoError(t, err)

	t.Run("DPU_BecomesUnassignedAfterTenantDeletion", func(t *testing.T) {
		dpuA, err := s.Get("dpu-a")
		require.NoError(t, err, "DPU should still exist after tenant deletion")
		assert.Nil(t, dpuA.TenantID, "DPU should have NULL tenant_id after tenant deletion")
	})

	t.Run("TenantB_ResourcesUnchanged", func(t *testing.T) {
		// Tenant B should still exist
		tenantB, err := s.GetTenant("tenant-b")
		require.NoError(t, err)
		assert.Equal(t, "Tenant B", tenantB.Name)

		// Tenant B's DPU should still be assigned
		dpuB, err := s.Get("dpu-b")
		require.NoError(t, err)
		require.NotNil(t, dpuB.TenantID)
		assert.Equal(t, "tenant-b", *dpuB.TenantID)

		// Tenant B's hosts should still be visible
		hosts, err := s.ListAgentHosts("tenant-b")
		require.NoError(t, err)
		require.Len(t, hosts, 1)
		assert.Equal(t, "gpu-node-b", hosts[0].Hostname)
	})

	t.Run("ListDPUsByTenant_StillWorks", func(t *testing.T) {
		dpus, err := s.ListDPUsByTenant("tenant-b")
		require.NoError(t, err)
		require.Len(t, dpus, 1)
		assert.Equal(t, "dpu-b", dpus[0].ID)
	})
}

// TestTenantDeletionCascade verifies the cascade behavior when a tenant is deleted:
// - DPUs become unassigned (ON DELETE SET NULL)
// - SSH CAs become unassigned (ON DELETE SET NULL)
// - Operator-tenant memberships are deleted (ON DELETE CASCADE)
// - Authorizations must be deleted manually first (no CASCADE)
func TestTenantDeletionCascade(t *testing.T) {
	SetInsecureMode(true)
	defer SetInsecureMode(false)

	s := setupTestStore(t)

	// Create tenant
	err := s.AddTenant("tenant-to-delete", "Deletable Tenant", "", "", nil)
	require.NoError(t, err)

	tenantID := "tenant-to-delete"

	// Create and assign DPU
	err = s.Add("dpu-cascade", "bf3-cascade", "192.168.1.10", 50051)
	require.NoError(t, err)
	err = s.AssignDPUToTenant("dpu-cascade", tenantID)
	require.NoError(t, err)

	// Create SSH CA for tenant
	err = s.CreateSSHCA("ca-cascade", "ssh-ca-cascade", []byte("pubkey"), []byte("privkey"), "ed25519", &tenantID)
	require.NoError(t, err)

	// Create operator and add to tenant
	err = s.CreateOperator("op-cascade", "cascade@example.com", "Cascade Operator")
	require.NoError(t, err)
	err = s.AddOperatorToTenant("op-cascade", tenantID, "admin")
	require.NoError(t, err)

	// Verify setup
	deps, err := s.GetTenantDependencies(tenantID)
	require.NoError(t, err)
	assert.True(t, deps.HasAny(), "Tenant should have dependencies before deletion")
	assert.Len(t, deps.DPUs, 1)
	assert.Len(t, deps.Operators, 1)
	assert.Len(t, deps.CAs, 1)

	// Delete tenant (no authorizations blocking it)
	err = s.RemoveTenant(tenantID)
	require.NoError(t, err)

	t.Run("DPU_BecomesUnassigned", func(t *testing.T) {
		dpu, err := s.Get("dpu-cascade")
		require.NoError(t, err, "DPU should still exist")
		assert.Nil(t, dpu.TenantID, "DPU tenant_id should be NULL")
	})

	t.Run("SSHCA_BecomesUnassigned", func(t *testing.T) {
		ca, err := s.GetSSHCA("ssh-ca-cascade")
		require.NoError(t, err, "SSH CA should still exist")
		assert.Nil(t, ca.TenantID, "SSH CA tenant_id should be NULL")
	})

	t.Run("OperatorTenantMembership_IsDeleted", func(t *testing.T) {
		memberships, err := s.GetOperatorTenants("op-cascade")
		require.NoError(t, err)
		assert.Len(t, memberships, 0, "Operator should have no tenant memberships")
	})

	t.Run("Operator_StillExists", func(t *testing.T) {
		op, err := s.GetOperator("op-cascade")
		require.NoError(t, err, "Operator should still exist")
		assert.Equal(t, "cascade@example.com", op.Email)
	})
}

// TestTenantDeletionBlockedByAuthorizations verifies that a tenant cannot be deleted
// while authorizations reference it (no ON DELETE CASCADE on authorizations table).
func TestTenantDeletionBlockedByAuthorizations(t *testing.T) {
	SetInsecureMode(true)
	defer SetInsecureMode(false)

	s := setupTestStore(t)

	// Create tenant
	err := s.AddTenant("blocked-tenant", "Blocked Tenant", "", "", nil)
	require.NoError(t, err)

	tenantID := "blocked-tenant"

	// Create operator
	err = s.CreateOperator("op-blocking", "blocking@example.com", "Blocking Operator")
	require.NoError(t, err)
	err = s.AddOperatorToTenant("op-blocking", tenantID, "admin")
	require.NoError(t, err)

	// Create SSH CA for tenant
	err = s.CreateSSHCA("ca-blocking", "ssh-ca-blocking", []byte("pubkey"), []byte("privkey"), "ed25519", &tenantID)
	require.NoError(t, err)

	// Create authorization that blocks tenant deletion
	err = s.CreateAuthorization("auth-blocking", "op-blocking", tenantID, []string{"ca-blocking"}, []string{"all"}, "system", nil)
	require.NoError(t, err)

	t.Run("CannotDeleteTenantWithAuthorizations", func(t *testing.T) {
		err := s.RemoveTenant(tenantID)
		require.Error(t, err, "Should fail to delete tenant with authorizations")
		assert.Contains(t, err.Error(), "constraint failed")
	})

	t.Run("CanDeleteAfterRemovingAuthorization", func(t *testing.T) {
		// Remove the blocking authorization
		err := s.DeleteAuthorization("auth-blocking")
		require.NoError(t, err)

		// Now tenant deletion should succeed
		err = s.RemoveTenant(tenantID)
		require.NoError(t, err)

		// Verify tenant is gone
		_, err = s.GetTenant(tenantID)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})
}

// TestMultiTenantAuthorizationIsolation verifies that authorizations are properly
// scoped to tenants and one tenant's authorizations don't leak to another.
func TestMultiTenantAuthorizationIsolation(t *testing.T) {
	SetInsecureMode(true)
	defer SetInsecureMode(false)

	s := setupTestStore(t)

	// Create two tenants
	err := s.AddTenant("tenant-a", "Tenant A", "", "", nil)
	require.NoError(t, err)
	err = s.AddTenant("tenant-b", "Tenant B", "", "", nil)
	require.NoError(t, err)

	// Create operators
	err = s.CreateOperator("op-a", "alice@tenant-a.com", "Alice")
	require.NoError(t, err)
	err = s.CreateOperator("op-b", "bob@tenant-b.com", "Bob")
	require.NoError(t, err)

	// Add operators to their respective tenants
	err = s.AddOperatorToTenant("op-a", "tenant-a", "admin")
	require.NoError(t, err)
	err = s.AddOperatorToTenant("op-b", "tenant-b", "admin")
	require.NoError(t, err)

	// Create CAs for each tenant
	tenantA := "tenant-a"
	tenantB := "tenant-b"
	err = s.CreateSSHCA("ca-a", "ssh-ca-a", []byte("pubkey-a"), []byte("privkey-a"), "ed25519", &tenantA)
	require.NoError(t, err)
	err = s.CreateSSHCA("ca-b", "ssh-ca-b", []byte("pubkey-b"), []byte("privkey-b"), "ed25519", &tenantB)
	require.NoError(t, err)

	// Create authorizations
	err = s.CreateAuthorization("auth-a", "op-a", "tenant-a", []string{"ca-a"}, []string{"all"}, "system", nil)
	require.NoError(t, err)
	err = s.CreateAuthorization("auth-b", "op-b", "tenant-b", []string{"ca-b"}, []string{"all"}, "system", nil)
	require.NoError(t, err)

	t.Run("TenantA_OnlySeesOwnAuthorizations", func(t *testing.T) {
		auths, err := s.ListAuthorizationsByTenant("tenant-a")
		require.NoError(t, err)
		require.Len(t, auths, 1)
		assert.Equal(t, "auth-a", auths[0].ID)
		assert.Equal(t, "op-a", auths[0].OperatorID)
	})

	t.Run("TenantB_OnlySeesOwnAuthorizations", func(t *testing.T) {
		auths, err := s.ListAuthorizationsByTenant("tenant-b")
		require.NoError(t, err)
		require.Len(t, auths, 1)
		assert.Equal(t, "auth-b", auths[0].ID)
		assert.Equal(t, "op-b", auths[0].OperatorID)
	})

	t.Run("OperatorA_CannotAccessTenantBCA", func(t *testing.T) {
		// Operator A should be authorized for CA A
		authorized, err := s.CheckCAAuthorization("op-a", "ca-a")
		require.NoError(t, err)
		assert.True(t, authorized, "Operator A should be authorized for CA A")

		// Operator A should NOT be authorized for CA B
		authorized, err = s.CheckCAAuthorization("op-a", "ca-b")
		require.NoError(t, err)
		assert.False(t, authorized, "Operator A should NOT be authorized for CA B")
	})

	t.Run("OperatorB_CannotAccessTenantACA", func(t *testing.T) {
		// Operator B should be authorized for CA B
		authorized, err := s.CheckCAAuthorization("op-b", "ca-b")
		require.NoError(t, err)
		assert.True(t, authorized, "Operator B should be authorized for CA B")

		// Operator B should NOT be authorized for CA A
		authorized, err = s.CheckCAAuthorization("op-b", "ca-a")
		require.NoError(t, err)
		assert.False(t, authorized, "Operator B should NOT be authorized for CA A")
	})
}

// TestMultipleDPUsPerTenant verifies that a tenant can have multiple DPUs
// and all are properly isolated and retrievable.
func TestMultipleDPUsPerTenant(t *testing.T) {
	SetInsecureMode(true)
	defer SetInsecureMode(false)

	s := setupTestStore(t)

	// Create tenant
	err := s.AddTenant("multi-dpu-tenant", "Multi-DPU Tenant", "", "", nil)
	require.NoError(t, err)

	// Create multiple DPUs
	for i := 1; i <= 5; i++ {
		dpuID := "dpu-" + string(rune('0'+i))
		dpuName := "bf3-node-" + string(rune('0'+i))
		err := s.Add(dpuID, dpuName, "192.168.1."+string(rune('0'+i)), 50051)
		require.NoError(t, err)
		err = s.AssignDPUToTenant(dpuID, "multi-dpu-tenant")
		require.NoError(t, err)
	}

	t.Run("AllDPUs_AssignedToTenant", func(t *testing.T) {
		dpus, err := s.ListDPUsByTenant("multi-dpu-tenant")
		require.NoError(t, err)
		assert.Len(t, dpus, 5, "Tenant should have 5 DPUs")

		count, err := s.GetTenantDPUCount("multi-dpu-tenant")
		require.NoError(t, err)
		assert.Equal(t, 5, count)
	})

	t.Run("UnassignOne_ReducesCount", func(t *testing.T) {
		err := s.UnassignDPUFromTenant("dpu-1")
		require.NoError(t, err)

		dpus, err := s.ListDPUsByTenant("multi-dpu-tenant")
		require.NoError(t, err)
		assert.Len(t, dpus, 4, "Tenant should have 4 DPUs after unassign")

		// Verify the unassigned DPU has no tenant
		dpu, err := s.Get("dpu-1")
		require.NoError(t, err)
		assert.Nil(t, dpu.TenantID)
	})
}

// TestReassignDPUBetweenTenants verifies that a DPU can be moved from one tenant
// to another and isolation is maintained.
func TestReassignDPUBetweenTenants(t *testing.T) {
	SetInsecureMode(true)
	defer SetInsecureMode(false)

	s := setupTestStore(t)

	// Create two tenants
	err := s.AddTenant("tenant-source", "Source Tenant", "", "", nil)
	require.NoError(t, err)
	err = s.AddTenant("tenant-dest", "Destination Tenant", "", "", nil)
	require.NoError(t, err)

	// Create and assign DPU to source tenant
	err = s.Add("mobile-dpu", "bf3-mobile", "192.168.1.100", 50051)
	require.NoError(t, err)
	err = s.AssignDPUToTenant("mobile-dpu", "tenant-source")
	require.NoError(t, err)

	// Verify initial assignment
	dpus, err := s.ListDPUsByTenant("tenant-source")
	require.NoError(t, err)
	require.Len(t, dpus, 1)

	dpus, err = s.ListDPUsByTenant("tenant-dest")
	require.NoError(t, err)
	assert.Len(t, dpus, 0)

	// Reassign to destination tenant
	err = s.AssignDPUToTenant("mobile-dpu", "tenant-dest")
	require.NoError(t, err)

	t.Run("SourceTenant_NoLongerHasDPU", func(t *testing.T) {
		dpus, err := s.ListDPUsByTenant("tenant-source")
		require.NoError(t, err)
		assert.Len(t, dpus, 0, "Source tenant should no longer have the DPU")
	})

	t.Run("DestTenant_NowHasDPU", func(t *testing.T) {
		dpus, err := s.ListDPUsByTenant("tenant-dest")
		require.NoError(t, err)
		require.Len(t, dpus, 1, "Destination tenant should now have the DPU")
		assert.Equal(t, "mobile-dpu", dpus[0].ID)
	})

	t.Run("DPU_HasCorrectTenant", func(t *testing.T) {
		dpu, err := s.Get("mobile-dpu")
		require.NoError(t, err)
		require.NotNil(t, dpu.TenantID)
		assert.Equal(t, "tenant-dest", *dpu.TenantID)
	})
}
