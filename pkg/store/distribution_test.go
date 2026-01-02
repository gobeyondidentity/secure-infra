package store

import (
	"os"
	"testing"
	"time"
)

func TestDistributionHistory(t *testing.T) {
	// Create temp database
	tmpFile, err := os.CreateTemp("", "distribution_test_*.db")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()

	store, err := Open(tmpFile.Name())
	if err != nil {
		t.Fatalf("failed to open store: %v", err)
	}
	defer store.Close()

	t.Run("RecordDistribution_Success", func(t *testing.T) {
		d := &Distribution{
			DPUName:             "bf3-lab-01",
			CredentialType:      "ssh-ca",
			CredentialName:      "prod-ca",
			Outcome:             DistributionOutcomeSuccess,
			AttestationStatus:   stringPtr("verified"),
			AttestationAgeSecs:  intPtr(120),
			InstalledPath:       stringPtr("/etc/ssh/ca.pub"),
			ErrorMessage:        nil,
		}

		err := store.RecordDistribution(d)
		if err != nil {
			t.Fatalf("RecordDistribution failed: %v", err)
		}

		if d.ID == 0 {
			t.Error("Distribution ID should be set after insert")
		}
	})

	t.Run("RecordDistribution_BlockedStale", func(t *testing.T) {
		d := &Distribution{
			DPUName:             "bf3-lab-02",
			CredentialType:      "ssh-ca",
			CredentialName:      "prod-ca",
			Outcome:             DistributionOutcomeBlockedStale,
			AttestationStatus:   stringPtr("verified"),
			AttestationAgeSecs:  intPtr(7200), // 2 hours - stale
			InstalledPath:       nil,           // Not installed
			ErrorMessage:        stringPtr("attestation too old: 2h0m0s > 1h0m0s"),
		}

		err := store.RecordDistribution(d)
		if err != nil {
			t.Fatalf("RecordDistribution blocked failed: %v", err)
		}
	})

	t.Run("RecordDistribution_BlockedFailed", func(t *testing.T) {
		d := &Distribution{
			DPUName:             "bf3-lab-03",
			CredentialType:      "ssh-ca",
			CredentialName:      "dev-ca",
			Outcome:             DistributionOutcomeBlockedFailed,
			AttestationStatus:   stringPtr("failed"),
			AttestationAgeSecs:  intPtr(60),
			InstalledPath:       nil,
			ErrorMessage:        stringPtr("attestation status is failed"),
		}

		err := store.RecordDistribution(d)
		if err != nil {
			t.Fatalf("RecordDistribution blocked-failed: %v", err)
		}
	})

	t.Run("RecordDistribution_Forced", func(t *testing.T) {
		d := &Distribution{
			DPUName:             "bf3-lab-01",
			CredentialType:      "ssh-ca",
			CredentialName:      "prod-ca",
			Outcome:             DistributionOutcomeForced,
			AttestationStatus:   stringPtr("failed"),
			AttestationAgeSecs:  intPtr(300),
			InstalledPath:       stringPtr("/etc/ssh/ca.pub"),
			ErrorMessage:        nil,
		}

		err := store.RecordDistribution(d)
		if err != nil {
			t.Fatalf("RecordDistribution forced failed: %v", err)
		}
	})

	t.Run("GetDistributionHistory_ByDPU", func(t *testing.T) {
		history, err := store.GetDistributionHistory("bf3-lab-01")
		if err != nil {
			t.Fatalf("GetDistributionHistory failed: %v", err)
		}

		if len(history) != 2 {
			t.Errorf("GetDistributionHistory returned %d records, want 2", len(history))
		}

		// Most recent first (forced)
		if history[0].Outcome != DistributionOutcomeForced {
			t.Errorf("First record outcome = %q, want %q", history[0].Outcome, DistributionOutcomeForced)
		}
		if history[1].Outcome != DistributionOutcomeSuccess {
			t.Errorf("Second record outcome = %q, want %q", history[1].Outcome, DistributionOutcomeSuccess)
		}
	})

	t.Run("GetDistributionHistory_ByDPU_NotFound", func(t *testing.T) {
		history, err := store.GetDistributionHistory("nonexistent")
		if err != nil {
			t.Fatalf("GetDistributionHistory failed: %v", err)
		}

		if len(history) != 0 {
			t.Errorf("GetDistributionHistory for nonexistent DPU returned %d records, want 0", len(history))
		}
	})

	t.Run("GetDistributionHistoryByCredential", func(t *testing.T) {
		history, err := store.GetDistributionHistoryByCredential("prod-ca")
		if err != nil {
			t.Fatalf("GetDistributionHistoryByCredential failed: %v", err)
		}

		if len(history) != 3 {
			t.Errorf("GetDistributionHistoryByCredential returned %d records, want 3", len(history))
		}

		// Verify all are for prod-ca
		for _, d := range history {
			if d.CredentialName != "prod-ca" {
				t.Errorf("CredentialName = %q, want %q", d.CredentialName, "prod-ca")
			}
		}
	})

	t.Run("GetDistributionHistoryByCredential_NotFound", func(t *testing.T) {
		history, err := store.GetDistributionHistoryByCredential("nonexistent-ca")
		if err != nil {
			t.Fatalf("GetDistributionHistoryByCredential failed: %v", err)
		}

		if len(history) != 0 {
			t.Errorf("GetDistributionHistoryByCredential for nonexistent CA returned %d records, want 0", len(history))
		}
	})

	t.Run("ListRecentDistributions", func(t *testing.T) {
		// We have 4 total distributions
		recent, err := store.ListRecentDistributions(2)
		if err != nil {
			t.Fatalf("ListRecentDistributions failed: %v", err)
		}

		if len(recent) != 2 {
			t.Errorf("ListRecentDistributions returned %d records, want 2", len(recent))
		}

		// Should be ordered by created_at DESC (most recent first)
		// The last insert was forced for bf3-lab-01
		if recent[0].DPUName != "bf3-lab-01" || recent[0].Outcome != DistributionOutcomeForced {
			t.Errorf("First recent = %s/%s, want bf3-lab-01/forced", recent[0].DPUName, recent[0].Outcome)
		}
	})

	t.Run("ListRecentDistributions_All", func(t *testing.T) {
		recent, err := store.ListRecentDistributions(100)
		if err != nil {
			t.Fatalf("ListRecentDistributions failed: %v", err)
		}

		if len(recent) != 4 {
			t.Errorf("ListRecentDistributions returned %d records, want 4", len(recent))
		}
	})
}

func TestDistributionNullableFields(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "distribution_nullable_test_*.db")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()

	store, err := Open(tmpFile.Name())
	if err != nil {
		t.Fatalf("failed to open store: %v", err)
	}
	defer store.Close()

	// Distribution with no attestation info (e.g., no attestation exists)
	d := &Distribution{
		DPUName:             "bf3-no-attestation",
		CredentialType:      "ssh-ca",
		CredentialName:      "test-ca",
		Outcome:             DistributionOutcomeBlockedFailed,
		AttestationStatus:   nil,
		AttestationAgeSecs:  nil,
		InstalledPath:       nil,
		ErrorMessage:        stringPtr("no attestation found"),
	}

	err = store.RecordDistribution(d)
	if err != nil {
		t.Fatalf("RecordDistribution with nulls failed: %v", err)
	}

	history, err := store.GetDistributionHistory("bf3-no-attestation")
	if err != nil {
		t.Fatalf("GetDistributionHistory failed: %v", err)
	}

	if len(history) != 1 {
		t.Fatalf("Expected 1 record, got %d", len(history))
	}

	retrieved := history[0]
	if retrieved.AttestationStatus != nil {
		t.Errorf("AttestationStatus should be nil, got %v", *retrieved.AttestationStatus)
	}
	if retrieved.AttestationAgeSecs != nil {
		t.Errorf("AttestationAgeSecs should be nil, got %v", *retrieved.AttestationAgeSecs)
	}
	if retrieved.InstalledPath != nil {
		t.Errorf("InstalledPath should be nil, got %v", *retrieved.InstalledPath)
	}
	if retrieved.ErrorMessage == nil || *retrieved.ErrorMessage != "no attestation found" {
		t.Errorf("ErrorMessage = %v, want 'no attestation found'", retrieved.ErrorMessage)
	}
}

func TestDistributionCreatedAt(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "distribution_time_test_*.db")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()

	store, err := Open(tmpFile.Name())
	if err != nil {
		t.Fatalf("failed to open store: %v", err)
	}
	defer store.Close()

	// Truncate to second precision since SQLite stores Unix timestamps
	before := time.Now().Truncate(time.Second)

	d := &Distribution{
		DPUName:        "bf3-time-test",
		CredentialType: "ssh-ca",
		CredentialName: "time-ca",
		Outcome:        DistributionOutcomeSuccess,
	}

	err = store.RecordDistribution(d)
	if err != nil {
		t.Fatalf("RecordDistribution failed: %v", err)
	}

	// Add 1 second to account for potential second boundary crossing
	after := time.Now().Add(time.Second).Truncate(time.Second)

	history, err := store.GetDistributionHistory("bf3-time-test")
	if err != nil {
		t.Fatalf("GetDistributionHistory failed: %v", err)
	}

	if len(history) != 1 {
		t.Fatalf("Expected 1 record, got %d", len(history))
	}

	createdAt := history[0].CreatedAt
	if createdAt.Before(before) || createdAt.After(after) {
		t.Errorf("CreatedAt %v not between %v and %v", createdAt, before, after)
	}
}

func TestDistributionNewFields(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "distribution_new_fields_test_*.db")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()

	store, err := Open(tmpFile.Name())
	if err != nil {
		t.Fatalf("failed to open store: %v", err)
	}
	defer store.Close()

	// Test recording with all new fields populated
	snapshot := `{"pcr0":"abc123","pcr1":"def456"}`
	d := &Distribution{
		DPUName:             "bf3-full-fields",
		CredentialType:      "ssh-ca",
		CredentialName:      "prod-ca",
		Outcome:             DistributionOutcomeForced,
		AttestationStatus:   stringPtr("stale"),
		AttestationAgeSecs:  intPtr(7200),
		InstalledPath:       stringPtr("/etc/ssh/ca.pub"),
		ErrorMessage:        nil,
		OperatorID:          "op-123",
		OperatorEmail:       "admin@example.com",
		TenantID:            "tenant-abc",
		AttestationSnapshot: &snapshot,
		BlockedReason:       stringPtr("attestation too old"),
		ForcedBy:            stringPtr("emergency-admin@example.com"),
	}

	err = store.RecordDistribution(d)
	if err != nil {
		t.Fatalf("RecordDistribution failed: %v", err)
	}

	// Retrieve and verify all fields
	history, err := store.GetDistributionHistory("bf3-full-fields")
	if err != nil {
		t.Fatalf("GetDistributionHistory failed: %v", err)
	}

	if len(history) != 1 {
		t.Fatalf("Expected 1 record, got %d", len(history))
	}

	r := history[0]
	if r.OperatorID != "op-123" {
		t.Errorf("OperatorID = %q, want %q", r.OperatorID, "op-123")
	}
	if r.OperatorEmail != "admin@example.com" {
		t.Errorf("OperatorEmail = %q, want %q", r.OperatorEmail, "admin@example.com")
	}
	if r.TenantID != "tenant-abc" {
		t.Errorf("TenantID = %q, want %q", r.TenantID, "tenant-abc")
	}
	if r.AttestationSnapshot == nil || *r.AttestationSnapshot != snapshot {
		t.Errorf("AttestationSnapshot = %v, want %q", r.AttestationSnapshot, snapshot)
	}
	if r.BlockedReason == nil || *r.BlockedReason != "attestation too old" {
		t.Errorf("BlockedReason = %v, want %q", r.BlockedReason, "attestation too old")
	}
	if r.ForcedBy == nil || *r.ForcedBy != "emergency-admin@example.com" {
		t.Errorf("ForcedBy = %v, want %q", r.ForcedBy, "emergency-admin@example.com")
	}
}

func TestListDistributionsByOperator(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "distribution_by_operator_test_*.db")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()

	store, err := Open(tmpFile.Name())
	if err != nil {
		t.Fatalf("failed to open store: %v", err)
	}
	defer store.Close()

	// Create distributions by different operators
	distributions := []*Distribution{
		{DPUName: "dpu-1", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeSuccess, OperatorID: "op-alice", OperatorEmail: "alice@example.com", TenantID: "t1"},
		{DPUName: "dpu-2", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeSuccess, OperatorID: "op-alice", OperatorEmail: "alice@example.com", TenantID: "t1"},
		{DPUName: "dpu-3", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeSuccess, OperatorID: "op-bob", OperatorEmail: "bob@example.com", TenantID: "t2"},
	}

	for _, d := range distributions {
		if err := store.RecordDistribution(d); err != nil {
			t.Fatalf("RecordDistribution failed: %v", err)
		}
	}

	// Query by Alice
	aliceDistributions, err := store.ListDistributionsByOperator("op-alice", 10)
	if err != nil {
		t.Fatalf("ListDistributionsByOperator failed: %v", err)
	}
	if len(aliceDistributions) != 2 {
		t.Errorf("Expected 2 distributions for Alice, got %d", len(aliceDistributions))
	}
	for _, d := range aliceDistributions {
		if d.OperatorID != "op-alice" {
			t.Errorf("OperatorID = %q, want %q", d.OperatorID, "op-alice")
		}
	}

	// Query by Bob
	bobDistributions, err := store.ListDistributionsByOperator("op-bob", 10)
	if err != nil {
		t.Fatalf("ListDistributionsByOperator failed: %v", err)
	}
	if len(bobDistributions) != 1 {
		t.Errorf("Expected 1 distribution for Bob, got %d", len(bobDistributions))
	}

	// Query by nonexistent operator
	noneDistributions, err := store.ListDistributionsByOperator("op-nobody", 10)
	if err != nil {
		t.Fatalf("ListDistributionsByOperator failed: %v", err)
	}
	if len(noneDistributions) != 0 {
		t.Errorf("Expected 0 distributions for unknown operator, got %d", len(noneDistributions))
	}
}

func TestListDistributionsByTenant(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "distribution_by_tenant_test_*.db")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()

	store, err := Open(tmpFile.Name())
	if err != nil {
		t.Fatalf("failed to open store: %v", err)
	}
	defer store.Close()

	// Create distributions for different tenants
	distributions := []*Distribution{
		{DPUName: "dpu-1", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeSuccess, OperatorID: "op-1", TenantID: "tenant-prod"},
		{DPUName: "dpu-2", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeSuccess, OperatorID: "op-1", TenantID: "tenant-prod"},
		{DPUName: "dpu-3", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeSuccess, OperatorID: "op-1", TenantID: "tenant-prod"},
		{DPUName: "dpu-4", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeSuccess, OperatorID: "op-2", TenantID: "tenant-dev"},
	}

	for _, d := range distributions {
		if err := store.RecordDistribution(d); err != nil {
			t.Fatalf("RecordDistribution failed: %v", err)
		}
	}

	// Query by tenant-prod
	prodDistributions, err := store.ListDistributionsByTenant("tenant-prod", 10)
	if err != nil {
		t.Fatalf("ListDistributionsByTenant failed: %v", err)
	}
	if len(prodDistributions) != 3 {
		t.Errorf("Expected 3 distributions for tenant-prod, got %d", len(prodDistributions))
	}

	// Query by tenant-dev
	devDistributions, err := store.ListDistributionsByTenant("tenant-dev", 10)
	if err != nil {
		t.Fatalf("ListDistributionsByTenant failed: %v", err)
	}
	if len(devDistributions) != 1 {
		t.Errorf("Expected 1 distribution for tenant-dev, got %d", len(devDistributions))
	}

	// Test limit
	limitedDistributions, err := store.ListDistributionsByTenant("tenant-prod", 2)
	if err != nil {
		t.Fatalf("ListDistributionsByTenant failed: %v", err)
	}
	if len(limitedDistributions) != 2 {
		t.Errorf("Expected 2 distributions with limit, got %d", len(limitedDistributions))
	}
}

func TestListDistributionsByOutcome(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "distribution_by_outcome_test_*.db")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()

	store, err := Open(tmpFile.Name())
	if err != nil {
		t.Fatalf("failed to open store: %v", err)
	}
	defer store.Close()

	// Create distributions with different outcomes
	distributions := []*Distribution{
		{DPUName: "dpu-1", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeSuccess, OperatorID: "op-1", TenantID: "t1"},
		{DPUName: "dpu-2", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeSuccess, OperatorID: "op-1", TenantID: "t1"},
		{DPUName: "dpu-3", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeBlockedStale, OperatorID: "op-1", TenantID: "t1"},
		{DPUName: "dpu-4", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeBlockedFailed, OperatorID: "op-1", TenantID: "t1"},
		{DPUName: "dpu-5", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeForced, OperatorID: "op-1", TenantID: "t1"},
	}

	for _, d := range distributions {
		if err := store.RecordDistribution(d); err != nil {
			t.Fatalf("RecordDistribution failed: %v", err)
		}
	}

	// Query success outcomes
	successDistributions, err := store.ListDistributionsByOutcome(DistributionOutcomeSuccess, 10)
	if err != nil {
		t.Fatalf("ListDistributionsByOutcome failed: %v", err)
	}
	if len(successDistributions) != 2 {
		t.Errorf("Expected 2 success distributions, got %d", len(successDistributions))
	}
	for _, d := range successDistributions {
		if d.Outcome != DistributionOutcomeSuccess {
			t.Errorf("Outcome = %q, want %q", d.Outcome, DistributionOutcomeSuccess)
		}
	}

	// Query blocked-stale outcomes
	staleDistributions, err := store.ListDistributionsByOutcome(DistributionOutcomeBlockedStale, 10)
	if err != nil {
		t.Fatalf("ListDistributionsByOutcome failed: %v", err)
	}
	if len(staleDistributions) != 1 {
		t.Errorf("Expected 1 blocked-stale distribution, got %d", len(staleDistributions))
	}

	// Query forced outcomes
	forcedDistributions, err := store.ListDistributionsByOutcome(DistributionOutcomeForced, 10)
	if err != nil {
		t.Fatalf("ListDistributionsByOutcome failed: %v", err)
	}
	if len(forcedDistributions) != 1 {
		t.Errorf("Expected 1 forced distribution, got %d", len(forcedDistributions))
	}
}

func TestListDistributionsInTimeRange(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "distribution_time_range_test_*.db")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()

	store, err := Open(tmpFile.Name())
	if err != nil {
		t.Fatalf("failed to open store: %v", err)
	}
	defer store.Close()

	// Record distributions now
	before := time.Now().Add(-time.Second)
	distributions := []*Distribution{
		{DPUName: "dpu-1", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeSuccess, OperatorID: "op-1", TenantID: "t1"},
		{DPUName: "dpu-2", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeSuccess, OperatorID: "op-1", TenantID: "t1"},
	}

	for _, d := range distributions {
		if err := store.RecordDistribution(d); err != nil {
			t.Fatalf("RecordDistribution failed: %v", err)
		}
	}
	after := time.Now().Add(time.Second)

	// Query within the time range
	inRangeDistributions, err := store.ListDistributionsInTimeRange(before, after, 10)
	if err != nil {
		t.Fatalf("ListDistributionsInTimeRange failed: %v", err)
	}
	if len(inRangeDistributions) != 2 {
		t.Errorf("Expected 2 distributions in range, got %d", len(inRangeDistributions))
	}

	// Query before the time range
	pastDistributions, err := store.ListDistributionsInTimeRange(before.Add(-time.Hour), before.Add(-time.Minute), 10)
	if err != nil {
		t.Fatalf("ListDistributionsInTimeRange failed: %v", err)
	}
	if len(pastDistributions) != 0 {
		t.Errorf("Expected 0 distributions in past range, got %d", len(pastDistributions))
	}

	// Query after the time range
	futureDistributions, err := store.ListDistributionsInTimeRange(after.Add(time.Minute), after.Add(time.Hour), 10)
	if err != nil {
		t.Fatalf("ListDistributionsInTimeRange failed: %v", err)
	}
	if len(futureDistributions) != 0 {
		t.Errorf("Expected 0 distributions in future range, got %d", len(futureDistributions))
	}
}

func TestListDistributionsWithFilters(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "distribution_filters_test_*.db")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()

	store, err := Open(tmpFile.Name())
	if err != nil {
		t.Fatalf("failed to open store: %v", err)
	}
	defer store.Close()

	// Create diverse set of distributions
	distributions := []*Distribution{
		{DPUName: "dpu-1", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeSuccess, OperatorID: "op-alice", TenantID: "tenant-prod"},
		{DPUName: "dpu-1", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeBlockedStale, OperatorID: "op-alice", TenantID: "tenant-prod"},
		{DPUName: "dpu-2", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeSuccess, OperatorID: "op-bob", TenantID: "tenant-prod"},
		{DPUName: "dpu-3", CredentialType: "ssh-ca", CredentialName: "ca-1", Outcome: DistributionOutcomeForced, OperatorID: "op-alice", TenantID: "tenant-dev"},
	}

	for _, d := range distributions {
		if err := store.RecordDistribution(d); err != nil {
			t.Fatalf("RecordDistribution failed: %v", err)
		}
	}

	t.Run("FilterByTargetDPU", func(t *testing.T) {
		results, err := store.ListDistributionsWithFilters(DistributionQueryOpts{
			TargetDPU: "dpu-1",
			Limit:     10,
		})
		if err != nil {
			t.Fatalf("ListDistributionsWithFilters failed: %v", err)
		}
		if len(results) != 2 {
			t.Errorf("Expected 2 distributions for dpu-1, got %d", len(results))
		}
	})

	t.Run("FilterByOperatorAndTenant", func(t *testing.T) {
		results, err := store.ListDistributionsWithFilters(DistributionQueryOpts{
			OperatorID: "op-alice",
			TenantID:   "tenant-prod",
			Limit:      10,
		})
		if err != nil {
			t.Fatalf("ListDistributionsWithFilters failed: %v", err)
		}
		if len(results) != 2 {
			t.Errorf("Expected 2 distributions for alice in tenant-prod, got %d", len(results))
		}
	})

	t.Run("FilterByOutcome", func(t *testing.T) {
		outcome := DistributionOutcomeSuccess
		results, err := store.ListDistributionsWithFilters(DistributionQueryOpts{
			Outcome: &outcome,
			Limit:   10,
		})
		if err != nil {
			t.Fatalf("ListDistributionsWithFilters failed: %v", err)
		}
		if len(results) != 2 {
			t.Errorf("Expected 2 success distributions, got %d", len(results))
		}
	})

	t.Run("FilterByMultipleCriteria", func(t *testing.T) {
		outcome := DistributionOutcomeSuccess
		results, err := store.ListDistributionsWithFilters(DistributionQueryOpts{
			OperatorID: "op-alice",
			Outcome:    &outcome,
			Limit:      10,
		})
		if err != nil {
			t.Fatalf("ListDistributionsWithFilters failed: %v", err)
		}
		if len(results) != 1 {
			t.Errorf("Expected 1 success distribution by alice, got %d", len(results))
		}
	})

	t.Run("FilterNoMatches", func(t *testing.T) {
		results, err := store.ListDistributionsWithFilters(DistributionQueryOpts{
			OperatorID: "op-nobody",
			Limit:      10,
		})
		if err != nil {
			t.Fatalf("ListDistributionsWithFilters failed: %v", err)
		}
		if len(results) != 0 {
			t.Errorf("Expected 0 distributions, got %d", len(results))
		}
	})

	t.Run("NoFilters", func(t *testing.T) {
		results, err := store.ListDistributionsWithFilters(DistributionQueryOpts{
			Limit: 10,
		})
		if err != nil {
			t.Fatalf("ListDistributionsWithFilters failed: %v", err)
		}
		if len(results) != 4 {
			t.Errorf("Expected 4 distributions with no filters, got %d", len(results))
		}
	})

	t.Run("WithLimit", func(t *testing.T) {
		results, err := store.ListDistributionsWithFilters(DistributionQueryOpts{
			Limit: 2,
		})
		if err != nil {
			t.Fatalf("ListDistributionsWithFilters failed: %v", err)
		}
		if len(results) != 2 {
			t.Errorf("Expected 2 distributions with limit, got %d", len(results))
		}
	})

	t.Run("NoLimit", func(t *testing.T) {
		results, err := store.ListDistributionsWithFilters(DistributionQueryOpts{})
		if err != nil {
			t.Fatalf("ListDistributionsWithFilters failed: %v", err)
		}
		if len(results) != 4 {
			t.Errorf("Expected 4 distributions with no limit, got %d", len(results))
		}
	})
}

// Helper functions for creating pointers to primitives
func stringPtr(s string) *string {
	return &s
}

func intPtr(i int) *int {
	return &i
}
