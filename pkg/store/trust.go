// Package store provides SQLite-based storage for DPU registry.
// This file contains methods for TrustRelationship entities (M2M trust).
package store

import (
	"database/sql"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// TrustType represents the type of trust relationship.
type TrustType string

const (
	TrustTypeSSHHost TrustType = "ssh_host"
	TrustTypeMTLS    TrustType = "mtls"
)

// TrustStatus represents the status of a trust relationship.
type TrustStatus string

const (
	TrustStatusActive    TrustStatus = "active"
	TrustStatusSuspended TrustStatus = "suspended"
)

// TrustRelationship represents a trust relationship between two DPUs.
type TrustRelationship struct {
	ID            string
	SourceDPUID   string      // Source device ID
	SourceDPUName string      // Source device name (for display)
	TargetDPUID   string      // Target device ID
	TargetDPUName string      // Target device name (for display)
	TenantID      string      // Tenant both devices belong to
	TrustType     TrustType   // ssh_host or mtls
	Bidirectional bool        // If true, trust goes both ways
	Status        TrustStatus // active or suspended
	SuspendReason *string     // Why suspended (e.g., "bf3-02 attestation failed")
	CreatedAt     time.Time
	UpdatedAt     time.Time
}

// generateTrustID generates a unique ID with format "tr_" + first 8 chars of UUID.
func generateTrustID() string {
	u := uuid.New().String()
	return "tr_" + u[:8]
}

// CreateTrustRelationship inserts a new trust relationship.
func (s *Store) CreateTrustRelationship(t *TrustRelationship) error {
	if t.ID == "" {
		t.ID = generateTrustID()
	}
	now := time.Now()
	if t.CreatedAt.IsZero() {
		t.CreatedAt = now
	}
	if t.UpdatedAt.IsZero() {
		t.UpdatedAt = now
	}
	if t.Status == "" {
		t.Status = TrustStatusActive
	}

	bidirectional := 0
	if t.Bidirectional {
		bidirectional = 1
	}

	var suspendReason sql.NullString
	if t.SuspendReason != nil {
		suspendReason = sql.NullString{String: *t.SuspendReason, Valid: true}
	}

	_, err := s.db.Exec(
		`INSERT INTO trust_relationships
		(id, source_dpu_id, source_dpu_name, target_dpu_id, target_dpu_name, tenant_id, trust_type, bidirectional, status, suspend_reason, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		t.ID, t.SourceDPUID, t.SourceDPUName, t.TargetDPUID, t.TargetDPUName,
		t.TenantID, string(t.TrustType), bidirectional, string(t.Status),
		suspendReason, t.CreatedAt.Unix(), t.UpdatedAt.Unix(),
	)
	if err != nil {
		return fmt.Errorf("failed to create trust relationship: %w", err)
	}
	return nil
}

// GetTrustRelationship retrieves a trust relationship by ID.
func (s *Store) GetTrustRelationship(id string) (*TrustRelationship, error) {
	row := s.db.QueryRow(
		`SELECT id, source_dpu_id, source_dpu_name, target_dpu_id, target_dpu_name, tenant_id, trust_type, bidirectional, status, suspend_reason, created_at, updated_at
		FROM trust_relationships WHERE id = ?`,
		id,
	)
	return s.scanTrustRelationship(row)
}

// ListTrustRelationships returns all trust relationships for a tenant.
func (s *Store) ListTrustRelationships(tenantID string) ([]*TrustRelationship, error) {
	rows, err := s.db.Query(
		`SELECT id, source_dpu_id, source_dpu_name, target_dpu_id, target_dpu_name, tenant_id, trust_type, bidirectional, status, suspend_reason, created_at, updated_at
		FROM trust_relationships WHERE tenant_id = ? ORDER BY created_at DESC`,
		tenantID,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to list trust relationships: %w", err)
	}
	defer rows.Close()

	return s.scanTrustRelationshipRows(rows)
}

// ListAllTrustRelationships returns all trust relationships across all tenants.
func (s *Store) ListAllTrustRelationships() ([]*TrustRelationship, error) {
	rows, err := s.db.Query(
		`SELECT id, source_dpu_id, source_dpu_name, target_dpu_id, target_dpu_name, tenant_id, trust_type, bidirectional, status, suspend_reason, created_at, updated_at
		FROM trust_relationships ORDER BY created_at DESC`,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to list all trust relationships: %w", err)
	}
	defer rows.Close()

	return s.scanTrustRelationshipRows(rows)
}

// ListTrustRelationshipsByDPU returns all trust relationships involving a specific DPU.
func (s *Store) ListTrustRelationshipsByDPU(dpuID string) ([]*TrustRelationship, error) {
	rows, err := s.db.Query(
		`SELECT id, source_dpu_id, source_dpu_name, target_dpu_id, target_dpu_name, tenant_id, trust_type, bidirectional, status, suspend_reason, created_at, updated_at
		FROM trust_relationships WHERE source_dpu_id = ? OR target_dpu_id = ? ORDER BY created_at DESC`,
		dpuID, dpuID,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to list trust relationships by DPU: %w", err)
	}
	defer rows.Close()

	return s.scanTrustRelationshipRows(rows)
}

// UpdateTrustStatus updates the status of a trust relationship.
func (s *Store) UpdateTrustStatus(id string, status TrustStatus, reason *string) error {
	now := time.Now().Unix()

	var suspendReason sql.NullString
	if reason != nil {
		suspendReason = sql.NullString{String: *reason, Valid: true}
	}

	result, err := s.db.Exec(
		`UPDATE trust_relationships SET status = ?, suspend_reason = ?, updated_at = ? WHERE id = ?`,
		string(status), suspendReason, now, id,
	)
	if err != nil {
		return fmt.Errorf("failed to update trust status: %w", err)
	}

	rows, _ := result.RowsAffected()
	if rows == 0 {
		return fmt.Errorf("trust relationship not found: %s", id)
	}
	return nil
}

// DeleteTrustRelationship removes a trust relationship by ID.
func (s *Store) DeleteTrustRelationship(id string) error {
	result, err := s.db.Exec(`DELETE FROM trust_relationships WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("failed to delete trust relationship: %w", err)
	}

	rows, _ := result.RowsAffected()
	if rows == 0 {
		return fmt.Errorf("trust relationship not found: %s", id)
	}
	return nil
}

// TrustRelationshipExists checks if a trust relationship exists between two DPUs for a specific trust type.
func (s *Store) TrustRelationshipExists(sourceID, targetID string, trustType TrustType) (bool, error) {
	var count int
	err := s.db.QueryRow(
		`SELECT COUNT(*) FROM trust_relationships
		WHERE source_dpu_id = ? AND target_dpu_id = ? AND trust_type = ?`,
		sourceID, targetID, string(trustType),
	).Scan(&count)
	if err != nil {
		return false, fmt.Errorf("failed to check trust relationship existence: %w", err)
	}
	return count > 0, nil
}

func (s *Store) scanTrustRelationship(row *sql.Row) (*TrustRelationship, error) {
	var tr TrustRelationship
	var trustType, status string
	var bidirectional int
	var suspendReason sql.NullString
	var createdAt, updatedAt int64

	err := row.Scan(&tr.ID, &tr.SourceDPUID, &tr.SourceDPUName, &tr.TargetDPUID, &tr.TargetDPUName,
		&tr.TenantID, &trustType, &bidirectional, &status, &suspendReason, &createdAt, &updatedAt)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("trust relationship not found")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to scan trust relationship: %w", err)
	}

	tr.TrustType = TrustType(trustType)
	tr.Status = TrustStatus(status)
	tr.Bidirectional = bidirectional != 0
	if suspendReason.Valid {
		tr.SuspendReason = &suspendReason.String
	}
	tr.CreatedAt = time.Unix(createdAt, 0)
	tr.UpdatedAt = time.Unix(updatedAt, 0)

	return &tr, nil
}

func (s *Store) scanTrustRelationshipRows(rows *sql.Rows) ([]*TrustRelationship, error) {
	var results []*TrustRelationship
	for rows.Next() {
		var tr TrustRelationship
		var trustType, status string
		var bidirectional int
		var suspendReason sql.NullString
		var createdAt, updatedAt int64

		err := rows.Scan(&tr.ID, &tr.SourceDPUID, &tr.SourceDPUName, &tr.TargetDPUID, &tr.TargetDPUName,
			&tr.TenantID, &trustType, &bidirectional, &status, &suspendReason, &createdAt, &updatedAt)
		if err != nil {
			return nil, fmt.Errorf("failed to scan trust relationship: %w", err)
		}

		tr.TrustType = TrustType(trustType)
		tr.Status = TrustStatus(status)
		tr.Bidirectional = bidirectional != 0
		if suspendReason.Valid {
			tr.SuspendReason = &suspendReason.String
		}
		tr.CreatedAt = time.Unix(createdAt, 0)
		tr.UpdatedAt = time.Unix(updatedAt, 0)

		results = append(results, &tr)
	}
	return results, rows.Err()
}
