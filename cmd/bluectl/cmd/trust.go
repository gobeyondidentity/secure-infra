package cmd

import (
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/beyondidentity/fabric-console/pkg/attestation"
	"github.com/beyondidentity/fabric-console/pkg/store"
	"github.com/spf13/cobra"
)

func init() {
	rootCmd.AddCommand(trustCmd)
	trustCmd.AddCommand(trustCreateCmd)
	trustCmd.AddCommand(trustListCmd)
	trustCmd.AddCommand(trustDeleteCmd)

	// Flags for trust create
	trustCreateCmd.Flags().StringP("source", "s", "", "Source DPU name (required)")
	trustCreateCmd.Flags().StringP("target", "t", "", "Target DPU name (required)")
	trustCreateCmd.Flags().String("type", "ssh_host", "Trust type: ssh_host, mtls")
	trustCreateCmd.Flags().Bool("bidirectional", false, "Create bidirectional trust")
	trustCreateCmd.MarkFlagRequired("source")
	trustCreateCmd.MarkFlagRequired("target")

	// Flags for trust list
	trustListCmd.Flags().String("tenant", "", "Filter by tenant")
}

var trustCmd = &cobra.Command{
	Use:   "trust",
	Short: "Manage trust relationships between DPUs",
	Long:  `Commands to create, list, and delete M2M trust relationships between DPUs.`,
}

var trustCreateCmd = &cobra.Command{
	Use:   "create",
	Short: "Create a trust relationship between DPUs",
	Long: `Create a trust relationship between two DPUs.

The source DPU is granted access to the target DPU. Use --bidirectional
to create mutual trust in both directions.

Trust types:
  ssh_host  - SSH host key trust (allows SSH access)
  mtls      - Mutual TLS trust (allows mTLS connections)

Examples:
  bluectl trust create --source bf3-01 --target bf3-02
  bluectl trust create --source bf3-01 --target bf3-02 --type mtls
  bluectl trust create --source bf3-01 --target bf3-02 --bidirectional`,
	RunE: func(cmd *cobra.Command, args []string) error {
		sourceName, _ := cmd.Flags().GetString("source")
		targetName, _ := cmd.Flags().GetString("target")
		trustType, _ := cmd.Flags().GetString("type")
		bidirectional, _ := cmd.Flags().GetBool("bidirectional")

		// Validate trust type
		if trustType != "ssh_host" && trustType != "mtls" {
			return fmt.Errorf("invalid trust type: %s (must be 'ssh_host' or 'mtls')", trustType)
		}

		// Look up source DPU
		sourceDPU, err := dpuStore.Get(sourceName)
		if err != nil {
			return fmt.Errorf("source DPU not found: %s", sourceName)
		}

		// Look up target DPU
		targetDPU, err := dpuStore.Get(targetName)
		if err != nil {
			return fmt.Errorf("target DPU not found: %s", targetName)
		}

		// Source and target must be different
		if sourceDPU.ID == targetDPU.ID {
			return fmt.Errorf("source and target must be different DPUs")
		}

		// Both DPUs must belong to a tenant
		if sourceDPU.TenantID == nil {
			return fmt.Errorf("source DPU '%s' is not assigned to a tenant", sourceName)
		}
		if targetDPU.TenantID == nil {
			return fmt.Errorf("target DPU '%s' is not assigned to a tenant", targetName)
		}

		// Both DPUs must belong to the same tenant
		if *sourceDPU.TenantID != *targetDPU.TenantID {
			return fmt.Errorf("source and target DPUs must belong to the same tenant")
		}

		tenantID := *sourceDPU.TenantID

		// Check if trust relationship already exists
		exists, err := dpuStore.TrustRelationshipExists(sourceDPU.ID, targetDPU.ID, store.TrustType(trustType))
		if err != nil {
			return fmt.Errorf("failed to check existing trust: %w", err)
		}
		if exists {
			return fmt.Errorf("trust relationship already exists between %s and %s for type %s", sourceName, targetName, trustType)
		}

		// Check attestation status for both DPUs (Phase 4: M2M Trust attestation gate)
		if err := checkAttestationForTrust(sourceName); err != nil {
			fmt.Fprintf(os.Stderr, "Hint: Run 'bluectl attestation %s' to verify device attestation\n", sourceName)
			return err
		}
		if err := checkAttestationForTrust(targetName); err != nil {
			fmt.Fprintf(os.Stderr, "Hint: Run 'bluectl attestation %s' to verify device attestation\n", targetName)
			return err
		}

		// Create trust relationship
		tr := &store.TrustRelationship{
			SourceDPUID:   sourceDPU.ID,
			SourceDPUName: sourceDPU.Name,
			TargetDPUID:   targetDPU.ID,
			TargetDPUName: targetDPU.Name,
			TenantID:      tenantID,
			TrustType:     store.TrustType(trustType),
			Bidirectional: bidirectional,
			Status:        store.TrustStatusActive,
		}

		if err := dpuStore.CreateTrustRelationship(tr); err != nil {
			return fmt.Errorf("failed to create trust relationship: %w", err)
		}

		if outputFormat != "table" {
			return formatOutput(tr)
		}

		// Format trust type for display
		typeDisplay := formatTrustType(trustType)

		fmt.Println("Trust relationship created:")
		if bidirectional {
			fmt.Printf("  %s <--> %s (%s, bidirectional)\n", sourceName, targetName, typeDisplay)
		} else {
			fmt.Printf("  %s --> %s (%s)\n", sourceName, targetName, typeDisplay)
		}
		fmt.Printf("  Status: %s\n", tr.Status)

		return nil
	},
}

var trustListCmd = &cobra.Command{
	Use:   "list",
	Short: "List trust relationships",
	Long: `List all trust relationships, optionally filtered by tenant.

Examples:
  bluectl trust list
  bluectl trust list --tenant acme
  bluectl trust list --output json`,
	RunE: func(cmd *cobra.Command, args []string) error {
		tenantFilter, _ := cmd.Flags().GetString("tenant")

		var relationships []*store.TrustRelationship
		var err error

		if tenantFilter != "" {
			// Look up tenant by name
			tenant, err := dpuStore.GetTenant(tenantFilter)
			if err != nil {
				return fmt.Errorf("tenant not found: %s", tenantFilter)
			}
			relationships, err = dpuStore.ListTrustRelationships(tenant.ID)
			if err != nil {
				return fmt.Errorf("failed to list trust relationships: %w", err)
			}
		} else {
			// List all trust relationships
			relationships, err = dpuStore.ListAllTrustRelationships()
			if err != nil {
				return fmt.Errorf("failed to list trust relationships: %w", err)
			}
		}

		if outputFormat != "table" {
			return formatOutput(relationships)
		}

		if len(relationships) == 0 {
			fmt.Println("No trust relationships found.")
			return nil
		}

		w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintln(w, "SOURCE\tTARGET\tTYPE\tDIRECTION\tSTATUS")
		for _, tr := range relationships {
			direction := "one-way"
			if tr.Bidirectional {
				direction = "bidirectional"
			}
			fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n",
				tr.SourceDPUName,
				tr.TargetDPUName,
				tr.TrustType,
				direction,
				tr.Status)
		}
		w.Flush()
		return nil
	},
}

var trustDeleteCmd = &cobra.Command{
	Use:   "delete <trust-id>",
	Short: "Delete a trust relationship",
	Long: `Delete a trust relationship by its ID.

Use 'bluectl trust list' to find the ID of the trust relationship to delete.

Examples:
  bluectl trust delete tr_a1b2c3d4`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		trustID := args[0]

		// Verify trust exists before deletion
		_, err := dpuStore.GetTrustRelationship(trustID)
		if err != nil {
			return fmt.Errorf("trust relationship not found: %s", trustID)
		}

		if err := dpuStore.DeleteTrustRelationship(trustID); err != nil {
			return fmt.Errorf("failed to delete trust relationship: %w", err)
		}

		fmt.Println("Trust relationship deleted.")
		return nil
	},
}

// formatTrustType returns a human-readable trust type string.
func formatTrustType(trustType string) string {
	switch strings.ToLower(trustType) {
	case "ssh_host":
		return "SSH host"
	case "mtls":
		return "mTLS"
	default:
		return trustType
	}
}

// checkAttestationForTrust verifies that a DPU has valid attestation for trust creation.
// Returns nil if attestation is verified and fresh, otherwise returns an error.
func checkAttestationForTrust(dpuName string) error {
	gate := attestation.NewGate(dpuStore)
	decision, err := gate.CanDistribute(dpuName)
	if err != nil {
		return fmt.Errorf("cannot create trust: '%s' attestation check failed", dpuName)
	}
	if !decision.Allowed {
		return fmt.Errorf("cannot create trust: '%s' attestation not verified", dpuName)
	}
	return nil
}
