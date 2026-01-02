package cmd

import (
	"fmt"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/beyondidentity/fabric-console/pkg/store"
	"github.com/google/uuid"
	"github.com/spf13/cobra"
)

func init() {
	rootCmd.AddCommand(operatorCmd)
	operatorCmd.AddCommand(operatorInviteCmd)
	operatorCmd.AddCommand(operatorListCmd)
	operatorCmd.AddCommand(operatorSuspendCmd)
	operatorCmd.AddCommand(operatorActivateCmd)

	// Flags for operator invite
	operatorInviteCmd.Flags().String("tenant", "", "Tenant to invite operator to (required)")
	operatorInviteCmd.Flags().String("role", "operator", "Role: admin or operator")
	operatorInviteCmd.MarkFlagRequired("tenant")

	// Flags for operator list
	operatorListCmd.Flags().String("tenant", "", "Filter by tenant")
}

var operatorCmd = &cobra.Command{
	Use:   "operator",
	Short: "Manage operators",
	Long:  `Commands to invite, list, suspend, and activate operators.`,
}

var operatorInviteCmd = &cobra.Command{
	Use:   "invite <email>",
	Short: "Invite an operator to a tenant",
	Long: `Generate an invite code for an operator to join a tenant.

The invite code should be shared with the operator, who will use it
with 'km init' to bind their KeyMaker to the control plane.

Examples:
  bluectl operator invite nelson@acme.com --tenant acme
  bluectl operator invite marcus@acme.com --tenant acme --role admin`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		email := args[0]
		tenantName, _ := cmd.Flags().GetString("tenant")
		role, _ := cmd.Flags().GetString("role")

		// Validate role
		if role != "admin" && role != "operator" {
			return fmt.Errorf("invalid role: %s (must be 'admin' or 'operator')", role)
		}

		// Get tenant
		tenant, err := dpuStore.GetTenant(tenantName)
		if err != nil {
			return fmt.Errorf("tenant not found: %s", tenantName)
		}

		// Check if operator exists, create if not
		op, err := dpuStore.GetOperatorByEmail(email)
		if err != nil {
			// Create new operator with pending status
			opID := "op_" + uuid.New().String()[:8]
			if err := dpuStore.CreateOperator(opID, email, ""); err != nil {
				return fmt.Errorf("failed to create operator: %w", err)
			}
			op, _ = dpuStore.GetOperator(opID)
		}

		// Add operator to tenant if not already a member
		_, existingRoleErr := dpuStore.GetOperatorRole(op.ID, tenant.ID)
		if existingRoleErr != nil {
			// Not a member yet, add them
			if err := dpuStore.AddOperatorToTenant(op.ID, tenant.ID, role); err != nil {
				// Ignore duplicate key errors
				if !strings.Contains(err.Error(), "UNIQUE constraint") {
					return fmt.Errorf("failed to add operator to tenant: %w", err)
				}
			}
		}

		// Generate invite code
		prefix := tenant.Name
		if len(prefix) > 4 {
			prefix = prefix[:4]
		}
		code := store.GenerateInviteCode(strings.ToUpper(prefix))

		// Store invite (hashed)
		invite := &store.InviteCode{
			ID:            "inv_" + uuid.New().String()[:8],
			CodeHash:      store.HashInviteCode(code),
			OperatorEmail: email,
			TenantID:      tenant.ID,
			Role:          role,
			CreatedBy:     "admin", // TODO: get from auth context
			ExpiresAt:     time.Now().Add(24 * time.Hour),
			Status:        "pending",
		}

		if err := dpuStore.CreateInviteCode(invite); err != nil {
			return fmt.Errorf("failed to create invite: %w", err)
		}

		fmt.Printf("Invite created for %s\n", email)
		fmt.Printf("Code: %s\n", code)
		fmt.Printf("Expires: %s\n", invite.ExpiresAt.Format(time.RFC3339))
		fmt.Println()
		fmt.Println("Share this code with the operator. They will run:")
		fmt.Printf("  km init\n")

		return nil
	},
}

var operatorListCmd = &cobra.Command{
	Use:   "list",
	Short: "List operators",
	Long: `List all operators, optionally filtered by tenant.

Examples:
  bluectl operator list
  bluectl operator list --tenant acme`,
	RunE: func(cmd *cobra.Command, args []string) error {
		tenantFilter, _ := cmd.Flags().GetString("tenant")

		var operators []*store.Operator
		var err error

		if tenantFilter != "" {
			tenant, err := dpuStore.GetTenant(tenantFilter)
			if err != nil {
				return fmt.Errorf("tenant not found: %s", tenantFilter)
			}
			operators, err = dpuStore.ListOperatorsByTenant(tenant.ID)
			if err != nil {
				return err
			}
		} else {
			operators, err = dpuStore.ListOperators()
			if err != nil {
				return err
			}
		}

		if outputFormat != "table" {
			return formatOutput(operators)
		}

		if len(operators) == 0 {
			fmt.Println("No operators found.")
			return nil
		}

		w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintln(w, "EMAIL\tSTATUS\tCREATED")
		for _, op := range operators {
			fmt.Fprintf(w, "%s\t%s\t%s\n",
				op.Email, op.Status, op.CreatedAt.Format("2006-01-02"))
		}
		w.Flush()
		return nil
	},
}

var operatorSuspendCmd = &cobra.Command{
	Use:   "suspend <email>",
	Short: "Suspend an operator",
	Long: `Suspend an operator. Their KeyMakers remain bound but all
authorization checks will fail until reactivated.

Examples:
  bluectl operator suspend marcus@acme.com`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		email := args[0]

		op, err := dpuStore.GetOperatorByEmail(email)
		if err != nil {
			return fmt.Errorf("operator not found: %s", email)
		}

		if op.Status == "suspended" {
			fmt.Printf("Operator %s is already suspended.\n", email)
			return nil
		}

		if err := dpuStore.UpdateOperatorStatus(op.ID, "suspended"); err != nil {
			return err
		}

		fmt.Printf("Operator %s suspended.\n", email)
		return nil
	},
}

var operatorActivateCmd = &cobra.Command{
	Use:   "activate <email>",
	Short: "Activate a suspended operator",
	Long: `Activate an operator who was previously suspended or pending.

Examples:
  bluectl operator activate marcus@acme.com`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		email := args[0]

		op, err := dpuStore.GetOperatorByEmail(email)
		if err != nil {
			return fmt.Errorf("operator not found: %s", email)
		}

		if op.Status == "active" {
			fmt.Printf("Operator %s is already active.\n", email)
			return nil
		}

		if err := dpuStore.UpdateOperatorStatus(op.ID, "active"); err != nil {
			return err
		}

		fmt.Printf("Operator %s activated.\n", email)
		return nil
	},
}
