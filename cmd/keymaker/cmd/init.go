package cmd

import (
	"bufio"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/google/uuid"
	"github.com/spf13/cobra"
	"golang.org/x/crypto/ssh"
)

func init() {
	rootCmd.AddCommand(initCmd)
	rootCmd.AddCommand(whoamiCmd)

	initCmd.Flags().String("name", "", "Custom name for this KeyMaker")
	initCmd.Flags().String("control-plane", "http://localhost:8080", "Control plane URL")
	initCmd.Flags().String("invite-code", "", "Invite code (will prompt if not provided)")
}

// KMConfig is stored in ~/.km/config.json
type KMConfig struct {
	KeyMakerID      string `json:"keymaker_id"`
	OperatorID      string `json:"operator_id"`
	OperatorEmail   string `json:"operator_email"`
	ControlPlaneURL string `json:"control_plane_url"`
	PrivateKeyPath  string `json:"private_key_path"`
}

var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Initialize and bind this KeyMaker to the control plane",
	Long: `Initialize a new KeyMaker by binding it to the control plane.

You will need an invite code from your administrator. The code is
generated with 'bluectl operator invite'.

This command:
1. Generates a new ed25519 keypair
2. Binds the public key to the control plane using your invite code
3. Stores configuration in ~/.km/config.json

Examples:
  km init
  km init --name workstation-home
  km init --control-plane https://fabric.acme.com`,
	RunE: runInit,
}

func runInit(cmd *cobra.Command, args []string) error {
	name, _ := cmd.Flags().GetString("name")
	controlPlane, _ := cmd.Flags().GetString("control-plane")
	inviteCode, _ := cmd.Flags().GetString("invite-code")

	// Check if already initialized
	configPath := getConfigPath()
	if _, err := os.Stat(configPath); err == nil {
		return fmt.Errorf("KeyMaker already initialized. Remove %s to re-initialize", configPath)
	}

	// Prompt for invite code if not provided
	if inviteCode == "" {
		fmt.Print("Enter invite code: ")
		reader := bufio.NewReader(os.Stdin)
		code, err := reader.ReadString('\n')
		if err != nil {
			return fmt.Errorf("failed to read invite code: %w", err)
		}
		inviteCode = strings.TrimSpace(code)
	}

	if inviteCode == "" {
		return fmt.Errorf("invite code is required")
	}

	// Generate keypair
	fmt.Println("Generating keypair...")
	pubKey, privKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return fmt.Errorf("failed to generate keypair: %w", err)
	}

	// Convert to SSH format
	sshPubKey, err := ssh.NewPublicKey(pubKey)
	if err != nil {
		return fmt.Errorf("failed to convert public key: %w", err)
	}
	sshPubKeyStr := strings.TrimSpace(string(ssh.MarshalAuthorizedKey(sshPubKey)))

	// Auto-generate name if not provided
	if name == "" {
		name = fmt.Sprintf("km-%s-%s", runtime.GOOS, uuid.New().String()[:4])
	}

	// Build bind request
	bindReq := map[string]string{
		"invite_code":        inviteCode,
		"public_key":         sshPubKeyStr,
		"platform":           runtime.GOOS,
		"secure_element":     "software",
		"device_fingerprint": uuid.New().String(),
		"device_name":        name,
	}

	// POST to control plane
	fmt.Printf("Binding to %s...\n", controlPlane)
	reqBody, _ := json.Marshal(bindReq)
	resp, err := http.Post(
		controlPlane+"/api/v1/keymakers/bind",
		"application/json",
		strings.NewReader(string(reqBody)),
	)
	if err != nil {
		return fmt.Errorf("failed to connect to control plane: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		var errResp struct {
			Error string `json:"error"`
		}
		json.NewDecoder(resp.Body).Decode(&errResp)
		if errResp.Error != "" {
			return fmt.Errorf("binding failed: %s", errResp.Error)
		}
		return fmt.Errorf("binding failed: HTTP %d", resp.StatusCode)
	}

	// Parse response
	var bindResp struct {
		KeyMakerID    string `json:"keymaker_id"`
		OperatorID    string `json:"operator_id"`
		OperatorEmail string `json:"operator_email"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&bindResp); err != nil {
		return fmt.Errorf("failed to parse response: %w", err)
	}

	// Save private key
	keyPath := getPrivateKeyPath()
	if err := os.MkdirAll(filepath.Dir(keyPath), 0700); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	privKeyPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "OPENSSH PRIVATE KEY",
		Bytes: privKey,
	})
	if err := os.WriteFile(keyPath, privKeyPEM, 0600); err != nil {
		return fmt.Errorf("failed to save private key: %w", err)
	}

	// Save config
	config := KMConfig{
		KeyMakerID:      bindResp.KeyMakerID,
		OperatorID:      bindResp.OperatorID,
		OperatorEmail:   bindResp.OperatorEmail,
		ControlPlaneURL: controlPlane,
		PrivateKeyPath:  keyPath,
	}

	configData, _ := json.MarshalIndent(config, "", "  ")
	if err := os.WriteFile(configPath, configData, 0600); err != nil {
		return fmt.Errorf("failed to save config: %w", err)
	}

	fmt.Printf("\nKeyMaker bound successfully!\n")
	fmt.Printf("  KeyMaker ID: %s\n", bindResp.KeyMakerID)
	fmt.Printf("  Operator:    %s\n", bindResp.OperatorEmail)
	fmt.Printf("  Config:      %s\n", configPath)

	return nil
}

var whoamiCmd = &cobra.Command{
	Use:   "whoami",
	Short: "Show current KeyMaker identity",
	RunE: func(cmd *cobra.Command, args []string) error {
		config, err := loadConfig()
		if err != nil {
			return fmt.Errorf("KeyMaker not initialized. Run 'km init' first")
		}

		if outputFormat != "table" {
			return formatOutput(config)
		}

		fmt.Printf("KeyMaker ID:    %s\n", config.KeyMakerID)
		fmt.Printf("Operator ID:    %s\n", config.OperatorID)
		fmt.Printf("Operator Email: %s\n", config.OperatorEmail)
		fmt.Printf("Control Plane:  %s\n", config.ControlPlaneURL)
		fmt.Printf("Private Key:    %s\n", config.PrivateKeyPath)

		return nil
	},
}

func getConfigDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".km")
}

func getConfigPath() string {
	return filepath.Join(getConfigDir(), "config.json")
}

func getPrivateKeyPath() string {
	return filepath.Join(getConfigDir(), "keymaker.pem")
}

func loadConfig() (*KMConfig, error) {
	data, err := os.ReadFile(getConfigPath())
	if err != nil {
		return nil, err
	}

	var config KMConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	return &config, nil
}
