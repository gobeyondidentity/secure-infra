# Secure Infrastructure Makefile
# Build commands for all project binaries

# Output directory
BIN_DIR := bin

# Binary names
AGENT := $(BIN_DIR)/agent
AGENT_ARM64 := $(BIN_DIR)/agent-arm64
BLUECTL := $(BIN_DIR)/bluectl
SERVER := $(BIN_DIR)/server
KM := $(BIN_DIR)/km
HOST_AGENT := $(BIN_DIR)/host-agent
DPUEMU := $(BIN_DIR)/dpuemu

.PHONY: all agent bluectl server km host-agent dpuemu test clean release help

# Default target: build all binaries
all: $(BIN_DIR)
	@echo "Building all binaries..."
	@go build -o $(AGENT) ./cmd/agent
	@echo "  $(AGENT)"
	@go build -o $(BLUECTL) ./cmd/bluectl
	@echo "  $(BLUECTL)"
	@go build -o $(KM) ./cmd/keymaker
	@echo "  $(KM)"
	@go build -o $(SERVER) ./cmd/server
	@echo "  $(SERVER)"
	@go build -o $(HOST_AGENT) ./cmd/host-agent
	@echo "  $(HOST_AGENT)"
	@go build -o $(DPUEMU) ./dpuemu/cmd/dpuemu
	@echo "  $(DPUEMU)"
	@echo "Done."

# Create bin directory if needed
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Build agent for local platform and cross-compile for BlueField (ARM64)
agent: $(BIN_DIR)
	@echo "Building agent..."
	@go build -o $(AGENT) ./cmd/agent
	@echo "  $(AGENT)"
	@echo "Cross-compiling agent for BlueField (linux/arm64)..."
	@GOOS=linux GOARCH=arm64 go build -o $(AGENT_ARM64) ./cmd/agent
	@echo "  $(AGENT_ARM64)"

# Build bluectl CLI
bluectl: $(BIN_DIR)
	@echo "Building bluectl..."
	@go build -o $(BLUECTL) ./cmd/bluectl
	@echo "  $(BLUECTL)"

# Build server
server: $(BIN_DIR)
	@echo "Building server..."
	@go build -o $(SERVER) ./cmd/server
	@echo "  $(SERVER)"

# Build keymaker CLI
km: $(BIN_DIR)
	@echo "Building km (keymaker)..."
	@go build -o $(KM) ./cmd/keymaker
	@echo "  $(KM)"

# Build host-agent
host-agent: $(BIN_DIR)
	@echo "Building host-agent..."
	@go build -o $(HOST_AGENT) ./cmd/host-agent
	@echo "  $(HOST_AGENT)"

# Build DPU emulator
dpuemu: $(BIN_DIR)
	@echo "Building dpuemu..."
	@go build -o $(DPUEMU) ./dpuemu/cmd/dpuemu
	@echo "  $(DPUEMU)"

# Run all tests
test:
	@echo "Running tests..."
	@go test ./...

# Remove bin directory contents
clean:
	@echo "Cleaning bin/..."
	@rm -rf $(BIN_DIR)/*
	@echo "Done."

# Build release binaries for multiple platforms
release: $(BIN_DIR)
	@echo "Building release binaries..."
	@echo ""
	@echo "darwin/arm64:"
	@GOOS=darwin GOARCH=arm64 go build -o $(BIN_DIR)/agent-darwin-arm64 ./cmd/agent
	@echo "  $(BIN_DIR)/agent-darwin-arm64"
	@GOOS=darwin GOARCH=arm64 go build -o $(BIN_DIR)/bluectl-darwin-arm64 ./cmd/bluectl
	@echo "  $(BIN_DIR)/bluectl-darwin-arm64"
	@GOOS=darwin GOARCH=arm64 go build -o $(BIN_DIR)/km-darwin-arm64 ./cmd/keymaker
	@echo "  $(BIN_DIR)/km-darwin-arm64"
	@echo ""
	@echo "linux/amd64:"
	@GOOS=linux GOARCH=amd64 go build -o $(BIN_DIR)/agent-linux-amd64 ./cmd/agent
	@echo "  $(BIN_DIR)/agent-linux-amd64"
	@GOOS=linux GOARCH=amd64 go build -o $(BIN_DIR)/bluectl-linux-amd64 ./cmd/bluectl
	@echo "  $(BIN_DIR)/bluectl-linux-amd64"
	@GOOS=linux GOARCH=amd64 go build -o $(BIN_DIR)/km-linux-amd64 ./cmd/keymaker
	@echo "  $(BIN_DIR)/km-linux-amd64"
	@GOOS=linux GOARCH=amd64 go build -o $(BIN_DIR)/host-agent-linux-amd64 ./cmd/host-agent
	@echo "  $(BIN_DIR)/host-agent-linux-amd64"
	@echo ""
	@echo "linux/arm64 (BlueField DPU):"
	@GOOS=linux GOARCH=arm64 go build -o $(BIN_DIR)/agent-linux-arm64 ./cmd/agent
	@echo "  $(BIN_DIR)/agent-linux-arm64"
	@GOOS=linux GOARCH=arm64 go build -o $(BIN_DIR)/bluectl-linux-arm64 ./cmd/bluectl
	@echo "  $(BIN_DIR)/bluectl-linux-arm64"
	@GOOS=linux GOARCH=arm64 go build -o $(BIN_DIR)/km-linux-arm64 ./cmd/keymaker
	@echo "  $(BIN_DIR)/km-linux-arm64"
	@echo ""
	@echo "Release build complete."

# =============================================================================
# Quickstart Demo Targets
# Run each step of the emulator quickstart guide
# =============================================================================

.PHONY: demo-clean demo-step1 demo-step2 demo-step3 demo-step4 demo-step5 demo-step6 demo-step7 demo-step8 demo-step9

# Reset environment for fresh start
demo-clean:
	@echo "=== Clean Slate ==="
	@rm -f ~/.local/share/bluectl/dpus.db
	@rm -f ~/.local/share/bluectl/key
	@rm -rf ~/.km
	@echo "Database and config removed. Ready for fresh start."

# Step 1: Start the server (runs in foreground)
demo-step1:
	@echo "=== Step 1: Start Server ==="
	@echo "Starting server on :18080..."
	@echo "Press Ctrl+C to stop"
	$(BIN_DIR)/server

# Step 2: Create tenant
demo-step2:
	@echo "=== Step 2: Create Tenant ==="
	$(BIN_DIR)/bluectl tenant add gpu-prod --description "GPU Production Cluster"

# Step 3: Start emulator (runs in foreground)
demo-step3:
	@echo "=== Step 3: Start DPU Emulator ==="
	@echo "Starting emulator on :50051..."
	@echo "Press Ctrl+C to stop"
	$(BIN_DIR)/dpuemu serve --port 50051 --fixture dpuemu/fixtures/bf3-static.json

# Step 4: Register DPU
demo-step4:
	@echo "=== Step 4: Register DPU ==="
	$(BIN_DIR)/bluectl dpu add localhost --name bf3
	$(BIN_DIR)/bluectl tenant assign gpu-prod bf3

# Step 5: Create operator (interactive for invite code)
demo-step5:
	@echo "=== Step 5: Create Operator ==="
	@echo "Creating invitation..."
	@$(BIN_DIR)/bluectl operator invite operator@example.com gpu-prod
	@echo ""
	@echo "Now run: bin/km init"
	@echo "Enter the invite code when prompted"

demo-step5-accept:
	@echo "=== Step 5b: Accept Invitation ==="
	$(BIN_DIR)/km init

demo-step5-verify:
	@echo "=== Step 5: Verify Operator ==="
	$(BIN_DIR)/km whoami

# Step 6: Create SSH CA and grant access
demo-step6:
	@echo "=== Step 6: Create SSH CA ==="
	$(BIN_DIR)/km ssh-ca create test-ca

demo-step6-grant:
	@echo "=== Step 6b: Grant Access ==="
	$(BIN_DIR)/bluectl operator grant operator@example.com gpu-prod test-ca bf3

# Step 7: Submit attestation
demo-step7:
	@echo "=== Step 7: Attestation ==="
	$(BIN_DIR)/bluectl attestation bf3

# Step 8: Push credentials
demo-step8:
	@echo "=== Step 8: Push Credentials ==="
	$(BIN_DIR)/km push ssh-ca test-ca bf3

# Step 9: Test host agent (optional)
demo-step9:
	@echo "=== Step 9: Host Agent ==="
	$(BIN_DIR)/host-agent --dpu-agent http://localhost:9443 --oneshot

# =============================================================================
# Help
# =============================================================================

# Show available targets
help:
	@echo "Secure Infrastructure Build Targets:"
	@echo ""
	@echo "  make all        Build all binaries (default)"
	@echo "  make agent      Build agent (local + ARM64 for BlueField)"
	@echo "  make bluectl    Build bluectl CLI"
	@echo "  make server     Build server"
	@echo "  make km         Build keymaker CLI"
	@echo "  make host-agent Build host-agent"
	@echo "  make dpuemu     Build DPU emulator"
	@echo "  make test       Run all tests"
	@echo "  make clean      Remove bin/ contents"
	@echo "  make release    Build release binaries for all platforms"
	@echo ""
	@echo "Quickstart Demo Targets:"
	@echo ""
	@echo "  make demo-clean       Reset environment (clean slate)"
	@echo "  make demo-step1       Start server (Terminal 1)"
	@echo "  make demo-step2       Create tenant"
	@echo "  make demo-step3       Start emulator (Terminal 2)"
	@echo "  make demo-step4       Register DPU"
	@echo "  make demo-step5       Create operator invitation"
	@echo "  make demo-step5-accept Accept invitation (km init)"
	@echo "  make demo-step5-verify Verify operator (km whoami)"
	@echo "  make demo-step6       Create SSH CA"
	@echo "  make demo-step6-grant Grant CA access"
	@echo "  make demo-step7       Submit attestation"
	@echo "  make demo-step8       Push credentials"
	@echo "  make demo-step9       Test host agent (optional)"
	@echo ""
	@echo "  make help       Show this help"
