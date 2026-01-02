#!/bin/bash
# Secure Infrastructure DPU Agent Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/nmelo/secure-infra/main/eng/deploy/install-agent.sh | sudo bash
#
# Environment variables:
#   CONTROL_PLANE_URL - Control plane URL (will prompt if not set)
#   VERSION           - Specific version to install (default: latest)

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

VERSION="${VERSION:-latest}"
GITHUB_REPO="nmelo/secure-infra"
BINARY_NAME="fabric-agent"
INSTALL_DIR="/usr/local/bin"
SERVICE_NAME="fabric-agent"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# =============================================================================
# Helper Functions
# =============================================================================

info() {
    echo "[INFO] $*"
}

warn() {
    echo "[WARN] $*" >&2
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root. Use: sudo bash install-agent.sh"
    fi
}

# Detect system architecture
detect_arch() {
    local arch
    arch=$(uname -m)

    case "$arch" in
        aarch64|arm64)
            echo "arm64"
            ;;
        x86_64|amd64)
            warn "Detected x86_64 architecture. DPU agent is designed for BlueField (arm64)."
            warn "Are you sure you want to install on this architecture?"
            read -r -p "Continue? [y/N] " response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                error "Installation cancelled."
            fi
            echo "amd64"
            ;;
        *)
            error "Unsupported architecture: $arch. BlueField DPUs use arm64."
            ;;
    esac
}

# Find available download tool
get_download_cmd() {
    if command -v curl &>/dev/null; then
        echo "curl"
    elif command -v wget &>/dev/null; then
        echo "wget"
    else
        error "Neither curl nor wget found. Please install one of them."
    fi
}

# Download file using available tool
download_file() {
    local url="$1"
    local output="$2"
    local downloader
    downloader=$(get_download_cmd)

    info "Downloading from: $url"

    if [[ "$downloader" == "curl" ]]; then
        curl -fsSL "$url" -o "$output"
    else
        wget -q "$url" -O "$output"
    fi
}

# Get the latest release version from GitHub
get_latest_version() {
    local downloader
    downloader=$(get_download_cmd)
    local api_url="https://api.github.com/repos/${GITHUB_REPO}/releases/latest"

    if [[ "$downloader" == "curl" ]]; then
        curl -fsSL "$api_url" | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/'
    else
        wget -qO- "$api_url" | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/'
    fi
}

# Prompt for control plane URL if not provided
get_control_plane_url() {
    if [[ -n "${CONTROL_PLANE_URL:-}" ]]; then
        echo "$CONTROL_PLANE_URL"
        return
    fi

    echo ""
    read -r -p "Enter Control Plane URL (e.g., https://control.example.com:8443): " url

    if [[ -z "$url" ]]; then
        error "Control Plane URL is required."
    fi

    echo "$url"
}

# =============================================================================
# Main Installation
# =============================================================================

main() {
    echo "=============================================="
    echo "  Secure Infrastructure DPU Agent Installer"
    echo "=============================================="
    echo ""

    # Pre-flight checks
    check_root

    info "Detecting architecture..."
    ARCH=$(detect_arch)
    info "Architecture: $ARCH"

    # Determine version to install
    if [[ "$VERSION" == "latest" ]]; then
        info "Fetching latest release version..."
        VERSION=$(get_latest_version)
        if [[ -z "$VERSION" ]]; then
            error "Could not determine latest version. Set VERSION env var manually."
        fi
    fi
    info "Installing version: $VERSION"

    # Get control plane URL
    CONTROL_PLANE_URL=$(get_control_plane_url)
    info "Control Plane URL: $CONTROL_PLANE_URL"

    # Construct download URL
    # Expected release asset name: fabric-agent-linux-arm64
    DOWNLOAD_URL="https://github.com/${GITHUB_REPO}/releases/download/${VERSION}/${BINARY_NAME}-linux-${ARCH}"

    # Download binary
    info "Downloading agent binary..."
    TEMP_FILE=$(mktemp)
    trap 'rm -f "$TEMP_FILE"' EXIT

    if ! download_file "$DOWNLOAD_URL" "$TEMP_FILE"; then
        error "Failed to download binary from: $DOWNLOAD_URL"
    fi

    # Validate download
    if [[ ! -s "$TEMP_FILE" ]]; then
        error "Downloaded file is empty or does not exist."
    fi

    # Install binary
    info "Installing binary to ${INSTALL_DIR}/${BINARY_NAME}..."
    install -m 755 "$TEMP_FILE" "${INSTALL_DIR}/${BINARY_NAME}"

    # Verify installation
    if [[ ! -x "${INSTALL_DIR}/${BINARY_NAME}" ]]; then
        error "Binary installation failed or file is not executable."
    fi
    info "Binary installed successfully."

    # Create systemd service file
    info "Creating systemd service..."
    cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Secure Infrastructure DPU Agent
Documentation=https://github.com/${GITHUB_REPO}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
ExecStart=${INSTALL_DIR}/${BINARY_NAME} --control-plane ${CONTROL_PLANE_URL}
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=false
ProtectSystem=full
ProtectHome=read-only

[Install]
WantedBy=multi-user.target
EOF

    info "Service file created at: $SERVICE_FILE"

    # Reload systemd and enable service
    info "Enabling and starting service..."
    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"
    systemctl start "$SERVICE_NAME"

    # Check service status
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo ""
        echo "=============================================="
        echo "  Installation Complete!"
        echo "=============================================="
        echo ""
        echo "The DPU agent is now running."
        echo ""
        echo "Useful commands:"
        echo "  Check status:    systemctl status ${SERVICE_NAME}"
        echo "  View logs:       journalctl -u ${SERVICE_NAME} -f"
        echo "  Restart:         systemctl restart ${SERVICE_NAME}"
        echo "  Stop:            systemctl stop ${SERVICE_NAME}"
        echo ""
    else
        warn "Service installed but may not be running correctly."
        warn "Check status with: systemctl status ${SERVICE_NAME}"
        warn "View logs with: journalctl -u ${SERVICE_NAME} -e"
        exit 1
    fi
}

main "$@"
