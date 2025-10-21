#!/bin/bash
# Local OptiX setup script for personal laptop/workstation
# Installs NVIDIA drivers, CUDA 12.8, and configures OptiX environment
#
# Usage: ./scripts/setup-optix-local.sh [path-to-optix-installer.sh]
#
# If OptiX installer path is not provided, the script will guide you to download it

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "  $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Do not run this script as root. It will use sudo when needed."
    exit 1
fi

print_header "OptiX Local Setup for Menger Development"
echo "This script will install:"
echo "  • NVIDIA GPU drivers"
echo "  • CUDA Toolkit 12.8"
echo "  • OptiX SDK 8.0"
echo "  • Development tools (g++, make, cmake)"
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    OS_VERSION=$VERSION_ID
    print_info "Detected OS: $PRETTY_NAME"
else
    print_error "Cannot detect OS. Only Ubuntu/Debian supported currently."
    exit 1
fi

# Only support Ubuntu for now
if [ "$OS" != "ubuntu" ] && [ "$OS" != "debian" ]; then
    print_error "This script currently only supports Ubuntu/Debian."
    print_info "For other distributions, please install manually:"
    print_info "  1. NVIDIA drivers (proprietary)"
    print_info "  2. CUDA Toolkit 12.8: https://developer.nvidia.com/cuda-downloads"
    print_info "  3. OptiX SDK: https://developer.nvidia.com/optix"
    exit 1
fi

# Check for NVIDIA GPU
print_header "Checking for NVIDIA GPU"
if lspci | grep -i nvidia > /dev/null; then
    GPU_INFO=$(lspci | grep -i nvidia | head -n1)
    print_success "NVIDIA GPU detected: $GPU_INFO"
else
    print_error "No NVIDIA GPU detected. This setup requires an NVIDIA GPU."
    print_info "If you have an NVIDIA GPU but it's not detected, check BIOS settings."
    exit 1
fi

# Step 1: Install NVIDIA drivers
print_header "Step 1: Installing NVIDIA Drivers"

if command -v nvidia-smi &> /dev/null; then
    CURRENT_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")
    print_info "NVIDIA driver already installed: $CURRENT_DRIVER"

    read -p "Reinstall/update driver? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping driver installation"
    else
        INSTALL_DRIVER=1
    fi
else
    INSTALL_DRIVER=1
fi

if [ "${INSTALL_DRIVER:-0}" -eq 1 ]; then
    print_info "Installing ubuntu-drivers-common..."
    sudo apt-get update
    sudo apt-get install -y ubuntu-drivers-common

    print_info "Installing NVIDIA driver (this may take a few minutes)..."
    sudo ubuntu-drivers install --gpgpu

    print_success "NVIDIA driver installed"
    print_warning "You may need to reboot before continuing"

    read -p "Reboot now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo reboot
    else
        print_info "Please reboot and run this script again to continue"
        exit 0
    fi
fi

# Verify driver
if ! nvidia-smi &> /dev/null; then
    print_error "NVIDIA driver not working. Please reboot and try again."
    exit 1
fi

# Step 2: Install CUDA 12.8
print_header "Step 2: Installing CUDA Toolkit 12.8"

if [ -d "/usr/local/cuda-12.8" ]; then
    print_info "CUDA 12.8 already installed at /usr/local/cuda-12.8"

    read -p "Reinstall CUDA? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping CUDA installation"
        INSTALL_CUDA=0
    else
        INSTALL_CUDA=1
    fi
else
    INSTALL_CUDA=1
fi

if [ "${INSTALL_CUDA:-0}" -eq 1 ]; then
    print_info "Adding CUDA repository..."

    # Add CUDA repository for Ubuntu 24.04
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update

    print_info "Installing CUDA 12.8 (this will take several minutes)..."
    sudo apt-get install -y cuda-toolkit-12-8

    print_success "CUDA 12.8 installed"
fi

# Step 3: Install development tools
print_header "Step 3: Installing Development Tools"

print_info "Installing g++, make, cmake..."
sudo apt-get install -y build-essential cmake

print_success "Development tools installed"

# Step 4: OptiX SDK
print_header "Step 4: Installing OptiX SDK"

OPTIX_INSTALLER="${1:-}"
OPTIX_INSTALL_DIR="/usr/local/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64"

if [ -d "$OPTIX_INSTALL_DIR" ]; then
    print_info "OptiX SDK already installed at $OPTIX_INSTALL_DIR"

    read -p "Reinstall OptiX? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping OptiX installation"
        INSTALL_OPTIX=0
    else
        INSTALL_OPTIX=1
    fi
else
    INSTALL_OPTIX=1
fi

if [ "${INSTALL_OPTIX:-0}" -eq 1 ]; then
    if [ -z "$OPTIX_INSTALLER" ] || [ ! -f "$OPTIX_INSTALLER" ]; then
        print_warning "OptiX SDK installer not provided or not found"
        echo ""
        echo "To get OptiX SDK:"
        echo "  1. Visit: https://developer.nvidia.com/optix"
        echo "  2. Sign in with NVIDIA Developer account (free)"
        echo "  3. Download: OptiX SDK 8.0.0 for Linux"
        echo "  4. Run this script again with the installer path:"
        echo "     ./scripts/setup-optix-local.sh ~/Downloads/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh"
        echo ""

        read -p "Enter path to OptiX installer (or press Enter to skip): " OPTIX_INSTALLER

        if [ -z "$OPTIX_INSTALLER" ] || [ ! -f "$OPTIX_INSTALLER" ]; then
            print_warning "Skipping OptiX installation - you'll need to install manually later"
            INSTALL_OPTIX=0
        fi
    fi

    if [ "${INSTALL_OPTIX:-0}" -eq 1 ]; then
        print_info "Installing OptiX SDK to $OPTIX_INSTALL_DIR..."

        # Make installer executable
        chmod +x "$OPTIX_INSTALLER"

        # Extract to /usr/local (requires sudo)
        sudo "$OPTIX_INSTALLER" --skip-license --prefix=/usr/local

        print_success "OptiX SDK installed"
    fi
fi

# Step 5: Configure environment variables
print_header "Step 5: Configuring Environment Variables"

# Detect shell
CURRENT_SHELL=$(basename "$SHELL")
print_info "Detected shell: $CURRENT_SHELL"

# Bash/Zsh configuration
if [ "$CURRENT_SHELL" = "bash" ] || [ "$CURRENT_SHELL" = "zsh" ]; then
    if [ "$CURRENT_SHELL" = "bash" ]; then
        RC_FILE="$HOME/.bashrc"
    else
        RC_FILE="$HOME/.zshrc"
    fi

    print_info "Configuring environment in $RC_FILE..."

    # Check if already configured
    if grep -q "CUDA for Menger OptiX development" "$RC_FILE" 2>/dev/null; then
        print_info "Environment already configured in $RC_FILE"
    else
        cat >> "$RC_FILE" << 'EOF'

# CUDA for Menger OptiX development
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.8
export OPTIX_ROOT=/usr/local/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
EOF
        print_success "Environment configured in $RC_FILE"
        print_warning "Run 'source $RC_FILE' or restart your shell to apply changes"
    fi
fi

# Fish shell configuration
if [ "$CURRENT_SHELL" = "fish" ] || [ -x "$(command -v fish)" ]; then
    print_info "Generating Fish shell configuration..."

    # Call the fish config script
    FISH_SCRIPT="$(dirname "$0")/setup-optix-env.fish"
    if [ -f "$FISH_SCRIPT" ]; then
        print_info "Fish configuration available at: $FISH_SCRIPT"
        print_info "To apply: source $FISH_SCRIPT"
    else
        print_warning "Fish setup script not found. Creating it..."
        ./scripts/generate-fish-config.sh
    fi
fi

# Step 6: Verification
print_header "Step 6: Running Verification"

# Source the environment for verification
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.8
export OPTIX_ROOT=/usr/local/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64

if [ -f "$(dirname "$0")/verify-optix.sh" ]; then
    bash "$(dirname "$0")/verify-optix.sh"
else
    print_warning "Verification script not found. Skipping verification."
fi

# Final message
print_header "Setup Complete!"
echo ""
print_success "OptiX development environment is configured"
echo ""
echo "Next steps:"
echo "  1. Restart your shell or run: source ~/${CURRENT_SHELL}rc"
echo "  2. Verify installation: ./scripts/verify-optix.sh"
echo "  3. Read the OptiX implementation plan: cat OptiX.md"
echo "  4. Start development: see Phase 1 in OptiX.md"
echo ""

if [ "${INSTALL_OPTIX:-0}" -eq 0 ] && [ ! -d "$OPTIX_INSTALL_DIR" ]; then
    print_warning "OptiX SDK not installed - download and install manually"
    echo "  Download: https://developer.nvidia.com/optix"
    echo "  Run: sudo ./NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh --prefix=/usr/local"
fi