#!/usr/bin/env fish
# Fish shell environment configuration for OptiX development
#
# Usage:
#   source scripts/setup-optix-env.fish
#
# To make permanent, add to ~/.config/fish/config.fish:
#   source /path/to/menger/scripts/setup-optix-env.fish

# CUDA 12.8 paths
set -gx PATH /usr/local/cuda-12.8/bin $PATH
set -gx LD_LIBRARY_PATH /usr/local/cuda-12.8/lib64 $LD_LIBRARY_PATH

# CUDA home directory
set -gx CUDA_HOME /usr/local/cuda-12.8

# OptiX SDK location
set -gx OPTIX_ROOT /usr/local/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64

# Optional: Add color-coded status message
if test -n "$PS1"
    set_color green
    echo "✓ OptiX environment configured for Fish shell"
    set_color normal
    echo "  CUDA_HOME: $CUDA_HOME"
    echo "  OPTIX_ROOT: $OPTIX_ROOT"
end

# Verification function for Fish
function verify-optix-env
    echo "Checking OptiX environment..."

    # Check nvcc
    if command -v nvcc > /dev/null
        set_color green
        echo "✓ nvcc found: "(nvcc --version | grep release)
        set_color normal
    else
        set_color red
        echo "✗ nvcc not found in PATH"
        set_color normal
    end

    # Check nvidia-smi
    if command -v nvidia-smi > /dev/null
        set_color green
        echo "✓ nvidia-smi found: driver version "(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)
        set_color normal
    else
        set_color red
        echo "✗ nvidia-smi not found"
        set_color normal
    end

    # Check OptiX headers
    if test -f "$OPTIX_ROOT/include/optix.h"
        set_color green
        echo "✓ OptiX headers found at $OPTIX_ROOT"
        set_color normal
    else
        set_color red
        echo "✗ OptiX headers not found at $OPTIX_ROOT"
        set_color normal
    end

    echo ""
    echo "For full verification, run: ./scripts/verify-optix.sh"
end
