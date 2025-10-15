#!/bin/bash
# Comprehensive verification script for OptiX, CUDA, and NVIDIA driver installation
# Can be run standalone or as part of AMI build process

# Note: We do NOT use 'set -e' so the script continues checking even if some tests fail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

print_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

print_info() {
    echo -e "  $1"
}

# Main verification
print_header "OptiX Installation Verification"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo ""

# 1. Check NVIDIA Driver
print_header "1. NVIDIA Driver"
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
    if [ -n "$DRIVER_VERSION" ]; then
        print_success "NVIDIA driver installed: version $DRIVER_VERSION"

        # Try to get GPU name
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
        if [ -n "$GPU_NAME" ]; then
            print_info "$GPU_NAME"
        fi

        # Check driver version for OptiX compatibility
        DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
        if [ "$DRIVER_MAJOR" -ge 520 ]; then
            print_success "Driver version $DRIVER_VERSION supports OptiX 8.x"
        elif [ "$DRIVER_MAJOR" -ge 450 ]; then
            print_warning "Driver version $DRIVER_VERSION supports OptiX 7.x but not 8.x"
        else
            print_fail "Driver version $DRIVER_VERSION is too old (need 450+)"
        fi
    else
        print_fail "NVIDIA driver found but failed to query version"
    fi
else
    print_fail "nvidia-smi not found"
fi

# 2. Check GPU
print_header "2. GPU Detection"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -n1)
    if [ -n "$GPU_COUNT" ] && [ "$GPU_COUNT" != "0" ]; then
        print_success "Detected $GPU_COUNT GPU(s)"

        GPU_INFO=$(nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader 2>/dev/null)
        if [ -n "$GPU_INFO" ]; then
            echo "$GPU_INFO" | while IFS=, read -r idx name compute_cap memory; do
                print_info "GPU $idx: $name (Compute $compute_cap, $memory)"
            done
        fi
    else
        print_fail "No GPUs detected by nvidia-smi"
    fi
else
    print_fail "Cannot detect GPU (nvidia-smi not available)"
fi

# 3. Check CUDA
print_header "3. CUDA Installation"

# Check nvcc
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    print_success "nvcc found: CUDA $CUDA_VERSION"
    print_info "Location: $(which nvcc)"
else
    print_fail "nvcc not found in PATH"
fi

# Check CUDA paths
if [ -d "/usr/local/cuda" ]; then
    print_success "CUDA directory exists: /usr/local/cuda"
    CUDA_LINK=$(readlink -f /usr/local/cuda || echo "/usr/local/cuda")
    print_info "Points to: $CUDA_LINK"
else
    print_warning "Standard CUDA directory /usr/local/cuda not found"
fi

# Check for specific CUDA 12.8
if [ -d "/usr/local/cuda-12.8" ]; then
    print_success "CUDA 12.8 directory exists: /usr/local/cuda-12.8"
else
    print_warning "CUDA 12.8 directory not found at /usr/local/cuda-12.8"
fi

# Check CUDA libraries
print_header "4. CUDA Libraries"
CUDA_LIBS=("libcudart.so" "libcuda.so")
for lib in "${CUDA_LIBS[@]}"; do
    if ldconfig -p 2>/dev/null | grep -q "$lib"; then
        LIB_PATH=$(ldconfig -p 2>/dev/null | grep "$lib" | head -n1 | awk '{print $NF}')
        print_success "$lib found: $LIB_PATH"
    else
        print_warning "$lib not found in ldconfig cache"
    fi
done

# 5. Check OptiX
print_header "5. OptiX SDK"

# Check OPTIX_ROOT environment variable
if [ -n "$OPTIX_ROOT" ]; then
    print_success "OPTIX_ROOT is set: $OPTIX_ROOT"
else
    print_warning "OPTIX_ROOT environment variable not set"
fi

# Find OptiX installation
OPTIX_PATHS=(
    "$OPTIX_ROOT"
    "/opt/optix"
    "$HOME/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64"
    "$HOME/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64"
    "/usr/local/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64"
    "/usr/local/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64"
)

OPTIX_FOUND=""
for path in "${OPTIX_PATHS[@]}"; do
    if [ -n "$path" ] && [ -d "$path" ]; then
        if [ -f "$path/include/optix.h" ]; then
            OPTIX_FOUND="$path"
            break
        fi
    fi
done

if [ -n "$OPTIX_FOUND" ]; then
    print_success "OptiX SDK found: $OPTIX_FOUND"

    # Check for key headers
    HEADERS=("optix.h" "optix_stubs.h" "optix_types.h")
    for header in "${HEADERS[@]}"; do
        if [ -f "$OPTIX_FOUND/include/$header" ]; then
            print_success "Header found: $header"
        else
            print_fail "Header missing: $header"
        fi
    done

    # Try to detect OptiX version
    if [ -f "$OPTIX_FOUND/include/optix.h" ]; then
        OPTIX_VERSION=$(grep "OPTIX_VERSION" "$OPTIX_FOUND/include/optix.h" | head -n1 | awk '{print $NF}')
        if [ -n "$OPTIX_VERSION" ]; then
            print_info "OptiX version constant: $OPTIX_VERSION"
        fi
    fi
else
    print_fail "OptiX SDK not found in common locations"
    print_info "Searched: ${OPTIX_PATHS[*]}"
fi

# 6. Check Environment Variables
print_header "6. Environment Variables"

# PATH
if echo "$PATH" | grep -q "cuda"; then
    print_success "CUDA found in PATH"
    echo "$PATH" | tr ':' '\n' | grep cuda | while read -r p; do
        print_info "$p"
    done
else
    print_warning "CUDA not found in PATH"
fi

# LD_LIBRARY_PATH
if [ -n "$LD_LIBRARY_PATH" ]; then
    if echo "$LD_LIBRARY_PATH" | grep -q "cuda"; then
        print_success "CUDA found in LD_LIBRARY_PATH"
        echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep cuda | while read -r p; do
            print_info "$p"
        done
    else
        print_warning "CUDA not in LD_LIBRARY_PATH"
    fi
else
    print_warning "LD_LIBRARY_PATH not set"
fi

# CUDA_HOME
if [ -n "$CUDA_HOME" ]; then
    if [ -d "$CUDA_HOME" ]; then
        print_success "CUDA_HOME is set and valid: $CUDA_HOME"
    else
        print_warning "CUDA_HOME is set but directory doesn't exist: $CUDA_HOME"
    fi
else
    print_warning "CUDA_HOME environment variable not set"
fi

# 7. Compilation Test (optional, requires g++)
print_header "7. Compilation Test"

if [ -n "$OPTIX_FOUND" ] && command -v g++ &> /dev/null; then
    TEST_FILE="/tmp/test_optix_$$.cpp"
    TEST_OBJ="/tmp/test_optix_$$.o"

    cat > "$TEST_FILE" << 'EOF'
#include <optix.h>
#include <optix_stubs.h>

int main() {
    // Just test that headers are valid
    OptixDeviceContext context = nullptr;
    (void)context;
    return 0;
}
EOF

    if g++ -I"$OPTIX_FOUND/include" -c "$TEST_FILE" -o "$TEST_OBJ" 2>/dev/null; then
        print_success "OptiX headers compile successfully"
        rm -f "$TEST_OBJ"
    else
        print_fail "Failed to compile test program with OptiX headers"
    fi
    rm -f "$TEST_FILE"
else
    if [ -z "$OPTIX_FOUND" ]; then
        print_warning "Skipping compilation test (OptiX not found)"
    elif ! command -v g++ &> /dev/null; then
        print_warning "Skipping compilation test (g++ not available)"
    fi
fi

# 8. System Info
print_header "8. System Information"
print_info "Kernel: $(uname -r)"

if [ -f /etc/os-release ]; then
    OS_NAME=$(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d '"' -f2)
    print_info "OS: ${OS_NAME:-Unknown}"
else
    print_info "OS: Unknown (no /etc/os-release)"
fi

print_info "Architecture: $(uname -m)"

if command -v free &> /dev/null; then
    TOTAL_MEM=$(free -h 2>/dev/null | grep Mem | awk '{print $2}')
    if [ -n "$TOTAL_MEM" ]; then
        print_info "Total Memory: $TOTAL_MEM"
    fi
fi

# Summary
print_header "Summary"
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
fi
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
fi

echo ""
if [ $FAILED -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ OptiX environment is correctly configured${NC}"
    exit 0
elif [ $FAILED -eq 0 ]; then
    echo -e "${YELLOW}✓ OptiX environment is functional but has some warnings${NC}"
    echo -e "${YELLOW}  Review the warnings above - they may not affect functionality${NC}"
    exit 0
else
    echo -e "${RED}✗ OptiX environment has issues that need to be resolved${NC}"
    echo ""
    echo "Common fixes:"
    echo "  • Install NVIDIA drivers: sudo ubuntu-drivers install --gpgpu"
    echo "  • Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
    echo "  • Download OptiX SDK: https://developer.nvidia.com/optix"
    echo "  • Set environment variables in ~/.bashrc:"
    echo "      export PATH=/usr/local/cuda/bin:\$PATH"
    echo "      export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    echo "      export OPTIX_ROOT=/path/to/optix"
    exit 1
fi
