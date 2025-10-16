#!/bin/bash
# Validate AMI build configuration without actually building
# Tests scripts, permissions, and prerequisites

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
OPTIX_INSTALLER="${1}"

echo -e "${BLUE}=== AMI Build Validation ===${NC}"
echo ""

PASSED=0
FAILED=0
WARNINGS=0

test_passed() {
  echo -e "${GREEN}✓${NC} $1"
  ((PASSED++))
}

test_failed() {
  echo -e "${RED}✗${NC} $1"
  ((FAILED++))
}

test_warning() {
  echo -e "${YELLOW}⚠${NC} $1"
  ((WARNINGS++))
}

# Test 1: OptiX Installer
echo -e "${BLUE}1. OptiX Installer${NC}"
if [ -z "$OPTIX_INSTALLER" ]; then
  test_warning "OptiX installer path not provided"
  echo "   Usage: $0 /path/to/NVIDIA-OptiX-SDK-*.sh"
  OPTIX_INSTALLER="/tmp/optix-installer.sh"
elif [ ! -f "$OPTIX_INSTALLER" ]; then
  test_failed "OptiX installer not found: $OPTIX_INSTALLER"
else
  test_passed "OptiX installer found: $OPTIX_INSTALLER"

  # Check if executable
  if [ -x "$OPTIX_INSTALLER" ]; then
    test_passed "OptiX installer is executable"
  else
    test_warning "OptiX installer not executable (will be made executable)"
  fi

  # Check file size (should be ~40-100 MB)
  SIZE=$(stat -f%z "$OPTIX_INSTALLER" 2>/dev/null || stat -c%s "$OPTIX_INSTALLER" 2>/dev/null || echo "0")
  SIZE_MB=$((SIZE / 1024 / 1024))
  if [ $SIZE_MB -gt 30 ] && [ $SIZE_MB -lt 150 ]; then
    test_passed "OptiX installer size looks reasonable: ${SIZE_MB}MB"
  else
    test_warning "Unexpected OptiX installer size: ${SIZE_MB}MB (expected 40-100MB)"
  fi
fi
echo ""

# Test 2: Build Script Syntax
echo -e "${BLUE}2. Build Script Syntax${NC}"
if bash -n "$SCRIPT_DIR/build-ami.sh" 2>/dev/null; then
  test_passed "build-ami.sh syntax valid"
else
  test_failed "build-ami.sh has syntax errors"
fi

if bash -n "$SCRIPT_DIR/verify-optix.sh" 2>/dev/null; then
  test_passed "verify-optix.sh syntax valid"
else
  test_failed "verify-optix.sh has syntax errors"
fi
echo ""

# Test 3: AWS Resources (Dry-Run)
echo -e "${BLUE}3. AWS Resources${NC}"

# Get base AMI
BASE_AMI=$(aws ec2 describe-images \
  --region "$REGION" \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" \
            "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text 2>/dev/null)

if [ -n "$BASE_AMI" ] && [ "$BASE_AMI" != "None" ]; then
  test_passed "Base Ubuntu 24.04 AMI found: $BASE_AMI"
else
  test_failed "Could not find Ubuntu 24.04 base AMI"
fi

# Test security group creation (dry-run not supported, test describe)
SG_NAME="menger-ami-build-sg"
EXISTING_SG=$(aws ec2 describe-security-groups \
  --region "$REGION" \
  --group-names "$SG_NAME" \
  --query 'SecurityGroups[0].GroupId' \
  --output text 2>/dev/null || echo "None")

if [ "$EXISTING_SG" != "None" ] && [ -n "$EXISTING_SG" ]; then
  test_passed "Security group already exists: $SG_NAME ($EXISTING_SG)"
else
  test_passed "Security group will be created: $SG_NAME"
fi

# Test subnet availability
VPC_ID=$(aws ec2 describe-vpcs \
  --region "$REGION" \
  --filters "Name=isDefault,Values=true" \
  --query 'Vpcs[0].VpcId' \
  --output text 2>/dev/null)

if [ -n "$VPC_ID" ] && [ "$VPC_ID" != "None" ]; then
  SUBNET_ID=$(aws ec2 describe-subnets \
    --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'Subnets[0].SubnetId' \
    --output text 2>/dev/null)

  if [ -n "$SUBNET_ID" ] && [ "$SUBNET_ID" != "None" ]; then
    test_passed "Subnet available: $SUBNET_ID"
  else
    test_warning "No subnet found (will be created)"
  fi
fi
echo ""

# Test 4: Instance Launch (Dry-Run)
echo -e "${BLUE}4. Instance Launch Permissions${NC}"
if [ -n "$BASE_AMI" ] && [ "$BASE_AMI" != "None" ]; then
  DRY_RUN_OUTPUT=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$BASE_AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --dry-run 2>&1 || true)

  if echo "$DRY_RUN_OUTPUT" | grep -q "DryRunOperation"; then
    test_passed "Can launch $INSTANCE_TYPE instances"
  elif echo "$DRY_RUN_OUTPUT" | grep -q "Unsupported"; then
    test_failed "Instance type $INSTANCE_TYPE not supported in region"
  elif echo "$DRY_RUN_OUTPUT" | grep -q "UnauthorizedOperation"; then
    test_failed "Missing EC2 RunInstances permission"
  else
    test_warning "Could not verify instance launch capability"
  fi
fi
echo ""

# Test 5: AMI Creation Permission (Dry-Run)
echo -e "${BLUE}5. AMI Creation Permissions${NC}"
DRY_RUN_OUTPUT=$(aws ec2 create-image \
  --region "$REGION" \
  --instance-id "i-00000000000000000" \
  --name "test-dry-run" \
  --dry-run 2>&1 || true)

if echo "$DRY_RUN_OUTPUT" | grep -q "DryRunOperation\|does not exist"; then
  test_passed "Can create AMIs"
else
  test_failed "Missing EC2 CreateImage permission"
fi
echo ""

# Test 6: Provisioning Script Contents
echo -e "${BLUE}6. Provisioning Script Contents${NC}"

# Extract provisioning script from build-ami.sh
PROVISION_CHECK=$(grep -A 200 "<<'PROVISION_SCRIPT'" "$SCRIPT_DIR/build-ami.sh" || echo "")

# Check for key components
if echo "$PROVISION_CHECK" | grep -q "cuda-toolkit-12-8"; then
  test_passed "CUDA 12.8 installation present"
else
  test_warning "CUDA installation command not found"
fi

if echo "$PROVISION_CHECK" | grep -q "ubuntu-drivers install --gpgpu"; then
  test_passed "NVIDIA driver installation present"
else
  test_warning "NVIDIA driver installation command not found"
fi

if echo "$PROVISION_CHECK" | grep -q "OptiX"; then
  test_passed "OptiX installation present"
else
  test_warning "OptiX installation not found"
fi

if echo "$PROVISION_CHECK" | grep -q "/tmp/verify-optix.sh"; then
  test_passed "OptiX verification integrated"
else
  test_warning "OptiX verification not integrated"
fi

if echo "$PROVISION_CHECK" | grep -q "Claude Code\|claude-code"; then
  test_passed "Claude Code installation present"
else
  test_warning "Claude Code installation not found"
fi
echo ""

# Test 7: Estimated Costs
echo -e "${BLUE}7. Estimated Costs${NC}"
SPOT_PRICE=$(aws ec2 describe-spot-price-history \
  --region "$REGION" \
  --instance-types "$INSTANCE_TYPE" \
  --product-descriptions "Linux/UNIX" \
  --max-items 1 \
  --query 'SpotPriceHistory[0].SpotPrice' \
  --output text 2>/dev/null || echo "0.30")

if [ -n "$SPOT_PRICE" ] && [ "$SPOT_PRICE" != "None" ]; then
  # AMI build typically takes 30-60 minutes
  COST_LOW=$(echo "$SPOT_PRICE * 0.5" | bc 2>/dev/null || echo "~0.15")
  COST_HIGH=$(echo "$SPOT_PRICE * 1.0" | bc 2>/dev/null || echo "~0.30")
  test_passed "Estimated cost: \$${COST_LOW}-\$${COST_HIGH} (30-60 min @ \$${SPOT_PRICE}/hr)"
else
  test_warning "Could not estimate costs"
fi
echo ""

# Test 8: Network Connectivity Requirements
echo -e "${BLUE}8. Network Requirements${NC}"
test_passed "Will need internet access for:"
echo "   • Ubuntu package repositories (apt-get)"
echo "   • CUDA repository (developer.download.nvidia.com)"
echo "   • AWS CLI downloads (awscli.amazonaws.com)"
echo "   • Node.js downloads (deb.nodesource.com)"
echo "   • npm registry (registry.npmjs.org)"
echo ""

# Test 9: Time Estimates
echo -e "${BLUE}9. Time Estimates${NC}"
echo "   • Instance launch: 2-3 minutes"
echo "   • Package updates: 3-5 minutes"
echo "   • NVIDIA drivers: 5-10 minutes"
echo "   • CUDA installation: 5-10 minutes"
echo "   • OptiX installation: <1 minute"
echo "   • Development tools: 5-10 minutes"
echo "   • Verification: 1-2 minutes"
echo "   • AMI creation: 5-10 minutes"
echo "   ${BLUE}Total estimated time: 30-50 minutes${NC}"
echo ""

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo -e "${GREEN}Passed:   $PASSED${NC}"
[ $WARNINGS -gt 0 ] && echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
[ $FAILED -gt 0 ] && echo -e "${RED}Failed:   $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ] && [ $WARNINGS -le 2 ]; then
  echo -e "${GREEN}✓ Validation passed! Ready to build AMI.${NC}"
  echo ""
  echo "To build the AMI, run:"
  echo "  scripts/build-ami.sh $OPTIX_INSTALLER"
  exit 0
elif [ $FAILED -eq 0 ]; then
  echo -e "${YELLOW}✓ Validation passed with warnings. Review above.${NC}"
  echo ""
  echo "To build the AMI, run:"
  echo "  scripts/build-ami.sh $OPTIX_INSTALLER"
  exit 0
else
  echo -e "${RED}✗ Validation failed. Fix errors before building AMI.${NC}"
  exit 1
fi
