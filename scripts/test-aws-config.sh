#!/bin/bash
# Test AWS configuration and permissions without actually creating resources
# Uses AWS dry-run mode to validate configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
REGION="${AWS_REGION:-us-east-1}"
AVAILABILITY_ZONE=""
INSTANCE_TYPE="g4dn.xlarge"
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --region)
      REGION="$2"
      shift 2
      ;;
    --availability-zone)
      AVAILABILITY_ZONE="$2"
      REGION="${AVAILABILITY_ZONE%?}"
      shift 2
      ;;
    --instance-type)
      INSTANCE_TYPE="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    -h|--help)
      cat <<EOF
Usage: $0 [OPTIONS]

Test AWS configuration without creating resources.

OPTIONS:
  --region REGION          AWS region (default: us-east-1)
  --availability-zone AZ   AWS availability zone
  --instance-type TYPE     Instance type to test (default: g4dn.xlarge)
  --verbose                Show detailed output
  -h, --help               Show this help message

EXAMPLES:
  # Test default configuration
  $0

  # Test specific region and instance type
  $0 --region us-west-2 --instance-type g5.xlarge

  # Test with verbose output
  $0 --verbose

EOF
      exit 0
      ;;
    *)
      echo -e "${RED}Error: Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

echo -e "${BLUE}=== AWS Configuration Test ===${NC}"
echo "Region:        $REGION"
[ -n "$AVAILABILITY_ZONE" ] && echo "AZ:            $AVAILABILITY_ZONE"
echo "Instance Type: $INSTANCE_TYPE"
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

# Test 1: AWS CLI Installation
echo -e "${BLUE}1. AWS CLI${NC}"
if command -v aws &> /dev/null; then
  AWS_VERSION=$(aws --version 2>&1 | cut -d' ' -f1)
  test_passed "AWS CLI installed: $AWS_VERSION"
else
  test_failed "AWS CLI not found"
fi
echo ""

# Test 2: AWS Credentials
echo -e "${BLUE}2. AWS Credentials${NC}"
if aws sts get-caller-identity &> /dev/null; then
  ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
  USER_ARN=$(aws sts get-caller-identity --query Arn --output text 2>/dev/null)
  test_passed "Credentials valid: $USER_ARN"
  [ "$VERBOSE" = true ] && echo "   Account ID: $ACCOUNT_ID"
else
  test_failed "AWS credentials not configured or invalid"
  echo "   Run: aws configure"
fi
echo ""

# Test 3: Region Accessibility
echo -e "${BLUE}3. Region Accessibility${NC}"
if aws ec2 describe-regions --region "$REGION" --region-names "$REGION" &> /dev/null; then
  test_passed "Region accessible: $REGION"
else
  test_failed "Cannot access region: $REGION"
fi
echo ""

# Test 4: Default VPC
echo -e "${BLUE}4. Default VPC${NC}"
VPC_ID=$(aws ec2 describe-vpcs \
  --region "$REGION" \
  --filters "Name=isDefault,Values=true" \
  --query 'Vpcs[0].VpcId' \
  --output text 2>/dev/null)

if [ "$VPC_ID" != "None" ] && [ -n "$VPC_ID" ]; then
  test_passed "Default VPC found: $VPC_ID"

  # Check for subnets
  SUBNET_COUNT=$(aws ec2 describe-subnets \
    --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'Subnets | length(@)' \
    --output text 2>/dev/null || echo "0")

  if [ "$SUBNET_COUNT" -gt 0 ]; then
    test_passed "Found $SUBNET_COUNT subnet(s) in default VPC"
  else
    test_warning "No subnets in default VPC (will be created automatically)"
  fi
else
  test_warning "No default VPC (one will be used or created)"
fi
echo ""

# Test 5: Instance Type Availability
echo -e "${BLUE}5. Instance Type Availability${NC}"
INSTANCE_AVAILABLE=$(aws ec2 describe-instance-type-offerings \
  --region "$REGION" \
  --location-type "availability-zone" \
  --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
  --query 'InstanceTypeOfferings[0].InstanceType' \
  --output text 2>/dev/null)

if [ "$INSTANCE_AVAILABLE" = "$INSTANCE_TYPE" ]; then
  test_passed "Instance type available: $INSTANCE_TYPE"

  # Show availability zones
  if [ "$VERBOSE" = true ]; then
    AZS=$(aws ec2 describe-instance-type-offerings \
      --region "$REGION" \
      --location-type "availability-zone" \
      --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
      --query 'InstanceTypeOfferings[*].Location' \
      --output text 2>/dev/null)
    echo "   Available in AZs: $AZS"
  fi
else
  test_failed "Instance type not available in region: $INSTANCE_TYPE"
  echo "   Run: scripts/nvidia-spot.sh --list-instances --region $REGION"
fi
echo ""

# Test 6: Spot Instance Pricing
echo -e "${BLUE}6. Spot Instance Pricing${NC}"
SPOT_PRICE=$(aws ec2 describe-spot-price-history \
  --region "$REGION" \
  --instance-types "$INSTANCE_TYPE" \
  --product-descriptions "Linux/UNIX" \
  --max-items 1 \
  --query 'SpotPriceHistory[0].SpotPrice' \
  --output text 2>/dev/null)

if [ -n "$SPOT_PRICE" ] && [ "$SPOT_PRICE" != "None" ]; then
  test_passed "Current spot price: \$$SPOT_PRICE/hour"
else
  test_warning "Could not retrieve spot price"
fi
echo ""

# Test 7: SSH Key
echo -e "${BLUE}7. SSH Key${NC}"
SSH_KEY="${HOME}/.ssh/id_rsa.pub"
if [ -f "$SSH_KEY" ]; then
  test_passed "SSH public key found: $SSH_KEY"
else
  test_warning "SSH public key not found: $SSH_KEY"
  echo "   Generate with: ssh-keygen -t rsa -b 4096"
fi
echo ""

# Test 8: IAM Permissions (Dry-Run Tests)
echo -e "${BLUE}8. IAM Permissions (Dry-Run)${NC}"

# Get a base AMI for testing
BASE_AMI=$(aws ec2 describe-images \
  --region "$REGION" \
  --owners amazon \
  --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
            "Name=state,Values=available" \
  --query 'Images[0].ImageId' \
  --output text 2>/dev/null)

if [ -z "$BASE_AMI" ] || [ "$BASE_AMI" = "None" ]; then
  test_warning "Could not find base AMI for testing"
else
  # Test EC2 RunInstances permission
  if aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$BASE_AMI" \
    --instance-type "t3.micro" \
    --dry-run 2>&1 | grep -q "DryRunOperation"; then
    test_passed "EC2 RunInstances permission"
  else
    test_failed "Missing EC2 RunInstances permission"
  fi

  # Test EC2 CreateImage permission
  if aws ec2 create-image \
    --region "$REGION" \
    --instance-id "i-00000000000000000" \
    --name "test-dry-run" \
    --dry-run 2>&1 | grep -q "DryRunOperation\|does not exist"; then
    test_passed "EC2 CreateImage permission"
  else
    test_failed "Missing EC2 CreateImage permission"
  fi

  # Test EC2 CreateSnapshot permission
  if aws ec2 create-snapshot \
    --region "$REGION" \
    --volume-id "vol-00000000000000000" \
    --dry-run 2>&1 | grep -q "DryRunOperation\|does not exist"; then
    test_passed "EC2 CreateSnapshot permission"
  else
    test_failed "Missing EC2 CreateSnapshot permission"
  fi

  # Test EC2 TerminateInstances permission
  if aws ec2 terminate-instances \
    --region "$REGION" \
    --instance-ids "i-00000000000000000" \
    --dry-run 2>&1 | grep -q "DryRunOperation\|does not exist"; then
    test_passed "EC2 TerminateInstances permission"
  else
    test_failed "Missing EC2 TerminateInstances permission"
  fi
fi
echo ""

# Test 9: Terraform
echo -e "${BLUE}9. Terraform${NC}"
if command -v terraform &> /dev/null; then
  TERRAFORM_VERSION=$(terraform version -json 2>/dev/null | jq -r .terraform_version 2>/dev/null || terraform version | head -1)
  test_passed "Terraform installed: $TERRAFORM_VERSION"

  # Test Terraform configuration
  TERRAFORM_DIR="$PROJECT_ROOT/terraform"
  if [ -f "$TERRAFORM_DIR/main.tf" ]; then
    cd "$TERRAFORM_DIR"
    if terraform init -backend=false > /dev/null 2>&1; then
      test_passed "Terraform configuration valid"
    else
      test_warning "Terraform initialization failed"
    fi
  fi
else
  test_failed "Terraform not found"
fi
echo ""

# Test 10: Required Tools
echo -e "${BLUE}10. Required Tools${NC}"
TOOLS=("jq" "rsync" "ssh" "scp")
for tool in "${TOOLS[@]}"; do
  if command -v "$tool" &> /dev/null; then
    test_passed "$tool installed"
  else
    test_warning "$tool not found (required for state management)"
  fi
done
echo ""

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo -e "${GREEN}Passed:   $PASSED${NC}"
[ $WARNINGS -gt 0 ] && echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
[ $FAILED -gt 0 ] && echo -e "${RED}Failed:   $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ] && [ $WARNINGS -eq 0 ]; then
  echo -e "${GREEN}✓ All tests passed! Ready to create resources.${NC}"
  exit 0
elif [ $FAILED -eq 0 ]; then
  echo -e "${YELLOW}✓ Tests passed with warnings. Review warnings above.${NC}"
  exit 0
else
  echo -e "${RED}✗ Some tests failed. Fix errors before proceeding.${NC}"
  echo ""
  echo "Common fixes:"
  echo "  • Configure AWS credentials: aws configure"
  echo "  • Install missing tools: sudo apt-get install jq rsync"
  echo "  • Check IAM permissions for EC2 operations"
  exit 1
fi
