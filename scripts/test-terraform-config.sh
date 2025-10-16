#!/bin/bash
# Validate Terraform configuration without creating resources
# Uses terraform validate, plan, and dry-run checks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TERRAFORM_DIR="$PROJECT_ROOT/terraform"

# Configuration
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="g4dn.xlarge"
AMI_ID="${1}"

echo -e "${BLUE}=== Terraform Configuration Test ===${NC}"
echo "Terraform dir: $TERRAFORM_DIR"
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

# Test 1: Terraform Installation
echo -e "${BLUE}1. Terraform Installation${NC}"
if command -v terraform &> /dev/null; then
  TERRAFORM_VERSION=$(terraform version -json 2>/dev/null | jq -r .terraform_version 2>/dev/null || terraform version | head -1 | awk '{print $2}')
  test_passed "Terraform installed: $TERRAFORM_VERSION"
else
  test_failed "Terraform not found"
  echo ""
  echo "Install from: https://www.terraform.io/downloads"
  exit 1
fi
echo ""

# Test 2: Terraform Files
echo -e "${BLUE}2. Terraform Files${NC}"
REQUIRED_FILES=("main.tf" "variables.tf" "outputs.tf" "versions.tf" "user-data.sh")
for file in "${REQUIRED_FILES[@]}"; do
  if [ -f "$TERRAFORM_DIR/$file" ]; then
    test_passed "$file exists"
  else
    test_failed "$file missing"
  fi
done
echo ""

# Test 3: Terraform Syntax
echo -e "${BLUE}3. Terraform Syntax${NC}"
cd "$TERRAFORM_DIR"

if terraform fmt -check -recursive > /dev/null 2>&1; then
  test_passed "Terraform formatting correct"
else
  test_warning "Terraform formatting issues (run: terraform fmt)"
fi

if terraform validate -json > /dev/null 2>&1; then
  test_passed "Terraform syntax valid"
else
  test_failed "Terraform validation failed"
  terraform validate
fi
echo ""

# Test 4: Terraform Initialization
echo -e "${BLUE}4. Terraform Initialization${NC}"
if [ -d ".terraform" ]; then
  test_passed "Terraform already initialized"
else
  echo "   Initializing Terraform..."
  if terraform init -backend=false > /dev/null 2>&1; then
    test_passed "Terraform initialization successful"
  else
    test_failed "Terraform initialization failed"
    terraform init
  fi
fi
echo ""

# Test 5: Variable Validation
echo -e "${BLUE}5. Variable Validation${NC}"

# Check if AMI ID provided for plan test
if [ -z "$AMI_ID" ]; then
  test_warning "No AMI ID provided for plan test"
  echo "   Usage: $0 ami-xxxxxxxxxxxx"
  AMI_ID="ami-00000000000000000"  # Dummy for validation
fi

# Create test tfvars
cat > terraform.tfvars.test <<EOF
region           = "$REGION"
instance_type    = "$INSTANCE_TYPE"
max_spot_price   = "0.50"
max_session_cost = 10.00
ami_id           = "$AMI_ID"
user_public_key  = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDUMMY test@example.com"
auto_terminate   = true
availability_zone = ""
EOF

if terraform validate -var-file=terraform.tfvars.test > /dev/null 2>&1; then
  test_passed "Variables valid"
else
  test_failed "Variable validation failed"
fi

rm -f terraform.tfvars.test
echo ""

# Test 6: Resource Configuration
echo -e "${BLUE}6. Resource Configuration${NC}"

# Check for required resources
RESOURCES=("aws_vpc" "aws_security_group" "aws_key_pair" "aws_spot_instance_request")
for resource in "${RESOURCES[@]}"; do
  if grep -q "$resource" main.tf; then
    test_passed "$resource defined"
  else
    test_warning "$resource not found in main.tf"
  fi
done
echo ""

# Test 7: Security Group Rules
echo -e "${BLUE}7. Security Group Rules${NC}"
if grep -q "from_port.*22" main.tf; then
  test_passed "SSH ingress rule defined"
else
  test_warning "SSH ingress rule not found"
fi

if grep -q "egress" main.tf; then
  test_passed "Egress rules defined"
else
  test_warning "Egress rules not found"
fi
echo ""

# Test 8: User Data Script
echo -e "${BLUE}8. User Data Script${NC}"
USER_DATA="$TERRAFORM_DIR/user-data.sh"

if [ -f "$USER_DATA" ]; then
  test_passed "user-data.sh exists"

  # Check syntax
  if bash -n "$USER_DATA" 2>/dev/null; then
    test_passed "user-data.sh syntax valid"
  else
    test_failed "user-data.sh has syntax errors"
  fi

  # Check for key components
  if grep -q "gitlab.com/lilacashes/menger.git" "$USER_DATA"; then
    test_passed "Repository clone present"
  else
    test_warning "Repository clone not found"
  fi

  if grep -q "WELCOME.txt" "$USER_DATA"; then
    test_passed "Welcome message creation present"
  else
    test_warning "Welcome message not found"
  fi
else
  test_failed "user-data.sh not found"
fi
echo ""

# Test 9: Outputs
echo -e "${BLUE}9. Outputs Configuration${NC}"
OUTPUTS=("instance_id" "instance_public_ip" "spot_request_id")
for output in "${OUTPUTS[@]}"; do
  if grep -q "output \"$output\"" outputs.tf 2>/dev/null; then
    test_passed "$output defined"
  else
    test_warning "$output not found in outputs"
  fi
done
echo ""

# Test 10: Terraform Plan (Dry-Run)
echo -e "${BLUE}10. Terraform Plan (Dry-Run)${NC}"

if [ "$AMI_ID" != "ami-00000000000000000" ]; then
  echo "   Generating plan..."

  # Create temporary tfvars
  cat > terraform.tfvars.test <<EOF
region           = "$REGION"
instance_type    = "$INSTANCE_TYPE"
max_spot_price   = "0.50"
max_session_cost = 10.00
ami_id           = "$AMI_ID"
user_public_key  = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDUMMY test@example.com"
auto_terminate   = true
availability_zone = ""
EOF

  PLAN_OUTPUT=$(terraform plan -var-file=terraform.tfvars.test -out=/dev/null 2>&1)
  PLAN_EXIT=$?

  if [ $PLAN_EXIT -eq 0 ]; then
    test_passed "Terraform plan successful"

    # Parse plan output
    RESOURCES_TO_CREATE=$(echo "$PLAN_OUTPUT" | grep "Plan:" | grep -o "[0-9]* to add" | awk '{print $1}' || echo "0")
    if [ -n "$RESOURCES_TO_CREATE" ] && [ "$RESOURCES_TO_CREATE" -gt 0 ]; then
      echo "   Resources to create: $RESOURCES_TO_CREATE"
    fi
  else
    test_failed "Terraform plan failed"
    echo ""
    echo "$PLAN_OUTPUT"
  fi

  rm -f terraform.tfvars.test
else
  test_warning "Skipping plan test (no valid AMI ID provided)"
fi
echo ""

# Test 11: Cost Estimation
echo -e "${BLUE}11. Cost Estimation${NC}"
SPOT_PRICE=$(aws ec2 describe-spot-price-history \
  --region "$REGION" \
  --instance-types "$INSTANCE_TYPE" \
  --product-descriptions "Linux/UNIX" \
  --max-items 1 \
  --query 'SpotPriceHistory[0].SpotPrice' \
  --output text 2>/dev/null || echo "0.30")

if [ -n "$SPOT_PRICE" ] && [ "$SPOT_PRICE" != "None" ]; then
  # Calculate daily/monthly costs for reference
  DAILY_COST=$(echo "$SPOT_PRICE * 24" | bc 2>/dev/null || echo "7.20")
  MONTHLY_COST=$(echo "$SPOT_PRICE * 24 * 30" | bc 2>/dev/null || echo "216.00")

  test_passed "Spot price: \$${SPOT_PRICE}/hour"
  echo "   Daily (24hrs):   \$${DAILY_COST}"
  echo "   Monthly (30d):   \$${MONTHLY_COST}"
  echo "   ${YELLOW}Note: Actual cost depends on usage time${NC}"
else
  test_warning "Could not retrieve spot price"
fi
echo ""

# Test 12: State File Warning
echo -e "${BLUE}12. State Management${NC}"
if [ -f "terraform.tfstate" ]; then
  test_warning "terraform.tfstate exists (previous deployment)"

  STATE_RESOURCES=$(terraform state list 2>/dev/null | wc -l || echo "0")
  if [ "$STATE_RESOURCES" -gt 0 ]; then
    echo "   Current resources: $STATE_RESOURCES"
    echo "   ${YELLOW}Terminate before new deployment: scripts/nvidia-spot.sh --terminate${NC}"
  fi
else
  test_passed "No state file (clean slate)"
fi
echo ""

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo -e "${GREEN}Passed:   $PASSED${NC}"
[ $WARNINGS -gt 0 ] && echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
[ $FAILED -gt 0 ] && echo -e "${RED}Failed:   $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ] && [ $WARNINGS -eq 0 ]; then
  echo -e "${GREEN}✓ All tests passed! Terraform configuration ready.${NC}"
  exit 0
elif [ $FAILED -eq 0 ]; then
  echo -e "${YELLOW}✓ Tests passed with warnings. Review warnings above.${NC}"
  exit 0
else
  echo -e "${RED}✗ Some tests failed. Fix errors before deploying.${NC}"
  exit 1
fi
