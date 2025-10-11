#!/bin/bash
# Main wrapper script for launching and managing NVIDIA GPU spot instances

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TERRAFORM_DIR="$PROJECT_ROOT/terraform"

# Default configuration
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="g4dn.xlarge"
MAX_SPOT_PRICE="0.50"
MAX_SESSION_COST="10.00"
AUTO_TERMINATE="true"
AMI_ID=""
SSH_KEY="${HOME}/.ssh/id_rsa.pub"
COMMAND=""
LIST_INSTANCES=false
LIST_RUNNING=false
TERMINATE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Usage function
usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Launch and manage NVIDIA GPU spot instances for menger development.

OPTIONS:
  --region REGION          AWS region (default: from ~/.aws/config or us-east-1)
  --instance-type TYPE     Instance type (default: g4dn.xlarge)
  --max-price PRICE        Maximum spot price per hour (default: 0.50)
  --max-cost COST          Maximum total session cost (default: 10.00)
  --ami-id AMI_ID          Custom AMI ID (required for launch)
  --list-instances         List available NVIDIA instances and spot prices
  --list-running           Show currently running instances managed by this script
  --terminate              Terminate all running instances and clean up resources
  --command "CMD"          Run command and auto-terminate
  --no-auto-terminate      Disable auto-termination on logout
  --ssh-key PATH           Path to SSH public key (default: ~/.ssh/id_rsa.pub)
  -h, --help               Show this help message

EXAMPLES:
  # List available instance types
  $0 --list-instances

  # Launch with defaults
  $0 --ami-id ami-xxxxxxxxxxxx

  # Check running instances
  $0 --list-running

  # Terminate running instances
  $0 --terminate

  # Specify instance type and max price
  $0 --ami-id ami-xxxxxxxxxxxx --instance-type g5.xlarge --max-price 0.75

  # Run command and terminate
  $0 --ami-id ami-xxxxxxxxxxxx --command "cd menger && sbt test"

BEFORE FIRST USE:
  1. Build custom AMI:
     scripts/build-ami.sh /path/to/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh

  2. Note the AMI ID from the build output

  3. Launch instance with the AMI ID:
     $0 --ami-id ami-xxxxxxxxxxxx

EOF
  exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --region)
      REGION="$2"
      shift 2
      ;;
    --instance-type)
      INSTANCE_TYPE="$2"
      shift 2
      ;;
    --max-price)
      MAX_SPOT_PRICE="$2"
      shift 2
      ;;
    --max-cost)
      MAX_SESSION_COST="$2"
      shift 2
      ;;
    --ami-id)
      AMI_ID="$2"
      shift 2
      ;;
    --list-instances)
      LIST_INSTANCES=true
      shift
      ;;
    --list-running)
      LIST_RUNNING=true
      shift
      ;;
    --terminate)
      TERMINATE=true
      shift
      ;;
    --command)
      COMMAND="$2"
      AUTO_TERMINATE="true"
      shift 2
      ;;
    --no-auto-terminate)
      AUTO_TERMINATE="false"
      shift
      ;;
    --ssh-key)
      SSH_KEY="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo -e "${RED}Error: Unknown option: $1${NC}"
      usage
      ;;
  esac
done

# List instances if requested
if [ "$LIST_INSTANCES" = true ]; then
  bash "$SCRIPT_DIR/list-instances.sh" "$REGION"
  exit 0
fi

# List running instances if requested
if [ "$LIST_RUNNING" = true ]; then
  cd "$TERRAFORM_DIR"

  if [ ! -f "terraform.tfstate" ] || [ ! -s "terraform.tfstate" ]; then
    echo -e "${YELLOW}No terraform state found. No instances currently managed.${NC}"
    exit 0
  fi

  # Check if there are any resources in the state
  RESOURCE_COUNT=$(terraform state list 2>/dev/null | wc -l)
  if [ "$RESOURCE_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No active instances found.${NC}"
    exit 0
  fi

  echo -e "${GREEN}=== Running Instances ===${NC}"
  echo ""

  # Get instance info from terraform state
  INSTANCE_ID=$(terraform output -raw instance_id 2>/dev/null || echo "N/A")
  INSTANCE_IP=$(terraform output -raw instance_public_ip 2>/dev/null || echo "N/A")
  SPOT_REQUEST_ID=$(terraform output -raw spot_request_id 2>/dev/null || echo "N/A")

  if [ "$INSTANCE_ID" != "N/A" ] && [ -n "$INSTANCE_ID" ]; then
    # Get additional details from AWS
    INSTANCE_INFO=$(aws ec2 describe-instances \
      --region "$REGION" \
      --instance-ids "$INSTANCE_ID" \
      --query 'Reservations[0].Instances[0].[InstanceType,State.Name,LaunchTime,Tags[?Key==`Project`].Value|[0]]' \
      --output text 2>/dev/null || echo "")

    if [ -n "$INSTANCE_INFO" ]; then
      read -r INST_TYPE STATE LAUNCH_TIME PROJECT <<< "$INSTANCE_INFO"

      echo "Instance ID:       $INSTANCE_ID"
      echo "Spot Request ID:   $SPOT_REQUEST_ID"
      echo "Public IP:         $INSTANCE_IP"
      echo "Instance Type:     $INST_TYPE"
      echo "State:             $STATE"
      echo "Project:           $PROJECT"
      echo "Launch Time:       $LAUNCH_TIME"
      echo ""
      echo -e "${YELLOW}To terminate this instance:${NC}"
      echo "  $0 --terminate"
      echo ""
      echo -e "${YELLOW}To connect:${NC}"
      echo "  ssh -X ubuntu@$INSTANCE_IP"
    else
      echo -e "${YELLOW}Instance tracked in terraform but not found in AWS.${NC}"
      echo "Instance may have been terminated externally."
      echo ""
      echo -e "${YELLOW}To clean up terraform state:${NC}"
      echo "  cd $TERRAFORM_DIR && terraform destroy"
    fi
  else
    echo -e "${YELLOW}No active instances found.${NC}"
  fi

  exit 0
fi

# Terminate instances if requested
if [ "$TERMINATE" = true ]; then
  cd "$TERRAFORM_DIR"

  if [ ! -f "terraform.tfstate" ] || [ ! -s "terraform.tfstate" ]; then
    echo -e "${YELLOW}No terraform state found. Nothing to terminate.${NC}"
    exit 0
  fi

  # Check if there are any resources
  RESOURCE_COUNT=$(terraform state list 2>/dev/null | wc -l)
  if [ "$RESOURCE_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No active instances to terminate.${NC}"
    exit 0
  fi

  # Get instance info before destroying
  INSTANCE_ID=$(terraform output -raw instance_id 2>/dev/null || echo "")

  echo -e "${YELLOW}Terminating infrastructure...${NC}"

  # Destroy with minimal output
  if terraform destroy -auto-approve > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Instance terminated${NC}"
    [ -n "$INSTANCE_ID" ] && echo "  Instance ID: $INSTANCE_ID"
  else
    echo -e "${RED}Error: Failed to destroy infrastructure${NC}"
    echo "Run manually: cd $TERRAFORM_DIR && terraform destroy"
    exit 1
  fi

  exit 0
fi

# Validate AMI ID
if [ -z "$AMI_ID" ]; then
  echo -e "${RED}Error: AMI ID is required${NC}"
  echo ""
  echo "Build AMI first:"
  echo "  scripts/build-ami.sh /path/to/NVIDIA-OptiX-SDK-installer.sh"
  echo ""
  echo "Or list available instances to choose instance type:"
  echo "  $0 --list-instances --region $REGION"
  exit 1
fi

# Validate SSH key
if [ ! -f "$SSH_KEY" ]; then
  echo -e "${RED}Error: SSH public key not found: $SSH_KEY${NC}"
  exit 1
fi

# Read SSH public key
SSH_PUBLIC_KEY=$(cat "$SSH_KEY")

# Get git configuration from local machine
GIT_USER_NAME=$(git config --get user.name || echo "")
GIT_USER_EMAIL=$(git config --get user.email || echo "")

if [ -z "$GIT_USER_NAME" ] || [ -z "$GIT_USER_EMAIL" ]; then
  echo -e "${YELLOW}Warning: Git user.name or user.email not configured locally${NC}"
  echo "Configure with:"
  echo "  git config --global user.name \"Your Name\""
  echo "  git config --global user.email \"your.email@example.com\""
  echo ""
fi

echo -e "${GREEN}=== NVIDIA Spot Instance Configuration ===${NC}"
echo "Region:            $REGION"
echo "Instance Type:     $INSTANCE_TYPE"
echo "Max Spot Price:    \$${MAX_SPOT_PRICE}/hour"
echo "Max Session Cost:  \$${MAX_SESSION_COST}"
echo "AMI ID:            $AMI_ID"
echo "Auto-terminate:    $AUTO_TERMINATE"
[ -n "$COMMAND" ] && echo "Command:           $COMMAND"
echo ""

# Change to terraform directory
cd "$TERRAFORM_DIR"

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
region           = "$REGION"
instance_type    = "$INSTANCE_TYPE"
max_spot_price   = "$MAX_SPOT_PRICE"
max_session_cost = $MAX_SESSION_COST
ami_id           = "$AMI_ID"
user_public_key  = "$SSH_PUBLIC_KEY"
auto_terminate   = $AUTO_TERMINATE
EOF

# Initialize terraform if needed
if [ ! -d ".terraform" ]; then
  echo -e "${YELLOW}Initializing Terraform...${NC}"
  terraform init
fi

# Apply terraform configuration
echo -e "${YELLOW}Launching spot instance...${NC}"
terraform apply -auto-approve

# Extract instance information
INSTANCE_IP=$(terraform output -raw instance_public_ip)
INSTANCE_ID=$(terraform output -raw instance_id)

echo -e "${GREEN}Instance launched successfully!${NC}"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $INSTANCE_IP"
echo ""

# Wait for SSH to be ready
echo -e "${YELLOW}Waiting for SSH to be ready...${NC}"
for i in {1..30}; do
  if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes ubuntu@$INSTANCE_IP exit 2>/dev/null; then
    break
  fi
  echo -n "."
  sleep 10
done
echo ""

# Wait for user-data to complete
echo -e "${YELLOW}Waiting for instance initialization...${NC}"
ssh -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP \
  'bash -c "while [ ! -f /var/log/user-data.log ] || ! grep -q \"Initialization complete\" /var/log/user-data.log 2>/dev/null; do sleep 5; done"'

echo -e "${GREEN}Instance ready!${NC}"
echo ""

# Configure git identity on remote instance
if [ -n "$GIT_USER_NAME" ] && [ -n "$GIT_USER_EMAIL" ]; then
  echo -e "${YELLOW}Configuring git identity...${NC}"
  ssh -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP \
    "bash -c \"git config --global user.name \\\"$GIT_USER_NAME\\\" && \
     git config --global user.email \\\"$GIT_USER_EMAIL\\\"\""
  echo -e "${GREEN}Git configured: $GIT_USER_NAME <$GIT_USER_EMAIL>${NC}"
  echo ""
fi

# Upload and start auto-terminate daemon if enabled
if [ "$AUTO_TERMINATE" = "true" ]; then
  echo -e "${YELLOW}Setting up auto-terminate daemon...${NC}"
  scp -o StrictHostKeyChecking=no "$SCRIPT_DIR/auto-terminate.sh" ubuntu@$INSTANCE_IP:/tmp/
  ssh -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP \
    'bash -c "sudo mv /tmp/auto-terminate.sh /usr/local/bin/ && \
     sudo chmod +x /usr/local/bin/auto-terminate.sh && \
     sudo nohup /usr/local/bin/auto-terminate.sh > /dev/null 2>&1 &"'
  echo -e "${GREEN}Auto-terminate daemon started${NC}"
  echo ""
fi

# Display welcome message
ssh -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP 'cat ~/WELCOME.txt'
echo ""

# Run command if specified
if [ -n "$COMMAND" ]; then
  echo -e "${YELLOW}Executing command: $COMMAND${NC}"
  ssh -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP "bash -c $(printf '%q' "$COMMAND")"
  echo ""
  echo -e "${GREEN}Command completed${NC}"
  echo -e "${YELLOW}Instance will auto-terminate after grace period${NC}"
  exit 0
fi

# Connect with X11 forwarding
echo -e "${GREEN}Connecting to instance with X11 forwarding...${NC}"
echo -e "${YELLOW}Note: Instance will auto-terminate after you log out (if enabled)${NC}"
echo ""

ssh -X -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP

# After disconnection
echo ""
echo -e "${YELLOW}Disconnected from instance${NC}"

if [ "$AUTO_TERMINATE" = "true" ]; then
  echo -e "${YELLOW}Instance will auto-terminate after grace period (5 minutes)${NC}"
  echo "To destroy immediately, run:"
  echo "  cd $TERRAFORM_DIR && terraform destroy"
else
  read -p "Destroy instance now? [y/N] " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Destroying instance...${NC}"
    terraform destroy -auto-approve
    echo -e "${GREEN}Instance destroyed${NC}"
  else
    echo "Instance left running. To destroy later:"
    echo "  cd $TERRAFORM_DIR && terraform destroy"
  fi
fi
