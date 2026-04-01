#!/bin/bash
# Main wrapper script for launching and managing NVIDIA GPU spot instances

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TERRAFORM_DIR="$PROJECT_ROOT/terraform"

# Default configuration
REGION="${AWS_REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-}"
AVAILABILITY_ZONE=""
INSTANCE_TYPE="g4dn.xlarge"
MAX_SPOT_PRICE="0.50"
MAX_SESSION_COST="10.00"
AUTO_TERMINATE="true"
AMI_ID=""
SSH_KEY=""
for _candidate in "${HOME}/.ssh/id_ed25519.pub" "${HOME}/.ssh/id_ecdsa.pub" "${HOME}/.ssh/id_rsa.pub"; do
  if [ -f "$_candidate" ]; then
    SSH_KEY="$_candidate"
    break
  fi
done
COMMAND=""
LIST_INSTANCES=false
LIST_RUNNING=false
TERMINATE=false
LIST_STATES=false
LIST_AMIS=false
MENGER_BRANCH="main"
RETRIEVE_GLOB=""
RETRIEVE_TO="./artifacts"
RESTORE_STATE=""
SAVE_STATE=""
NO_AUTO_RESTORE=false

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
  --region REGION            AWS region (default: from ~/.aws/config or us-east-1)
  --availability-zone AZ     AWS availability zone (e.g., us-east-1a). Region is derived from AZ
  --instance-type TYPE       Instance type (default: g4dn.xlarge)
  --max-price PRICE          Maximum spot price per hour (default: 0.50)
  --max-cost COST            Maximum total session cost (default: 10.00)
  --ami-id AMI_ID            Custom AMI ID (optional if ami-registry.tsv has an entry for this region)
  --list-instances           List available NVIDIA instances and spot prices
  --list-running             Show currently running instances managed by this script
  --terminate                Terminate all running instances and clean up resources
  --list-states              List all saved instance states
  --list-amis                List AMIs built for this project (from ami-registry.tsv)
  --menger-branch BRANCH     Git branch to clone on the instance (default: main)
  --retrieve GLOB            After --command, rsync files matching GLOB from ~/GLOB on instance
  --retrieve-to DIR          Local destination for --retrieve (default: ./artifacts)
  --restore-state NAME       Restore specific saved state on launch
  --save-state NAME          Save instance state with given name before shutdown
  --no-auto-restore          Don't automatically restore 'last' state on launch
  --command "CMD"            Run command and auto-terminate
  --no-auto-terminate        Disable auto-termination on logout
  --ssh-key PATH             Path to SSH public key (auto-detected from ~/.ssh/)
  --aws-profile PROFILE      AWS credentials profile (default: $AWS_PROFILE env var)
  -h, --help                 Show this help message

EXAMPLES:
  # Launch with defaults (AMI ID read from registry automatically)
  $0

  # Launch with a specific branch
  $0 --menger-branch feature/my-branch

  # Launch in specific availability zone (useful for GPU availability, region derived)
  $0 --availability-zone eu-central-1b

  # Check running instances
  $0 --list-running

  # Terminate running instances
  $0 --terminate

  # List saved states
  $0 --list-states

  # List all AMIs built for this project
  $0 --list-amis

  # Restore from specific state
  $0 --restore-state before-refactor

  # Save state before shutdown
  $0 --save-state my-checkpoint

  # Specify instance type and max price
  $0 --instance-type g5.xlarge --max-price 0.75

  # Run a render and retrieve the output image
  $0 --command "menger-app --optix --sponge-type cube-sponge --level 3 --save-name out.png" --retrieve "*.png"

  # Launch a specific branch
  $0 --menger-branch feature/my-branch

BEFORE FIRST USE:
  1. Build custom AMI:
     scripts/build-ami.sh /path/to/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh

  2. AMI ID is saved automatically to scripts/ami-registry.tsv
     Run: $0 --list-amis

  3. Launch instance (AMI ID is read from registry automatically):
     $0

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
    --availability-zone)
      AVAILABILITY_ZONE="$2"
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
    --list-states)
      LIST_STATES=true
      shift
      ;;
    --restore-state)
      RESTORE_STATE="$2"
      shift 2
      ;;
    --save-state)
      SAVE_STATE="$2"
      shift 2
      ;;
    --no-auto-restore)
      NO_AUTO_RESTORE=true
      shift
      ;;
    --list-amis)
      LIST_AMIS=true
      shift
      ;;
    --menger-branch)
      MENGER_BRANCH="$2"
      shift 2
      ;;
    --retrieve)
      RETRIEVE_GLOB="$2"
      shift 2
      ;;
    --retrieve-to)
      RETRIEVE_TO="$2"
      shift 2
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
    --aws-profile)
      AWS_PROFILE="$2"
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

# Derive region from availability zone if specified
if [ -n "$AVAILABILITY_ZONE" ]; then
  # Extract region by removing the last character (the AZ letter)
  REGION="${AVAILABILITY_ZONE%?}"
  echo -e "${YELLOW}Note: Using region $REGION derived from availability zone $AVAILABILITY_ZONE${NC}"
  echo ""
fi

# List instances if requested
if [ "$LIST_INSTANCES" = true ]; then
  bash "$SCRIPT_DIR/list-instances.sh" "$REGION"
  exit 0
fi

# List saved states if requested
if [ "$LIST_STATES" = true ]; then
  bash "$SCRIPT_DIR/list-spot-states.sh"
  exit 0
fi

# List AMIs if requested
if [ "$LIST_AMIS" = true ]; then
  AMI_REGISTRY="$SCRIPT_DIR/ami-registry.tsv"
  AMI_COUNT=0
  if [ -f "$AMI_REGISTRY" ]; then
    AMI_COUNT=$(grep -c '^[^#]' "$AMI_REGISTRY" 2>/dev/null || true)
    AMI_COUNT=${AMI_COUNT:-0}
  fi
  if [ "$AMI_COUNT" -eq 0 ] 2>/dev/null || [ -z "$AMI_COUNT" ]; then
    echo -e "${YELLOW}No AMIs found in registry.${NC}"
    echo "Build an AMI first: scripts/build-ami.sh /path/to/NVIDIA-OptiX-SDK-installer.sh"
    exit 0
  fi
  echo -e "${GREEN}=== Menger AMI Registry ===${NC}"
  printf "%-15s %-25s %-40s %s\n" "REGION" "AMI ID" "NAME" "BUILT"
  printf '%0.s-' {1..95}; echo
  grep '^[^#]' "$AMI_REGISTRY" | while IFS=$'\t' read -r reg id name ts; do
    printf "%-15s %-25s %-40s %s\n" "$reg" "$id" "$name" "$ts"
  done
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

# Resolve AMI ID — use explicit --ami-id or look up registry for current region
AMI_REGISTRY="$SCRIPT_DIR/ami-registry.tsv"
if [ -z "$AMI_ID" ]; then
  if [ -f "$AMI_REGISTRY" ]; then
    AMI_ID=$(grep '^[^#]' "$AMI_REGISTRY" | awk -F'\t' -v region="$REGION" '$1==region{id=$2} END{print id}')
  fi
  if [ -z "$AMI_ID" ]; then
    echo -e "${RED}Error: No AMI found for region '$REGION'${NC}"
    echo ""
    echo "Build an AMI first:"
    echo "  scripts/build-ami.sh /path/to/NVIDIA-OptiX-SDK-installer.sh"
    echo ""
    echo "Or pass an explicit AMI ID:"
    echo "  $0 --ami-id ami-xxxxxxxxxxxx"
    echo ""
    echo "To see AMIs built for other regions:"
    echo "  $0 --list-amis"
    exit 1
  fi
  echo -e "${YELLOW}Using AMI from registry: $AMI_ID (region: $REGION)${NC}"
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
[ -n "$AVAILABILITY_ZONE" ] && echo "Availability Zone: $AVAILABILITY_ZONE"
echo "Instance Type:     $INSTANCE_TYPE"
echo "Max Spot Price:    \$${MAX_SPOT_PRICE}/hour"
echo "Max Session Cost:  \$${MAX_SESSION_COST}"
echo "AMI ID:            $AMI_ID"
echo "Menger Branch:     $MENGER_BRANCH"
echo "Auto-terminate:    $AUTO_TERMINATE"
[ -n "$AWS_PROFILE" ] && echo "AWS Profile:       $AWS_PROFILE"
[ -n "$COMMAND" ] && echo "Command:           $COMMAND"
echo ""

# Change to terraform directory
cd "$TERRAFORM_DIR"

# Export AWS_PROFILE so Terraform and all AWS CLI calls use the right credentials
if [ -n "$AWS_PROFILE" ]; then
  export AWS_PROFILE
fi

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
region             = "$REGION"
availability_zone  = "$AVAILABILITY_ZONE"
instance_type      = "$INSTANCE_TYPE"
max_spot_price     = "$MAX_SPOT_PRICE"
max_session_cost   = $MAX_SESSION_COST
ami_id             = "$AMI_ID"
user_public_key    = "$SSH_PUBLIC_KEY"
auto_terminate     = $AUTO_TERMINATE
menger_branch      = "$MENGER_BRANCH"
EOF

# Initialize terraform if needed
if [ ! -d ".terraform" ]; then
  echo -e "${YELLOW}Initializing Terraform...${NC}"
  if ! terraform init > /dev/null 2>&1; then
    echo -e "${RED}Error: Terraform initialization failed${NC}"
    terraform init
    exit 1
  fi
fi

# Apply terraform configuration
echo -e "${YELLOW}Launching spot instance...${NC}"
# Capture stdout; suppress plan/refresh noise but show error blocks on failure
# Use 'set +e' so a non-zero exit doesn't trigger set -e before we can capture $?
set +e
TF_OUTPUT=$(terraform apply -auto-approve -compact-warnings 2>&1)
TF_EXIT=$?
set -e
if [ $TF_EXIT -ne 0 ]; then
  echo -e "${RED}Error: Terraform apply failed${NC}"
  echo "$TF_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g' | grep -E '(^\s*(│|Error:|╷|╵)|^Error )' | sed 's/^[[:space:]]*│ *//'
  exit 1
fi

# Extract instance information
INSTANCE_IP=$(terraform output -raw instance_public_ip)
INSTANCE_ID=$(terraform output -raw instance_id)

echo -e "${GREEN}Instance launched successfully!${NC}"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $INSTANCE_IP"
echo ""

# Wait for SSH to be ready (up to 5 minutes / 30 attempts)
echo -e "${YELLOW}Waiting for SSH to be ready...${NC}"
SSH_READY=false
for i in {1..30}; do
  if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes ubuntu@$INSTANCE_IP exit 2>/dev/null; then
    SSH_READY=true
    break
  fi
  echo -n "."
  sleep 10
done
echo ""
if [ "$SSH_READY" = false ]; then
  echo -e "${RED}Error: Instance did not become reachable after 5 minutes.${NC}"
  echo "Instance ID: $INSTANCE_ID   IP: $INSTANCE_IP"
  echo ""
  echo "Diagnose with:"
  echo "  aws ec2 get-console-output --instance-id $INSTANCE_ID --region $REGION"
  echo ""
  echo "Destroy with:"
  echo "  cd $TERRAFORM_DIR && terraform destroy"
  exit 1
fi

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

# Restore state if requested
BACKUP_BASE="${SPOT_STATES_DIR:-$HOME/.aws/spot-states}"
STATE_TO_RESTORE=""

if [ -n "$RESTORE_STATE" ]; then
  # Explicit state restoration requested
  STATE_TO_RESTORE="$RESTORE_STATE"
elif [ "$NO_AUTO_RESTORE" = false ] && [ -d "$BACKUP_BASE/last" ]; then
  # Auto-restore 'last' state if it exists
  STATE_TO_RESTORE="last"
  echo -e "${YELLOW}Note: Auto-restoring 'last' saved state (use --no-auto-restore to skip)${NC}"
  echo ""
fi

if [ -n "$STATE_TO_RESTORE" ]; then
  echo -e "${YELLOW}Restoring state: $STATE_TO_RESTORE${NC}"
  if bash "$SCRIPT_DIR/restore-spot-state.sh" "$STATE_TO_RESTORE" "$INSTANCE_IP"; then
    echo -e "${GREEN}✓ State restored successfully${NC}"
  else
    echo -e "${RED}Warning: State restoration failed${NC}"
    echo "Continuing with fresh instance..."
  fi
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

  # Retrieve artifacts if requested
  if [ -n "$RETRIEVE_GLOB" ]; then
    echo -e "${YELLOW}Retrieving artifacts matching: $RETRIEVE_GLOB${NC}"
    mkdir -p "$RETRIEVE_TO"
    if rsync -az -e "ssh -o StrictHostKeyChecking=no" \
        "ubuntu@$INSTANCE_IP:/home/ubuntu/$RETRIEVE_GLOB" \
        "$RETRIEVE_TO/"; then
      echo -e "${GREEN}✓ Artifacts saved to $RETRIEVE_TO/${NC}"
      ls -lh "$RETRIEVE_TO/"
    else
      echo -e "${RED}Warning: Artifact retrieval failed${NC}"
      echo "Retrieve manually:"
      echo "  rsync -az ubuntu@$INSTANCE_IP:/home/ubuntu/$RETRIEVE_GLOB $RETRIEVE_TO/"
    fi
  fi

  echo -e "${YELLOW}Instance will auto-terminate after grace period${NC}"
  exit 0
fi

# Connect with X11 forwarding
echo -e "${GREEN}Connecting to instance with X11 forwarding...${NC}"
echo -e "${YELLOW}Note: Instance will auto-terminate after you log out (if enabled)${NC}"
echo ""

# Poll for spot termination notice in background; trigger backup immediately if detected
SPOT_INTERRUPTED=false
(
  while true; do
    sleep 5
    termination=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 -o BatchMode=yes \
      ubuntu@$INSTANCE_IP \
      'curl -sf -H "X-aws-ec2-metadata-token: $(curl -sf -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 30" http://169.254.169.254/latest/api/token 2>/dev/null)" http://169.254.169.254/latest/meta-data/spot/termination-time 2>/dev/null' \
      2>/dev/null)
    if [ -n "$termination" ]; then
      echo "" >&2
      echo -e "${RED}⚠ Spot termination notice received! Termination at: $termination${NC}" >&2
      echo -e "${YELLOW}Starting emergency state backup...${NC}" >&2
      bash "$SCRIPT_DIR/backup-spot-state.sh" "last" "$INSTANCE_IP" >&2 && \
        echo -e "${GREEN}✓ Emergency backup complete${NC}" >&2 || \
        echo -e "${RED}Emergency backup failed — instance may already be gone${NC}" >&2
      break
    fi
  done
) &
SPOT_POLLER_PID=$!

ssh -X -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP

kill $SPOT_POLLER_PID 2>/dev/null || true

# After disconnection
echo ""
echo -e "${YELLOW}Disconnected from instance${NC}"

if [ "$AUTO_TERMINATE" = "true" ]; then
  # Backup state before termination
  STATE_NAME_TO_SAVE="${SAVE_STATE:-last}"

  echo -e "${YELLOW}Backing up instance state: $STATE_NAME_TO_SAVE${NC}"
  if bash "$SCRIPT_DIR/backup-spot-state.sh" "$STATE_NAME_TO_SAVE" "$INSTANCE_IP"; then
    echo -e "${GREEN}✓ State backed up successfully${NC}"
  else
    echo -e "${RED}Warning: State backup failed${NC}"
    read -p "Continue with termination? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Termination cancelled. Instance left running."
      echo "To destroy later:"
      echo "  cd $TERRAFORM_DIR && terraform destroy"
      exit 0
    fi
  fi
  echo ""

  echo -e "${YELLOW}Waiting 5 minutes before auto-terminating instance...${NC}"
  echo "Press Ctrl+C to cancel auto-termination and keep instance running"
  echo ""

  # Wait 5 minutes (300 seconds)
  if sleep 300; then
    echo -e "${YELLOW}Grace period expired. Terminating instance...${NC}"
    cd "$TERRAFORM_DIR"
    if terraform destroy -auto-approve > /dev/null 2>&1; then
      echo -e "${GREEN}✓ Instance terminated${NC}"
    else
      echo -e "${RED}Error: Failed to destroy infrastructure${NC}"
      echo "Run manually: cd $TERRAFORM_DIR && terraform destroy"
      exit 1
    fi
  else
    echo ""
    echo -e "${YELLOW}Auto-termination cancelled. Instance left running.${NC}"
    echo "To destroy later:"
    echo "  cd $TERRAFORM_DIR && terraform destroy"
  fi
else
  read -p "Destroy instance now? [y/N] " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Destroying instance...${NC}"
    cd "$TERRAFORM_DIR"
    terraform destroy -auto-approve
    echo -e "${GREEN}Instance destroyed${NC}"
  else
    echo "Instance left running. To destroy later:"
    echo "  cd $TERRAFORM_DIR && terraform destroy"
  fi
fi
