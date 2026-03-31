#!/bin/bash
# Script to build custom AMI with CUDA 12.8, OptiX, and dev tools
# Alternative to Packer for users who prefer AWS CLI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMI_REGISTRY="$SCRIPT_DIR/ami-registry.tsv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
OPTIX_INSTALLER=""
DEREGISTER_OLD=false
DEREGISTER_ID=""

usage() {
  cat <<EOF
Usage: $0 [OPTIONS] <path-to-optix-installer.sh>
       $0 --deregister <ami-id>
       $0 --list

Build a custom AMI with CUDA 12.8, OptiX, Scala, and dev tools.

OPTIONS:
  --region REGION          AWS region (default: \$AWS_REGION or us-east-1)
  --deregister-old         Deregister previous AMIs for this region after successful build
  --deregister AMI-ID      Deregister a specific AMI and remove it from the registry
  --list                   List AMIs in the registry (same as nvidia-spot.sh --list-amis)
  -h, --help               Show this help message

Download OptiX SDK from: https://developer.nvidia.com/optix
EOF
  exit 1
}

# --- Helper: deregister one AMI and its snapshots, remove from registry ---
deregister_ami() {
  local ami_id="$1"
  local region="$2"
  echo -e "${YELLOW}Deregistering AMI $ami_id in $region...${NC}"

  # Get snapshot IDs before deregistering
  local snapshots
  snapshots=$(aws ec2 describe-images --region "$region" --image-ids "$ami_id" \
    --query 'Images[0].BlockDeviceMappings[].Ebs.SnapshotId' --output text 2>/dev/null || true)

  # Deregister
  if aws ec2 deregister-image --region "$region" --image-id "$ami_id" 2>/dev/null; then
    echo -e "${GREEN}✓ AMI $ami_id deregistered${NC}"
  else
    echo -e "${YELLOW}⚠ AMI $ami_id not found in AWS (may already be deregistered)${NC}"
  fi

  # Delete snapshots
  for snap in $snapshots; do
    if [ -n "$snap" ] && [ "$snap" != "None" ] && [ "$snap" != "null" ]; then
      if aws ec2 delete-snapshot --region "$region" --snapshot-id "$snap" 2>/dev/null; then
        echo -e "${GREEN}✓ Snapshot $snap deleted${NC}"
      else
        echo -e "${YELLOW}⚠ Could not delete snapshot $snap${NC}"
      fi
    fi
  done

  # Remove from registry
  if [ -f "$AMI_REGISTRY" ]; then
    local tmp
    tmp=$(mktemp)
    grep -v "	$ami_id	" "$AMI_REGISTRY" > "$tmp" || true
    mv "$tmp" "$AMI_REGISTRY"
    echo "Removed $ami_id from registry."
  fi
}

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      REGION="$2"; shift 2 ;;
    --deregister-old)
      DEREGISTER_OLD=true; shift ;;
    --deregister)
      DEREGISTER_ID="$2"; shift 2 ;;
    --list)
      if [ ! -f "$AMI_REGISTRY" ] || [ "$(grep -c '^[^#]' "$AMI_REGISTRY" 2>/dev/null || true)" -eq 0 ]; then
        echo "No AMIs in registry."
      else
        printf "%-15s %-25s %-40s %s\n" "REGION" "AMI ID" "NAME" "BUILT"
        printf '%0.s-' {1..95}; echo
        grep '^[^#]' "$AMI_REGISTRY" | while IFS=$'\t' read -r reg id name ts; do
          printf "%-15s %-25s %-40s %s\n" "$reg" "$id" "$name" "$ts"
        done
      fi
      exit 0 ;;
    -h|--help)
      usage ;;
    -*)
      echo "Unknown option: $1"; usage ;;
    *)
      OPTIX_INSTALLER="$1"; shift ;;
  esac
done

# --- Standalone --deregister ---
if [ -n "$DEREGISTER_ID" ]; then
  # Look up region from registry if not overridden
  if [ -f "$AMI_REGISTRY" ]; then
    REG_REGION=$(grep "	$DEREGISTER_ID	" "$AMI_REGISTRY" | cut -f1 | head -1)
    [ -n "$REG_REGION" ] && REGION="$REG_REGION"
  fi
  deregister_ami "$DEREGISTER_ID" "$REGION"
  exit 0
fi

if [ -z "$OPTIX_INSTALLER" ]; then
  echo "Error: OptiX installer path is required."
  usage
fi

if [ ! -f "$OPTIX_INSTALLER" ]; then
  echo "Error: OptiX installer not found at: $OPTIX_INSTALLER"
  exit 1
fi

echo "=== Building custom AMI for menger NVIDIA development ==="
echo "Region: $REGION"
echo "Instance Type: $INSTANCE_TYPE"
echo "OptiX Installer: $OPTIX_INSTALLER"

# Get latest Ubuntu 24.04 AMI
echo "=== Finding base Ubuntu AMI ==="
CANONICAL=099720109477
BASE_AMI=$(aws ec2 describe-images \
  --region "$REGION" \
  --owners "$CANONICAL" \
  --filters \
    "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" \
    "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text)

echo "Base AMI: $BASE_AMI"

# Create temporary key pair
KEY_NAME="menger-ami-build-$(date +%s)"
KEY_FILE="/tmp/${KEY_NAME}.pem"
echo "=== Creating temporary key pair: $KEY_NAME ==="
aws ec2 create-key-pair \
  --region "$REGION" \
  --key-name "$KEY_NAME" \
  --query 'KeyMaterial' \
  --output text > "$KEY_FILE"
chmod 400 "$KEY_FILE"

# Get or create security group
SG_NAME="menger-ami-build-sg"
echo "=== Checking for security group: $SG_NAME ==="
SG_ID=$(aws ec2 describe-security-groups \
  --region "$REGION" \
  --group-names "$SG_NAME" \
  --query 'SecurityGroups[0].GroupId' \
  --output text 2>/dev/null) || {
  # SG doesn't exist, create it
  echo "Security group not found, creating..."
  SG_ID=$(aws ec2 create-security-group \
    --region "$REGION" \
    --group-name "$SG_NAME" \
    --description "Temporary SG for AMI building" \
    --query 'GroupId' \
    --output text)
}

# Allow SSH
aws ec2 authorize-security-group-ingress \
  --region "$REGION" \
  --group-id "$SG_ID" \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0 2>/dev/null || true

echo "Security Group: $SG_ID"

# Get or find subnet
if [ -z "$SUBNET_ID" ]; then
  echo "=== Finding subnet in default VPC ==="
  VPC_ID=$(aws ec2 describe-vpcs \
    --region "$REGION" \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' \
    --output text)

  SUBNET_ID=$(aws ec2 describe-subnets \
    --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'Subnets[0].SubnetId' \
    --output text)

  if [ "$SUBNET_ID" = "None" ] || [ -z "$SUBNET_ID" ]; then
    echo "Error: No subnets found in default VPC. Creating one..."
    SUBNET_ID=$(aws ec2 create-subnet \
      --region "$REGION" \
      --vpc-id "$VPC_ID" \
      --cidr-block "172.31.0.0/20" \
      --query 'Subnet.SubnetId' \
      --output text)
    aws ec2 modify-subnet-attribute \
      --region "$REGION" \
      --subnet-id "$SUBNET_ID" \
      --map-public-ip-on-launch
  fi
fi

echo "Subnet: $SUBNET_ID"

# Launch temporary instance
echo "=== Launching temporary build instance ==="
INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$BASE_AMI" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --subnet-id "$SUBNET_ID" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=menger-ami-builder}]" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance ID: $INSTANCE_ID"

# Cleanup function
BUILD_STATUS="failed: script exited unexpectedly"
cleanup() {
  echo "=== Cleaning up ==="
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" 2>/dev/null || true
  aws ec2 delete-key-pair --region "$REGION" --key-name "$KEY_NAME" 2>/dev/null || true
  rm -f "$KEY_FILE"
  echo "Note: Security group $SG_ID left for reuse. Delete manually if needed:"
  echo "  aws ec2 delete-security-group --region $REGION --group-id $SG_ID"
  echo ""
  echo "========================================"
  echo "$BUILD_STATUS"
  echo "========================================"
}
trap cleanup EXIT

# Wait for instance to be running
echo "=== Waiting for instance to be running ==="
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "Public IP: $PUBLIC_IP"
echo "Waiting 60 seconds for SSH to be ready..."
sleep 60

# SSH options
SSH_OPTS="-i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

# Copy OptiX installer and verification script
echo "=== Uploading OptiX installer ==="
scp $SSH_OPTS "$OPTIX_INSTALLER" ubuntu@$PUBLIC_IP:/tmp/optix-installer.sh

echo "=== Uploading verification script ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scp $SSH_OPTS "$SCRIPT_DIR/verify-optix.sh" ubuntu@$PUBLIC_IP:/tmp/verify-optix.sh

# Run provisioning script
echo "=== Running provisioning script (this will take 15-30 minutes) ==="
ssh $SSH_OPTS ubuntu@$PUBLIC_IP 'bash -s' <<'PROVISION_SCRIPT'
set -e
echo "Starting provisioning at $(date)"

# Update system
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# Basic tools
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential git curl wget vim htop tmux unzip jq software-properties-common cmake

# Install AWS CLI (required for auto-terminate daemon)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
unzip -q /tmp/awscliv2.zip -d /tmp
sudo /tmp/aws/install
rm -rf /tmp/aws /tmp/awscliv2.zip

# Install Node.js (required for Claude Code)
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs

# Install Claude Code
sudo npm install -g @anthropic-ai/claude-code

# Fish shell
sudo apt-add-repository ppa:fish-shell/release-3 -y
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y fish

# X11 support
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  xauth x11-apps libx11-dev libxext-dev libxrender-dev libxtst-dev libxi-dev xvfb

# Configure X11 forwarding
sudo sed -i 's/#X11Forwarding yes/X11Forwarding yes/' /etc/ssh/sshd_config
sudo sed -i 's/#X11DisplayOffset 10/X11DisplayOffset 10/' /etc/ssh/sshd_config
sudo sed -i 's/#X11UseLocalhost yes/X11UseLocalhost no/' /etc/ssh/sshd_config

# Install CUDA 12.8 + NVIDIA driver from CUDA repository
# Install driver via CUDA repo (not ubuntu-drivers, which requires a GPU to autodetect)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-8 cuda-drivers
rm cuda-keyring_1.1-1_all.deb

# Install OptiX
sudo mkdir -p /opt/optix
chmod +x /tmp/optix-installer.sh
sudo /tmp/optix-installer.sh --skip-license --prefix=/opt/optix
rm /tmp/optix-installer.sh

# Set environment variables system-wide and for all users

# 1. System-wide environment file
sudo tee /etc/profile.d/cuda-optix.sh > /dev/null <<'ENVEOF'
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export OPTIX_ROOT=/opt/optix
ENVEOF
sudo chmod +x /etc/profile.d/cuda-optix.sh

# 2. Add to /etc/environment for system-wide availability
sudo sed -i 's|^PATH="|PATH="/usr/local/cuda-12.8/bin:|' /etc/environment
echo 'CUDA_HOME="/usr/local/cuda-12.8"' | sudo tee -a /etc/environment
echo 'OPTIX_ROOT="/opt/optix"' | sudo tee -a /etc/environment
echo 'LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64"' | sudo tee -a /etc/environment

# 3. Add to existing ubuntu user (bash)
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export OPTIX_ROOT=/opt/optix' >> ~/.bashrc

# 4. Add to existing ubuntu user (fish)
mkdir -p ~/.config/fish
cat >> ~/.config/fish/config.fish <<'FISHEOF'
set -x PATH /usr/local/cuda-12.8/bin $PATH
set -x LD_LIBRARY_PATH /usr/local/cuda-12.8/lib64 $LD_LIBRARY_PATH
set -x CUDA_HOME /usr/local/cuda-12.8
set -x OPTIX_ROOT /opt/optix
FISHEOF

# 5. Add to skeleton for new users (bash)
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' | sudo tee -a /etc/skel/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/skel/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.8' | sudo tee -a /etc/skel/.bashrc
echo 'export OPTIX_ROOT=/opt/optix' | sudo tee -a /etc/skel/.bashrc

# 6. Add to skeleton for new users (fish)
sudo mkdir -p /etc/skel/.config/fish
sudo tee /etc/skel/.config/fish/config.fish > /dev/null <<'FISHSKELEOF'
set -x PATH /usr/local/cuda-12.8/bin $PATH
set -x LD_LIBRARY_PATH /usr/local/cuda-12.8/lib64 $LD_LIBRARY_PATH
set -x CUDA_HOME /usr/local/cuda-12.8
set -x OPTIX_ROOT /opt/optix
FISHSKELEOF

# Java
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-17-jdk

# sbt
echo 'deb https://repo.scala-sbt.org/scalasbt/debian all main' | sudo tee /etc/apt/sources.list.d/sbt.list
curl -sL 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823' | sudo apt-key add
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y sbt

# Coursier
curl -fL https://github.com/coursier/launchers/raw/master/cs-x86_64-pc-linux.gz | gzip -d | sudo tee /usr/local/bin/cs > /dev/null
sudo chmod +x /usr/local/bin/cs

# CMake wrapper to suppress sbt-jni version parsing warning
sudo tee /usr/local/bin/cmake > /dev/null <<'CMAKEEOF'
#!/bin/bash
# Wrapper for cmake that filters out the sbt-jni version parsing warning
# Redirect stderr to stdout, filter, then redirect back to stderr
exec 3>&1
/usr/bin/cmake "$@" 2>&1 >&3 3>&- | grep -v -E "(CMake Warning:|Ignoring extra path from command line|/build/[0-9]+\")" >&2
exit ${PIPESTATUS[0]}
CMAKEEOF
sudo chmod +x /usr/local/bin/cmake

# IntelliJ IDEA
sudo snap install intellij-idea-community --classic

# Clean up
sudo apt-get autoremove -y
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

echo "Provisioning completed at $(date)"
PROVISION_SCRIPT

echo "=== Provisioning complete ==="

# Verify OptiX installation
echo "=== Verifying OptiX installation ==="
ssh $SSH_OPTS ubuntu@$PUBLIC_IP 'bash -s' <<'VERIFY_SCRIPT'
# Set environment variables for verification
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export OPTIX_ROOT=/opt/optix

# Run verification (--no-gpu: build instance has no GPU driver; checks are warnings not failures)
chmod +x /tmp/verify-optix.sh
/tmp/verify-optix.sh --no-gpu
VERIFY_SCRIPT

if [ $? -ne 0 ]; then
    echo ""
    echo "=== ERROR: OptiX verification failed ==="
    echo "AMI build aborted. Please review the verification output above."
    echo ""
    BUILD_STATUS="FAILED: OptiX verification failed. Review the output above for details."
    exit 1
fi

echo "=== OptiX verification successful ==="

# Stop instance
echo "=== Stopping instance ==="
aws ec2 stop-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-stopped --region "$REGION" --instance-ids "$INSTANCE_ID"

# Create AMI
AMI_NAME="menger-nvidia-dev-$(date +%Y%m%d-%H%M%S)"
echo "=== Creating AMI: $AMI_NAME ==="
AMI_ID=$(aws ec2 create-image \
  --region "$REGION" \
  --instance-id "$INSTANCE_ID" \
  --name "$AMI_NAME" \
  --description "Menger NVIDIA dev environment with CUDA 12.8, OptiX, Scala, IntelliJ" \
  --tag-specifications "ResourceType=image,Tags=[{Key=Name,Value=$AMI_NAME},{Key=Project,Value=menger},{Key=CUDAVersion,Value=12.8}]" \
  --query 'ImageId' \
  --output text)

echo "AMI ID: $AMI_ID"
echo "=== Waiting for AMI to be available (up to 20 minutes) ==="
AMI_READY=false
for i in $(seq 1 80); do
  AMI_STATE=$(aws ec2 describe-images --region "$REGION" --image-ids "$AMI_ID" \
    --query 'Images[0].State' --output text 2>/dev/null || echo "unknown")
  if [ "$AMI_STATE" = "available" ]; then
    AMI_READY=true
    break
  elif [ "$AMI_STATE" = "failed" ]; then
    BUILD_STATUS="FAILED: AMI $AMI_ID entered failed state. Check AWS console for details."
    exit 1
  fi
  echo -n "."
  sleep 15
done
echo ""

if [ "$AMI_READY" = false ]; then
  BUILD_STATUS="TIMED OUT: AMI $AMI_ID was created but did not reach 'available' state within 20 minutes.
  The AMI may still become available. Check with:
    aws ec2 describe-images --region $REGION --image-ids $AMI_ID --query 'Images[0].State'
  Once available, add it to the registry manually:
    ts=\$(date -u +%Y-%m-%dT%H:%M:%SZ) ; printf '%s\t%s\t%s\t%s\n' '$REGION' '$AMI_ID' '$AMI_NAME' \"\$ts\" >> scripts/ami-registry.tsv"
  exit 1
fi

# Record AMI ID in version-controlled registry
AMI_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
printf '%s\t%s\t%s\t%s\n' "$REGION" "$AMI_ID" "$AMI_NAME" "$AMI_TIMESTAMP" >> "$AMI_REGISTRY"

# Deregister previous AMIs for this region if requested
if [ "$DEREGISTER_OLD" = true ]; then
  echo "=== Deregistering previous AMIs for region $REGION ==="
  grep '^[^#]' "$AMI_REGISTRY" | grep "^$REGION	" | grep -v "	$AMI_ID	" | \
  while IFS=$'\t' read -r reg old_id old_name old_ts; do
    deregister_ami "$old_id" "$reg"
  done
fi

echo ""
echo "=== AMI Build Complete ==="
echo "AMI ID:   $AMI_ID"
echo "AMI Name: $AMI_NAME"
echo "Region:   $REGION"
echo "Registry: $AMI_REGISTRY"
echo ""
BUILD_STATUS="SUCCESS: AMI $AMI_ID ($AMI_NAME) is available in $REGION."
