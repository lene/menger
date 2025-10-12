#!/bin/bash
# Script to build custom AMI with CUDA 12.8, OptiX, and dev tools
# Alternative to Packer for users who prefer AWS CLI

set -e

# Configuration
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
OPTIX_INSTALLER="${1:-}"

if [ -z "$OPTIX_INSTALLER" ]; then
  echo "Usage: $0 <path-to-optix-installer.sh>"
  echo "Download OptiX SDK from: https://developer.nvidia.com/optix"
  exit 1
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
cleanup() {
  echo "=== Cleaning up ==="
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" 2>/dev/null || true
  aws ec2 delete-key-pair --region "$REGION" --key-name "$KEY_NAME" 2>/dev/null || true
  rm -f "$KEY_FILE"
  echo "Note: Security group $SG_ID left for reuse. Delete manually if needed:"
  echo "  aws ec2 delete-security-group --region $REGION --group-id $SG_ID"
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

# Copy OptiX installer
echo "=== Uploading OptiX installer ==="
scp $SSH_OPTS "$OPTIX_INSTALLER" ubuntu@$PUBLIC_IP:/tmp/optix-installer.sh

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
  build-essential git curl wget vim htop tmux unzip jq software-properties-common

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

# Install NVIDIA drivers
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install --gpgpu

# Install CUDA 12.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-8
rm cuda-keyring_1.1-1_all.deb

# Install OptiX
sudo mkdir -p /opt/optix
chmod +x /tmp/optix-installer.sh
sudo /tmp/optix-installer.sh --skip-license --prefix=/opt/optix
rm /tmp/optix-installer.sh

# Set environment variables in skeleton
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' | sudo tee -a /etc/skel/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/skel/.bashrc
echo 'export OPTIX_ROOT=/opt/optix' | sudo tee -a /etc/skel/.bashrc
sudo mkdir -p /etc/skel/.config/fish
echo 'set -x PATH /usr/local/cuda-12.8/bin $PATH' | sudo tee /etc/skel/.config/fish/config.fish
echo 'set -x LD_LIBRARY_PATH /usr/local/cuda-12.8/lib64 $LD_LIBRARY_PATH' | sudo tee -a /etc/skel/.config/fish/config.fish
echo 'set -x OPTIX_ROOT /opt/optix' | sudo tee -a /etc/skel/.config/fish/config.fish

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

# IntelliJ IDEA
sudo snap install intellij-idea-community --classic

# Clean up
sudo apt-get autoremove -y
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

echo "Provisioning completed at $(date)"
PROVISION_SCRIPT

echo "=== Provisioning complete ==="

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
echo "=== Waiting for AMI to be available (this may take several minutes) ==="
aws ec2 wait image-available --region "$REGION" --image-ids "$AMI_ID"

echo ""
echo "=== AMI Build Complete ==="
echo "AMI ID: $AMI_ID"
echo "AMI Name: $AMI_NAME"
echo "Region: $REGION"
echo ""
echo "To use this AMI, update terraform/terraform.tfvars:"
echo "  ami_id = \"$AMI_ID\""
echo ""
