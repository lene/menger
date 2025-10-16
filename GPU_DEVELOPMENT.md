# GPU Development Guide

Comprehensive guide for NVIDIA GPU development using automated AWS EC2 spot instances with CUDA
and OptiX.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Workflow](#workflow)
  - [1. AMI Creation (One-Time Setup)](#1-ami-creation-one-time-setup)
  - [2. Launching Instances](#2-launching-instances)
  - [3. Working on the Instance](#3-working-on-the-instance)
  - [4. State Management](#4-state-management)
  - [5. Termination](#5-termination)
- [Testing & Validation](#testing--validation)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Instance Types & Costs](#instance-types--costs)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## Overview

This system provides automated provisioning of AWS EC2 spot instances with NVIDIA GPUs for
CUDA/OptiX development. It includes:

**Features:**
- **One-command launch** of pre-configured GPU instances
- **State management** - automatically save and restore your work between sessions
- **Auto-termination** - instance shuts down 5 minutes after logout (prevents forgotten instances)
- **X11 forwarding** - run GUI applications (IntelliJ, menger) remotely
- **Pre-installed** - CUDA 12.8, OptiX, Scala, sbt, IntelliJ, Fish shell
- **Dry-run mode** - validate configuration without creating resources
- **Cost-effective** spot instances (typically 50-70% cheaper than on-demand)
- **Cost controls** - configurable maximum hourly and session costs

**Use Cases:**
- CUDA/OptiX development without local NVIDIA GPU
- Testing on different GPU architectures
- Access to high-performance GPUs (V100, A10G, etc.)
- Development from any machine (laptop, desktop, even tablet with SSH)

---

## Prerequisites

Before you begin, ensure you have:

### Local Machine
- **AWS CLI** configured with credentials
  ```bash
  aws configure
  ```
- **Terraform** >= 1.5.0
  ```bash
  # Check version
  terraform version
  ```
- **SSH key pair** at `~/.ssh/id_rsa.pub` (or custom path)
  ```bash
  # Generate if needed
  ssh-keygen -t rsa -b 4096
  ```
- **Git** configured with user.name and user.email
  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "your@email.com"
  ```
- **Required tools**: `jq`, `rsync` (for state management)
  ```bash
  # Ubuntu/Debian
  sudo apt-get install jq rsync

  # macOS
  brew install jq rsync
  ```

### AWS Account
- **IAM permissions** for EC2 operations:
  - `ec2:RunInstances`
  - `ec2:CreateImage`
  - `ec2:TerminateInstances`
  - `ec2:Describe*`
  - `ec2:CreateSecurityGroup`
  - `ec2:CreateKeyPair`

### OptiX SDK
- Download from [developer.nvidia.com/optix](https://developer.nvidia.com/optix)
- Requires free NVIDIA developer account
- Linux x86_64 version (e.g., `NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh`)

---

## Quick Start

```bash
# 1. Test your AWS configuration (optional but recommended)
./scripts/test-aws-config.sh

# 2. Build custom AMI with NVIDIA drivers and dev tools (one-time, ~30-60 min, $0.15-0.30)
./scripts/validate-ami-build.sh /path/to/NVIDIA-OptiX-SDK-*.sh  # Validate first
./scripts/build-ami.sh /path/to/NVIDIA-OptiX-SDK-*.sh           # Then build

# 3. Launch instance (note AMI ID from step 2)
./scripts/nvidia-spot.sh --ami-id ami-xxxxxxxxxxxx

# 4. Work on the instance (automatically connected via SSH with X11 forwarding)
nvidia-smi                    # Check GPU
cd ~/workspace/menger
sbt compile
sbt test

# 5. Logout (Ctrl+D) - instance auto-saves state and terminates after 5 minutes
```

**Next launch** (resumes where you left off):
```bash
./scripts/nvidia-spot.sh --ami-id ami-xxxxxxxxxxxx
# Your work is automatically restored from 'last' saved state!
# No need to specify --restore-state unless you want a different checkpoint
```

---

## Workflow

### 1. AMI Creation (One-Time Setup)

The AMI (Amazon Machine Image) is a pre-configured template with all development tools
installed. You create it once and reuse it for all instances.

#### Step 1.1: Download OptiX

1. Go to https://developer.nvidia.com/optix
2. Sign in or create free NVIDIA developer account
3. Download OptiX SDK for Linux x86_64 (e.g., version 9.0.0)
4. Note the path to the downloaded `.sh` file

#### Step 1.2: Validate Configuration (Recommended)

Before the expensive AMI build, validate your setup:

```bash
./scripts/validate-ami-build.sh /path/to/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh
```

This checks:
-  OptiX installer exists and is valid
-  AWS credentials and permissions
-  Base Ubuntu AMI availability
-  Script syntax
-  Estimated costs and time

#### Step 1.3: Build AMI

```bash
./scripts/build-ami.sh /path/to/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh
```

**What happens:**
1. Launches temporary g4dn.xlarge instance (~$0.30/hr)
2. Installs Ubuntu 24.04 updates
3. Installs NVIDIA drivers (compatible with GPU)
4. Installs CUDA Toolkit 12.8
5. Installs OptiX SDK
6. Installs development tools:
   - Java 17 JDK
   - sbt (Scala build tool)
   - Coursier
   - IntelliJ IDEA Community
   - Fish shell
   - X11 support
   - AWS CLI
   - Claude Code
7. Configures environment variables (CUDA_HOME, OPTIX_ROOT, etc.)
8. Verifies installation with comprehensive tests
9. Creates AMI snapshot
10. Terminates temporary instance

**Time:** 30-60 minutes
**Cost:** $0.15-$0.30 (one-time)

**Output:** AMI ID (e.g., `ami-0a1b2c3d4e5f6g7h8`)
- Save this ID for launching instances
- AMI can be reused indefinitely at no additional cost

#### Step 1.4: List Available Instance Types

```bash
./scripts/nvidia-spot.sh --list-instances
```

Shows available GPU instance types in your region with current spot prices. Note that retrieving the
available instances can take a minute or two.

---

### 2. Launching Instances

#### Basic Launch

```bash
./scripts/nvidia-spot.sh --ami-id ami-xxxxxxxxxxxx
```

This:
1. Creates Terraform configuration
2. Launches spot instance with your AMI
3. Waits for instance to be ready
4. Configures git identity (from your local machine)
5. **Auto-restores** your last saved state (if exists)
6. Connects via SSH with X11 forwarding

#### Custom Launch Options

```bash
# Different region
./scripts/nvidia-spot.sh --ami-id ami-xxx --region us-west-2

# Specific availability zone (useful for GPU availability)
./scripts/nvidia-spot.sh --ami-id ami-xxx --availability-zone us-east-1b

# Different instance type
./scripts/nvidia-spot.sh --ami-id ami-xxx --instance-type g5.xlarge

# Higher spot price (for better availability)
./scripts/nvidia-spot.sh --ami-id ami-xxx --max-price 0.75

# Restore specific state instead of 'last'
./scripts/nvidia-spot.sh --ami-id ami-xxx --restore-state my-checkpoint

# Fresh start (don't restore previous state)
./scripts/nvidia-spot.sh --ami-id ami-xxx --no-auto-restore

# Create named checkpoint on logout
./scripts/nvidia-spot.sh --ami-id ami-xxx --save-state before-refactor
```

#### Running Commands Without Interactive Session

```bash
# Run command and auto-terminate
./scripts/nvidia-spot.sh --ami-id ami-xxx --command "cd menger && sbt test"

# Useful for CI/CD or batch processing
./scripts/nvidia-spot.sh --ami-id ami-xxx --command "nvidia-smi && nvcc --version"
```

---

### 3. Working on the Instance

Once connected, you're in a fully configured development environment.

#### Verify GPU

```bash
# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Check OptiX (compile test)
cd /opt/optix
ls include/  # Should see optix.h, optix_stubs.h, etc.
```

#### Development Workflow

```bash
# Navigate to workspace (menger repo already cloned)
cd ~/workspace/menger

# Compile
sbt compile

# Run tests
sbt test

# Run application
sbt run

# Run with X11 forwarding (3D visualization)
xvfb-run -a sbt "run --level 2 --timeout 5"

# Start IntelliJ IDEA (GUI via X11)
intellij-idea-community &
```

#### Multiple Terminal Sessions

You can connect additional SSH sessions manually:

```bash
# From another local terminal
ssh -X ubuntu@<instance-ip>
```

**Note:** Auto-termination is triggered when you logout from the **main session** created by the
nvidia-spot.sh script, regardless of other SSH connections. If you need to keep the instance
running while working across multiple sessions, use `--no-auto-terminate` when launching.

#### Git Workflow

Your git identity is automatically configured from your local machine:

```bash
git status
git add .
git commit -m "Implement feature"
git push
```

---

### 4. State Management

State management automatically saves your work between sessions.

#### Automatic State Management

**On logout:**
- State is automatically saved as 'last'
- Stored locally at: `~/.aws/spot-states/last/` (on your local machine, not AWS)
- Includes:
  - Workspace (`~/workspace/`)
  - Configurations (`~/.config/`)
  - SSH keys (`~/.ssh/`)
  - Shell histories
  - Git repositories with uncommitted changes

**On next launch:**
- The 'last' state is automatically restored from `~/.aws/spot-states/last/`
- Everything is restored automatically to the new instance
- Continue exactly where you left off
- Use `--no-auto-restore` to skip automatic restoration

#### Manual State Management

```bash
# List all saved states
./scripts/nvidia-spot.sh --list-states

# Create named checkpoint (while instance is running)
./scripts/backup-spot-state.sh my-checkpoint

# Restore specific state on launch
./scripts/nvidia-spot.sh --ami-id ami-xxx --restore-state my-checkpoint

# Save with custom name on logout
./scripts/nvidia-spot.sh --ami-id ami-xxx --save-state before-big-change
```

#### State Management Scripts

**Backup state** (manual):
```bash
./scripts/backup-spot-state.sh STATE_NAME [INSTANCE_IP]
```

**Restore state** (manual):
```bash
./scripts/restore-spot-state.sh STATE_NAME [INSTANCE_IP]
```

**List states**:
```bash
./scripts/list-spot-states.sh
```

**Cleanup old states**:
```bash
# Dry-run to see what would be deleted
./scripts/cleanup-spot-states.sh --dry-run --older-than-days 30

# Actually delete
./scripts/cleanup-spot-states.sh --older-than-days 30

# Keep only 5 most recent
./scripts/cleanup-spot-states.sh --keep-recent 5
```

#### What Gets Saved

**Included** (~5-10 GB, 2-5 minutes):
- ✅ All files in `~/workspace/`
- ✅ Configuration files (`~/.config/`)
- ✅ SSH keys and config
- ✅ Git repositories (including uncommitted changes)
- ✅ Shell histories (bash, fish)
- ✅ Dotfiles (`.gitconfig`, `.vimrc`, `.tmux.conf`)

**Excluded** (regenerated on demand):
- ❌ Build artifacts (`target/`, `*.class`)
- ❌ IDE caches (`.metals/`, `.bsp/`)
- ❌ SBT/Ivy caches (downloaded fresh)
- ❌ Temporary files

**Storage:** States are stored locally at `~/.aws/spot-states/` (no AWS costs).

---

### 5. Termination

#### Automatic Termination (Default)

1. Logout from SSH (Ctrl+D or `exit`)
2. Instance **backs up your state** to local machine
3. Waits **5 minutes** (grace period)
4. Terminates instance automatically

You can cancel auto-termination during the grace period with Ctrl+C.

#### Manual Termination

```bash
# Terminate immediately
./scripts/nvidia-spot.sh --terminate

# Or using Terraform directly
cd terraform
terraform destroy -auto-approve
```

#### Keep Instance Running

```bash
# Launch without auto-termination
./scripts/nvidia-spot.sh --ami-id ami-xxx --no-auto-terminate

# On logout, you'll be prompted:
# Destroy instance now? [y/N]
```

#### Check Running Instances

```bash
./scripts/nvidia-spot.sh --list-running
```

---

## Testing & Validation

Test your configuration **without creating resources** using dry-run mode.

### Test Scripts Overview

| Script | Purpose | Cost | Time |
|--------|---------|------|------|
| `test-aws-config.sh` | AWS CLI, credentials, permissions | $0 | 2-3 min |
| `validate-ami-build.sh` | AMI build configuration | $0 | 2-3 min |
| `test-terraform-config.sh` | Terraform configuration + plan | $0 | 2-3 min |
| `test-state-management.sh` | State management scripts | $0 | 1-2 min |

### 1. Test AWS Configuration

```bash
./scripts/test-aws-config.sh

# With options
./scripts/test-aws-config.sh --region us-west-2 --instance-type g5.xlarge --verbose
```

**Tests:**
- AWS CLI installation
- AWS credentials validity
- Region accessibility
- VPC and subnet availability
- Instance type availability
- Spot pricing
- IAM permissions (using `--dry-run`)
- Required tools (jq, rsync, etc.)

### 2. Validate AMI Build

```bash
./scripts/validate-ami-build.sh /path/to/NVIDIA-OptiX-SDK-*.sh
```

**Tests:**
- OptiX installer validity
- Build script syntax
- AWS resources (base AMI, security group)
- Instance launch permissions
- AMI creation permissions
- Provisioning script contents
- Cost and time estimates

### 3. Test Terraform Configuration

```bash
./scripts/test-terraform-config.sh ami-xxxxxxxxxxxx
```

**Tests:**
- Terraform installation
- Syntax and formatting
- Variable validation
- **Generates Terraform plan** (shows exactly what will be created)
- Cost estimation
- State file status

### 4. Test State Management

```bash
./scripts/test-state-management.sh
```

**Tests:**
- Script existence and syntax
- Required tools
- Mock state creation
- List/restore/cleanup functionality
- No AWS credentials required

### AWS Dry-Run Mode

AWS supports `--dry-run` for many EC2 operations to validate requests without executing them:

```bash
# Test instance launch
aws ec2 run-instances \
  --image-id ami-xxx \
  --instance-type g4dn.xlarge \
  --dry-run

# Expected output if you have permissions:
# "DryRunOperation: Request would have succeeded, but DryRun flag is set"

# Expected output if missing permissions:
# "UnauthorizedOperation: You are not authorized..."
```

Dry-run validates:
- ✅ IAM permissions
- ✅ Parameter correctness
- ✅ Service quotas
- ✅ Resource availability

But cannot test:
- ❌ Actual software installation
- ❌ Network connectivity
- ❌ Runtime errors
- ❌ Spot availability

### Recommended Testing Workflow

**Before AMI build:**
```bash
./scripts/test-aws-config.sh
./scripts/validate-ami-build.sh /path/to/optix.sh
# Fix any issues, then build
./scripts/build-ami.sh /path/to/optix.sh
```

**Before instance launch:**
```bash
./scripts/test-terraform-config.sh ami-xxx
# Review plan, then launch
./scripts/nvidia-spot.sh --ami-id ami-xxx
```

**Small-scale test** (optional, ~$0.10):
```bash
./scripts/nvidia-spot.sh --ami-id ami-xxx \
  --command "nvidia-smi && nvcc --version && echo 'Success'"
```

**For detailed testing documentation, see [TESTING.md](TESTING.md).**

---

## CLI Reference

### nvidia-spot.sh

Main CLI for managing spot instances.

**Usage:**
```bash
./scripts/nvidia-spot.sh [OPTIONS]
```

**Options:**
- `--ami-id AMI_ID` - Custom AMI ID (required for launch)
- `--region REGION` - AWS region (default: from `~/.aws/config` or `us-east-1`)
- `--availability-zone AZ` - AWS availability zone (e.g., `us-east-1a`)
- `--instance-type TYPE` - Instance type (default: `g4dn.xlarge`)
- `--max-price PRICE` - Maximum spot price per hour (default: `0.50`)
- `--max-cost COST` - Maximum session cost (default: `10.00`)
- `--ssh-key PATH` - Path to SSH public key (default: `~/.ssh/id_rsa.pub`)
- `--list-instances` - List available NVIDIA instances and spot prices
- `--list-running` - Show currently running instances
- `--list-states` - List all saved instance states
- `--terminate` - Terminate all running instances
- `--restore-state NAME` - Restore specific saved state on launch
- `--save-state NAME` - Save state with given name before shutdown
- `--no-auto-restore` - Don't automatically restore 'last' state
- `--command "CMD"` - Run command and auto-terminate
- `--no-auto-terminate` - Disable auto-termination on logout
- `-h, --help` - Show help message

**Examples:**
```bash
# List instances in different region
./scripts/nvidia-spot.sh --list-instances --region eu-west-1

# Launch with specific availability zone
./scripts/nvidia-spot.sh --ami-id ami-xxx --availability-zone us-east-1c

# Launch larger instance with higher price
./scripts/nvidia-spot.sh --ami-id ami-xxx \
  --instance-type g5.2xlarge --max-price 1.00

# Restore named checkpoint
./scripts/nvidia-spot.sh --ami-id ami-xxx --restore-state my-checkpoint

# Run tests and terminate
./scripts/nvidia-spot.sh --ami-id ami-xxx --command "cd menger && sbt test"
```

### Other Scripts

**AMI Management:**
```bash
./scripts/build-ami.sh OPTIX_INSTALLER    # Build custom AMI
./scripts/validate-ami-build.sh INSTALLER # Validate before building
./scripts/verify-optix.sh                 # Verify OptiX installation
```

**State Management:**
```bash
./scripts/backup-spot-state.sh NAME [IP]  # Backup state
./scripts/restore-spot-state.sh NAME [IP] # Restore state
./scripts/list-spot-states.sh             # List states
./scripts/cleanup-spot-states.sh OPTIONS  # Cleanup old states
```

**Testing:**
```bash
./scripts/test-aws-config.sh OPTIONS      # Test AWS setup
./scripts/validate-ami-build.sh INSTALLER # Validate AMI build
./scripts/test-terraform-config.sh AMI_ID # Test Terraform
./scripts/test-state-management.sh        # Test state scripts
```

**Instance Management:**
```bash
./scripts/list-instances.sh [REGION]      # List GPU instances and prices
```

---

## Configuration

### Environment Variables

Customize behavior with environment variables:

```bash
# Custom state storage directory
export SPOT_STATES_DIR=~/my-spot-states

# AWS region
export AWS_REGION=us-west-2

# Instance type for AMI build
export INSTANCE_TYPE=g4dn.xlarge
```

### Terraform Variables

Located at `terraform/terraform.tfvars` (auto-generated by `nvidia-spot.sh`):

```hcl
region             = "us-east-1"
availability_zone  = "us-east-1a"  # Optional
instance_type      = "g4dn.xlarge"
max_spot_price     = "0.50"
max_session_cost   = 10.00
ami_id             = "ami-xxxxxxxxxxxx"
auto_terminate     = true
```

### Auto-Termination Behavior

- **Triggered on:** SSH logout/disconnect
- **State backup:** Runs immediately after logout
- **Grace period:** 5-minute countdown starts after backup completes
- **Location:** Countdown runs on your local machine (not the instance)
- **Cancellable:** Press Ctrl+C during the 5-minute grace period to keep instance running
- **Note:** Auto-termination is tied to the main SSH session created by the script. Additional
  SSH connections you manually create won't prevent termination after the main session ends.

---

## Instance Types & Costs

### Recommended GPU Instance Types

| Instance | GPU | vCPU | RAM | Use Case | Typical Spot Price |
|----------|-----|------|-----|----------|-------------------|
| **g4dn.xlarge** | T4 | 4 | 16GB | Development, cost-effective | $0.15-0.30/hr |
| **g4dn.2xlarge** | T4 | 8 | 32GB | Development + more resources | $0.25-0.45/hr |
| **g5.xlarge** | A10G | 4 | 16GB | Newer GPU, better performance | $0.30-0.60/hr |
| **g5.2xlarge** | A10G | 8 | 32GB | Production workloads | $0.45-0.85/hr |
| **p3.2xlarge** | V100 | 8 | 61GB | High-performance computing | $0.90-1.50/hr |

**Note:** Spot prices vary by region and availability. Run
`./scripts/nvidia-spot.sh --list-instances` for current prices.

### Cost Examples

**Typical development session (g4dn.xlarge @ $0.20/hr):**
- 2 hours: $0.40
- 4 hours: $0.80
- 8 hours: $1.60
- 40 hours/week: $8.00

**With state management:**
- Work is saved automatically
- No cost when not actively using
- Optimal cost-effectiveness

**AMI Build (one-time):**
- Cost: $0.15-0.30
- Reusable indefinitely

**Storage costs:**
- AMI snapshot: ~$2.50/month (50GB used)
- Local state backups: $0 (on your machine)

---

## Troubleshooting

### AMI Build Issues

**OptiX installer fails:**
```bash
# Ensure correct Linux x86_64 version downloaded
ls -lh /path/to/NVIDIA-OptiX-SDK-*.sh

# Check if executable
chmod +x /path/to/NVIDIA-OptiX-SDK-*.sh
```

**NVIDIA driver installation fails:**
```bash
# Use GPU-capable instance type
export INSTANCE_TYPE=g4dn.xlarge
./scripts/build-ami.sh /path/to/optix.sh
```

**Build timeout:**
- Normal build time: 30-60 minutes
- Check CloudWatch logs in AWS console
- SSH to build instance to debug: `ssh ubuntu@<build-ip>`

### Instance Launch Issues

**Spot request not fulfilled:**
```bash
# Check current prices
./scripts/nvidia-spot.sh --list-instances

# Try higher max price
./scripts/nvidia-spot.sh --ami-id ami-xxx --max-price 0.75

# Try different availability zone
./scripts/nvidia-spot.sh --ami-id ami-xxx --availability-zone us-east-1b

# Try different region
./scripts/nvidia-spot.sh --ami-id ami-xxx --region us-west-2
```

**AMI not found:**
```bash
# Ensure AMI is in the same region
aws ec2 describe-images --image-ids ami-xxx --region us-east-1

# Copy AMI to different region if needed
aws ec2 copy-image --source-region us-east-1 --source-image-id ami-xxx \
  --region us-west-2 --name "menger-nvidia-dev"
```

**Permission denied:**
```bash
# Check SSH key
ls -l ~/.ssh/id_rsa.pub

# Use custom key
./scripts/nvidia-spot.sh --ami-id ami-xxx --ssh-key /path/to/key.pub
```

### Connection Issues

**SSH timeout:**
```bash
# Instance initialization takes 2-3 minutes
# Wait and check status
./scripts/nvidia-spot.sh --list-running

# Check security group allows SSH from your IP
aws ec2 describe-security-groups --group-names menger-nvidia-dev-sg
```

**X11 not working:**
```bash
# On macOS, ensure XQuartz is running
open -a XQuartz

# On Linux, ensure X11 forwarding enabled
ssh -X -o ForwardX11=yes ubuntu@<ip>

# Test X11
xeyes
```

### State Management Issues

**Backup fails:**
```bash
# Check required tools installed
which rsync jq

# Check connectivity
ssh ubuntu@<ip> exit

# Check disk space locally
df -h ~/.aws/spot-states/
```

**Restore fails:**
```bash
# List available states
./scripts/list-spot-states.sh

# Check state exists
ls -la ~/.aws/spot-states/STATE_NAME/

# Manual restore
./scripts/restore-spot-state.sh STATE_NAME <ip>
```

**jq not found:**
```bash
# Ubuntu/Debian
sudo apt-get install jq rsync

# macOS
brew install jq rsync
```

### Auto-Terminate Issues

**Instance doesn't terminate:**
```bash
# Check if all SSH sessions closed
./scripts/nvidia-spot.sh --list-running

# Manual termination
./scripts/nvidia-spot.sh --terminate
```

**State not saved:**
```bash
# Manually backup before terminate
./scripts/backup-spot-state.sh my-backup <ip>
```

### AWS Credentials Issues

**Credentials not configured:**
```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region, Output format

# Or use environment variables
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx
export AWS_REGION=us-east-1
```

**Permission denied errors:**
```bash
# Test permissions with dry-run
aws ec2 run-instances --image-id ami-xxx --instance-type t2.micro --dry-run

# Check IAM permissions include:
# - ec2:RunInstances
# - ec2:CreateImage
# - ec2:TerminateInstances
# - ec2:Describe*
```

---

## Advanced Topics

### Custom AMI Modifications

Edit `scripts/build-ami.sh` provisioning script to add/modify:
- Additional software packages
- Environment variables
- System configurations
- User preferences

### Persistent Data with EFS

For data that persists across sessions without state management:

```bash
# Create EFS (one-time)
aws efs create-file-system --region us-east-1

# Get file system ID
aws efs describe-file-systems

# Mount in user-data (edit terraform/user-data.sh):
sudo mount -t efs fs-xxxxx:/ /mnt/persistent
```

### Multiple AMIs

Create different AMIs for different purposes by modifying the build script.

### Cost Monitoring

Set up AWS Budget alerts:

```bash
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget file://budget.json
```

### Terraform Customization

For advanced Terraform modifications, see `terraform/` directory:
- `main.tf` - Resource definitions
- `variables.tf` - Input variables
- `outputs.tf` - Output values
- `versions.tf` - Provider versions
- `user-data.sh` - Instance initialization

See [terraform/README.md](terraform/README.md) for technical details.

### Integration with CI/CD

Use for automated testing:

```bash
# In CI pipeline
./scripts/nvidia-spot.sh --ami-id $AMI_ID \
  --command "cd menger && sbt clean compile test" \
  --max-cost 5.00

# Instance auto-terminates after tests complete
```

---

## Summary

This GPU development system provides:

✅ **Easy setup** - One-time AMI build, then launch instances with single command
✅ **Cost-effective** - Spot pricing + auto-termination = pay only for active use
✅ **Productive** - Full dev environment with state management
✅ **Safe** - Testing tools catch issues before spending money
✅ **Flexible** - Multiple instance types, regions, configurations

**Typical workflow:**
1. Build AMI once (~$0.30, 40 minutes)
2. Launch instance when needed (~$0.20/hour)
3. Work with full GPU access
4. Logout - work auto-saved
5. Next time: pick up exactly where you left off

**Questions or issues?** See [Troubleshooting](#troubleshooting) or open an issue.