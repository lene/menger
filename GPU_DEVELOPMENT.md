# GPU Development Guide

Comprehensive guide for NVIDIA GPU development using either local workstation setup or automated
AWS EC2 spot instances with CUDA and OptiX.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
  - [Automated Setup (Recommended)](#automated-setup-recommended)
  - [Manual Setup](#manual-setup)
  - [Fish Shell Configuration](#fish-shell-configuration)
  - [Verification](#verification)
- [Quick Start](#quick-start)
- [AWS Cloud Development](#aws-cloud-development)
  - [1. AMI Creation (One-Time Setup)](#1-ami-creation-one-time-setup)
  - [2. Launching Instances](#2-launching-instances)
  - [3. Working on the Instance](#3-working-on-the-instance)
  - [4. State Management](#4-state-management)
  - [5. Termination](#5-termination)
- [Testing & Validation](#testing--validation)
- [Detailed Testing Guide](#detailed-testing-guide)
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

## Local Development Setup

For developers who have an NVIDIA GPU on their local machine, you can set up OptiX development
without using AWS cloud resources.

### Automated Setup (Recommended)

The automated setup script handles NVIDIA drivers, CUDA 12.8, and OptiX SDK installation:

```bash
# Download OptiX SDK first from https://developer.nvidia.com/optix
# Then run:
./scripts/setup-optix-local.sh /path/to/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh
```

**What it does:**
1. ✅ Checks for NVIDIA GPU
2. ✅ Installs/updates NVIDIA drivers
3. ✅ Installs CUDA Toolkit 12.8
4. ✅ Installs OptiX SDK
5. ✅ Installs development tools (g++, cmake)
6. ✅ Configures environment variables (Bash/Zsh)
7. ✅ Generates Fish shell configuration
8. ✅ Runs comprehensive verification

**Supported platforms:** Ubuntu 22.04+, Debian 12+

**Time:** 15-30 minutes (includes driver installation and potential reboot)

### Manual Setup

If you prefer manual setup or the automated script doesn't work for your system:

#### 1. Install NVIDIA Drivers

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ubuntu-drivers-common
sudo ubuntu-drivers install --gpgpu

# Verify
nvidia-smi
```

#### 2. Install CUDA Toolkit 12.8

```bash
# Add CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA
sudo apt-get install -y cuda-toolkit-12-8

# Verify
/usr/local/cuda-12.8/bin/nvcc --version
```

#### 3. Install OptiX SDK

```bash
# Download from https://developer.nvidia.com/optix (requires free account)
# Make executable and run:
chmod +x NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh
sudo ./NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh --prefix=/usr/local
```

#### 4. Install Development Tools

```bash
sudo apt-get install -y build-essential cmake git
```

#### 5. Configure Environment Variables

**For Bash/Zsh:**
```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.8
export OPTIX_ROOT=/usr/local/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64

# Apply changes
source ~/.bashrc  # or ~/.zshrc
```

**For Fish shell:** See next section.

### Fish Shell Configuration

Fish shell users can use the provided environment configuration script:

```fish
# One-time setup: Add to ~/.config/fish/config.fish
source /path/to/menger/scripts/setup-optix-env.fish

# Or source manually each session
source scripts/setup-optix-env.fish
```

The Fish configuration script:
- ✅ Sets `PATH` with CUDA binaries
- ✅ Sets `LD_LIBRARY_PATH` with CUDA libraries
- ✅ Sets `CUDA_HOME` and `OPTIX_ROOT`
- ✅ Provides quick verification function: `verify-optix-env`
- ✅ Shows color-coded status on load

**Quick verification in Fish:**
```fish
verify-optix-env
```

### Verification

After setup (automated or manual), verify your installation:

```bash
# Run comprehensive verification
./scripts/verify-optix.sh
```

This checks:
- ✅ NVIDIA driver version and compatibility
- ✅ GPU detection and compute capability
- ✅ CUDA installation and tools
- ✅ CUDA libraries in system
- ✅ OptiX SDK headers and version
- ✅ Environment variables
- ✅ Compilation test with OptiX headers
- ✅ System information

**Example output:**
```
=== OptiX Installation Verification ===

1. NVIDIA Driver
✓ NVIDIA driver installed: version 545.29.06
✓ Driver version 545.29.06 supports OptiX 8.x
  NVIDIA GeForce RTX 3080

2. GPU Detection
✓ Detected 1 GPU(s)
  GPU 0: NVIDIA GeForce RTX 3080 (Compute 8.6, 10240 MiB)

3. CUDA Installation
✓ nvcc found: CUDA 12.8
  Location: /usr/local/cuda-12.8/bin/nvcc

...

=== Summary ===
Passed: 18
✓ OptiX environment is correctly configured
```

### Local Development Workflow

Once setup is complete:

```bash
cd ~/workspace/menger

# Compile project
sbt compile

# Run tests
sbt test

# Run application (interactive mode)
sbt run

# Run specific sponge type
sbt "run --level 2 --sponge-type tesseract-sponge"

# Run with animation
sbt "run --level 1 --animate frames=10:rot-y=0-360"
```

### Troubleshooting Local Setup

**Driver installation requires reboot:**
```bash
# After driver install
sudo reboot

# Then re-run setup or continue manually
```

**CUDA not in PATH:**
```bash
# Check environment
echo $PATH | grep cuda
echo $CUDA_HOME

# Re-source shell config
source ~/.bashrc  # or ~/.zshrc or config.fish
```

**OptiX headers not found:**
```bash
# Check installation
ls -la /usr/local/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/include/

# Verify OPTIX_ROOT
echo $OPTIX_ROOT

# Re-run verification
./scripts/verify-optix.sh
```

**Compilation fails:**
```bash
# Check g++ installation
g++ --version

# Install if missing
sudo apt-get install build-essential cmake
```

---

## AWS Cloud Development

For users without a local NVIDIA GPU, or who want access to high-performance cloud GPUs, the
project provides fully automated AWS EC2 spot instance provisioning.

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

Test your AWS configuration **without creating resources or incurring costs**. All test scripts use AWS dry-run mode to validate permissions, resources, and configurations before actual deployment.

### Quick Reference

| Script | What It Tests | Cost | Time |
|--------|--------------|------|------|
| `test-aws-config.sh` | AWS CLI, credentials, IAM permissions, resources | $0 | 2-3 min |
| `validate-ami-build.sh` | AMI build config, OptiX installer, permissions | $0 | 2-3 min |
| `test-terraform-config.sh` | Terraform config + plan generation | $0 | 2-3 min |
| `test-state-management.sh` | State management scripts (no AWS required) | $0 | 1-2 min |

### Quick Start Testing

**Before building AMI:**
```bash
./scripts/test-aws-config.sh
./scripts/validate-ami-build.sh /path/to/NVIDIA-OptiX-SDK-*.sh
```

**Before launching instance:**
```bash
./scripts/test-terraform-config.sh ami-xxxxxxxxxxxx
```

**Before using state management:**
```bash
./scripts/test-state-management.sh
```

**Recommendation:** Run all test scripts before major operations (AMI build, instance launch) to catch 90%+ of issues at zero cost.

For comprehensive testing documentation including detailed explanations, example outputs, troubleshooting, and limitations, see [Detailed Testing Guide](#detailed-testing-guide) below.

---

## Detailed Testing Guide

This section provides comprehensive documentation for testing your AWS infrastructure configuration
**without creating resources or incurring costs**. All test scripts use AWS dry-run mode and local
validation to catch issues before spending money.

### Understanding AWS Dry-Run Mode

AWS supports `--dry-run` for many EC2 operations. This validates your request without actually
executing it.

#### What Dry-Run Tests

✅ **Permissions**: Checks if you have the IAM permissions to perform the action
✅ **Parameters**: Validates that all parameters are correct
✅ **Quotas**: Checks if you've reached service limits
✅ **Resources**: Validates that referenced resources exist

#### Supported Commands

```bash
# Test instance launch
aws ec2 run-instances \
  --image-id ami-xxx \
  --instance-type g4dn.xlarge \
  --dry-run

# Test AMI creation
aws ec2 create-image \
  --instance-id i-xxx \
  --name "test-ami" \
  --dry-run

# Test snapshot creation
aws ec2 create-snapshot \
  --volume-id vol-xxx \
  --dry-run

# Test instance termination
aws ec2 terminate-instances \
  --instance-ids i-xxx \
  --dry-run
```

#### Expected Dry-Run Output

**Success** (you have permissions):
```
An error occurred (DryRunOperation) when calling the RunInstances operation:
Request would have succeeded, but DryRun flag is set.
```

**Failure** (missing permissions):
```
An error occurred (UnauthorizedOperation) when calling the RunInstances operation:
You are not authorized to perform this operation.
```

---

### Test Script 1: AWS Configuration (`test-aws-config.sh`)

**Purpose**: Validate AWS CLI, credentials, permissions, and region setup.

#### What It Tests

- ✅ AWS CLI installation and version
- ✅ AWS credentials validity (using `sts get-caller-identity`)
- ✅ Region accessibility
- ✅ Default VPC and subnet availability
- ✅ Instance type availability in region
- ✅ Spot instance pricing
- ✅ SSH key existence
- ✅ IAM permissions using `--dry-run` mode
- ✅ Terraform installation
- ✅ Required tools (jq, rsync, ssh, scp)

#### Usage

```bash
# Test default region
./scripts/test-aws-config.sh

# Test specific region
./scripts/test-aws-config.sh --region us-west-2

# Test specific instance type
./scripts/test-aws-config.sh --instance-type g5.xlarge

# Verbose output
./scripts/test-aws-config.sh --verbose
```

#### What It Does NOT Do

- ❌ Does not create any AWS resources
- ❌ Does not modify existing resources
- ❌ Does not incur any costs

#### Example Output

```
=== AWS Configuration Test ===
Region:        us-east-1
Instance Type: g4dn.xlarge

1. AWS CLI
✓ AWS CLI installed: aws-cli/2.13.0

2. AWS Credentials
✓ Credentials valid: arn:aws:iam::123456789012:user/myuser

3. Region Accessibility
✓ Region accessible: us-east-1

...

=== Summary ===
Passed:   15
Warnings: 2
✓ Tests passed with warnings. Review warnings above.
```

---

### Test Script 2: AMI Build Validation (`validate-ami-build.sh`)

**Purpose**: Validate AMI build configuration before running the expensive build process.

#### What It Tests

- ✅ OptiX installer file existence and size
- ✅ Build script syntax (`build-ami.sh`, `verify-optix.sh`)
- ✅ AWS resources (base AMI, security group, subnet)
- ✅ Instance launch permissions (dry-run)
- ✅ AMI creation permissions (dry-run)
- ✅ Provisioning script contents (CUDA, NVIDIA drivers, OptiX, Claude Code)
- ✅ Estimated costs and time
- ✅ Network connectivity requirements

#### Usage

```bash
# Validate with OptiX installer
./scripts/validate-ami-build.sh /path/to/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh

# Just validate scripts (without OptiX installer)
./scripts/validate-ami-build.sh
```

#### What It Does NOT Do

- ❌ Does not launch any instances
- ❌ Does not create AMIs
- ❌ Does not install any software

#### Example Output

```
=== AMI Build Validation ===

1. OptiX Installer
✓ OptiX installer found: /tmp/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh
✓ OptiX installer is executable
✓ OptiX installer size looks reasonable: 45MB

...

7. Estimated Costs
✓ Estimated cost: $0.15-$0.30 (30-60 min @ $0.30/hr)

...

✓ Validation passed! Ready to build AMI.

To build the AMI, run:
  scripts/build-ami.sh /tmp/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh
```

---

### Test Script 3: Terraform Configuration (`test-terraform-config.sh`)

**Purpose**: Validate Terraform configuration and generate plan without creating resources.

#### What It Tests

- ✅ Terraform installation and version
- ✅ All required Terraform files exist
- ✅ Terraform syntax and formatting
- ✅ Terraform initialization
- ✅ Variable validation
- ✅ Resource configuration (VPC, security group, spot instance, etc.)
- ✅ Security group rules
- ✅ User data script syntax
- ✅ Output definitions
- ✅ **Terraform plan generation** (dry-run equivalent)
- ✅ Cost estimation
- ✅ State file status

#### Usage

```bash
# Validate configuration (basic)
./scripts/test-terraform-config.sh

# Validate and generate plan with AMI
./scripts/test-terraform-config.sh ami-xxxxxxxxxxxx
```

#### What It Does NOT Do

- ❌ Does not apply Terraform changes
- ❌ Does not create any AWS resources
- ❌ Does not modify state file

#### Example Output

```
=== Terraform Configuration Test ===

1. Terraform Installation
✓ Terraform installed: v1.5.0

...

10. Terraform Plan (Dry-Run)
   Generating plan...
✓ Terraform plan successful
   Resources to create: 5

11. Cost Estimation
✓ Spot price: $0.30/hour
   Daily (24hrs):   $7.20
   Monthly (30d):   $216.00
   Note: Actual cost depends on usage time

...

✓ All tests passed! Terraform configuration ready.
```

#### Terraform Plan Explained

Terraform's `plan` command is the dry-run equivalent for infrastructure:

```bash
cd terraform

# Generate plan
terraform plan

# Generate plan and save it
terraform plan -out=plan.tfplan

# Show what would be created
terraform show plan.tfplan
```

**What `terraform plan` shows**:
- Resources to be created (+)
- Resources to be modified (~)
- Resources to be destroyed (-)
- Estimated changes

**Example output**:
```
Terraform will perform the following actions:

  # aws_spot_instance_request.nvidia_dev will be created
  + resource "aws_spot_instance_request" "nvidia_dev" {
      + ami                    = "ami-xxx"
      + instance_type          = "g4dn.xlarge"
      + spot_price             = "0.50"
      ...
    }

Plan: 5 to add, 0 to change, 0 to destroy.
```

---

### Test Script 4: State Management (`test-state-management.sh`)

**Purpose**: Test state management scripts using mock data (no instance required).

#### What It Tests

- ✅ All state management scripts exist
- ✅ Script syntax validation
- ✅ Script permissions (executable)
- ✅ Required tools (rsync, ssh, scp, jq, tar)
- ✅ Mock state creation and structure
- ✅ List states functionality
- ✅ Restore script (list mode)
- ✅ Cleanup script (dry-run mode)
- ✅ Cleanup filters (--older-than-days, --keep-recent)
- ✅ Environment variable override (SPOT_STATES_DIR)
- ✅ Metadata parsing
- ✅ Directory structure validation

#### Usage

```bash
# Run all state management tests
./scripts/test-state-management.sh
```

#### What It Does NOT Do

- ❌ Does not connect to any instances
- ❌ Does not require AWS credentials
- ❌ Does not modify real state backups

#### Example Output

```
=== State Management Test ===
Test directory: /tmp/tmp.abcdefg123

1. Script Files
✓ backup-spot-state.sh exists
✓ restore-spot-state.sh exists
✓ list-spot-states.sh exists
✓ cleanup-spot-states.sh exists

...

6. List States Script
✓ Lists test-state-1
✓ Lists test-state-2
✓ Lists 'last' state
✓ Highlights 'last' as auto-saved
✓ Counts states correctly

...

✓ All tests passed! State management system working.
```

---

### Testing Workflows

#### Before Building AMI

1. **Test AWS setup**:
   ```bash
   ./scripts/test-aws-config.sh --region us-east-1
   ```

2. **Validate AMI build**:
   ```bash
   ./scripts/validate-ami-build.sh /path/to/optix-installer.sh
   ```

3. **Fix any errors**, then proceed with actual build:
   ```bash
   ./scripts/build-ami.sh /path/to/optix-installer.sh
   ```

#### Before Launching Spot Instance

1. **Test Terraform config**:
   ```bash
   ./scripts/test-terraform-config.sh ami-xxxxxxxxxxxx
   ```

2. **Review plan output** to ensure resources look correct

3. **Launch instance**:
   ```bash
   ./scripts/nvidia-spot.sh --ami-id ami-xxxxxxxxxxxx
   ```

#### Before Using State Management

1. **Test state management**:
   ```bash
   ./scripts/test-state-management.sh
   ```

2. **Verify all required tools** are installed

3. **Use state management**:
   ```bash
   ./scripts/nvidia-spot.sh --ami-id ami-xxx --save-state my-checkpoint
   ```

---

### Limitations of Dry-Run Testing

While dry-run testing is very useful, it has limitations:

#### What Dry-Run CANNOT Test

❌ **Actual software installation**: Can't verify CUDA/OptiX actually install correctly
❌ **Network connectivity**: Can't test if downloads work
❌ **Timing issues**: Can't measure actual provisioning time
❌ **Runtime errors**: Can't catch errors in user-data scripts
❌ **Spot availability**: Can't guarantee spot instances are actually available
❌ **Complex interactions**: Can't test multi-step provisioning flows

#### Recommended: Small Test Run

For complete validation, consider a **small-scale test run**:

```bash
# Use smallest GPU instance for testing
./scripts/nvidia-spot.sh \
  --ami-id ami-xxxxxxxxxxxx \
  --instance-type g4dn.xlarge \
  --command "nvidia-smi && nvcc --version && echo 'Test passed'"

# This will:
# - Launch instance
# - Run verification commands
# - Auto-terminate
# Cost: ~$0.10 (20 minutes @ $0.30/hr)
```

---

### Cost Estimates for Testing

#### Dry-Run Testing (Test Scripts)
- **Cost**: $0 (completely free)
- **Time**: 2-5 minutes per script

#### Small Test Instance
- **Cost**: ~$0.10 (20-minute test)
- **Time**: 25-30 minutes total

#### AMI Build
- **Cost**: $0.15-$0.30 (30-60 minute build)
- **Time**: 35-65 minutes total
- **One-time**: AMI can be reused indefinitely

#### Full Development Session
- **Cost**: Varies by usage (e.g., 4 hours @ $0.30 = $1.20)
- **With state management**: Resume where you left off

---

### Testing Troubleshooting

#### Test Fails: "AWS credentials not configured"

```bash
# Configure AWS credentials
aws configure

# Or use environment variables
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx
export AWS_REGION=us-east-1
```

#### Test Fails: "Missing permissions"

Check IAM policy includes:
- `ec2:RunInstances`
- `ec2:CreateImage`
- `ec2:CreateSnapshot`
- `ec2:TerminateInstances`
- `ec2:Describe*`

#### Terraform Plan Fails

```bash
# Re-initialize Terraform
cd terraform
rm -rf .terraform .terraform.lock.hcl
terraform init
```

#### State Management Tests Fail: "jq not found"

```bash
# Ubuntu/Debian
sudo apt-get install jq rsync

# macOS
brew install jq rsync
```

---

### Testing Summary

| Script | What It Tests | Requires AWS | Creates Resources | Cost |
|--------|--------------|--------------|-------------------|------|
| `test-aws-config.sh` | AWS setup, permissions | ✅ | ❌ | $0 |
| `validate-ami-build.sh` | AMI build config | ✅ | ❌ | $0 |
| `test-terraform-config.sh` | Terraform config | ✅ | ❌ | $0 |
| `test-state-management.sh` | State scripts | ❌ | ❌ | $0 |

**Recommendation**: Run all test scripts before each major operation (AMI build, instance launch).
This catches 90%+ of issues before incurring any costs.

For complete validation, follow up with a small test instance run (~$0.10).

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