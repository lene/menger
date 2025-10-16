# Testing Guide for AWS Infrastructure

This guide explains how to test your AWS configuration, AMI builds, and spot instance infrastructure **without actually creating resources**.

## Quick Start

Run all tests at once:

```bash
# Test AWS configuration and permissions
./scripts/test-aws-config.sh

# Validate AMI build configuration
./scripts/validate-ami-build.sh /path/to/NVIDIA-OptiX-SDK-*.sh

# Test Terraform configuration
./scripts/test-terraform-config.sh ami-xxxxxxxxxxxx

# Test state management scripts
./scripts/test-state-management.sh
```

---

## Test Scripts Overview

### 1. AWS Configuration Test (`test-aws-config.sh`)

**Purpose**: Validate AWS CLI, credentials, permissions, and region setup.

**What it tests**:
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

**Usage**:
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

**What it does NOT do**:
- ❌ Does not create any AWS resources
- ❌ Does not modify existing resources
- ❌ Does not incur any costs

**Example output**:
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

### 2. AMI Build Validation (`validate-ami-build.sh`)

**Purpose**: Validate AMI build configuration before running the expensive build process.

**What it tests**:
- ✅ OptiX installer file existence and size
- ✅ Build script syntax (`build-ami.sh`, `verify-optix.sh`)
- ✅ AWS resources (base AMI, security group, subnet)
- ✅ Instance launch permissions (dry-run)
- ✅ AMI creation permissions (dry-run)
- ✅ Provisioning script contents (CUDA, NVIDIA drivers, OptiX, Claude Code)
- ✅ Estimated costs and time
- ✅ Network connectivity requirements

**Usage**:
```bash
# Validate with OptiX installer
./scripts/validate-ami-build.sh /path/to/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh

# Just validate scripts (without OptiX installer)
./scripts/validate-ami-build.sh
```

**What it does NOT do**:
- ❌ Does not launch any instances
- ❌ Does not create AMIs
- ❌ Does not install any software

**Example output**:
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

### 3. Terraform Configuration Test (`test-terraform-config.sh`)

**Purpose**: Validate Terraform configuration and generate plan without creating resources.

**What it tests**:
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

**Usage**:
```bash
# Validate configuration (basic)
./scripts/test-terraform-config.sh

# Validate and generate plan with AMI
./scripts/test-terraform-config.sh ami-xxxxxxxxxxxx
```

**What it does NOT do**:
- ❌ Does not apply Terraform changes
- ❌ Does not create any AWS resources
- ❌ Does not modify state file

**Example output**:
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

---

### 4. State Management Test (`test-state-management.sh`)

**Purpose**: Test state management scripts using mock data (no instance required).

**What it tests**:
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

**Usage**:
```bash
# Run all state management tests
./scripts/test-state-management.sh
```

**What it does NOT do**:
- ❌ Does not connect to any instances
- ❌ Does not require AWS credentials
- ❌ Does not modify real state backups

**Example output**:
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

## AWS Dry-Run Mode Explained

AWS supports `--dry-run` for many EC2 operations. This validates your request without actually executing it.

### What Dry-Run Tests

✅ **Permissions**: Checks if you have the IAM permissions to perform the action
✅ **Parameters**: Validates that all parameters are correct
✅ **Quotas**: Checks if you've reached service limits
✅ **Resources**: Validates that referenced resources exist

### Supported Commands

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

### Expected Dry-Run Output

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

## Terraform Plan (Dry-Run Equivalent)

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

## Testing Workflow

### Before Building AMI

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

### Before Launching Spot Instance

1. **Test Terraform config**:
   ```bash
   ./scripts/test-terraform-config.sh ami-xxxxxxxxxxxx
   ```

2. **Review plan output** to ensure resources look correct

3. **Launch instance**:
   ```bash
   ./scripts/nvidia-spot.sh --ami-id ami-xxxxxxxxxxxx
   ```

### Before Using State Management

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

## Limitations of Dry-Run Testing

While dry-run testing is very useful, it has limitations:

### What Dry-Run CANNOT Test

❌ **Actual software installation**: Can't verify CUDA/OptiX actually install correctly
❌ **Network connectivity**: Can't test if downloads work
❌ **Timing issues**: Can't measure actual provisioning time
❌ **Runtime errors**: Can't catch errors in user-data scripts
❌ **Spot availability**: Can't guarantee spot instances are actually available
❌ **Complex interactions**: Can't test multi-step provisioning flows

### Recommended: Small Test Run

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

## Cost Estimates for Testing

### Dry-Run Testing (These Scripts)
- **Cost**: $0 (completely free)
- **Time**: 2-5 minutes per script

### Small Test Instance
- **Cost**: ~$0.10 (20-minute test)
- **Time**: 25-30 minutes total

### AMI Build
- **Cost**: $0.15-$0.30 (30-60 minute build)
- **Time**: 35-65 minutes total
- **One-time**: AMI can be reused indefinitely

### Full Development Session
- **Cost**: Varies by usage (e.g., 4 hours @ $0.30 = $1.20)
- **With state management**: Resume where you left off

---

## Troubleshooting

### Test Fails: "AWS credentials not configured"

```bash
# Configure AWS credentials
aws configure

# Or use environment variables
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx
export AWS_REGION=us-east-1
```

### Test Fails: "Missing permissions"

Check IAM policy includes:
- `ec2:RunInstances`
- `ec2:CreateImage`
- `ec2:CreateSnapshot`
- `ec2:TerminateInstances`
- `ec2:Describe*`

### Terraform Plan Fails

```bash
# Re-initialize Terraform
cd terraform
rm -rf .terraform .terraform.lock.hcl
terraform init
```

### State Management Tests Fail: "jq not found"

```bash
# Ubuntu/Debian
sudo apt-get install jq rsync

# macOS
brew install jq rsync
```

---

## Summary

| Script | What It Tests | Requires AWS | Creates Resources | Cost |
|--------|--------------|--------------|-------------------|------|
| `test-aws-config.sh` | AWS setup, permissions | ✅ | ❌ | $0 |
| `validate-ami-build.sh` | AMI build config | ✅ | ❌ | $0 |
| `test-terraform-config.sh` | Terraform config | ✅ | ❌ | $0 |
| `test-state-management.sh` | State scripts | ❌ | ❌ | $0 |

**Recommendation**: Run all test scripts before each major operation (AMI build, instance launch). This catches 90%+ of issues before incurring any costs.

For complete validation, follow up with a small test instance run (~$0.10).
