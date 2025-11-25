# Scripts Directory

Infrastructure-as-code scripts for AWS GPU development workflow and local environment setup.

## Quick Reference

```bash
# Launch GPU spot instance
./nvidia-spot.sh launch

# Setup local development environment
./setup-optix-local.sh

# Verify OptiX installation
./verify-optix.sh

# List available GPU instances and pricing
./list-instances.sh

# Manage saved instance states
./list-spot-states.sh
./cleanup-spot-states.sh --keep-recent 5
```

## Script Categories

### AWS GPU Spot Instance Management

**Primary CLI Tool:**
- `nvidia-spot.sh` (16 KB) - **Main entry point** for all spot instance operations
  - Launch instances with custom AMIs
  - SSH connection management
  - Instance termination
  - State backup/restore integration
  - Auto-termination support
  - Cost tracking
  - See `terraform/README.md` for complete guide

**State Management:**
- `backup-spot-state.sh` - Save workspace, configs, SSH keys after session
- `restore-spot-state.sh` - Restore previous state to new instance
- `list-spot-states.sh` - View all saved states with metadata
- `cleanup-spot-states.sh` - Remove old states (with --keep-recent, --older-than-days)

**Instance Discovery:**
- `list-instances.sh` - Show GPU instance types (g4dn, g5, p3, p4d, p5) with spot prices

**Auto-Termination:**
- `auto-terminate.sh` - Daemon running on spot instances (installed via user-data)
  - Monitors SSH sessions
  - Terminates idle instances after grace period (default: 5 minutes)
  - Logs to `/var/log/auto-terminate.log`
  - Critical for cost control

### Infrastructure Setup

**AMI Building:**
- `build-ami.sh` (12 KB) - Build custom AMI with CUDA 12.8, OptiX SDK 9.0, dev tools
  - Ubuntu 24.04 base
  - NVIDIA drivers (580.x+)
  - Java, Scala, sbt
  - Integrated verification via `verify-optix.sh`
  - See `terraform/README.md` for usage

**Local Development Setup:**
- `setup-optix-local.sh` (9.1 KB) - Interactive setup for local machines
  - Installs NVIDIA drivers, CUDA, OptiX
  - Configures environment variables
  - Supports Bash, Zsh, Fish shells
  - Ubuntu/Debian only
- `setup-optix-env.fish` - Fish shell integration (called by setup-optix-local.sh)
  - Sets CUDA_HOME, OPTIX_ROOT, PATH, LD_LIBRARY_PATH
  - Provides `verify_optix` function for Fish users

### Verification & Validation

**OptiX Installation Verification:**
- `verify-optix.sh` (9.1 KB) - **Comprehensive verification** of OptiX, CUDA, drivers
  - Checks NVIDIA driver version and GPU detection
  - Verifies CUDA toolkit installation
  - Locates OptiX SDK and validates headers
  - Performs compilation test
  - Called by `build-ami.sh`, useful standalone
  - Provides diagnostic summary with pass/fail counts

**AWS Configuration Testing:**
- `test-aws-config.sh` (9.0 KB) - Validate AWS credentials and permissions
  - Tests AWS CLI installation
  - Verifies EC2 launch permissions (dry-run)
  - Checks instance type availability
  - Retrieves spot pricing
  - Validates SSH key setup
  - Comprehensive permission verification
  - **Use when**: AWS setup fails or troubleshooting access issues

**Terraform Validation:**
- `test-terraform-config.sh` (8.1 KB) - Validate Terraform configuration without provisioning
  - Syntax validation (`terraform validate`)
  - Plan generation and analysis
  - Resource definition checks
  - Cost estimation
  - State management verification
  - **Use when**: Modifying Terraform configs or troubleshooting deployments

**State Management Testing:**
- `test-state-management.sh` (9.1 KB) - Unit tests for state management scripts
  - Creates mock state directories
  - Tests backup/restore/cleanup logic
  - Validates metadata parsing
  - Dry-run testing of cleanup filters
  - **Use when**: Developing or debugging state management features

**AMI Build Validation:**
- `validate-ami-build.sh` (8.3 KB) - Pre-flight checks before building AMI
  - Verifies OptiX installer availability and size
  - Checks build script syntax
  - Tests AWS permissions
  - Validates provisioning scripts
  - Estimates build costs and time
  - **Use when**: Before running `build-ami.sh` to catch issues early

## Usage Patterns

### First-Time Setup

```bash
# 1. Validate AWS configuration
./test-aws-config.sh

# 2. Validate Terraform setup
./test-terraform-config.sh

# 3. Build custom AMI (one-time, ~20 minutes)
./build-ami.sh

# 4. Launch your first spot instance
./nvidia-spot.sh launch
```

### Daily Development Workflow

```bash
# Launch instance (auto-restores last state)
./nvidia-spot.sh launch

# Work on instance via SSH...

# Exit SSH - automatic backup triggers

# Later: list saved states
./list-spot-states.sh

# Restore specific state
./nvidia-spot.sh launch --restore-state my-feature-branch

# Cleanup old states
./cleanup-spot-states.sh --keep-recent 5 --dry-run
./cleanup-spot-states.sh --keep-recent 5  # Execute
```

### Local Development Setup

```bash
# Interactive setup (prompts for choices)
./setup-optix-local.sh

# Verify installation
./verify-optix.sh

# For Fish shell users
setup-optix-env.fish
```

### Troubleshooting

```bash
# Verify OptiX/CUDA installation
./verify-optix.sh

# Test AWS credentials and permissions
./test-aws-config.sh

# Validate Terraform configuration
./test-terraform-config.sh

# Validate AMI build prerequisites
./validate-ami-build.sh

# Test state management scripts
./test-state-management.sh
```

## Script Dependencies

### Required Tools

**All workflows:**
- bash (4.0+)
- AWS CLI (v2)
- jq (JSON processor)

**Spot instance management:**
- Terraform
- SSH client
- rsync

**Local setup:**
- curl
- pkexec (for CLAUDE.md compliance, replaces sudo)

**AMI building:**
- OptiX SDK 9.0 installer (download from NVIDIA)

### Environment Variables

Scripts respect these environment variables:

- `AWS_PROFILE` - AWS credentials profile
- `AWS_REGION` - Default region (scripts use us-east-1 by default)
- `OPTIX_ROOT` - OptiX SDK installation path
- `CUDA_HOME` - CUDA toolkit path

## Cost Management

**Spot Instances:**
- g4dn.xlarge: ~$0.15-0.30/hour (typical)
- Auto-termination prevents runaway costs
- State backup/restore enables quick session switching

**AMI Storage:**
- Custom AMI: ~8 GB snapshot storage
- Cost: ~$0.40/month (negligible)

**State Backups:**
- Stored locally (no AWS storage cost)
- Typical size: 100 MB - 2 GB per state
- Use `cleanup-spot-states.sh` to manage disk usage

## Documentation

**Primary Documentation:**
- `terraform/README.md` - Complete AWS GPU development guide
- `docs/INSTALLATION_FROM_SCRATCH.md` - Local setup guide
- `docs/TROUBLESHOOTING.md` - Common issues and solutions

**Script Help:**
All scripts support `--help` flag for detailed usage:
```bash
./nvidia-spot.sh --help
./build-ami.sh --help
./verify-optix.sh --help
# etc.
```

## Maintenance

**Last Updated:** November 2025 (all scripts)

**Active Development:**
- All scripts actively maintained
- Part of core infrastructure-as-code workflow
- Regular updates for AWS API changes and OptiX SDK versions

**Testing:**
- Infrastructure validation scripts (`test-*.sh`, `validate-*.sh`) provide automated testing
- Run validation scripts after updates to verify functionality

## Archive

**Obsolete Scripts:**
- `archive/test-optix.cpp` - Simple header compilation test
  - Replaced by comprehensive `verify-optix.sh`
  - Kept for reference

See `archive/README.md` for details on archived scripts.

---

*For questions about AWS setup or script usage, see `terraform/README.md`*
