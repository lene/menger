# Scripts Directory

Infrastructure-as-code scripts for AWS GPU development workflow and local environment setup.

## Quick Reference

```bash
# Launch GPU spot instance (AMI looked up from registry automatically)
./nvidia-spot.sh

# Or specify a branch
./nvidia-spot.sh --menger-branch feature/my-branch

# Run a render non-interactively and retrieve the output
./nvidia-spot.sh --command "menger-app --optix --sponge-type cube-sponge --level 3 --save-name out.png" --retrieve "*.png"

# See all AMIs built for this project
./nvidia-spot.sh --list-amis

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
  - AMI registry lookup (no need to pass `--ami-id` manually)
  - Branch selection with `--menger-branch`
  - Artifact retrieval with `--retrieve`
  - Launch instances with custom AMIs
  - SSH connection management
  - Instance termination
  - State backup/restore integration
  - Auto-termination support
  - Cost tracking
  - Spot termination poller: background process polls EC2 metadata every 5 s; triggers emergency
    backup automatically if AWS signals reclamation (~2 min advance warning)
  - See [docs/guide/cloud.md](../docs/guide/cloud.md) for complete guide

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
  - Works alongside the **spot termination poller** in `nvidia-spot.sh`, which handles
    AWS-initiated reclamation (fires emergency backup on notice)

### Infrastructure Setup

**AMI Building:**
- `build-ami.sh` (12 KB) - Build custom AMI with CUDA 12.8, OptiX SDK 9.0, dev tools
  - Ubuntu 24.04 base
  - NVIDIA drivers (580.x+)
  - Java, Scala, sbt, nvtop, htop, Fish shell
  - Integrated verification via `verify-optix.sh`
  - AMI ID auto-saved to `ami-registry.tsv` (TSV: `region<TAB>ami-id<TAB>name<TAB>timestamp-UTC`)
  - Management subcommands:
    - `--list` — list all AMIs in registry
    - `--deregister <ami-id>` — deregister one AMI from AWS and remove from registry
    - `--deregister-old` — deregister previous AMIs for the region after a fresh build
    - `--copy <ami-id> --to-regions REGION[,...]` — copy an existing AMI to other regions
    - `--copy-to-regions REGION[,...]` — after a fresh build, also copy to these regions
  - See [docs/guide/cloud.md](../docs/guide/cloud.md) for usage

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

# 3. Build custom AMI (one-time, ~20 minutes; us-east-1 recommended — eu-central-1 has GPU spot
#    capacity shortages)
AWS_PROFILE=personal ./build-ami.sh /path/to/NVIDIA-OptiX-SDK-9.0.0-linux64.sh

# 4. Launch your first spot instance (AMI ID is read from registry automatically)
AWS_PROFILE=personal ./nvidia-spot.sh

# Or with a non-default branch
AWS_PROFILE=personal ./nvidia-spot.sh --menger-branch feature/my-branch
```

### Daily Development Workflow

```bash
# Launch instance (AMI looked up from registry, auto-restores last state)
# Instance initialization (sbt stage) takes ~5 min; progress in /var/log/user-data.log
AWS_PROFILE=personal ./nvidia-spot.sh

# Launch a specific branch
AWS_PROFILE=personal ./nvidia-spot.sh --menger-branch feature/my-branch

# Run a render and retrieve output images
AWS_PROFILE=personal ./nvidia-spot.sh \
  --command "menger-app --optix --sponge-type cube-sponge --level 3 --save-name out.png" \
  --retrieve "*.png"
# Output saved to ./artifacts/out.png

# Work on instance via SSH...

# Exit SSH - automatic backup triggers, then instance terminates

# Later: list saved states
./list-spot-states.sh

# Restore specific state
AWS_PROFILE=personal ./nvidia-spot.sh --restore-state my-feature-branch

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

**SSH timeout on launch:** If the instance launches but SSH never becomes reachable, diagnose
with the EC2 console output and then tear down with `cd terraform ; terraform destroy`.

**Instance initialization hangs:** The `sbt stage` step inside `user-data.sh` can take ~5 min.
Follow progress: `ssh ubuntu@<IP> 'tail -f /var/log/user-data.log'`

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

- `AWS_PROFILE` - AWS credentials profile (also: `--aws-profile PROFILE` flag on `nvidia-spot.sh`)
- `AWS_REGION` - Default region (scripts use us-east-1 by default; recommended — eu-central-1
  has persistent GPU spot capacity shortages)
- `OPTIX_ROOT` - OptiX SDK installation path
- `CUDA_HOME` - CUDA toolkit path

**SSH key auto-detection:** `--ssh-key` is optional. `nvidia-spot.sh` tries
`~/.ssh/id_ed25519.pub`, `~/.ssh/id_ecdsa.pub`, `~/.ssh/id_rsa.pub` in that order.

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
- `docs/guide/cloud.md` - Complete AWS GPU development guide
- `terraform/README.md` - Terraform infrastructure reference
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

**Last Updated:** March 2026 (nvidia-spot.sh, build-ami.sh, user-data.sh)

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

*For questions about AWS setup or script usage, see `docs/guide/cloud.md`*
