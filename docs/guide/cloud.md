# Menger — Cloud GPU Development

**Version**: 0.5.6
**Last Updated**: April 2026

← [User Guide Index](../USER_GUIDE.md)

---

## Overview

`scripts/nvidia-spot.sh` manages the full lifecycle of AWS EC2 GPU spot instances for
development and rendering. A single command launches an instance, waits for it to
initialize, and drops you into an SSH session. When you log out, your workspace is
automatically backed up and the instance is terminated.

The workflow is designed so that you pay only for the time the instance is running. Typical
sessions cost $0.10–$2.00.

---

## Prerequisites

### Required tools (local machine)

- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
- [Terraform](https://developer.hashicorp.com/terraform/install) (1.x)
- `jq`, `rsync`, `ssh`

### AWS credentials

Configure a profile in `~/.aws/config` and `~/.aws/credentials`:

```ini
# ~/.aws/config
[profile personal]
region = us-east-1

# ~/.aws/credentials
[personal]
aws_access_key_id     = AKIA...
aws_secret_access_key = ...
```

Pass `--aws-profile personal` (or set `AWS_PROFILE=personal`) on every invocation.

### SSH key

The script auto-detects your SSH public key by trying, in order:
`~/.ssh/id_ed25519.pub`, `~/.ssh/id_ecdsa.pub`, `~/.ssh/id_rsa.pub`.
Use `--ssh-key /path/to/key.pub` to override.

---

## First-Time Setup

### 1. Build the AMI (one-time, ~20 minutes)

The custom AMI pre-installs CUDA 12.8, OptiX 9.0, JVM 21, sbt, Fish shell, nvtop, and htop so
that launch time stays short.

```bash
# Download OptiX SDK 9.0 installer from https://developer.nvidia.com/optix
# then:
AWS_PROFILE=personal ./scripts/build-ami.sh /path/to/NVIDIA-OptiX-SDK-9.0.0-linux64.sh
```

The AMI ID is saved automatically to `scripts/ami-registry.tsv`. Subsequent launches look up the
newest AMI for the target region from this file — you do not need to pass `--ami-id` manually.

**Rebuild** when you need to pick up infrastructure changes (new system packages, CUDA updates,
etc.). After rebuilding, deregister the old AMI to avoid accruing snapshot storage costs:

```bash
AWS_PROFILE=personal ./scripts/build-ami.sh /path/to/installer.sh --deregister-old
```

### 2. Verify the registry

```bash
./scripts/nvidia-spot.sh --list-amis
```

---

## Launching an Instance

### Bare launch (interactive session)

```bash
AWS_PROFILE=personal ./scripts/nvidia-spot.sh
```

The script:
1. Looks up the newest AMI for `us-east-1` from `ami-registry.tsv`
2. Runs `terraform apply` to request a `g4dn.xlarge` spot instance
3. Waits for SSH to become available (~1 min)
4. Waits for instance initialization to complete (see below)
5. Auto-restores your last saved workspace state (if one exists)
6. Connects you via SSH with X11 forwarding

### Launch a specific branch

```bash
AWS_PROFILE=personal ./scripts/nvidia-spot.sh --menger-branch feature/my-branch
```

### Launch a different instance type

```bash
AWS_PROFILE=personal ./scripts/nvidia-spot.sh --instance-type g5.xlarge --max-price 0.75
```

### Instance initialization time

The first time an instance boots, `user-data.sh` clones the requested branch and runs
`sbt stage` to build `menger-app`. This takes **~5 minutes** and is logged to
`/var/log/user-data.log` on the instance. The script waits for the sentinel line
`Initialization complete` before connecting you.

Once initialized, `menger-app` is on PATH and ready to use.

---

## Running Renders

### Non-interactive (headless) render

Use `--command` to run a command on the instance, then `--retrieve` to pull output files
back to your local machine:

```bash
AWS_PROFILE=personal ./scripts/nvidia-spot.sh \
  --command "menger-app --optix --sponge-type cube-sponge --level 3 --save-name out.png" \
  --retrieve "*.png"
```

Output files are saved to `./artifacts/` (override with `--retrieve-to DIR`).

After the command completes, the instance auto-terminates.

### Interactive session

Without `--command`, the script opens an SSH session with X11 forwarding. You can run
commands interactively, use `nvtop` to monitor GPU utilization, etc.:

```bash
menger-app --optix --objects 'type=sphere:ior=1.5:size=1.5' --save-name sphere.png
nvtop        # GPU monitor
htop         # CPU/memory monitor
```

---

## State Management

### Automatic backup

When you exit an SSH session, the script automatically backs up your workspace
(files in `~/workspace/menger`, SSH keys, configs) to `~/.aws/spot-states/last/` on your
local machine. This happens before the instance terminates.

### Restore state on next launch

The last saved state is auto-restored by default:

```bash
AWS_PROFILE=personal ./scripts/nvidia-spot.sh   # restores 'last' automatically
```

To restore a named checkpoint:

```bash
AWS_PROFILE=personal ./scripts/nvidia-spot.sh --restore-state before-refactor
```

To skip auto-restore:

```bash
AWS_PROFILE=personal ./scripts/nvidia-spot.sh --no-auto-restore
```

### Save a named checkpoint

```bash
AWS_PROFILE=personal ./scripts/nvidia-spot.sh --save-state my-checkpoint
```

### List and clean up saved states

```bash
./scripts/list-spot-states.sh
./scripts/cleanup-spot-states.sh --keep-recent 5 --dry-run
./scripts/cleanup-spot-states.sh --keep-recent 5    # execute
```

---

## Spot Termination Protection

A background poller runs during your SSH session. Every 5 seconds it checks the instance
metadata service for a spot termination notice. If AWS signals that the instance will be
reclaimed, the script:

1. Prints a warning to your terminal
2. Immediately triggers an emergency state backup (`backup-spot-state.sh last`)
3. Exits the poller

This gives you ~2 minutes of advance warning before the instance disappears.

---

## Cost Control

| Mechanism | Default | Override |
|-----------|---------|----------|
| Max spot price | $0.50/hr | `--max-price PRICE` |
| Max session cost | $10.00 | `--max-cost COST` |
| Auto-terminate on logout | enabled | `--no-auto-terminate` |

### Auto-terminate daemon

An `auto-terminate.sh` daemon is installed on the instance via `user-data.sh`. It monitors SSH
sessions and terminates the instance after a grace period (default: 5 minutes) of no active
sessions. Logs: `/var/log/auto-terminate.log`.

### Manually terminate all running instances

```bash
AWS_PROFILE=personal ./scripts/nvidia-spot.sh --terminate
```

### Check running instances

```bash
AWS_PROFILE=personal ./scripts/nvidia-spot.sh --list-running
```

---

## AMI Registry

`scripts/ami-registry.tsv` is a plain TSV file listing all AMIs built for this project:

```
# Menger AMI registry — one line per build, newest last
# Format: region<TAB>ami-id<TAB>name<TAB>timestamp-UTC
us-east-1    ami-0618c45ee9f73c83d    menger-nvidia-dev-20260331-103218    2026-03-31T10:32:18Z
eu-central-1 ami-0c38fdd3d7b622e3b    menger-nvidia-dev-20260401-163642    2026-04-01T14:58:23Z
```

The script picks the **last** entry for the target region, so newer builds take precedence
automatically.

### Management commands

```bash
# List all AMIs in registry
./scripts/build-ami.sh --list

# Deregister a specific AMI (removes from AWS and registry)
AWS_PROFILE=personal ./scripts/build-ami.sh --deregister ami-0123456789abcdef0

# Deregister old AMIs after a new build
AWS_PROFILE=personal ./scripts/build-ami.sh /path/to/installer.sh --deregister-old

# Copy an existing AMI to additional regions
AWS_PROFILE=personal ./scripts/build-ami.sh \
  --copy ami-0618c45ee9f73c83d \
  --to-regions eu-central-1,ap-southeast-1

# Build and immediately copy to other regions
AWS_PROFILE=personal ./scripts/build-ami.sh /path/to/installer.sh \
  --copy-to-regions eu-central-1,ap-southeast-1

# Combine: build, copy to other regions, and deregister old AMIs everywhere
AWS_PROFILE=personal ./scripts/build-ami.sh /path/to/installer.sh \
  --copy-to-regions eu-central-1,ap-southeast-1 \
  --deregister-old
```

### When to rebuild the AMI

- System package updates (kernel, CUDA, drivers)
- New dev tools to pre-install
- PTX packaging changes (`build.sbt` `resourceGenerators`)
- Major Menger version changes that require a clean `sbt stage`

For most development work the existing AMI is fine — `user-data.sh` clones the latest
branch and re-stages `menger-app` on every launch.

---

## Troubleshooting

### SSH timeout on launch

If the script prints `Error: Instance did not become reachable after 5 minutes`, the
instance launched but SSH is unreachable. This usually means the AMI or security group
is misconfigured.

Diagnose with:

```bash
aws --profile personal ec2 get-console-output \
  --instance-id <INSTANCE_ID> --region us-east-1
```

Destroy the stuck resources:

```bash
cd terraform ; terraform destroy
```

### eu-central-1 GPU capacity

Both `g4dn` and `g5` spot instances in `eu-central-1` have persistent capacity shortages.
Use `us-east-1` (the default) unless you have a specific reason to use another region.

### Instance initialization hangs

If the script appears stuck after "Waiting for instance initialization...", the `sbt stage`
step inside `user-data.sh` may have failed. SSH in separately and check:

```bash
ssh ubuntu@<INSTANCE_IP>
tail -f /var/log/user-data.log
```

### Terraform state left over after failed launch

```bash
cd terraform ; terraform destroy
```

This cleans up any dangling EC2 resources and resets Terraform state.

---

← [User Guide Index](../USER_GUIDE.md) | → [Usage & Rendering](user-guide.md)
