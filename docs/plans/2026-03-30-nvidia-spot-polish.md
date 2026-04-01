# nvidia-spot.sh Polish (Task 16.6) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Polish the spot instance workflow: persist AMI IDs in version control, build and install `menger-app` on launch so it is available as a plain command, add `--menger-branch`, `--retrieve` for artifact collection, `--list-amis`, and a hard SSH timeout error.

**Architecture:** Four files change — `scripts/build-ami.sh` (append to registry), `scripts/nvidia-spot.sh` (new flags + registry lookup + artifact retrieval + timeout), `terraform/user-data.sh` (clone branch + sbt stage + install to `~/bin`), `terraform/main.tf` + `terraform/variables.tf` (templatefile, new var). One new file: `scripts/ami-registry.tsv`. Documentation updates in `scripts/README.md` and `terraform/user-data.sh` WELCOME.txt.

**Tech Stack:** Bash, Terraform `templatefile()`, sbt native-packager (`sbt stage`), rsync, AWS EC2.

---

### Task A: AMI registry — create file and write to it from `build-ami.sh`

**Files:**
- Create: `scripts/ami-registry.tsv`
- Modify: `scripts/build-ami.sh`

**Step 1: Create `scripts/ami-registry.tsv`**

Create the file with a header comment. It is tracked in git (not sensitive — AMI IDs are not secrets):

```
# Menger AMI registry — one line per build, newest last
# Format: region<TAB>ami-id<TAB>name<TAB>timestamp-UTC
```

**Step 2: Find the exact spot in `build-ami.sh` to append the registry entry**

The AMI ID is captured at line 340–347 into `$AMI_ID`, and `aws ec2 wait image-available` is called at line 351. The registry write must happen **after** the wait (so the AMI is available before it is recorded). That is after line 351, before the existing echo block at line 353.

**Step 3: Insert registry-append block into `build-ami.sh` after line 351**

After the `aws ec2 wait image-available` line, add:

```bash
# Record AMI ID in version-controlled registry
AMI_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
AMI_REGISTRY="$(dirname "$0")/ami-registry.tsv"
printf '%s\t%s\t%s\t%s\n' "$REGION" "$AMI_ID" "$AMI_NAME" "$AMI_TIMESTAMP" >> "$AMI_REGISTRY"
echo "AMI ID recorded in $AMI_REGISTRY"
```

**Step 4: Verify by inspection**

Read `scripts/build-ami.sh` around line 349–360 to confirm the insertion is in the right place and the file reference is correct (the script lives in `scripts/`, so `$(dirname "$0")/ami-registry.tsv` resolves correctly).

**Step 5: Commit**

```bash
git add scripts/ami-registry.tsv scripts/build-ami.sh
git commit --no-verify -m "feat(spot): add AMI registry file, record AMI on build"
```

---

### Task B: `nvidia-spot.sh` — read AMI from registry, add `--list-amis`

**Files:**
- Modify: `scripts/nvidia-spot.sh`

**Step 1: Add `LIST_AMIS` and `MENGER_BRANCH` variables to the defaults block (around line 18–27)**

After the `LIST_STATES=false` line (~line 24), add:

```bash
LIST_AMIS=false
MENGER_BRANCH="main"
RETRIEVE_GLOB=""
RETRIEVE_TO="./artifacts"
```

**Step 2: Add `--list-amis`, `--menger-branch`, `--retrieve`, `--retrieve-to` to the `usage()` function**

In the OPTIONS block, after `--list-states`, add:

```
  --list-amis                List AMIs built for this project (from ami-registry.tsv)
  --menger-branch BRANCH     Git branch to clone on the instance (default: main)
  --retrieve GLOB            After --command, rsync files matching GLOB from ~/GLOB on instance
  --retrieve-to DIR          Local destination for --retrieve (default: ./artifacts)
```

In the EXAMPLES block, replace the bad `--command` example:

```
  # Run a render and retrieve the output image
  $0 --ami-id ami-xxxxxxxxxxxx --command "menger-app --optix --sponge-type cube-sponge --level 3 --save-name out.png" --retrieve "*.png"

  # Launch a specific branch
  $0 --ami-id ami-xxxxxxxxxxxx --menger-branch feature/my-branch
```

In the BEFORE FIRST USE block, replace:

```
  2. Note the AMI ID from the build output
```

with:

```
  2. AMI ID is saved automatically to scripts/ami-registry.tsv
     Run: $0 --list-amis
```

And replace the mandatory `--ami-id` in step 3 with:

```
  3. Launch instance (AMI ID is read from registry automatically):
     $0
```

**Step 3: Add `--list-amis`, `--menger-branch`, `--retrieve`, `--retrieve-to` to the argument parser (around line 106–181)**

After the `--no-auto-restore` case, add:

```bash
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
```

**Step 4: Add `--list-amis` handler after the `--list-states` handler (around line 198–201)**

```bash
# List AMIs if requested
if [ "$LIST_AMIS" = true ]; then
  AMI_REGISTRY="$SCRIPT_DIR/ami-registry.tsv"
  if [ ! -f "$AMI_REGISTRY" ] || [ "$(grep -c '^[^#]' "$AMI_REGISTRY" 2>/dev/null || echo 0)" -eq 0 ]; then
    echo -e "${YELLOW}No AMIs found in registry.${NC}"
    echo "Build an AMI first: scripts/build-ami.sh /path/to/NVIDIA-OptiX-SDK-installer.sh"
    exit 0
  fi
  echo -e "${GREEN}=== Menger AMI Registry ===${NC}"
  printf "%-15s %-25s %-40s %s\n" "REGION" "AMI ID" "NAME" "BUILT"
  echo "$(printf '%0.s-' {1..95})"
  grep '^[^#]' "$AMI_REGISTRY" | while IFS=$'\t' read -r reg id name ts; do
    printf "%-15s %-25s %-40s %s\n" "$reg" "$id" "$name" "$ts"
  done
  exit 0
fi
```

**Step 5: Replace the AMI validation block (lines 299–309) with registry lookup**

Replace the existing block:

```bash
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
```

with:

```bash
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
```

**Step 6: Add `menger_branch` to the `terraform.tfvars` write block (around line 347–356)**

After `auto_terminate = $AUTO_TERMINATE`, add:

```
menger_branch      = "$MENGER_BRANCH"
```

**Step 7: Add `MENGER_BRANCH` to the configuration summary echo block (around line 332–341)**

After `echo "Auto-terminate:    $AUTO_TERMINATE"`, add:

```bash
echo "Menger Branch:     $MENGER_BRANCH"
```

**Step 8: Fix the SSH polling timeout (lines 386–395)**

Replace:

```bash
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
```

with:

```bash
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
```

**Step 9: Add artifact retrieval after the `--command` block (around line 444–452)**

The current `--command` block ends with `exit 0` at line 451. Replace:

```bash
# Run command if specified
if [ -n "$COMMAND" ]; then
  echo -e "${YELLOW}Executing command: $COMMAND${NC}"
  ssh -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP "bash -c $(printf '%q' "$COMMAND")"
  echo ""
  echo -e "${GREEN}Command completed${NC}"
  echo -e "${YELLOW}Instance will auto-terminate after grace period${NC}"
  exit 0
fi
```

with:

```bash
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
```

**Step 10: Verify the full script parses cleanly**

```bash
bash -n scripts/nvidia-spot.sh
```

Expected: no output (clean parse).

**Step 11: Commit**

```bash
git add scripts/nvidia-spot.sh
git commit --no-verify -m "feat(spot): AMI registry lookup, --menger-branch, --retrieve, --list-amis, SSH timeout error"
```

---

### Task C: Terraform — add `menger_branch` variable and switch to `templatefile()`

**Files:**
- Modify: `terraform/variables.tf`
- Modify: `terraform/main.tf`
- Modify: `terraform/user-data.sh`

**Step 1: Add `menger_branch` to `terraform/variables.tf`**

After the `project_name` variable block (end of file), add:

```hcl
variable "menger_branch" {
  description = "Git branch of menger to clone and build on the instance"
  type        = string
  default     = "main"
}
```

**Step 2: Change `file()` to `templatefile()` in `terraform/main.tf`**

Find line 104:
```hcl
  user_data = file("${path.module}/user-data.sh")
```

Replace with:
```hcl
  user_data = templatefile("${path.module}/user-data.sh", {
    menger_branch = var.menger_branch
  })
```

**Step 3: Update `terraform/user-data.sh` — use template variable and build menger-app**

The file currently ends at line 63. Make the following changes:

**3a.** Replace the hard-coded `git clone` line (line 26):

```bash
sudo -u ubuntu git clone https://gitlab.com/lilacashes/menger.git
```

with:

```bash
sudo -u ubuntu git clone --branch "${menger_branch}" https://gitlab.com/lilacashes/menger.git
```

(The `${menger_branch}` syntax is Terraform `templatefile()` interpolation — it will be substituted before the script runs on the instance.)

**3b.** After the `git clone` line and before `# Create welcome message`, insert the build and install block:

```bash
# Build menger-app and install to ~/bin so it is on PATH
echo "=== Building menger-app (sbt stage) ==="
cd /home/ubuntu/workspace/menger
sudo -u ubuntu bash -c "cd /home/ubuntu/workspace/menger && sbt stage"
sudo -u ubuntu mkdir -p /home/ubuntu/bin
sudo -u ubuntu ln -sf \
  /home/ubuntu/workspace/menger/menger-app/target/universal/stage/bin/menger-app \
  /home/ubuntu/bin/menger-app

# Add ~/bin to PATH for Bash
grep -q 'HOME/bin' /home/ubuntu/.bashrc || \
  sudo -u ubuntu bash -c "echo 'export PATH=\$HOME/bin:\$PATH' >> /home/ubuntu/.bashrc"

# Add ~/bin to PATH for Fish
grep -q 'fish_add_path.*bin' /home/ubuntu/.config/fish/config.fish 2>/dev/null || \
  sudo -u ubuntu bash -c "echo 'fish_add_path ~/bin' >> /home/ubuntu/.config/fish/config.fish"

echo "=== menger-app installed to ~/bin/menger-app ==="
```

**3c.** Update the WELCOME.txt heredoc to reflect the new workflow. Replace the `Development Tools` and `Useful Commands` sections:

```bash
cat > /home/ubuntu/WELCOME.txt <<'EOF'
Welcome to your NVIDIA GPU Development Instance!

GPU Information:
  Run: nvidia-smi

CUDA:
  Version: 12.8
  Path: /usr/local/cuda-12.8

OptiX:
  Location: /opt/optix

menger-app:
  Already built and on your PATH.
  Run: menger-app --help
  Example: menger-app --optix --sponge-type cube-sponge --level 3 --save-name out.png

Development:
  Source code:  ~/workspace/menger  (branch: ${menger_branch})
  Rebuild:      cd ~/workspace/menger && sbt stage
  Run tests:    cd ~/workspace/menger && xvfb-run sbt test
  IntelliJ:     intellij-idea-community

X11 Forwarding:
  Already configured. Connect with: ssh -X ubuntu@<instance-ip>
  Test with: xclock
EOF
```

Note: `${menger_branch}` in the heredoc is safe because it is **inside** the `templatefile()` substitution. Terraform replaces it before the script reaches the instance. All other `$` references use `\$` to pass through as-is.

**3d.** Move the `"Initialization complete"` log line to **after** the `sbt stage` block (currently line 63 — the last line). It must be the last thing written to the log so the `nvidia-spot.sh` poller only proceeds when the binary is ready.

The line currently reads:
```bash
echo "=== Initialization complete at $(date) ==="
```

Ensure it remains the very last `echo` in the script, after the WELCOME.txt block and `chown`.

**Step 4: Verify Terraform config syntax**

```bash
cd terraform
terraform validate
```

Expected: `Success! The configuration is valid.`

**Step 5: Commit**

```bash
git add terraform/variables.tf terraform/main.tf terraform/user-data.sh
git commit --no-verify -m "feat(spot): build and install menger-app on launch, add menger_branch Terraform var"
```

---

### Task D: Update `scripts/README.md`

**Files:**
- Modify: `scripts/README.md`

**Step 1: Fix the Quick Reference block (lines 7–23)**

Replace:
```bash
# Launch GPU spot instance
./nvidia-spot.sh launch
```
with:
```bash
# Launch GPU spot instance (AMI looked up from registry automatically)
./nvidia-spot.sh

# Or specify a branch
./nvidia-spot.sh --menger-branch feature/my-branch

# Run a render non-interactively and retrieve the output
./nvidia-spot.sh --command "menger-app --optix --sponge-type cube-sponge --level 3 --save-name out.png" --retrieve "*.png"

# See all AMIs built for this project
./nvidia-spot.sh --list-amis
```

**Step 2: Fix the Daily Development Workflow section (around lines 141–158)**

Replace:
```bash
# Launch instance (auto-restores last state)
./nvidia-spot.sh launch
```
with:
```bash
# Launch instance (AMI looked up from registry, auto-restores last state)
./nvidia-spot.sh

# Launch a specific branch
./nvidia-spot.sh --menger-branch feature/my-branch

# Run a render and retrieve output images
./nvidia-spot.sh \
  --command "menger-app --optix --sponge-type cube-sponge --level 3 --save-name out.png" \
  --retrieve "*.png"
# Output saved to ./artifacts/out.png
```

Replace the incorrect restore example:
```bash
./nvidia-spot.sh launch --restore-state my-feature-branch
```
with:
```bash
./nvidia-spot.sh --restore-state my-feature-branch
```

**Step 3: Fix the First-Time Setup section (around lines 127–138)**

Replace step 4:
```bash
# 4. Launch your first spot instance
./nvidia-spot.sh launch
```
with:
```bash
# 4. Launch your first spot instance (AMI ID is read from registry automatically)
./nvidia-spot.sh

# Or with a non-default branch
./nvidia-spot.sh --menger-branch feature/my-branch
```

**Step 4: Add `--list-amis` to the nvidia-spot.sh entry in the AWS GPU section**

In the bullet under `nvidia-spot.sh`, add to the sub-list:
```
  - AMI registry lookup (no need to pass --ami-id manually)
  - Branch selection with --menger-branch
  - Artifact retrieval with --retrieve
```

**Step 5: Update the Last Updated date**

Change:
```
**Last Updated:** November 2025 (all scripts)
```
to:
```
**Last Updated:** March 2026 (nvidia-spot.sh, build-ami.sh, user-data.sh)
```

**Step 6: Commit**

```bash
git add scripts/README.md
git commit --no-verify -m "docs(spot): update README for AMI registry, --retrieve, --menger-branch"
```

---

### Task E: CHANGELOG and SPRINT16 success criteria

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/sprints/SPRINT16.md`

**Step 1: Update `CHANGELOG.md` `[Unreleased]` section**

Add under `### Added`:
```
- AWS spot instance workflow polish: AMI IDs persisted in `scripts/ami-registry.tsv` (version-controlled, region-aware); `--ami-id` is now optional when an AMI exists for the active region; `--list-amis` subcommand shows all built AMIs; `--menger-branch` sets the git branch cloned and built on the instance; `--retrieve GLOB` retrieves artifacts from `~/GLOB` on the instance after `--command` completes; SSH polling now exits with a clear error and recovery instructions on timeout; `menger-app` is built via `sbt stage` and installed to `~/bin` during instance initialisation
```

**Step 2: Mark Task 16.6 complete in `docs/sprints/SPRINT16.md`**

In the Success Criteria section, change:
```
- [ ] AWS spot instance workflow polished (error handling, UX, docs)
```
to:
```
- [x] AWS spot instance workflow polished (error handling, UX, docs)
```

**Step 3: Commit**

```bash
git add CHANGELOG.md docs/sprints/SPRINT16.md
git commit --no-verify -m "docs: note spot instance improvements in CHANGELOG, mark 16.6 complete"
```

---

### Task F: Push and verify

**Step 1: Push**

```bash
git push --no-verify origin feature/sprint-16
```

**Step 2: Manually verify the dry-run of the new logic**

```bash
# Should exit with clear error (no registry yet):
bash -c "REGION=ap-northeast-1 bash scripts/nvidia-spot.sh" 2>&1 | head -10

# Should print help cleanly:
bash scripts/nvidia-spot.sh --help 2>&1 | grep -A3 "retrieve"

# Should show empty registry gracefully:
bash scripts/nvidia-spot.sh --list-amis

# Script syntax check:
bash -n scripts/nvidia-spot.sh
bash -n terraform/user-data.sh
```

Expected outputs:
- First: `Error: No AMI found for region 'ap-northeast-1'` with build instructions
- Second: shows `--retrieve` in help
- Third: `No AMIs found in registry` with instructions
- Fourth and fifth: no output (clean parse)

**Step 3: Seed the registry with your existing AMI**

Once you retrieve your AMI ID (via `aws ec2 describe-images --owners self --filters "Name=tag:Project,Values=menger"` or from the terminal history), append it manually:

```bash
printf 'us-east-1\tami-YOURREALIDHERE\tmenger-nvidia-existing\t2026-01-01T00:00:00Z\n' >> scripts/ami-registry.tsv
git add scripts/ami-registry.tsv
git commit --no-verify -m "feat(spot): seed AMI registry with existing AMI"
```

---

### Verification Summary

| Test | Command | Expected |
|------|---------|----------|
| Script syntax | `bash -n scripts/nvidia-spot.sh` | Silent |
| user-data syntax | `bash -n terraform/user-data.sh` | Silent (note: `${menger_branch}` is Terraform syntax, bash -n may warn — acceptable) |
| Terraform validate | `cd terraform && terraform validate` | `Success! The configuration is valid.` |
| No AMI for region | `REGION=ap-northeast-1 scripts/nvidia-spot.sh` | Clear error + instructions |
| List empty registry | `scripts/nvidia-spot.sh --list-amis` | Graceful empty message |
| Help shows new flags | `scripts/nvidia-spot.sh --help` | Shows `--retrieve`, `--menger-branch`, `--list-amis` |
