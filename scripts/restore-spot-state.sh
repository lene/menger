#!/bin/bash
# Restore saved state to spot instance
# Usage: restore-spot-state.sh [STATE_NAME] [INSTANCE_IP]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_BASE="${SPOT_STATES_DIR:-$HOME/.aws/spot-states}"
STATE_NAME="${1}"
INSTANCE_IP="${2}"

# Get instance IP from terraform if not provided
if [ -z "$INSTANCE_IP" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  TERRAFORM_DIR="$(dirname "$SCRIPT_DIR")/terraform"

  if [ -f "$TERRAFORM_DIR/terraform.tfstate" ]; then
    cd "$TERRAFORM_DIR"
    INSTANCE_IP=$(terraform output -raw instance_public_ip 2>/dev/null || echo "")
  fi
fi

# List states if none specified
if [ -z "$STATE_NAME" ]; then
  echo -e "${BLUE}=== Available Saved States ===${NC}"
  echo ""

  if [ ! -d "$BACKUP_BASE" ] || [ -z "$(ls -A "$BACKUP_BASE" 2>/dev/null)" ]; then
    echo -e "${YELLOW}No saved states found${NC}"
    echo "States are stored in: $BACKUP_BASE"
    exit 0
  fi

  for state_dir in "$BACKUP_BASE"/*; do
    if [ -d "$state_dir" ]; then
      state=$(basename "$state_dir")

      if [ -f "$state_dir/metadata.json" ]; then
        timestamp=$(jq -r .timestamp "$state_dir/metadata.json" 2>/dev/null || echo "unknown")
        git_branch=$(jq -r .git_branch "$state_dir/metadata.json" 2>/dev/null || echo "unknown")
        git_commit=$(jq -r .git_commit "$state_dir/metadata.json" 2>/dev/null || echo "unknown")
        size=$(cat "$state_dir/.size" 2>/dev/null || echo "unknown")

        echo -e "${GREEN}$state${NC}"
        echo "  Saved:  $timestamp"
        echo "  Size:   $size"
        echo "  Git:    $git_branch @ $git_commit"
        echo ""
      else
        size=$(cat "$state_dir/.size" 2>/dev/null || du -sh "$state_dir" 2>/dev/null | cut -f1 || echo "unknown")
        echo -e "${GREEN}$state${NC}"
        echo "  Size:   $size"
        echo ""
      fi
    fi
  done

  echo "To restore a state:"
  echo "  $0 STATE_NAME [INSTANCE_IP]"
  echo ""
  exit 0
fi

if [ -z "$INSTANCE_IP" ]; then
  echo -e "${RED}Error: No instance IP provided or found${NC}"
  echo "Usage: $0 STATE_NAME [INSTANCE_IP]"
  exit 1
fi

BACKUP_DIR="$BACKUP_BASE/$STATE_NAME"

if [ ! -d "$BACKUP_DIR" ]; then
  echo -e "${RED}Error: State not found: $STATE_NAME${NC}"
  echo "Run without arguments to list available states:"
  echo "  $0"
  exit 1
fi

echo -e "${BLUE}=== Restoring Spot Instance State ===${NC}"
echo "State name:    $STATE_NAME"
echo "Source:        $BACKUP_DIR"
echo "Target:        ubuntu@$INSTANCE_IP"
echo ""

# Show state info
if [ -f "$BACKUP_DIR/metadata.json" ]; then
  timestamp=$(jq -r .timestamp "$BACKUP_DIR/metadata.json" 2>/dev/null || echo "unknown")
  git_branch=$(jq -r .git_branch "$BACKUP_DIR/metadata.json" 2>/dev/null || echo "unknown")
  git_commit=$(jq -r .git_commit "$BACKUP_DIR/metadata.json" 2>/dev/null || echo "unknown")

  echo "  Saved:       $timestamp"
  echo "  Git branch:  $git_branch @ $git_commit"
  echo ""
fi

# Wait for instance to be ready
echo -e "${YELLOW}Waiting for instance to be ready...${NC}"
for i in {1..60}; do
  if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
    ubuntu@$INSTANCE_IP exit 2>/dev/null; then
    break
  fi

  if [ $i -eq 60 ]; then
    echo -e "${RED}Error: Instance not responding after 5 minutes${NC}"
    exit 1
  fi

  echo -n "."
  sleep 5
done
echo ""
echo -e "${GREEN}✓ Instance ready${NC}"
echo ""

# Function to restore a directory
restore_dir() {
  local src="$1"
  local dst="$2"
  local desc="$3"

  if [ -d "$src" ] && [ -n "$(ls -A "$src" 2>/dev/null)" ]; then
    echo -e "${YELLOW}Restoring $desc...${NC}"

    # Create parent directory on remote
    ssh ubuntu@$INSTANCE_IP "mkdir -p $dst" 2>/dev/null || true

    rsync -az --delete --progress \
      -e "ssh -o StrictHostKeyChecking=no" \
      "$src/" \
      ubuntu@$INSTANCE_IP:$dst/ 2>&1 | grep -v "^[[:space:]]*$" || true

    echo -e "${GREEN}✓ $desc restored${NC}"
  else
    echo -e "${YELLOW}⚠ $desc not found in backup, skipping${NC}"
  fi
  echo ""
}

# Restore critical directories
restore_dir "$BACKUP_DIR/workspace" "/home/ubuntu/workspace" "workspace"
restore_dir "$BACKUP_DIR/config" "/home/ubuntu/.config" "configuration"
restore_dir "$BACKUP_DIR/ssh" "/home/ubuntu/.ssh" "SSH keys"

# Restore individual files
echo -e "${YELLOW}Restoring dotfiles...${NC}"

restore_file() {
  local file="$1"
  local desc="$2"

  if [ -f "$BACKUP_DIR/$file" ]; then
    ssh ubuntu@$INSTANCE_IP "mkdir -p $(dirname $file)" 2>/dev/null || true
    scp -q "$BACKUP_DIR/$file" ubuntu@$INSTANCE_IP:$file 2>/dev/null || true
    echo "  ✓ $desc"
  fi
}

restore_file "/home/ubuntu/.gitconfig" "Git config"
restore_file "/home/ubuntu/.bash_history" "Bash history"
restore_file "/home/ubuntu/.local/share/fish/fish_history" "Fish history"
restore_file "/home/ubuntu/.vimrc" "Vim config"
restore_file "/home/ubuntu/.tmux.conf" "Tmux config"

echo -e "${GREEN}✓ Dotfiles restored${NC}"
echo ""

# Fix permissions
echo -e "${YELLOW}Fixing permissions...${NC}"
ssh ubuntu@$INSTANCE_IP 'bash -c "chmod 700 ~/.ssh 2>/dev/null; chmod 600 ~/.ssh/* 2>/dev/null; true"'
echo -e "${GREEN}✓ Permissions fixed${NC}"
echo ""

# Summary
echo -e "${GREEN}=== Restore complete ===${NC}"
echo "State:     $STATE_NAME"
echo "Target:    ubuntu@$INSTANCE_IP"
echo ""

# Show git status on restored instance
if ssh ubuntu@$INSTANCE_IP '[ -d ~/workspace/menger/.git ]' 2>/dev/null; then
  echo -e "${BLUE}Git status on instance:${NC}"
  ssh ubuntu@$INSTANCE_IP 'cd ~/workspace/menger && git status -sb' 2>/dev/null || true
  echo ""
fi

echo -e "${BLUE}Note: Build caches not restored (will be regenerated on first build)${NC}"
echo "This speeds up restore time. First 'sbt compile' may take longer."
echo ""
