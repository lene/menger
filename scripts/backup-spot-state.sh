#!/bin/bash
# Backup spot instance state to local storage
# Usage: backup-spot-state.sh [STATE_NAME] [INSTANCE_IP]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_BASE="${SPOT_STATES_DIR:-$HOME/.aws/spot-states}"
STATE_NAME="${1:-last}"
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

if [ -z "$INSTANCE_IP" ]; then
  echo -e "${RED}Error: No instance IP provided or found${NC}"
  echo "Usage: $0 [STATE_NAME] [INSTANCE_IP]"
  exit 1
fi

BACKUP_DIR="$BACKUP_BASE/$STATE_NAME"

echo -e "${BLUE}=== Backing up spot instance state ===${NC}"
echo "State name:    $STATE_NAME"
echo "Instance IP:   $INSTANCE_IP"
echo "Destination:   $BACKUP_DIR"
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Test SSH connectivity
echo -e "${YELLOW}Testing connection...${NC}"
if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP exit 2>/dev/null; then
  echo -e "${RED}Error: Cannot connect to instance${NC}"
  exit 1
fi
echo -e "${GREEN}✓ Connected${NC}"
echo ""

# Create exclusion file
EXCLUDE_FILE=$(mktemp)
cat > "$EXCLUDE_FILE" <<'EOF'
# Build artifacts
*.class
target/
.bsp/
.metals/
.bloop/
project/target/
project/project/

# Large caches (can be regenerated)
.ivy2/cache/
.sbt/boot/scala-*/
.cache/JetBrains/*/log/
.cache/JetBrains/*/tmp/
.local/share/Trash/

# Temporary files
*.log
*.tmp
*.swp
*.swo
*~
.nv/
/tmp/

# Node modules (if any)
node_modules/
EOF

# Function to rsync a directory
rsync_dir() {
  local src="$1"
  local dst="$2"
  local desc="$3"

  echo -e "${YELLOW}Syncing $desc...${NC}"

  if ssh ubuntu@$INSTANCE_IP "[ -d $src ]" 2>/dev/null; then
    rsync -az --delete --progress \
      --exclude-from="$EXCLUDE_FILE" \
      -e "ssh -o StrictHostKeyChecking=no" \
      ubuntu@$INSTANCE_IP:$src/ \
      "$dst/" 2>&1 | grep -v "^[[:space:]]*$" || true
    echo -e "${GREEN}✓ $desc complete${NC}"
  else
    echo -e "${YELLOW}⚠ $src not found, skipping${NC}"
  fi
  echo ""
}

# Backup critical directories
rsync_dir "/home/ubuntu/workspace" "$BACKUP_DIR/workspace" "workspace"
rsync_dir "/home/ubuntu/.config" "$BACKUP_DIR/config" "configuration"
rsync_dir "/home/ubuntu/.ssh" "$BACKUP_DIR/ssh" "SSH keys"

# Backup individual files
echo -e "${YELLOW}Backing up dotfiles...${NC}"

backup_file() {
  local file="$1"
  local desc="$2"

  if ssh ubuntu@$INSTANCE_IP "[ -f $file ]" 2>/dev/null; then
    mkdir -p "$(dirname "$BACKUP_DIR/$file")"
    scp -q ubuntu@$INSTANCE_IP:$file "$BACKUP_DIR/$file" 2>/dev/null || true
    echo "  ✓ $desc"
  fi
}

backup_file "/home/ubuntu/.gitconfig" "Git config"
backup_file "/home/ubuntu/.bash_history" "Bash history"
backup_file "/home/ubuntu/.local/share/fish/fish_history" "Fish history"
backup_file "/home/ubuntu/.vimrc" "Vim config"
backup_file "/home/ubuntu/.tmux.conf" "Tmux config"

echo -e "${GREEN}✓ Dotfiles complete${NC}"
echo ""

# Collect metadata
echo -e "${YELLOW}Collecting metadata...${NC}"

INSTANCE_TYPE=$(ssh ubuntu@$INSTANCE_IP 'curl -s http://169.254.169.254/latest/meta-data/instance-type' 2>/dev/null || echo "unknown")
AMI_ID=$(ssh ubuntu@$INSTANCE_IP 'curl -s http://169.254.169.254/latest/meta-data/ami-id' 2>/dev/null || echo "unknown")

# Get git repository information
GIT_INFO=$(ssh ubuntu@$INSTANCE_IP 'bash -s' <<'REMOTE_GIT'
cd ~/workspace/menger 2>/dev/null || exit 0
if [ -d .git ]; then
  BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
  COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
  STATUS=$(git status --porcelain 2>/dev/null | wc -l)
  echo "$BRANCH|$COMMIT|$STATUS"
fi
REMOTE_GIT
)

if [ -n "$GIT_INFO" ]; then
  IFS='|' read -r GIT_BRANCH GIT_COMMIT GIT_CHANGED <<< "$GIT_INFO"
else
  GIT_BRANCH="unknown"
  GIT_COMMIT="unknown"
  GIT_CHANGED="0"
fi

# Create metadata JSON
cat > "$BACKUP_DIR/metadata.json" <<EOF
{
  "state_name": "$STATE_NAME",
  "timestamp": "$(date -Iseconds)",
  "instance_ip": "$INSTANCE_IP",
  "instance_type": "$INSTANCE_TYPE",
  "ami_id": "$AMI_ID",
  "git_branch": "$GIT_BRANCH",
  "git_commit": "$GIT_COMMIT",
  "git_changed_files": $GIT_CHANGED,
  "backup_tool_version": "1.0"
}
EOF

echo -e "${GREEN}✓ Metadata saved${NC}"
echo ""

# Calculate backup size
echo -e "${YELLOW}Calculating backup size...${NC}"
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)
echo "$BACKUP_SIZE" > "$BACKUP_DIR/.size"

echo -e "${GREEN}✓ Backup size: $BACKUP_SIZE${NC}"
echo ""

# Cleanup
rm -f "$EXCLUDE_FILE"

# Summary
echo -e "${GREEN}=== Backup complete ===${NC}"
echo "State:     $STATE_NAME"
echo "Location:  $BACKUP_DIR"
echo "Size:      $BACKUP_SIZE"
echo "Git:       $GIT_BRANCH @ $GIT_COMMIT"
[ "$GIT_CHANGED" != "0" ] && echo -e "${YELLOW}Warning:   $GIT_CHANGED uncommitted changes${NC}"
echo ""

# Optional: Backup caches separately (user can run manually)
echo -e "${BLUE}Note: Build caches not backed up (can be regenerated)${NC}"
echo "To backup caches too (adds ~5-10 GB, 10-15 min):"
echo "  $0 $STATE_NAME-with-caches $INSTANCE_IP"
echo "  Then run: scripts/backup-spot-caches.sh $STATE_NAME-with-caches $INSTANCE_IP"
echo ""
