#!/bin/bash
# List all saved spot instance states with detailed information
# Usage: list-spot-states.sh [--detailed]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BACKUP_BASE="${SPOT_STATES_DIR:-$HOME/.aws/spot-states}"
DETAILED=false

# Parse arguments
if [ "$1" = "--detailed" ]; then
  DETAILED=true
fi

echo -e "${BLUE}=== Saved Spot Instance States ===${NC}"
echo "Location: $BACKUP_BASE"
echo ""

# Check if backup directory exists
if [ ! -d "$BACKUP_BASE" ]; then
  echo -e "${YELLOW}No states directory found${NC}"
  echo "States will be saved to: $BACKUP_BASE"
  echo ""
  echo "To create your first state, run:"
  echo "  scripts/backup-spot-state.sh STATE_NAME INSTANCE_IP"
  exit 0
fi

# Check for states
STATE_COUNT=0
TOTAL_SIZE=0

# Create temporary file for sorting
TMP_FILE=$(mktemp)

# Collect state information
for state_dir in "$BACKUP_BASE"/*; do
  if [ -d "$state_dir" ]; then
    state=$(basename "$state_dir")
    STATE_COUNT=$((STATE_COUNT + 1))

    # Get metadata
    if [ -f "$state_dir/metadata.json" ]; then
      timestamp=$(jq -r .timestamp "$state_dir/metadata.json" 2>/dev/null || echo "unknown")
      git_branch=$(jq -r .git_branch "$state_dir/metadata.json" 2>/dev/null || echo "unknown")
      git_commit=$(jq -r .git_commit "$state_dir/metadata.json" 2>/dev/null || echo "unknown")
      git_changed=$(jq -r .git_changed_files "$state_dir/metadata.json" 2>/dev/null || echo "0")
      instance_type=$(jq -r .instance_type "$state_dir/metadata.json" 2>/dev/null || echo "unknown")
      ami_id=$(jq -r .ami_id "$state_dir/metadata.json" 2>/dev/null || echo "unknown")
    else
      timestamp="unknown"
      git_branch="unknown"
      git_commit="unknown"
      git_changed="0"
      instance_type="unknown"
      ami_id="unknown"
    fi

    # Get size
    if [ -f "$state_dir/.size" ]; then
      size=$(cat "$state_dir/.size")
    else
      size=$(du -sh "$state_dir" 2>/dev/null | cut -f1 || echo "unknown")
    fi

    # Estimate age
    if [ "$timestamp" != "unknown" ]; then
      timestamp_epoch=$(date -d "$timestamp" +%s 2>/dev/null || echo "0")
      now_epoch=$(date +%s)
      age_seconds=$((now_epoch - timestamp_epoch))
      age_days=$((age_seconds / 86400))

      if [ $age_days -eq 0 ]; then
        age="today"
      elif [ $age_days -eq 1 ]; then
        age="1 day ago"
      elif [ $age_days -lt 7 ]; then
        age="$age_days days ago"
      elif [ $age_days -lt 30 ]; then
        age_weeks=$((age_days / 7))
        age="$age_weeks weeks ago"
      else
        age_months=$((age_days / 30))
        age="$age_months months ago"
      fi
    else
      age="unknown"
      timestamp_epoch=0
    fi

    # Store information for sorting (by timestamp, newest first)
    echo "$timestamp_epoch|$state|$timestamp|$age|$size|$git_branch|$git_commit|$git_changed|$instance_type|$ami_id" >> "$TMP_FILE"
  fi
done

# Check if any states found
if [ $STATE_COUNT -eq 0 ]; then
  echo -e "${YELLOW}No saved states found${NC}"
  echo ""
  echo "To create your first state, run:"
  echo "  scripts/backup-spot-state.sh STATE_NAME INSTANCE_IP"
  rm -f "$TMP_FILE"
  exit 0
fi

# Sort by timestamp (newest first) and display
sort -t'|' -k1 -rn "$TMP_FILE" | while IFS='|' read -r timestamp_epoch state timestamp age size git_branch git_commit git_changed instance_type ami_id; do
  # State name
  if [ "$state" = "last" ]; then
    echo -e "${CYAN}● $state${NC} ${YELLOW}(auto-saved)${NC}"
  else
    echo -e "${GREEN}● $state${NC}"
  fi

  # Basic info
  echo "  Saved:     $age ($timestamp)"
  echo "  Size:      $size"

  # Git info
  if [ "$git_branch" != "unknown" ] && [ "$git_branch" != "null" ]; then
    if [ "$git_changed" != "0" ] && [ "$git_changed" != "null" ]; then
      echo -e "  Git:       $git_branch @ $git_commit ${YELLOW}($git_changed uncommitted)${NC}"
    else
      echo "  Git:       $git_branch @ $git_commit"
    fi
  fi

  # Detailed info
  if [ "$DETAILED" = true ]; then
    if [ "$instance_type" != "unknown" ] && [ "$instance_type" != "null" ]; then
      echo "  Instance:  $instance_type"
    fi
    if [ "$ami_id" != "unknown" ] && [ "$ami_id" != "null" ]; then
      echo "  AMI:       $ami_id"
    fi
  fi

  echo ""
done

# Summary
rm -f "$TMP_FILE"

# Calculate total size
TOTAL_SIZE_BYTES=$(du -sb "$BACKUP_BASE" 2>/dev/null | cut -f1 || echo "0")
if [ $TOTAL_SIZE_BYTES -gt 0 ]; then
  TOTAL_SIZE=$(du -sh "$BACKUP_BASE" 2>/dev/null | cut -f1)
else
  TOTAL_SIZE="unknown"
fi

echo -e "${BLUE}Summary:${NC}"
echo "  Total states: $STATE_COUNT"
echo "  Total size:   $TOTAL_SIZE"
echo ""

echo -e "${BLUE}Commands:${NC}"
echo "  Restore state:    scripts/restore-spot-state.sh STATE_NAME [INSTANCE_IP]"
echo "  Backup current:   scripts/backup-spot-state.sh STATE_NAME [INSTANCE_IP]"
echo "  Cleanup old:      scripts/cleanup-spot-states.sh [--dry-run]"
echo ""

# Show detailed flag hint
if [ "$DETAILED" = false ]; then
  echo -e "${YELLOW}Tip: Use --detailed flag for more information${NC}"
  echo ""
fi
